"""Embedding：把文本变成向量（对标 MimirQ 的 app/rag/embedding/）。

离线优先设计：
- DeterministicEmbedder：纯哈希词法向量，零依赖零 key，用于跑通链路。
- OpenAICompatibleEmbedder：真实语义嵌入，配置 key 后启用。
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from abc import ABC, abstractmethod

import httpx

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

MODEL_NAME = "retriqa-deterministic-v1"


class Embedder(ABC):
    """所有嵌入实现的统一接口。"""

    dim: int

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        """把一批文本编码成向量。"""


class DeterministicEmbedder(Embedder):
    """离线确定性词法哈希嵌入。

    原理（抄自 MimirQ 的 deterministic_test）：
    每个词 / 字符 n-gram 哈希到 256 个桶里，加正负权重。
    好处：确定性、可离线、能跑通"入库→检索"管道。
    缺点：没有语义，只做字面匹配 —— 这就是为什么它只用于开发模式。
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _add_feature(self, vector: list[float], feature: str, weight: float) -> None:
        digest = hashlib.blake2b(f"{MODEL_NAME}:{feature}".encode("utf-8", "ignore"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % len(vector)
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign * weight

    def _encode_one(self, text: str) -> list[float]:
        norm = unicodedata.normalize("NFKC", str(text or "")).casefold()[:4096]
        vector = [0.0] * self.dim
        features = 0

        for token in _WORD_RE.findall(norm):
            self._add_feature(vector, f"w:{token}", 2.0)
            features += 1

        # 对中文特别重要：字符二元组 / 三元组，让"旷工"和"连续旷工"字面相近
        compact = "".join(ch for ch in norm if not ch.isspace())
        for width, weight in ((2, 0.35), (3, 0.5)):
            for i in range(max(0, len(compact) - width + 1)):
                self._add_feature(vector, f"c:{width}:{compact[i:i + width]}", weight)
                features += 1

        if features == 0:
            self._add_feature(vector, "empty", 1.0)
        return vector

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(t) for t in texts]


class OpenAICompatibleEmbedder(Embedder):
    """真正的语义嵌入（OpenAI 兼容 /embeddings 端点）。

    演示：LLM_API_BASE = https://api.siliconflow.cn/v1，
          模型可用 BAAI/bge-m3（1024 维）。
    """

    def __init__(self, base_url: str, api_key: str, model: str, dim: int = 1024) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        resp = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in data]


def build_embedder(cfg) -> Embedder:
    """工厂：mock 模式或没配 key -> 离线；配了就上真模型。"""
    if not cfg.mock_mode and cfg.embedding_api_base and cfg.embedding_api_key:
        return OpenAICompatibleEmbedder(
            base_url=cfg.embedding_api_base or cfg.llm_api_base,
            api_key=cfg.embedding_api_key or cfg.llm_api_key,
            model=cfg.embedding_model,
        )
    return DeterministicEmbedder(dim=cfg.embedding_dim if hasattr(cfg, "embedding_dim") else 256)