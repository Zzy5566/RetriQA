"""LLM 封装（对标 MimirQ 的 app/rag/engine.py 里的 ChatOpenAI）。

离线优先：
- MockLLM：不看 prompt，直接基于检索证据拼一段"演示答案"，并逐字流出。
- OpenAICompatibleLLM：真正调用 /chat/completions。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import httpx


class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, stream: bool = False) -> str | Iterator[str]:
        """生成回答；stream=True 时返回逐块迭代器。"""


class MockLLM(LLM):
    """离线演示：把检索到的证据原样吐出几段，并提示这是 mock 模式。"""

    def generate(self, prompt: str, *, stream: bool = False) -> str | Iterator[str]:
        # prompt 里我们用特殊的占位标记注入了证据，这里简单截取引用
        answer = (
            "[Mock 模式 · 离线演示]\n"
            "这是基于检索证据生成的演示回答。配置真实 LLM 后这里将是模型原文。\n"
            "要点如下：\n"
        )
        for line in prompt.splitlines():
            if line.startswith("【证据】"):
                answer += line[:160] + "\n"
        if stream:
            return _tokenize(answer)
        return answer


class OpenAICompatibleLLM(LLM):
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, *, stream: bool = False) -> str | Iterator[str]:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "stream": stream,
        }
        if stream:
            return self._stream(url, headers, payload)
        resp = httpx.post(url, headers=headers, json={**payload, "stream": False}, timeout=120.0)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _stream(self, url: str, headers: dict, payload: dict) -> Iterator[str]:
        with httpx.stream("POST", url, headers=headers, json=payload, timeout=120.0) as r:
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = data
                try:
                    import json
                    obj = json.loads(data)
                    delta = obj["choices"][0]["delta"].get("content") or ""
                    chunk = delta
                except Exception:
                    pass
                if chunk:
                    yield chunk


def build_llm(cfg) -> LLM:
    if not cfg.mock_mode and cfg.llm_api_base and cfg.llm_api_key:
        return OpenAICompatibleLLM(base_url=cfg.llm_api_base, api_key=cfg.llm_api_key, model=cfg.llm_model)
    return MockLLM()


def _tokenize(text: str, size: int = 6) -> Iterator[str]:
    for i in range(0, len(text), size):
        yield text[i:i + size]