"""知识库存储：Chroma（向量） + rank_bm25（词法） + RRF 融合。

对标 MimirQ 的 app/storage/vector/ + app/rag/retrieval/hybrid/fusion.py。
这个文件就是你简历里"混合检索 + RRF 融合"的代码实体。
"""
from __future__ import annotations

import re
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from .chunks import split_text
from .embeddings import Embedder

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")
_RRF_K = 60  # RRF 常数，避免第一名拿满分


class Hit(dict):
    """一条检索命中的 chunk（dict 便于 JSON 序列化）。"""


class KnowledgeStore:
    def __init__(self, embedder: Embedder, *, persist_dir: str, collection: str = "knowledge", top_k: int = 5) -> None:
        self.embedder = embedder
        self.top_k = top_k
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.col = self.client.get_or_create_collection(
            collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25_chunks: list[tuple[str, str]] = []
        self._bm25: BM25Okapi | None = None

    # ---------------- 入库 ----------------
    def add_document(self, text: str, source: str) -> int:
        chunks = split_text(text, chunk_size=500, overlap=50)
        ids = [f"{source}-{i}" for i in range(len(chunks))]
        # 用 upsert：同一文件名再次上传会覆盖旧内容，不会炸 id 冲突
        self.col.upsert(
            ids=ids,
            documents=chunks,
            embeddings=self.embedder.encode(chunks),
            metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))],
        )
        self._rebuild_bm25()
        return len(chunks)

    def _rebuild_bm25(self) -> None:
        data = self.col.get(include=["documents"])
        self._bm25_chunks = list(zip(data["ids"], data["documents"]))
        tokenized = [self._bm25_terms(d) for _, d in self._bm25_chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    # ---------------- 检索 ----------------
    def search(self, query: str, *, k: int | None = None) -> list[Hit]:
        k = k or self.top_k
        vector_hits = self._search_vector(query, k)
        bm25_hits = self._search_bm25(query, k)
        return self._rrf_fuse(vector_hits, bm25_hits, k)

    def _search_vector(self, query: str, k: int) -> list[Hit]:
        qemb = self.embedder.encode([query])[0]
        res = self.col.query(query_embeddings=[qemb], n_results=max(k * 2, 3))
        hits: list[Hit] = []
        for doc_id, dist, meta, doc in zip(
            res["ids"][0], res["distances"][0], res["metadatas"][0], res["documents"][0]
        ):
            sim = max(0.0, 1.0 - float(dist))  # cosine distance -> 相似度
            hits.append(Hit(chunk_id=doc_id, text=doc, source=meta["source"], score=sim, channel="vector"))
        return hits

    def _search_bm25(self, query: str, k: int) -> list[Hit]:
        if not self._bm25 or not self._bm25_chunks:
            return []
        scores = self._bm25.get_scores(self._bm25_terms(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        hits: list[Hit] = []
        for i in order:
            if scores[i] <= 0:
                continue
            cid, doc = self._bm25_chunks[i]
            meta = self.col.get(ids=[cid], include=["metadatas"])["metadatas"][0]
            hits.append(Hit(chunk_id=cid, text=doc, source=meta["source"], score=float(scores[i]), channel="bm25"))
        return hits

    @staticmethod
    def _bm25_terms(text: str) -> list[str]:
        return _TOKEN_RE.findall(str(text).lower())

    @staticmethod
    def _rrf_fuse(vector_hits: list[Hit], bm25_hits: list[Hit], k: int) -> list[Hit]:
        """Reciprocal Rank Fusion：不信任任何单一通道的分数，按"名次"加权。"""
        fused: dict[str, dict[str, Any]] = {}
        for hits in (vector_hits, bm25_hits):
            for rank, hit in enumerate(hits):
                bucket = fused.setdefault(
                    hit["chunk_id"], {"hit": hit, "rr": 0.0, "channels": [], "scores": {}}
                )
                bucket["rr"] += 1.0 / (_RRF_K + rank + 1)
                bucket["channels"].append(hit["channel"])
                bucket["scores"][hit["channel"]] = round(hit["score"], 4)
        ordered = sorted(fused.values(), key=lambda v: v["rr"], reverse=True)[:k]
        return [
            Hit(
                chunk_id=v["hit"]["chunk_id"],
                text=v["hit"]["text"],
                source=v["hit"]["source"],
                score=round(v["rr"], 4),
                channels=v["channels"],
                scores=v["scores"],
            )
            for v in ordered
        ]

    def count(self) -> int:
        return self.col.count()