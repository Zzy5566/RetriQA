"""FastAPI 应用入口（对标 MimirQ 的 app/main.py）。"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.chat import router as chat_router
from .api.documents import router as documents_router
from .config import settings
from .rag.embeddings import build_embedder
from .rag.graph import build_self_rag_graph
from .rag.llm import build_llm
from .rag.store import KnowledgeStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 组装"依赖三角"：embedder -> store -> graph；llm 独立
    embedder = build_embedder(settings)
    llm = build_llm(settings)
    store = KnowledgeStore(
        embedder,
        persist_dir=settings.chroma_dir,
        top_k=settings.top_k,
    )
    graph = build_self_rag_graph(store, llm, max_attempts=settings.max_attempts)
    app.state.store = store
    app.state.graph = graph
    app.state.max_attempts = settings.max_attempts
    mode = "offline (deterministic + mock)" if settings.mock_mode else "real (OpenAI-compatible)"
    print(f"[RetriQA] mode={mode}  chunks={store.count()}  graph nodes={[n for n in graph.get_graph().nodes]}")
    yield


app = FastAPI(title="RetriQA - 个人知识库 RAG 问答", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3002", "http://127.0.0.1:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")


@app.get("/")
async def root() -> dict:
    return {"service": "RetriQA", "docs": "/docs", "health": "/health"}


@app.get("/health")
async def health() -> dict:
    store = app.state.store
    return {"ok": True, "mode": "offline" if settings.mock_mode else "real", "chunks": store.count()}