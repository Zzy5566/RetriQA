"""聊天接口：SSE 流式输出（对标 MimirQ 的 POST /api/v1/chat/stream）。

SSE 事件流：
  data: {"type":"start",    "question": ...}
  data: {"type":"token",    "content": ...}   # 逐块吐出答案
  data: {"type":"evidence", "citations": [...]}  # 证据（简历亮点：答案可溯源）
  data: {"type":"done",     "answer": ..., "attempts": n, "abstain": bool}
"""
from __future__ import annotations

import json
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..models import ChatRequest
from ..rag.graph import new_initial_state

router = APIRouter(prefix="/chat", tags=["chat"])

_TOKEN_SIZE = 6  # mock 模式模拟流式时每块吐几个字符


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@router.post("/stream")
async def chat_stream(request: Request, req: ChatRequest) -> StreamingResponse:
    graph = request.app.state.graph
    max_attempts = int(request.app.state.max_attempts)

    async def gen() -> AsyncGenerator[str, None]:
        yield _sse({"type": "start", "question": req.question})

        state = new_initial_state(req.question, max_attempts=max_attempts)
        result = dict(graph.invoke(state))

        answer = str(result.get("answer") or "")
        for i in range(0, len(answer), _TOKEN_SIZE):
            yield _sse({"type": "token", "content": answer[i:i + _TOKEN_SIZE]})

        yield _sse(
            {
                "type": "evidence",
                "citations": result.get("citations") or [],
            }
        )
        yield _sse(
            {
                "type": "done",
                "answer": answer,
                "attempts": int(result.get("attempts", 0)),
                "abstain": bool(result.get("abstain", False)),
                "retrieval_query_used": result.get("retrieval_query"),
            }
        )

    return StreamingResponse(gen(), media_type="text/event-stream")