"""文档入库接口。"""
from __future__ import annotations

from fastapi import APIRouter, File, Request, UploadFile

from ..models import DocUploadResult

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocUploadResult)
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="文本文件（.md/.txt）"),
) -> DocUploadResult:
    raw = await file.read()
    # 优先按 UTF-8 解码，失败退回 GBK（国内文档常见）
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("gbk", errors="ignore")

    store = request.app.state.store
    doc_id = file.filename or f"doc-{len(raw)}"
    chunks = store.add_document(text, source=doc_id)
    return DocUploadResult(doc_id=doc_id, chunks=chunks, message=f"入库成功，切为 {chunks} 块")


@router.get("/count")
async def document_count(request: Request) -> dict:
    store = request.app.state.store
    return {"chunks": store.count()}