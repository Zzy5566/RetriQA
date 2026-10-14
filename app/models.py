"""API 的 Pydantic Schema（对标 MimirQ 的 app/api/schemas/）。"""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000, description="用户问题")


class EvidenceItem(BaseModel):
    source: str
    text: str
    score: float
    channels: list[str] = Field(default_factory=list)


class ChatResult(BaseModel):
    question: str = Field(..., description="接口收到的原始问题")
    answer: str = Field(..., description="生成的答案")
    citations: list[EvidenceItem] = Field(default_factory=list)
    attempts: int = Field(description="Self-RAG 实际执行的检索轮数")
    abstain: bool = Field(description="是否因证据不足而拒答")


class DocUploadResult(BaseModel):
    doc_id: str
    chunks: int
    message: str