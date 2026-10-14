"""RetriQA 配置（对标 MimirQ 的 app/core/config.py）。"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 离线优先：true = deterministic 嵌入 + Mock LLM，零 key 可跑
    mock_mode: bool = True

    # LLM（OpenAI 兼容）
    llm_api_base: str = "https://api.siliconflow.cn/v1"
    llm_api_key: str = ""
    llm_model: str = "Qwen/Qwen3-32B"

    # Embedding（留空复用 LLM）
    embedding_api_base: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "BAAI/bge-m3"

    # RAG
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    max_attempts: int = 2
    chroma_dir: str = "./chroma_data"

    # 服务
    host: str = "127.0.0.1"
    port: int = 8100


settings = Settings()