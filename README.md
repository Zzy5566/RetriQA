# RetriQA — 个人知识库 RAG 问答（学习/简历项目）

基于 **FastAPI + LangGraph + ChromaDB** 的最小可用 RAG 问答服务，
结构对标 MimirQ，但刻意精简到"一个能讲清楚、能跑通、能扩展"的规模。

## 亮点（简历可写）

- **离线优先**：内置 deterministic 嵌入 + Mock LLM，零 API Key 即可跑通整条
  "上传 → 切块 → 入库 → 混合检索 → Self-RAG 生成"链路。
- **混合检索 + RRF 融合**：向量通道（Chroma）+ 词法通道（BM25），按名次融合。
- **LangGraph Self-RAG 条件边**：`retrieve → grade → generate | revise | abstain`，
  有界纠错循环 + 无证据拒答（grounding guard）。
- **SSE 流式问答**：`tokens + evidence` 事件，前端可做打字机效果 + 证据卡片。
- 接真实模型只需填 `.env`（OpenAI 兼容，如硅基流动）。

## 目录结构

```
retriqa/
├── app/
│   ├── main.py          # FastAPI 入口：组装 store/graph
│   ├── config.py        # pydantic-settings 配置
│   ├── models.py        # API Schema
│   ├── rag/
│   │   ├── embeddings.py  # 嵌入（离线/真实两套实现 + 工厂）
│   │   ├── chunks.py      # 递归切块
│   │   ├── store.py       # Chroma + BM25 + RRF 融合（混合检索）
│   │   ├── llm.py         # LLM（Mock / OpenAI 兼容）
│   │   └── graph.py       # ★ LangGraph Self-RAG 条件边图
│   └── api/
│       ├── documents.py   # POST /api/v1/documents/upload
│       └── chat.py        # POST /api/v1/chat/stream (SSE)
├── demo.py              # 离线 CLI 演示（推荐先跑这个理解图）
├── run.py               # 启动服务
└── sample_data/考勤管理制度.md
```

## 快速开始（离线，零 Key）

```bash
# 1) 建环境（建议 python 3.11/3.12）
conda create -n retriqa python=3.12 -y
conda activate retriqa
pip install -r requirements.txt          # 国内可加 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 2) 看 LangGraph 逐节点怎么跑（推荐先做这个）
python demo.py

# 3) 起服务
copy .env.example .env     # 保持 MOCK_MODE=true
python run.py              # http://127.0.0.1:8100/docs

# 4) 体验
# 上传文档：
curl -X POST -F "file=@sample_data/考勤管理制度.md" http://127.0.0.1:8100/api/v1/documents/upload

# 流式问答（SSE）。注意：Windows PowerShell 直接 -d 中文会变 GBK 乱码，
# 先把请求体写成 UTF-8 文件再发（macOS/Linux 可直接 -d '{"question":"..."}'）：
{
  '{"question":"旷工几天会被辞退？"}' | Out-File -Encoding utf8 body.json
  curl -N -X POST -H "Content-Type: application/json" --data-binary "@body.json" `
    http://127.0.0.1:8100/api/v1/chat/stream
}
```

## 接真实模型

编辑 `.env`：

```ini
MOCK_MODE=false
LLM_API_BASE=https://api.siliconflow.cn/v1
LLM_API_KEY=sk-xxxx
LLM_MODEL=Qwen/Qwen3-32B
EMBEDDING_API_BASE=https://api.siliconflow.cn/v1   # 留空则复用 LLM
EMBEDDING_API_KEY=sk-xxxx
EMBEDDING_MODEL=BAAI/bge-m3
```

> 换 embedding 后，旧的 `chroma_data/` 索引作废，请删除后重建（和 MimirQ 的提示一致）。

## 学习对照表（这个项目 vs MimirQ）

| 概念 | RetriQA | MimirQ 可参考的实现 |
|---|---|---|
| 嵌入 | `rag/embeddings.py` | `app/rag/embedding/providers/deterministic_test.py` |
| 切块 | `rag/chunks.py` | `app/parsing/chunking/` |
| 混合检索+融合 | `rag/store.py` | `app/rag/retriever.py` + `app/rag/retrieval/hybrid/fusion.py` |
| LangGraph | `rag/graph.py` | `app/rag/pipelines/langgraph.py` |
| SSE | `api/chat.py` | `app/api/v1/chat.py` |