"""离线 CLI 演示：不需要起服务，直接看 LangGraph 逐节点跑。

用法（项目根目录）：
    python demo.py

会连续问两个问题，覆盖两条路径：
  1. "旷工几天会被辞退？"  -> 能检索到证据 -> relevant -> generate（常态）
  2. "盖印章流程是什么？"  -> 检索不到 -> revise 改查一次 -> 仍无 -> abstain（拒答）
"""
from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.rag.embeddings import build_embedder
from app.rag.graph import build_self_rag_graph, new_initial_state
from app.rag.llm import build_llm
from app.rag.store import KnowledgeStore

SAMPLE = Path(__file__).parent / "sample_data" / "考勤管理制度.md"
QUESTIONS = ["旷工几天会被辞退？", "盖印章的流程是什么？"]


def run_one(graph, store, question: str) -> None:
    print(f"\n【问题】{question}\n")
    state = new_initial_state(question, max_attempts=settings.max_attempts)
    # stream_mode="updates"：一次一步，亲眼看到图怎么走
    for step in graph.stream(state, stream_mode="updates"):
        for node, update in step.items():
            if node == "retrieve":
                print(f"  · [retrieve ] 捞到 {len(update.get('docs', []))} 块 (第{update.get('attempts')}轮)")
            elif node == "grade":
                print(f"  · [grade    ] 路由 -> {update.get('_route')}  (重叠词={update.get('grade_overlap')})")
            elif node == "revise":
                print(f"  · [revise   ] 查询改写为: {update.get('retrieval_query')}")
            elif node == "abstain":
                print(f"  · [abstain  ] 拒答: {update.get('answer')}")
            elif node == "generate":
                print("  · [generate ] 生成回答")
    result = dict(graph.invoke(state))
    print("-" * 50)
    print("答案：", (result.get("answer") or "")[:200].replace("\n", " "))
    print(f"检索轮数={result.get('attempts')}  abstain={result.get('abstain', False)}")
    for c in result.get("citations", [])[:2]:
        print(f"  证据[{c.get('source')}] score={c.get('score')} channels={c.get('channels')} :: {str(c.get('text'))[:36]}")


def main() -> None:
    store = KnowledgeStore(build_embedder(settings), persist_dir=settings.chroma_dir, top_k=settings.top_k)
    # 每次重灌示例文档（upsert，同文件名覆盖）
    text = SAMPLE.read_text(encoding="utf-8")
    n = store.add_document(text, source=SAMPLE.name)
    print(f"[ingest] {SAMPLE.name} -> {n} chunks, 库内共 {store.count()} 块")

    llm = build_llm(settings)
    graph = build_self_rag_graph(store, llm, max_attempts=settings.max_attempts)

    for q in QUESTIONS:
        run_one(graph, store, q)


if __name__ == "__main__":
    main()