"""Self-RAG 编排图（对标 MimirQ 的 app/rag/pipelines/langgraph.py）。

这是你简历项目的核心技术点 —— 用 LangGraph 的条件边（conditional edges）
实现"检索 → 打分 → 决定走生成 / 改查重试 / 拒答"的自反思循环：

    retrieve ──> grade ──(相关)──────> generate ──> END
                  │
                  ├──(证据不足&可重试)──> revise ──> retrieve（回环，有界）
                  └──(证据不足&已达上限)──> abstain ──> END   ← grounding guard

关键点：
1. 状态（RAGState）在所有节点间流动 —— 这是 LangGraph 与普通函数链的本质区别。
2. 有界循环：attempts 达到 max_attempts 就不再 revise，绝不无限重试。
3. 拒答（abstain）：检索不到证据时宁可说"不知道"，这是 RAG 防幻觉的经典设计。
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from .llm import LLM
from .store import KnowledgeStore

ABSTAIN_MESSAGE = "未能在知识库中检索到足够的相关资料，无法回答该问题。"


class RAGState(TypedDict, total=False):
    question: str              # 用户原始问题
    retrieval_query: str       # 实际拿去检索的查询（revise 后可能被改写）
    docs: list[dict[str, Any]]      # 命中的 chunks
    citations: list[dict[str, Any]] # 供前端展示的证据
    answer: str
    attempts: int              # 已执行的检索轮数
    max_attempts: int          # 有界重试上限
    abstain: bool
    abstain_reason: str
    grade_overlap: int | None
    _route: str                # grade 节点的路由指令（不给业务层看，故带下划线）


def _query_terms(query: str) -> list[str]:
    """把查询切成检索用词：英文词（>=3） + 中文二元组。

    中文按单字切会混入大量"的/是/什么"等噪音，二元组才有区分度。
    """
    text = str(query).lower()
    terms: list[str] = []
    for m in re.finditer(r"[a-zA-Z0-9_]{3,}", text):
        terms.append(m.group(0))
    cjk_runs = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    for run in cjk_runs:
        terms.extend(run[i:i + 2] for i in range(len(run) - 1))
    return terms


def _relevant_overlap(query: str, doc: str) -> int:
    """粗糙的相关性：查询与证据共有多少个词/二元组。
    真实项目会用 LLM 或重排器；这里刻意保持可读，便于理解图的结构。"""
    return len(set(_query_terms(query)) & set(_query_terms(doc)))


def _broaden_query(state: RAGState) -> str:
    """revise 节点：从已捞到的证据里抽出高频词，扩展查询，扩大下一轮召回。"""
    old = str(state.get("retrieval_query") or state.get("question") or "").strip()
    docs = state.get("docs") or []
    if not old:
        return old
    if not docs:
        return old
    freq = Counter(_query_terms(docs[0].get("text", "")))
    extra = [term for term, _ in freq.most_common(3) if term not in _query_terms(old)][:2]
    candidate = f"{old} {' '.join(extra)}".strip()
    return candidate if candidate else old


def _build_prompt(state: RAGState) -> str:
    lines = [f"问题：{state.get('question')}", "", "以下是检索到的相关资料：", ""]
    for i, c in enumerate(state.get("citations") or [], start=1):
        lines.append(f"【证据{i}】(来源:{c.get('source')}) {c.get('text')}")
        lines.append("")
    lines.append("请仅依据上述证据回答，并标注引用了哪条证据。若无法回答则明确说明。")
    return "\n".join(lines)


def build_self_rag_graph(store: KnowledgeStore, llm: LLM, *, max_attempts: int = 2) -> Any:
    max_attempts = max(1, int(max_attempts))

    def retrieve(state: RAGState) -> dict[str, Any]:
        """节点1：混合检索（向量 + BM25 + RRF 融合）。"""
        query = str(state.get("retrieval_query") or state.get("question"))
        docs = store.search(query, k=5)
        citations = [
            {
                "source": d.get("source"),
                "text": d.get("text"),
                "score": d.get("score"),
                "channels": d.get("channels", []),
            }
            for d in docs
        ]
        return {
            "docs": docs,
            "citations": citations,
            "attempts": int(state.get("attempts", 0)) + 1,
        }

    def grade(state: RAGState) -> dict[str, Any]:
        """节点2（条件路由的心脏）：评估证据质量，写 _route 指令。

        关键点：相关性用【原问题】判定，而不是检索改写后的查询 ——
        否则 revise 把证据里的词塞进查询后，相关性就会"自证有罪"。
        """
        docs = state.get("docs") or []
        attempts = int(state.get("attempts", 0))
        question = str(state.get("question") or state.get("retrieval_query") or "")

        if not docs:
            # 一张都没捞到：还能改查就改查，否则拒答
            return {"_route": "insufficient" if attempts < max_attempts else "abstain"}

        overlaps = [_relevant_overlap(question, str(d.get("text", ""))) for d in docs]
        best = max(overlaps) if overlaps else 0
        if best > 0:
            return {"_route": "relevant", "grade_overlap": best}
        # 有证据但都不相关：还有次数就 revise；已达上限则拒答（不拿不相关证据硬编）
        return {"_route": "insufficient" if attempts < max_attempts else "abstain", "grade_overlap": best}

    def generate(state: RAGState) -> dict[str, Any]:
        """节点3：用证据构建 prompt，交给 LLM 生成。"""
        prompt = _build_prompt(state)
        return {"answer": str(llm.generate(prompt, stream=False))}

    def revise(state: RAGState) -> dict[str, Any]:
        """节点4：改写查询，回去重新检索。"""
        return {"retrieval_query": _broaden_query(state)}

    def abstain(state: RAGState) -> dict[str, Any]:
        """节点5：grounding guard —— 无证据则拒答，不编造。"""
        return {"answer": ABSTAIN_MESSAGE, "abstain": True, "abstain_reason": "no_evidence"}

    def decide_route(state: RAGState) -> str:
        return str(state.get("_route") or "insufficient")

    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("generate", generate)
    g.add_node("revise", revise)
    g.add_node("abstain", abstain)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    # ★ 条件边：一条边按运行时结果分三条路
    g.add_conditional_edges(
        "grade",
        decide_route,
        {"relevant": "generate", "insufficient": "revise", "abstain": "abstain"},
    )
    g.add_edge("revise", "retrieve")
    g.add_edge("generate", END)
    g.add_edge("abstain", END)

    return g.compile()


def new_initial_state(question: str, *, max_attempts: int) -> RAGState:
    return RAGState(
        question=question,
        retrieval_query=question,
        attempts=0,
        max_attempts=max_attempts,
    )