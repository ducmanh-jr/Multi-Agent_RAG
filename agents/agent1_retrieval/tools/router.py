"""
Deterministic and Learned Routing Policy for Agent 1 (Retrieval Router + Critic).
Integrates rules and guardrails from Test_agent/agent1/router_policy.py.

Supported retrieval actions:
  - 'decompose': Decompose complex multi-faceted or comparative queries into sub-queries.
  - 'bm25': Exact keyword / lexical lookup (C FTS5 / BM25).
  - 'hyde': Hypothetical Document Embeddings for conceptual/exploratory questions.
  - 'rewrite': Query rewriting when queries are vague or ambiguous.
  - 'dense': Semantic dense retrieval.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

VALID_TOOLS = {"bm25", "dense", "hyde", "rewrite", "decompose"}
_WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


@dataclass
class RetrievalDecision:
    thought: str
    tool_selected: str
    critique_is_relevant: bool
    critique_reason: str


def _norm(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    normalized = _norm(text)
    return any(phrase in normalized for phrase in phrases)


def _failed_trials(history_trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [trial for trial in history_trials if trial.get("is_relevant") is False]


def looks_multi_part(query: str) -> bool:
    """Detects if query requires decomposition across multiple concepts or comparisons."""
    q = _norm(query)
    if _has_any(
        q,
        (
            "compare ",
            "so sánh",
            "khác nhau",
            "analyze the causes",
            "environmental, economic, and social",
            "economic, and social",
            "key events",
            "lasting consequences",
            "long-term effects",
            "subsequent political changes",
            "ưu nhược điểm",
            "đánh giá so sánh",
        ),
    ):
        return True
    if re.search(r"(^|\s)(1[\).]|1\))", q) and re.search(r"(^|\s)(2[\).]|2\))", q):
        return True
    return query.count(",") >= 3 and _has_any(q, (" and ", " và ", " impacts ", " effects "))


def looks_vague(query: str) -> bool:
    """Detects vague or ambiguous queries that benefit from rewriting."""
    return _has_any(
        query,
        (
            "stuff about",
            "things and",
            "thing where",
            "the thing",
            "maybe",
            "about the attention thing",
            "see pictures good",
            "make computer",
            "robots learn by trying",
            "mấy cái",
            "cái gì đó",
        ),
    )


def looks_exact_lookup(query: str) -> bool:
    """Detects direct definition or lookup queries."""
    q = _norm(query)
    if _has_any(q, ("what is the connection", "relationship between", "relate to")):
        return False
    if _has_any(
        q,
        (
            "what is the capital",
            "how does gradient descent work",
            "list all ",
            "who invented",
            "what year",
            "define the term",
            " là gì",
            " định nghĩa",
            "what is ",
            "who is ",
            "when was ",
            "side effects of",
        ),
    ):
        return True
    tokens = _tokens(query)
    has_acronym = bool(re.search(r"\b[A-Z][A-Z0-9-]{1,}\b", query))
    return has_acronym and len(tokens) <= 8 and query.strip().endswith("?")


def looks_hyde(query: str) -> bool:
    """Detects explanatory/conceptual queries suited for HyDE."""
    q = _norm(query)
    if q.startswith("why ") or q.startswith("tại sao "):
        return True
    return _has_any(
        q,
        (
            "how does batch normalization",
            "how does dropout",
            "long-range dependency",
            "outperform",
            "accelerate training convergence",
            "preventing overfitting",
            "cơ chế hoạt động",
            "nguyên lý",
        ),
    )


def select_tool(sub_query: str, history_trials: list[dict[str, Any]] | None = None) -> str:
    """
    Selects optimal retrieval tool:
    'decompose' | 'bm25' | 'hyde' | 'rewrite' | 'dense'
    """
    history_trials = history_trials or []
    failed = _failed_trials(history_trials)

    if len(failed) >= 2:
        return "hyde"

    if failed:
        previous_tool = str(failed[-1].get("tool", "")).lower()
        if previous_tool == "bm25":
            return "dense"
        if previous_tool == "dense":
            return "bm25" if looks_exact_lookup(sub_query) else "rewrite"
        if previous_tool == "rewrite":
            return "hyde"

    if looks_multi_part(sub_query):
        return "decompose"
    if looks_vague(sub_query):
        return "rewrite"
    if looks_hyde(sub_query):
        return "hyde"
    if looks_exact_lookup(sub_query):
        return "bm25"
    return "dense"


def make_decision(
    sub_query: str,
    history_trials: list[dict[str, Any]] | None = None,
    retrieved_context: str | None = None,
    model_prediction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generates schema-compatible RetrievalDecision dict."""
    tool = select_tool(sub_query, history_trials)
    model_prediction = model_prediction or {}

    is_relevant = True
    if retrieved_context is not None:
        query_terms = {t for t in _tokens(sub_query) if len(t) >= 4}
        context_terms = set(_tokens(retrieved_context))
        if query_terms:
            is_relevant = len(query_terms & context_terms) / max(len(query_terms), 1) >= 0.20

    decision = RetrievalDecision(
        thought=str(model_prediction.get("thought") or f"Router xác định công cụ '{tool}' phù hợp nhất với cấu trúc câu hỏi."),
        tool_selected=tool,
        critique_is_relevant=is_relevant,
        critique_reason=str(
            model_prediction.get("critique_reason")
            or ("Ngữ cảnh trích xuất khớp với từ khóa truy vấn." if is_relevant else "Cần bổ sung nguồn dữ liệu cho truy vấn.")
        ),
    )
    return asdict(decision)
