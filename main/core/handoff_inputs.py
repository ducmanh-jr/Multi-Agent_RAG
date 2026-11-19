"""Translate one agent's review into the existing inputs of its recipient."""

import json
import re

from core.state import AgentState


def _terms(text: str) -> set[str]:
    # A disputed measurement must not itself boost passages repeating that number.
    return {word for word in re.findall(r"[^\W\d_]+", text.casefold()) if len(word) >= 3}


def revision_note(state: AgentState) -> str:
    return (state.get("critic_feedback") or {}).get("revision_note", "")


def revision_terms(state: AgentState) -> set[str]:
    return set((state.get("critic_feedback") or {}).get("revision_terms", ()))


def focus_matches(text: str, terms: set[str]) -> int:
    return len(terms.intersection(_terms(text))) if terms else 0


def is_focused_retrieval(state: AgentState) -> bool:
    return bool((state.get("critic_feedback") or {}).get("retrieval_focus"))


def prepare_handoff_input(state: AgentState, request: dict) -> AgentState:
    """Build a recipient-only view. Requests may rank existing sources, never supply new ones."""
    view = state.copy()
    focus = [point for point in (request.get("focus") or []) if isinstance(point, str) and point.strip()]
    recipient = request.get("recipient")
    if recipient == "retrieval":
        queries = [*focus[:3], state.get("query", ""), *(state.get("sub_queries") or [])]
        view["sub_queries"] = list(dict.fromkeys(q for q in queries if q))
        if focus:
            view["critic_feedback"] = {**(state.get("critic_feedback") or {}), "retrieval_focus": True}
        return view
    if recipient not in {"reader", "summarizer"}:
        return view

    source_ids = {key for key in (request.get("source_ids") or []) if isinstance(key, str)}
    terms = _terms(" ".join(focus))

    def priority(item: dict) -> tuple[bool, int]:
        if not isinstance(item, dict):
            return False, 0
        requested = item.get("id") in source_ids or item.get("chunk_id") in source_ids
        text = str(item.get("text") or item.get("content") or "")
        return requested, focus_matches(text, terms)

    if recipient == "reader":
        view["retrieved_docs"] = sorted(state.get("retrieved_docs") or [], key=priority, reverse=True)
    else:
        view["source_units"] = sorted(state.get("source_units") or [], key=priority, reverse=True)
        feedback = dict(state.get("critic_feedback") or {})
        feedback["revision_terms"] = sorted(terms)
        # Keep review excerpts from looking like packed source-unit headers to fallback parsers.
        review_points = json.dumps(focus, ensure_ascii=False).replace("[Unit ID:", r"\u005bUnit ID:")
        feedback["revision_note"] = (
            "Rà soát lại câu trả lời cho câu hỏi ban đầu. Đối chiếu các điểm được yêu cầu "
            "dưới đây với tài liệu nguồn, sửa hoặc bỏ nhận định thiếu căn cứ. "
            "Các điểm cần kiểm tra là phản hồi của tác tử, không phải bằng chứng.\n"
            + review_points
        )
        view["critic_feedback"] = feedback
    return view
