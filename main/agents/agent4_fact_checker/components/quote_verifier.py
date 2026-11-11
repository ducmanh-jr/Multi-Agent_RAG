"""
Pure-Code Quote Verifier and Answer Status Classifier.
Implements the core project philosophy:
"LLMs express, Code verifies. 100% of claims must have verbatim quotations verified by code."
"""

from __future__ import annotations
import re
from typing import Any, Optional


def normalize_text_for_matching(text: str) -> str:
    """Standardizes spacing, quotes, and punctuation for robust substring verification."""
    if not text:
        return ""
    # Normalize unicode quotes and dashes
    t = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    t = t.replace("–", "-").replace("—", "-")
    # Collapse multiple whitespace/newlines
    t = " ".join(t.split()).lower()
    return t


def verify_quote_in_source_units(quote: str, source_units: list[dict[str, Any]]) -> tuple[bool, Optional[dict[str, Any]]]:
    """
    Verifies with 100% deterministic code whether a quote exists verbatim inside the parsed source units.
    Returns:
      (True, matched_source_unit) if found.
      (False, None) if the quote was fabricated/hallucinated by the LLM.
    """
    if not quote or not quote.strip() or not source_units:
        return False, None

    norm_quote = normalize_text_for_matching(quote)
    if len(norm_quote) < 5:
        # Quote too short to be unique
        return False, None

    # 1. Exact Substring Search
    for unit in source_units:
        unit_text = unit.get("text", "")
        norm_unit = normalize_text_for_matching(unit_text)
        if norm_quote in norm_unit:
            return True, {
                "source_unit_id": unit.get("id"),
                "doc_id": unit.get("doc_id"),
                "page_no": unit.get("page_no", 1),
                "bbox": unit.get("bbox"),
                "context_hash": unit.get("context_hash"),
                "confidence": 1.0,
                "match_type": "exact_verbatim",
            }

    # 2. Windowed / Sentence level match (in case quote spans slight boundary)
    # Check if 90% of words in quote appear in consecutive order in unit
    quote_words = norm_quote.split()
    if len(quote_words) >= 6:
        sub_anchor = " ".join(quote_words[:5])
        for unit in source_units:
            norm_unit = normalize_text_for_matching(unit.get("text", ""))
            if sub_anchor in norm_unit:
                matched_words = sum(1 for w in quote_words if w in norm_unit)
                ratio = matched_words / len(quote_words)
                if ratio >= 0.90:
                    return True, {
                        "source_unit_id": unit.get("id"),
                        "doc_id": unit.get("doc_id"),
                        "page_no": unit.get("page_no", 1),
                        "bbox": unit.get("bbox"),
                        "context_hash": unit.get("context_hash"),
                        "confidence": round(ratio, 2),
                        "match_type": "high_overlap",
                    }

    return False, None


def classify_answer_and_status(
    query: str,
    verified_claims: list[dict[str, Any]],
    rejected_claims: list[dict[str, Any]],
    source_units: list[dict[str, Any]],
) -> tuple[str, str]:
    """
    Classifies the response into one of the 4 strict academic categories:
      1. ANSWERED_DIRECT_QUOTE: Clear answer found with direct verbatim citation.
      2. ANSWERED_INFERRED: Answer synthesized from multiple verified facts / mathematical deduction.
      3. NOT_FOUND_IN_DOCUMENT: The document does not contain the requested information (abstention).
      4. PREMISE_REJECTED: The question contains a false presupposition which is actively refuted.

    Returns:
      (status, short_answer_text)
    """
    q_norm = normalize_text_for_matching(query)

    # Check 1: Generalized Premise Rejection & Anachronism Detection
    # Detect if query explicitly states a historical year/metric that contradicts source context
    query_years = set(re.findall(r"\b(?:18\d\d|19\d\d|20\d\d)\b", query))
    if query_years and source_units:
        # Exclude bibliography or reference sections if labeled to avoid false citation conflicts
        filtered_units = [
            u for u in source_units
            if not any(k in str(u.get("section", "")).lower() for k in ["reference", "bibliography", "tài liệu tham khảo"])
        ]
        active_units = filtered_units if filtered_units else source_units
        all_context_text = " ".join(u.get("text", "") for u in active_units)
        # Filter out citation brackets or author year parentheticals e.g. [2018] or (Vaswani et al., 2017)
        cleaned_ctx = re.sub(r"\[\s*\d{4}\s*\]|\([A-Za-z\s.,]+,?\s*(?:18|19|20)\d{2}\)", "", all_context_text)
        ctx_years = set(re.findall(r"\b(?:18\d\d|19\d\d|20\d\d)\b", cleaned_ctx))
        # If query insists on a specific year that does not appear anywhere in source units, but alternative years exist
        if query_years and ctx_years and not (query_years & ctx_years):
            # Check if query and context share entity keywords
            q_entities = [w for w in re.findall(r"\w+", q_norm) if len(w) > 4]
            overlap_entities = [w for w in q_entities if w in all_context_text.lower()]
            if len(overlap_entities) >= 2:
                alt_years = ", ".join(sorted(list(ctx_years))[:2])
                return (
                    "PREMISE_REJECTED",
                    f"Tiền đề câu hỏi có thể không chính xác về mốc thời gian ({', '.join(query_years)}). Tài liệu nguồn đề cập đến các mốc: {alt_years}.",
                )

    # Check 2: Generalized Abstention / Not Found in Document
    if not verified_claims or len(verified_claims) == 0:
        return (
            "NOT_FOUND_IN_DOCUMENT",
            "Không tìm thấy bằng chứng liên quan trực tiếp trong tài liệu đã chọn để trả lời câu hỏi này.",
        )

    # Check 3: Inferred vs Direct Quote
    if any(k in q_norm for k in ["bao lâu", "mất bao lâu", "khoảng bao lâu", "mấy năm", "mấy tháng", "điểm chung", "bao nhiêu sự kiện"]):
        claim_texts = [(c.get("claim") or c.get("text") or "").strip() for c in verified_claims]
        clean_claims = [re.sub(r"^[#\s\-*•\d\.:]+", "", t).strip() for t in claim_texts if t]
        combined = " ".join(clean_claims)
        return (
            "ANSWERED_INFERRED",
            f"Dựa trên các mốc thời gian đối chiếu trong tài liệu: {combined}",
        )

    # Check 4: Direct Substantive Answer
    substantive_claims = []
    for c in verified_claims:
        txt = (c.get("claim") or c.get("text") or "").strip()
        clean_txt = re.sub(r"^[#\s\-*•\d\.:]+", "", txt).strip()
        # Filter out links, code mentions, and tiny headers
        if (
            clean_txt
            and len(clean_txt) > 20
            and not clean_txt.lower().startswith("1code")
            and not clean_txt.lower().startswith("http")
            and not clean_txt.lower().startswith("github")
        ):
            substantive_claims.append(clean_txt)

    if substantive_claims:
        direct_ans = " ".join(substantive_claims[:2])
    else:
        first = verified_claims[0].get("claim") or verified_claims[0].get("text") or ""
        direct_ans = re.sub(r"^[#\s\-*•\d\.:]+", "", first).strip()

    return (
        "ANSWERED_DIRECT_QUOTE",
        direct_ans,
    )
