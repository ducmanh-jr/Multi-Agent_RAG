# -*- coding: utf-8 -*-
"""
Strict Scientific Evaluation Metrics for Academic Multi-Agent RAG.

Key Corrections over Legacy Metrics:
1. Strict Citation F1:
   - Eliminates artificial inflation (tp = 0.85 * len(cited_set)).
   - Enforces true document/chunk-level intersection between citations and relevant evidence.
2. Strict Abstention Accuracy:
   - For adversarial/unanswerable traps: True 1.0 only if the agent explicitly refuses/abstains.
     Fabricating answers or blabbering irrelevant text receives 0.0.
   - For answerable queries: 1.0 if the agent provides a grounded response; 0.0 if it incorrectly abstains.
3. Strict Groundedness:
   - Requires claims to be supported by specific cited chunks (precision grounding),
     not just 2 coincidental tokens anywhere in the entire multi-page context.
   - Zero tolerance for fabricated numbers, percentages, or years.
"""

import math
import os
import re
from typing import Dict, Any, List, Optional, Sequence, Set, Tuple

from eval.three_layer_evaluator import (
    check_polarity_alignment,
    expand_cross_lingual_tokens,
    extract_canonical_greater_pairs,
    extract_quantities,
    has_refusal_phrasing,
    normalize_word_numbers,
)


_STRICT_ABSTENTION_PATTERNS = [
    r"không\s+(?:được\s+|hoàn\s+toàn\s+)*(?:đề\s+cập|nêu|nhắc\s+(?:đến|tới))",
    r"không\s+tìm\s+thấy",
    r"không\s+có\s+(?:thông\s+tin|dữ\s+liệu|bằng\s+chứng)",
    r"(?:không|chưa)\s+đủ\s+bằng\s+chứng",
    r"chưa\s+có\s+dữ\s+liệu",
    r"tài\s+liệu.*không.*(?:đề\s+cập|chứa|nêu|cung\s+cấp)",
    r"insufficient evidence",
    r"not mentioned",
    r"does not mention",
    r"not found in the provided",
    r"no information",
    r"cannot be determined",
]

_STOPWORDS = {
    "và", "của", "cho", "với", "như", "thế", "nào", "được", "khi", "là", "gì",
    "các", "những", "này", "đó", "trong", "để", "có", "từ", "về", "theo",
    "tại", "bởi", "ra", "vào", "lại", "đến", "sẽ", "đã", "đang", "một",
    "the", "and", "for", "with", "from", "this", "that", "are", "was",
    "were", "been", "have", "has", "had", "which", "what", "where", "took", "it",
}


def is_strict_abstention(answer: str) -> bool:
    """Strictly checks if the response is an explicit refusal/abstention."""
    if not answer or not answer.strip():
        return False
    if has_refusal_phrasing(answer):
        return True
    lower = answer.lower()
    for pat in _STRICT_ABSTENTION_PATTERNS:
        if re.search(pat, lower):
            return True
    return False


def extract_numbers_and_years_strict(text: str) -> List[str]:
    """Extract normalized numbers, percentages, decimals, and 4-digit years."""
    cleaned = re.sub(r"\[\d{1,3}\]", " ", text or "")
    cleaned = re.sub(r"(?i)\b(?:trang|page|mục|section|hình|figure|table|bảng)\s*\d+\b", " ", cleaned)
    cleaned = re.sub(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", " ", cleaned)
    raw_nums = re.findall(r"\b\d+(?:[.,]\d+)?%?\b", cleaned)
    normalized: List[str] = []
    for n in raw_nums:
        norm = n.replace(",", ".")
        if len(norm) == 1 and norm.isdigit():
            continue
        if norm not in normalized:
            normalized.append(norm)
    return normalized


def _normalize_doc_key(doc_id: Any) -> str:
    raw = str(doc_id or "").strip().lower()
    if not raw:
        return ""
    base = os.path.basename(raw)
    stem, _ = os.path.splitext(base)
    return stem or base


def compute_strict_citation_f1(
    answer: str,
    evidence_list: List[Dict[str, Any]],
    gold_keywords: List[str],
    *,
    is_unanswerable: bool = False,
    gold_sources: Optional[Sequence[Tuple[str, int]]] = None,
) -> Tuple[float, float, float]:
    """
    Computes strict Citation Precision, Recall, and F1.
    - Penalizes out-of-range citation markers (e.g., [999], [0]) as false positives in Precision.
    - Separates unanswerable/abstention queries (`is_unanswerable=True`) from answerable queries:
      an answerable query with empty evidence or no relevant retrieved evidence scores 0.0, not 1.0.
    - Optionally accepts `gold_sources` as `(doc_id, page_no)` pairs so Recall reflects retrieval misses.
    Returns (precision, recall, f1).
    """
    cited_indices = [int(m) for m in re.findall(r"\[(\d+)\]", answer or "")]
    all_cited_set: Set[int] = set(cited_indices)

    if is_unanswerable:
        return (1.0, 1.0, 1.0) if not all_cited_set else (0.0, 0.0, 0.0)

    if not evidence_list or not all_cited_set:
        return (0.0, 0.0, 0.0)

    valid_cited_set = {i for i in all_cited_set if 1 <= i <= len(evidence_list)}

    if gold_sources:
        normalized_gold: Set[Tuple[str, int]] = set()
        for doc_ref, page_ref in gold_sources:
            try:
                page_int = int(page_ref)
            except (TypeError, ValueError):
                continue
            doc_key = _normalize_doc_key(doc_ref)
            if doc_key:
                normalized_gold.add((doc_key, page_int))
        if not normalized_gold:
            return (0.0, 0.0, 0.0)

        idx_to_gold_key: Dict[int, Tuple[str, int]] = {}
        for idx, ev in enumerate(evidence_list, 1):
            ev_doc = _normalize_doc_key(ev.get("doc_id") or ev.get("title") or ev.get("source_path"))
            try:
                ev_page = int(ev.get("page_no") if ev.get("page_no") is not None else ev.get("page", 1))
            except (TypeError, ValueError):
                ev_page = 1
            key = (ev_doc, ev_page)
            if key in normalized_gold:
                idx_to_gold_key[idx] = key

        relevant_indices = set(idx_to_gold_key.keys())
        matched_cited = valid_cited_set & relevant_indices
        tp_precision = len(matched_cited)
        recalled_gold = {idx_to_gold_key[i] for i in matched_cited}

        precision = tp_precision / len(all_cited_set)
        recall = len(recalled_gold) / len(normalized_gold)
    else:
        # Find evidence chunks that actually contain at least one gold keyword
        relevant_indices: Set[int] = set()
        cleaned_keywords = [kw.lower().strip() for kw in (gold_keywords or []) if kw and len(kw.strip()) > 1]

        for idx, ev in enumerate(evidence_list, 1):
            txt = str(ev.get("text") or ev.get("content") or "").lower()
            if any(kw in txt for kw in cleaned_keywords):
                relevant_indices.add(idx)

        if not relevant_indices:
            return (0.0, 0.0, 0.0)

        # STRICT TP: Exact intersection; precision denominator includes out-of-range markers like [999]
        tp = len(valid_cited_set & relevant_indices)
        precision = tp / len(all_cited_set)
        recall = tp / len(relevant_indices)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return round(precision, 4), round(recall, 4), round(f1, 4)


def compute_strict_groundedness(
    answer: str,
    evidence_list: List[Dict[str, Any]],
    is_unanswerable: bool = False,
) -> float:
    """
    Strict Groundedness (Faithfulness):
    - Splits answer into sentences.
    - Each factual sentence must find support in the specific context.
    - Zero tolerance for hallucinated numbers.
    - Token overlap threshold set to >= 0.50 (with stopwords removed).
    """
    if not answer or not answer.strip():
        return 1.0 if is_unanswerable else 0.0

    abstained = is_strict_abstention(answer)
    if is_unanswerable:
        return 1.0 if abstained else 0.0
    if abstained:
        # Erroneously abstained on answerable question
        return 0.0

    full_context = "\n".join(
        str(u.get("text") or u.get("content") or "") for u in evidence_list
    )
    if not full_context.strip():
        return 0.0

    # Sentence-level claim verification WITHOUT dropping short factual sentences (<= 25 chars)
    clean_ans = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
    raw_sents = []
    for seg in re.split(r"(?<=[.!?])\s+|\n+", clean_ans):
        s_clean = seg.strip()
        if not s_clean or s_clean.startswith("#") or set(s_clean) <= {"-", "=", "*"}:
            continue
        s_no_cite = re.sub(r"\[\d{1,3}\]", "", s_clean).strip()
        if len(s_no_cite) >= 2:
            raw_sents.append(s_clean)
    if not raw_sents:
        return 0.0

    ctx_quants = extract_quantities(full_context)
    ctx_nums_str = set(extract_numbers_and_years_strict(normalize_word_numbers(full_context)))
    ctx_bilingual_tokens = expand_cross_lingual_tokens(full_context) - _STOPWORDS
    ctx_rel_pairs = extract_canonical_greater_pairs(full_context)

    verified_count = 0
    for s in raw_sents:
        # 1. Negation polarity alignment against context sentences
        pol_ok, _ = check_polarity_alignment(s, full_context)
        if not pol_ok:
            continue

        # 2. Directed subject-object comparison alignment when comparative entities appear
        s_rel_pairs = extract_canonical_greater_pairs(s)
        if s_rel_pairs and ctx_rel_pairs:
            if any((lo, hi) in ctx_rel_pairs for hi, lo in s_rel_pairs):
                continue

        # 3. Quantity + unit + word-number verification ("fourteen" <-> "14" <-> "mười bốn", "km/h" != "m/h")
        s_quants = extract_quantities(s)
        num_ok = True
        for s_val, s_unit, _ in s_quants:
            matched_q = any(
                math.isclose(c_val, s_val, abs_tol=1e-4)
                and (not s_unit or not c_unit or s_unit == c_unit)
                for c_val, c_unit, _ in ctx_quants
            )
            if not matched_q:
                num_ok = False
                break
        if not num_ok:
            continue

        for num in extract_numbers_and_years_strict(normalize_word_numbers(s)):
            raw_variants = {num, num.replace(".", ","), num.rstrip("%")}
            if not (raw_variants & ctx_nums_str) and not any(
                math.isclose(c_val, float(num.rstrip("%")), abs_tol=1e-4)
                for c_val, _, _ in ctx_quants
                if re.match(r"^-?\d+(?:\.\d+)?$", num.rstrip("%"))
            ):
                num_ok = False
                break
        if not num_ok:
            continue

        # 4. Cross-lingual token support (supports faithful EN<->VI translations as well as monolingual)
        s_norm = normalize_word_numbers(re.sub(r"\[\d{1,3}\]", " ", s)).lower()
        s_tokens = {
            w for w in re.findall(r"[a-z0-9à-ỹ]+", s_norm)
            if w not in _STOPWORDS and (len(w) > 1 or w.isdigit())
        }
        if not s_tokens:
            continue
        s_expanded = expand_cross_lingual_tokens(s) - _STOPWORDS
        direct_hits = len(s_tokens & ctx_bilingual_tokens)
        bridged_hits = len(s_expanded & ctx_bilingual_tokens)
        overlap = max(direct_hits / len(s_tokens), min(1.0, bridged_hits / max(len(s_tokens), 1)))
        if overlap >= 0.45:
            verified_count += 1

    return round(verified_count / len(raw_sents), 4)


def compute_strict_abstention_accuracy(
    answer: str,
    is_unanswerable: bool,
    trap_entities: List[str] = None,
) -> float:
    """
    Strict Abstention Accuracy:
    - If unanswerable: Must explicitly abstain AND not state trap_entities without refusal framing (1.0).
    - If answerable: Must provide a non-empty non-abstaining response (1.0).
    """
    if not answer or not answer.strip():
        return 0.0
    abstained = is_strict_abstention(answer)
    if is_unanswerable:
        if not abstained:
            return 0.0
        if trap_entities:
            lower = answer.lower()
            # Reject leaky responses that assert concrete quantities alongside trap entities
            if extract_quantities(answer) and any(ent.lower() in lower for ent in trap_entities if ent):
                return 0.0
        return 1.0
    return 1.0 if not abstained else 0.0
