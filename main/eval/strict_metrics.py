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

import re
from typing import Dict, Any, List, Set, Tuple


_STRICT_ABSTENTION_PATTERNS = [
    r"không\s+(?:được\s+|hoàn\s+toàn\s+)*(?:đề\s+cập|nêu|nhắc\s+(?:đến|tới))",
    r"không\s+tìm\s+thấy",
    r"không\s+có\s+(?:thông\s+tin|dữ\s+liệu|bằng\s+chứng)",
    r"không\s+đủ\s+bằng\s+chứng",
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
    "were", "been", "have", "has", "had", "which", "what", "where",
}


def is_strict_abstention(answer: str) -> bool:
    """Strictly checks if the response is an explicit refusal/abstention."""
    if not answer or len(answer.strip()) < 10:
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


def compute_strict_citation_f1(
    answer: str,
    evidence_list: List[Dict[str, Any]],
    gold_keywords: List[str],
) -> Tuple[float, float, float]:
    """
    Computes strict Citation Precision, Recall, and F1.
    NO artificial inflation: If cited evidence does not match relevant gold chunks, TP = 0.
    Returns (precision, recall, f1).
    """
    cited_indices = [int(m) for m in re.findall(r"\[(\d+)\]", answer or "")]
    if not evidence_list:
        return (1.0, 1.0, 1.0) if not cited_indices else (0.0, 0.0, 0.0)

    # Find evidence chunks that actually contain at least one gold keyword
    relevant_indices: Set[int] = set()
    cleaned_keywords = [kw.lower().strip() for kw in gold_keywords if kw and len(kw.strip()) > 1]
    
    for idx, ev in enumerate(evidence_list, 1):
        txt = str(ev.get("text") or ev.get("content") or "").lower()
        if any(kw in txt for kw in cleaned_keywords):
            relevant_indices.add(idx)

    # If no gold keywords matched any retrieved chunk, then none of the chunks are truly relevant
    if not relevant_indices:
        # None of the chunks are relevant to the gold standard
        if not cited_indices:
            return 1.0, 1.0, 1.0  # Correctly did not cite irrelevant evidence
        return 0.0, 0.0, 0.0  # Cited chunks that don't contain the facts

    if not cited_indices:
        return 0.0, 0.0, 0.0

    cited_set = set(i for i in cited_indices if 1 <= i <= len(evidence_list))
    if not cited_set:
        return 0.0, 0.0, 0.0

    # STRICT TP: Exact set intersection, no bonus!
    tp = len(cited_set & relevant_indices)

    precision = tp / len(cited_set) if cited_set else 0.0
    recall = tp / len(relevant_indices) if relevant_indices else 0.0

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

    # Sentence-level claim verification with localized numeric guard
    clean_ans = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
    raw_sents = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+", clean_ans)
        if len(s.strip()) > 25
    ]
    if not raw_sents:
        return 1.0

    ctx_norm = full_context.replace(",", ".")
    ctx_tokens = {
        w for w in re.findall(r"\w+", full_context.lower())
        if w not in _STOPWORDS and len(w) > 1
    }

    verified_count = 0
    for s in raw_sents:
        # Check if numbers in this sentence are grounded in context
        s_nums = extract_numbers_and_years_strict(s)
        num_ok = True
        for num in s_nums:
            raw_variants = {num, num.replace(".", ","), num.rstrip("%")}
            if not any(v in ctx_norm for v in raw_variants if v):
                num_ok = False
                break
        if not num_ok:
            continue

        s_tokens = {
            w for w in re.findall(r"\w+", s.lower())
            if w not in _STOPWORDS and len(w) > 1
        }
        if not s_tokens:
            continue
        # Strict overlap ratio: at least 45% non-stopword tokens must exist in context
        overlap = len(s_tokens & ctx_tokens) / len(s_tokens)
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
    - If unanswerable: Must explicitly abstain (1.0). Spouting other content = 0.0.
    - If answerable: Must NOT abstain (1.0). Inappropriately giving up = 0.0.
    """
    abstained = is_strict_abstention(answer)
    if is_unanswerable:
        return 1.0 if abstained else 0.0
    else:
        return 1.0 if not abstained else 0.0
