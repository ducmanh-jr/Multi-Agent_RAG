# -*- coding: utf-8 -*-
"""
Core Evaluation Metrics for the 120-Question Bilingual Benchmark.
Measures:
1. Groundedness (Faithfulness): Verified claims / Total claims
2. Citation F1: Precision and Recall of [n] citations against ground-truth evidence
3. Numeric/Date Exactness: Exact match of numbers/dates in answer vs context & gold
4. Abstention Accuracy: Correct refusal/abstention on unanswerable adversarial traps
5. Latency (p50, p95)
6. Gemini Quota Usage (API calls per query)
"""

import re
from pathlib import PurePosixPath
from typing import Dict, Any, List
from core.citation_audit import get_citation_audit
from core.verifier import (
    check_numerical_and_date_consistency,
    deterministic_verify_claims,
    verify_evidence_span_provenance,
)
from eval.three_layer_evaluator import (
    REFUSAL_STATUSES,
    audit_abstention_and_display,
    evaluate_three_layer_case,
)


_ABSTENTION_PATTERNS = [
    r"không tìm thấy",
    r"không đề cập",
    r"không có thông tin",
    r"không đủ bằng chứng",
    r"không được nêu",
    r"không được đề cập",
    r"chưa có dữ liệu",
    r"insufficient evidence",
    r"not mentioned",
    r"does not mention",
    r"not found in the provided",
    r"no information",
    r"cannot be determined",
]


def is_abstention_response(answer: str) -> bool:
    """Detect whether the agent abstained or flagged missing information."""
    if not answer:
        return True
    lower = answer.lower()
    for pat in _ABSTENTION_PATTERNS:
        if re.search(pat, lower):
            return True
    return False


def _document_key(value: Any) -> str:
    name = PurePosixPath(str(value or "").replace("\\", "/")).name.casefold()
    return name.removesuffix(".pdf")


def _location(item: Dict[str, Any]) -> tuple[str, str]:
    document = item.get("doc_id") or item.get("document_id") or item.get("filename")
    return _document_key(document), str(item.get("page_no") or item.get("page") or "")


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


def _matches_target(item: Dict[str, Any], target: Dict[str, Any], *, citation: bool = False) -> bool:
    document, page = _location(item)
    if document != _document_key(target["document"]) or page != str(target["page"]):
        return False
    text = (item.get("quote") if citation else item.get("text")) or item.get("text") or item.get("content") or ""
    passage = target.get("passage")
    if passage:
        return _normalized_text(passage) in _normalized_text(text)
    keywords = (target.get("keywords") or target.get("anchors") or []) if citation else []
    return not keywords or any(keyword.casefold() in text.casefold() for keyword in keywords if keyword)


def _answer_citation_numbers(report: str) -> set[int]:
    # Report reference definitions and the verification ledger are outside the answer body.
    body = re.split(
        r"<!-- DUAL_REPORT_SPLIT -->|^\s*#{1,6}\s+(?:\d+\.\s*)?(?:Danh mục Tài liệu Trích dẫn|Evidence References|References|Sources)\b",
        report, maxsplit=1, flags=re.IGNORECASE | re.MULTILINE,
    )[0]
    body = re.sub(r"^\s*```[^\n]*\n.*?^\s*```[^\n]*", "", body, flags=re.MULTILINE | re.DOTALL)
    body = re.sub(r"^\s*#{1,6}\s+[^\n]*|`[^`\n]*`", "", body, flags=re.MULTILINE)
    return {int(marker) for marker in re.findall(r"(?<!\\)\[(\d+)\]", body)}


def _score_citations(
    result_state: Dict[str, Any], report: str, targets: List[Dict[str, Any]],
) -> float | None:
    if not targets:
        return None
    cited_numbers = _answer_citation_numbers(report)
    if not cited_numbers:
        return 0.0

    spans = result_state.get("evidence_spans") or []
    audit = get_citation_audit(result_state.get("reasoning_trace") or [])
    report_data = result_state.get("written_report_data") or {}
    references = audit.get("references", []) if audit else report_data.get("references", [])
    if audit:
        spans_by_id = {span.get("id"): span for span in spans if span.get("id")}
        candidates = {
            str(ref["ref_id"]): [span]
            for ref in references
            if ref.get("ref_id") is not None
            and (span := spans_by_id.get(ref.get("evidence_id"))) and _location(span) == _location(ref)
        }
    else:
        # The legacy Writer emitted the preview of the evidence span it assigned the ref to.
        candidates = {}
        for ref in references:
            preview = _normalized_text(ref.get("preview") or "")
            if not preview or ref.get("ref_id") is None:
                continue
            matching = [
                span for span in spans
                if _location(span) == _location(ref)
                and _normalized_text(span.get("quote") or span.get("text") or "").startswith(preview)
            ]
            if len(matching) == 1:
                candidates[str(ref["ref_id"])] = matching

    correct_numbers = 0
    covered_targets = set()
    for number in cited_numbers:
        matched = {
            index for index, target in enumerate(targets)
            if any(_matches_target(span, target, citation=True) for span in candidates.get(str(number), []))
        }
        if matched:
            correct_numbers += 1
            covered_targets.update(matched)
    if not correct_numbers:
        return 0.0
    precision = correct_numbers / len(cited_numbers)
    recall = len(covered_targets) / len(targets)
    return round(2 * precision * recall / (precision + recall), 4)


def _audit_citations_and_provenance(
    result_state: Dict[str, Any],
    report: str,
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Audit inline [n] citation markers and verify evidence span provenance against SourceUnits."""
    spans = [s for s in (result_state.get("evidence_spans") or []) if isinstance(s, dict)]
    audit = get_citation_audit(result_state.get("reasoning_trace") or [])
    report_data = result_state.get("written_report_data") or {}
    references = audit.get("references", []) if audit else report_data.get("references", [])

    # 1. Verify evidence span provenance using PR-02 verifier (supports PR-01 merged provenance).
    provenance_rate: float | None = None
    if spans and evidence:
        valid_spans = sum(1 for span in spans if verify_evidence_span_provenance(span, evidence)[0])
        provenance_rate = round(valid_spans / len(spans), 4)

    # 2. Detect orphan/fabricated inline [n] markers in the answer body.
    cited_numbers = _answer_citation_numbers(report)
    known_ref_ids = {
        int(ref["ref_id"])
        for ref in references
        if isinstance(ref, dict) and ref.get("ref_id") is not None and str(ref.get("ref_id")).isdigit()
    }
    orphan_markers = sorted(n for n in cited_numbers if n not in known_ref_ids)
    valid_citation_ratio: float | None = None
    if cited_numbers:
        valid_citation_ratio = round((len(cited_numbers) - len(orphan_markers)) / len(cited_numbers), 4)

    return {
        "citation_provenance_rate": provenance_rate,
        "orphan_citation_count": len(orphan_markers),
        "valid_citation_ratio": valid_citation_ratio,
    }


def evaluate_single_prediction(
    question_item: Dict[str, Any],
    result_state: Dict[str, Any],
    latency_sec: float,
) -> Dict[str, Any]:
    """Evaluate a single benchmark prediction against ground-truth metadata."""
    category = question_item.get("category", "single_hop")
    is_unanswerable = bool(question_item.get("is_unanswerable", False)) or category in {
        "adversarial_trap",
        "insufficient_evidence",
    }
    gold_numbers = list(question_item.get("gold_numbers") or [])
    if not gold_numbers and question_item.get("gold_values"):
        gold_numbers = [
            str(item["value"])
            for item in question_item.get("gold_values", [])
            if isinstance(item, dict) and item.get("value") is not None
        ]
    selected_doc = question_item.get("selected_doc", "")
    targets = [] if is_unanswerable else question_item.get("gold_sources", [])
    if not targets and not is_unanswerable and question_item.get("gold_page") and selected_doc:
        targets = [{
            "document": selected_doc,
            "page": question_item["gold_page"],
            "keywords": question_item.get("gold_keywords", []),
        }]

    report_data = result_state.get("written_report_data") or {}
    answer = (
        report_data.get("answer")
        or result_state.get("answer")
        or result_state.get("final_response")
        or ""
    )
    status = result_state.get("answer_status") or report_data.get("status")
    abstention_check = audit_abstention_and_display(
        answer=answer,
        answer_status=status,
        answerability="unanswerable" if is_unanswerable else str(question_item.get("answerability") or "answerable"),
        forbidden_entities=question_item.get("forbidden_entities") or (),
        forbidden_numbers=question_item.get("forbidden_numbers") or (),
    )
    if status:
        norm_st = str(status).strip().upper()
        abstained = (norm_st in REFUSAL_STATUSES) and not abstention_check["status_display_mismatch"]
    else:
        abstained = bool(answer and is_abstention_response(answer)) and not abstention_check["status_display_mismatch"]
    abstention_acc = float(bool(answer or abstained) and abstained == is_unanswerable and not abstention_check["status_display_mismatch"])
    evidence = (
        result_state.get("source_units")
        or result_state.get("retrieved_docs")
        or result_state.get("filtered_context")
        or result_state.get("retrieved_chunks")
        or []
    )
    full_context = "\n".join(
        f"{u.get('doc_id', '')} Trang {u.get('page_no', 1)}: {str(u.get('text') or u.get('content') or '')}"
        for u in evidence
    )

    # 1. Groundedness & Hallucination Rate
    state_claims = result_state.get("claims") or []
    if not state_claims and answer and not abstained:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if len(s.strip()) > 25]
        state_claims = [{"id": f"c_{i+1}", "claim": s} for i, s in enumerate(sents[:6])]

    verified_c = 0
    rejected_c = 0
    total_c = 0
    if abstained:
        # An explicit abstention makes no factual claim (false refusals are penalized via abstention_accuracy).
        groundedness = 1.0
        hallucination_rate = 0.0
    elif state_claims and evidence:
        verified, rejected, _ = deterministic_verify_claims(state_claims, evidence)
        verified_c = len(verified)
        rejected_c = len(rejected)
        total_c = verified_c + rejected_c
        groundedness = verified_c / total_c if total_c > 0 else 1.0
        hallucination_rate = rejected_c / total_c if total_c > 0 else 0.0
        if is_unanswerable:
            # Answering an unanswerable trap instead of abstaining is a hallucination.
            hallucination_rate = 1.0
            groundedness = 0.0
    elif is_unanswerable:
        groundedness = 0.0
        hallucination_rate = 1.0
    else:
        groundedness = 1.0 if result_state.get("fact_check_label") == "support" else 0.5
        hallucination_rate = round(1.0 - groundedness, 4)

    # 2. Numeric/Date, Compound-Unit & Derived-Claim Exactness
    three_layer = evaluate_three_layer_case(question_item, result_state)
    if isinstance(question_item.get("derivation_spec"), dict):
        derived_res = three_layer.get("derived_audit") or {}
        correct_numbers = bool(answer and not abstained and derived_res.get("passed"))
        consistency_issues = [] if derived_res.get("passed") else [str(derived_res.get("reason") or "derived_mismatch")]
    else:
        num_ok, date_ok, consistency_issues = check_numerical_and_date_consistency(answer, full_context)
        numeric_answer = re.sub(r"\[\d+\]|\b(?:trang|page)\s+\d+\b", " ", answer, flags=re.IGNORECASE)
        numeric_answer = numeric_answer.replace(",", ".")
        gold_present = all(
            re.search(r"(?<!\d)(?<!\d[.,])" + re.escape(str(number).replace(",", ".")) + r"(?!\d|[.,]\d)",
                      numeric_answer)
            for number in gold_numbers
        )
        correct_numbers = bool(answer and not abstained and num_ok and date_ok and gold_present)
    numeric_exactness = None if is_unanswerable else float(correct_numbers)

    # 3. Retrieval, Citation F1, and Citation Provenance Audit
    retrieved = result_state.get("seen_chunks") or result_state.get("retrieved_docs") or []
    retrieval_recall = (
        sum(any(_matches_target(chunk, target) for chunk in retrieved) for target in targets) / len(targets)
        if targets else None
    )
    rendered_report = result_state.get("final_report") or answer
    citation_f1 = _score_citations(result_state, rendered_report, targets)
    citation_audit_metrics = _audit_citations_and_provenance(result_state, rendered_report, evidence)

    return {
        "id": question_item.get("id"),
        "category": category,
        "language": question_item.get("language", "vi"),
        "selected_doc": selected_doc,
        "documents": question_item.get("documents") or ([selected_doc] if selected_doc else []),
        "groundedness": round(groundedness, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "unsupported_claim_count": rejected_c,
        "total_claim_count": total_c,
        "citation_f1": citation_f1,
        "citation_provenance_rate": citation_audit_metrics["citation_provenance_rate"],
        "orphan_citation_count": citation_audit_metrics["orphan_citation_count"],
        "valid_citation_ratio": citation_audit_metrics["valid_citation_ratio"],
        "unit_mismatch_count": len(consistency_issues),
        "retrieval_recall": round(retrieval_recall, 4) if retrieval_recall is not None else None,
        "numeric_exactness": numeric_exactness,
        "abstention_accuracy": round(abstention_acc, 4),
        "three_layer_passed": bool(three_layer["passed"]),
        "three_layer_failures": list(three_layer["failures"]),
        "latency_sec": round(latency_sec, 3),
        "retry_count": int(result_state.get("retry_count", 0)),
        "critic_loops": int(result_state.get("critic_iterations", 0)),
        "answer_preview": (answer[:180] + "...") if len(answer) > 180 else answer,
    }
