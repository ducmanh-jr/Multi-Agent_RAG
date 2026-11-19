"""Check report citations against the source units already available to the Writer."""

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Callable

from core.state import AgentState, compute_context_hash


def citation_location(item: dict[str, Any]) -> tuple[str, int] | None:
    doc_id = item.get("doc_id") or item.get("document_id")
    page = item.get("page_no") if "page_no" in item else item.get("page")
    if not isinstance(doc_id, str) or not doc_id.strip():
        return None
    if doc_id.strip().lower() in {"doc", "unknown"}:
        return None
    if isinstance(page, str) and page.isdecimal():
        page = int(page)
    if type(page) is not int or page < 1:
        return None
    return doc_id.strip(), page


def evidence_quote(item: dict[str, Any]) -> str:
    quote = item.get("quote") or item.get("text")
    return quote if isinstance(quote, str) else ""


def claim_text(item: dict[str, Any]) -> str:
    text = item.get("claim_text") or item.get("claim") or item.get("text")
    return text if isinstance(text, str) else ""


def published_claim_id(item: dict[str, Any], position: int) -> str:
    claim_id = item.get("claim_id") or item.get("id")
    return claim_id if isinstance(claim_id, str) and claim_id else f"claim_{position}"


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _contains(text: str, excerpt: str) -> bool:
    if not excerpt or excerpt not in text:
        return False
    start = r"(?<!\w)" if excerpt[0].isalnum() else ""
    end = r"(?!\w)" if excerpt[-1].isalnum() else ""
    return bool(re.search(start + re.escape(excerpt) + end, text))


_NUMBER = re.compile(r"(?<!\w)[−+-]?\d+(?:[.,/\-]\d+)*(?:\s*%|(?=\s|[^\w.,/\-]|[.,/\-](?!\d)|$))")
_LOCATOR = re.compile(r"\[\d+\]|\b(?:trang|page|mục|section|hình|figure|table|bảng)\s+\d+\b", re.I)


def _numeric_literals(text: str) -> set[str]:
    return {
        re.sub(r"\s+", "", value).replace("−", "-").replace(",", ".")
        for value in _NUMBER.findall(_LOCATOR.sub(" ", text))
    }


def _checked_spans(state: AgentState) -> tuple[dict[str, dict[str, Any]], int]:
    source_index = defaultdict(list)
    for unit in state.get("source_units") or []:
        if isinstance(unit, dict) and isinstance(unit.get("id"), str):
            source_index[unit["id"], citation_location(unit)].append(_normalized(evidence_quote(unit)))

    candidates = state.get("evidence_spans") or []
    counts = Counter(span["id"] for span in candidates if isinstance(span, dict) and isinstance(span.get("id"), str))
    checked = {}
    for span in candidates:
        if not isinstance(span, dict):
            continue
        evidence_id = span.get("id")
        unit_id = span.get("source_unit_id") or span.get("unit_id")
        location = citation_location(span)
        quote = evidence_quote(span)
        if not isinstance(evidence_id, str) or not evidence_id.strip() or counts[evidence_id] != 1:
            continue
        if not isinstance(unit_id, str) or not location or not quote.strip():
            continue
        source_hash = span.get("context_hash")
        if source_hash and source_hash != compute_context_hash(quote, *location):
            continue
        if any(_contains(text, _normalized(quote)) for text in source_index.get((unit_id, location), ())):
            checked[evidence_id] = span
    return checked, len(candidates) - len(checked)


def _link_ids(item: dict[str, Any]) -> list[str]:
    links = item.get("evidence_ids")
    if not isinstance(links, list) or not links:
        links = [item.get("source_unit_id") or item.get("unit_id")]
    return [link for link in links if isinstance(link, str) and link]


def audit_claim_citations(state: AgentState) -> dict[str, Any]:
    """Resolve explicit claim links and check numeric literals within the cited quotes."""
    checked, unverified_count = _checked_spans(state)
    by_unit = defaultdict(list)
    for evidence_id, span in checked.items():
        by_unit[span.get("source_unit_id") or span.get("unit_id")].append(evidence_id)

    claims_by_text = defaultdict(list)
    for claim in state.get("claims") or []:
        if isinstance(claim, dict) and claim_text(claim):
            claims_by_text[_normalized(claim_text(claim))].append(claim)

    verified = [
        (published_claim_id(claim, position), claim)
        for position, claim in enumerate(state.get("verified_claims") or [], start=1)
        if isinstance(claim, dict)
    ]
    counts = Counter(claim_id for claim_id, _ in verified)
    claims = {}
    for claim_id, check in verified:
        text = _normalized(claim_text(check))
        status = str(check.get("status", "")).lower()
        if not text or counts[claim_id] != 1 or status not in {"", "support", "supported"}:
            continue
        links = _link_ids(check)
        if not links:
            # Some Fact Checkers assign fresh IDs. Require an unambiguous matching claim text.
            matches = claims_by_text.get(text, [])
            exact_ids = [claim for claim in matches if (claim.get("id") or claim.get("claim_id")) == claim_id]
            matching_claims = exact_ids or matches
            if len(matching_claims) == 1:
                links = _link_ids(matching_claims[0])
        if not links:
            # Legacy checkers can omit links. A verbatim match on one source page is unambiguous.
            exact_spans = {
                eid: citation_location(span) for eid, span in checked.items()
                if _contains(_normalized(evidence_quote(span)), text)
            }
            if len(set(exact_spans.values())) == 1:
                links = list(exact_spans)

        evidence_ids = []
        missing_link = False
        for link in links:
            if link in checked:
                evidence_ids.append(link)
                continue
            unit_spans = by_unit.get(link, [])
            if len(unit_spans) > 1:
                unit_spans = [eid for eid in unit_spans if _contains(_normalized(evidence_quote(checked[eid])), text)]
            if unit_spans:
                evidence_ids.extend(unit_spans)
            else:
                missing_link = True

        evidence_ids = list(dict.fromkeys(evidence_ids))
        quoted_values = set().union(*(_numeric_literals(evidence_quote(checked[eid])) for eid in evidence_ids))
        missing_values = sorted(_numeric_literals(text) - quoted_values) if evidence_ids else []
        status = "verified"
        if not evidence_ids or missing_link:
            status = "unresolved_evidence"
        elif missing_values:
            status = "missing_numeric_evidence"
        claims[claim_id] = {
            "status": status,
            "evidence_ids": evidence_ids if status == "verified" else [],
            "missing_values": missing_values,
        }
    return {"claims": claims, "unverified_evidence_count": unverified_count, "references": []}


def write_with_citation_audit(
    state: AgentState, write: Callable[[AgentState], dict[str, Any]]
) -> dict[str, Any]:
    audit = audit_claim_citations(state)
    writer_state = state.copy()
    writer_state["verified_claims"] = []
    for position, claim in enumerate(state.get("verified_claims") or [], start=1):
        if isinstance(claim, dict):
            check = audit["claims"].get(published_claim_id(claim, position), {})
            claim = {**claim, "evidence_ids": check.get("evidence_ids", [])}
        writer_state["verified_claims"].append(claim)
    result = write(writer_state)
    report_data = result.get("written_report_data") or {}
    # Only the shared renderer owns the numbering. A raw external report has no trusted mapping.
    if isinstance(report_data, dict) and report_data.get("citation_renderer") in {"academic_report", "figure_report"}:
        if report_data.get("citation_renderer") == "academic_report":
            result["answer"] = report_data["answer"]
            result["short_answer"] = report_data["answer"]
            if report_data["status"] == "INSUFFICIENT_EVIDENCE":
                result["confidence"] = "low"
        approved_ids = {eid for check in audit["claims"].values() for eid in check["evidence_ids"]}
        audit["references"] = [
            ref for ref in report_data.get("citation_references", [])
            if isinstance(ref, dict) and ref.get("evidence_id") in approved_ids
        ]
    result["reasoning_trace"] = [
        *(result.get("reasoning_trace") or []),
        {"node": "writer_node", "action": "citation_audit", "citation_audit": audit},
    ]
    return result


def get_citation_audit(trace: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(trace):
        if isinstance(event, dict) and event.get("action") == "citation_audit":
            audit = event.get("citation_audit")
            if isinstance(audit, dict):
                return audit
    return {}
