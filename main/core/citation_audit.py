"""Check report citations and post-Writer prose against audited source units and verified claims."""

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Callable, Iterator

from core.claim_semantics import (
    check_math_and_domain_consistency,
    check_sentence_entailment_guards,
    expand_unicode_scripts_preserving_role,
    extract_structured_quantities,
    infer_and_bind_derived_claim,
    normalize_math_preserving_structure,
    verify_derived_claim,
)
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


def _math_normalized(text: str) -> str:
    """Normalizes PDF ligatures and Unicode math symbols while keeping ^ (superscript) vs _ (subscript) distinct."""
    return normalize_math_preserving_structure(text)


def _contains(text: str, excerpt: str) -> bool:
    if not excerpt:
        return False
    if excerpt in text:
        start = r"(?<!\w)" if excerpt[0].isalnum() else ""
        end = r"(?!\w)" if excerpt[-1].isalnum() else ""
        if re.search(start + re.escape(excerpt) + end, text):
            return True
    m_text = _math_normalized(text)
    m_excerpt = _math_normalized(excerpt)
    if not m_excerpt or m_excerpt not in m_text:
        return False
    start = r"(?<!\w)" if m_excerpt[0].isalnum() else ""
    end = r"(?!\w)" if m_excerpt[-1].isalnum() else ""
    return bool(re.search(start + re.escape(m_excerpt) + end, m_text))


_NUMBER = re.compile(r"(?<!\w)[−+-]?\d+(?:[.,/\-]\d+)*(?:\s*%|(?=\s|[^\w.,/\-]|[.,/\-](?!\d)|$))")
_SCRIPT_LITERAL_RE = re.compile(r"(?<=[A-Za-zΑ-Ωα-ω0-9\)\}])([\^_])\s*\{?\s*([−+-]?\d+)\s*\}?")
_LOCATOR = re.compile(
    r"\[\d+\]|\b(?:trang|page|mục|section|hình|figure|fig\.?|table|tab\.?|bảng|"
    r"equation|eq\.?|formula|công\s+thức|phương\s+trình|biểu\s+thức)\s*\(?\d+(?:\.\d+)*\)?(?!\w)",
    re.I,
)


def _strip_inline_math_dollars(text: str) -> str:
    out = re.sub(r"\$\$([\s\S]*?)\$\$", r" \1 ", text or "")
    out = re.sub(
        r"(?<!\\)\$([^\$\n]+?)\$",
        lambda m: f" {m.group(1)} " if not re.match(r"^\s*\d+(?:[.,]\d+)?\s*$", m.group(1)) else m.group(0),
        out,
    )
    return out


def _numeric_literals(text: str) -> set[str]:
    """
    Extracts numeric literals, dates, percentages, and explicit superscript (`^2`) vs subscript (`_2`)
    numeric indices so `x_2 = 4` is never accepted against `x^2 = 4`, while `x²` matches `x^2`.
    """
    cleaned = _LOCATOR.sub(" ", _strip_inline_math_dollars(text or ""))
    cleaned = expand_unicode_scripts_preserving_role(cleaned)
    script_tokens = {
        f"{m.group(1)}{m.group(2).replace('−', '-')}"
        for m in _SCRIPT_LITERAL_RE.finditer(cleaned)
    }
    without_scripts = _SCRIPT_LITERAL_RE.sub(" ", cleaned)
    plain_tokens = {
        re.sub(r"\s+", "", value).replace("−", "-").replace(",", ".")
        for value in _NUMBER.findall(without_scripts)
    }
    return plain_tokens | script_tokens


def _span_source_matches(state: AgentState) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]]]]:
    source_index = defaultdict(list)
    for unit in state.get("source_units") or []:
        if isinstance(unit, dict) and isinstance(unit.get("id"), str):
            text = _normalized(evidence_quote(unit))
            source_index[unit["id"], citation_location(unit)].append((text, unit))

    candidates = state.get("evidence_spans") or []
    counts = Counter(span["id"] for span in candidates if isinstance(span, dict) and isinstance(span.get("id"), str))
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
        excerpt = _normalized(quote)
        matches = [
            unit for text, unit in source_index.get((unit_id, location), ()) if _contains(text, excerpt)
        ]
        if matches:
            yield span, matches


def citation_source_units(state: AgentState, evidence_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Recover the exact Reader unit behind each requested, unambiguous citation."""
    if not evidence_ids:
        return {}
    return {
        span["id"]: matches[0] for span, matches in _span_source_matches(state)
        if span["id"] in evidence_ids and all(unit == matches[0] for unit in matches)
    }


def _link_ids(item: dict[str, Any]) -> list[str]:
    links = item.get("evidence_ids")
    if not isinstance(links, list) or not links:
        links = [item.get("source_unit_id") or item.get("unit_id")]
    return [link for link in links if isinstance(link, str) and link]


def audit_claim_citations(state: AgentState) -> dict[str, Any]:
    """
    Resolve explicit claim links, verify numeric literals & math scripts within cited quotes,
    check sentence-level polarity / comparative direction, and deterministically validate
    `derived_claim` calculations (`92% - 81% = 11 pp`).
    """
    checked = {span["id"]: span for span, _ in _span_source_matches(state)}
    unverified_count = len(state.get("evidence_spans") or []) - len(checked)
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
        status_raw = str(check.get("status", "")).lower()
        if not text or counts[claim_id] != 1 or status_raw not in {"", "support", "supported"}:
            continue
        links = _link_ids(check)
        matched_summary_claim: dict[str, Any] | None = None
        matches = claims_by_text.get(text, [])
        exact_ids = [claim for claim in matches if (claim.get("id") or claim.get("claim_id")) == claim_id]
        matching_claims = exact_ids or matches
        if len(matching_claims) == 1:
            matched_summary_claim = matching_claims[0]
            if not links:
                links = _link_ids(matched_summary_claim)
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
                narrowed = [eid for eid in unit_spans if _contains(_normalized(evidence_quote(checked[eid])), text)]
                if not narrowed:
                    from utils.bilingual import expand_bilingual_tokens

                    c_toks = {w for w in re.findall(r"[a-zA-Z0-9À-ỹ]+", text.lower()) if len(w) >= 3}
                    c_exp = expand_bilingual_tokens(text, c_toks)
                    scored_eids = sorted(
                        unit_spans,
                        key=lambda eid: len(c_exp & {w for w in re.findall(r"[a-zA-Z0-9À-ỹ]+", evidence_quote(checked[eid]).lower()) if len(w) >= 3}),
                        reverse=True,
                    )
                    best_ov = len(c_exp & {w for w in re.findall(r"[a-zA-Z0-9À-ỹ]+", evidence_quote(checked[scored_eids[0]]).lower()) if len(w) >= 3})
                    if best_ov > 0:
                        narrowed = [
                            eid for eid in scored_eids[:2]
                            if len(c_exp & {w for w in re.findall(r"[a-zA-Z0-9À-ỹ]+", evidence_quote(checked[eid]).lower()) if len(w) >= 3}) >= max(1, best_ov - 1)
                        ]
                unit_spans = narrowed or unit_spans
            if unit_spans:
                evidence_ids.extend(unit_spans)
            else:
                missing_link = True

        evidence_ids = list(dict.fromkeys(evidence_ids))
        cited_spans = [checked[eid] for eid in evidence_ids if eid in checked]
        combined_quotes = "\n".join(evidence_quote(sp) for sp in cited_spans)

        quoted_values = set().union(*(_numeric_literals(evidence_quote(sp)) for sp in cited_spans)) if cited_spans else set()
        missing_values = sorted(_numeric_literals(text) - quoted_values) if evidence_ids else []

        derivation_spec = (
            check.get("derivation")
            or check.get("derivation_spec")
            or ((matched_summary_claim or {}).get("derivation") or (matched_summary_claim or {}).get("derivation_spec"))
        )
        is_explicit_derived = (
            isinstance(derivation_spec, dict)
            or check.get("claim_type") == "derived_claim"
            or (matched_summary_claim or {}).get("claim_type") == "derived_claim"
        )
        derived_audit_info: dict[str, Any] | None = None

        if cited_spans and (is_explicit_derived or missing_values):
            candidate_deriv = (
                derivation_spec
                if isinstance(derivation_spec, dict)
                else infer_and_bind_derived_claim(text, cited_spans)
            )
            if isinstance(candidate_deriv, dict):
                d_res = verify_derived_claim(text, candidate_deriv, cited_spans)
                derived_audit_info = d_res
                if d_res.get("passed"):
                    # Only clear the derived number itself; any script mismatch like _2 vs ^2 still blocks!
                    script_missing = [mv for mv in missing_values if mv.startswith(("^", "_"))]
                    missing_values = script_missing
                elif is_explicit_derived and not missing_values:
                    missing_values = [str(d_res.get("reason") or "invalid_derivation")]

        status = "verified"
        if not evidence_ids or missing_link:
            status = "unresolved_evidence"
        elif cited_spans:
            sem_ok, _, _ = check_sentence_entailment_guards(text, combined_quotes)
            math_ok, _ = check_math_and_domain_consistency(text, combined_quotes)
            if not sem_ok or not math_ok:
                status = "semantic_or_math_conflict"
            elif missing_values:
                status = "missing_numeric_evidence"

        entry: dict[str, Any] = {
            "status": status,
            "evidence_ids": evidence_ids if status == "verified" else [],
            "missing_values": missing_values,
        }
        if derived_audit_info is not None and derived_audit_info.get("passed") and status == "verified":
            entry["claim_type"] = "derived_claim"
            entry["computed_value"] = derived_audit_info.get("computed_value")
            entry["expected_unit"] = derived_audit_info.get("expected_unit")
        claims[claim_id] = entry

    return {"claims": claims, "unverified_evidence_count": unverified_count, "references": []}


def _verify_post_writer_prose(
    published_answer: str,
    *,
    approved_claims: list[dict[str, Any]],
    approved_spans: list[dict[str, Any]],
    allowed_ref_ids: set[int],
) -> tuple[bool, str]:
    """
    Post-Writer Deterministic Gate:
    Verifies every sentence emitted by Writer (`answer` / `short_answer` / Executive Summary)
    against the approved `verified_claims` and `evidence_spans`.
    Blocks any hallucinated numbers (e.g., `999 layers` when claim says `12 layers`), unit changes,
    negation flips, comparative subject swaps (`A > B` -> `B > A`), subscript/superscript edits,
    or unauthorized `[k]` citation markers.
    """
    if not published_answer or not published_answer.strip():
        return False, "empty_writer_answer"

    # 1. Verify all [k] citation markers belong to allowed_ref_ids
    cited_nums = {
        int(m.group(1))
        for m in re.finditer(r"(?<!\\)\[(\d+)\](?!\()", published_answer)
    }
    if cited_nums - allowed_ref_ids:
        return False, f"unauthorized_citation_markers:{sorted(cited_nums - allowed_ref_ids)}"

    approved_claim_texts = [claim_text(c) for c in approved_claims if claim_text(c).strip()]
    approved_quotes = [evidence_quote(s) for s in approved_spans if evidence_quote(s).strip()]
    approved_corpus = "\n".join([*approved_claim_texts, *approved_quotes])

    allowed_literals: set[str] = set()
    for txt in [*approved_claim_texts, *approved_quotes]:
        allowed_literals |= _numeric_literals(txt)

    allowed_quantities = [
        q
        for txt in [*approved_claim_texts, *approved_quotes]
        for q in extract_structured_quantities(txt)
    ]

    # Unescape literal citations \[2] before checking numeric literals so source markers aren't flagged
    unescaped_ans = published_answer.replace(r"\[", "[").replace(r"\]", "]")
    # Strip fenced code blocks when splitting prose sentences
    prose_only = re.sub(r"```[\s\S]*?```|`[^`\n]+`", " ", unescaped_ans)

    ans_literals = _numeric_literals(prose_only)
    extra_literals = sorted(ans_literals - allowed_literals)
    if extra_literals:
        return False, f"unapproved_numeric_or_script_literals:{extra_literals}"

    # Check unit consistency on every extracted quantity in Writer prose
    for qv, qu, raw_str in extract_structured_quantities(prose_only):
        if qu in {"%", "pp", "km/h", "m/h", "m/s", "km", "m", "cm", "mm", "ms", "s", "gb", "mb", "kg", "g", "l", "ml", "day", "layer"}:
            if not any(
                math.isclose(qv, av, abs_tol=0.05) and (qu == au or (not au and qu == "layer"))
                for av, au, _ in allowed_quantities
            ):
                return False, f"unapproved_quantity_or_unit:{raw_str}{qu}"

    # Sentence-by-sentence semantic polarity, comparative direction & math AST check
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+", prose_only)
        if len(s.strip()) >= 8
    ]
    for sent in sentences:
        sem_ok, _, sem_reason = check_sentence_entailment_guards(sent, approved_corpus)
        if not sem_ok:
            return False, f"writer_semantic_drift:{sem_reason}"
        math_ok, math_reason = check_math_and_domain_consistency(sent, approved_corpus)
        if not math_ok:
            return False, f"writer_math_drift:{math_reason}"

    return True, "ok"


def write_with_citation_audit(
    state: AgentState, write: Callable[[AgentState], dict[str, Any]]
) -> dict[str, Any]:
    audit = audit_claim_citations(state)
    writer_state = state.copy()
    # Never expose raw unverified source_units to Writer; Writer operates strictly on audited claims & spans
    writer_state["source_units"] = []
    writer_state["verified_claims"] = []
    approved_claims: list[dict[str, Any]] = []
    for position, claim in enumerate(state.get("verified_claims") or [], start=1):
        if isinstance(claim, dict):
            check = audit["claims"].get(published_claim_id(claim, position), {})
            ev_ids = check.get("evidence_ids", [])
            claim = {**claim, "evidence_ids": ev_ids}
            if check.get("status") == "verified" and ev_ids:
                approved_claims.append(claim)
        writer_state["verified_claims"].append(claim)

    result = write(writer_state)
    report_data = result.get("written_report_data") or {}
    # Only the shared renderer owns the numbering. A raw external report has no trusted mapping.
    if isinstance(report_data, dict) and report_data.get("citation_renderer") in {"academic_report", "figure_report"}:
        approved_ids = {eid for check in audit["claims"].values() for eid in check["evidence_ids"]}
        approved_refs = [
            ref for ref in report_data.get("citation_references", [])
            if isinstance(ref, dict) and ref.get("evidence_id") in approved_ids
        ]
        if report_data.get("citation_renderer") == "academic_report":
            if report_data.get("status") == "ANSWERED_DIRECT_QUOTE" and approved_claims:
                allowed_ref_ids = {int(r["ref_id"]) for r in approved_refs if isinstance(r.get("ref_id"), int)}
                approved_spans = [
                    sp for sp in (state.get("evidence_spans") or [])
                    if isinstance(sp, dict) and sp.get("id") in approved_ids
                ]
                prose_ok, prose_reason = _verify_post_writer_prose(
                    str(report_data.get("answer") or ""),
                    approved_claims=approved_claims,
                    approved_spans=approved_spans,
                    allowed_ref_ids=allowed_ref_ids,
                )
                if not prose_ok:
                    # Deterministic fallback: revert mutated Writer prose to approved claims
                    from agents.agent5_writer.components.report_builder import build_academic_report

                    safe_md, safe_data = build_academic_report(
                        query=str(writer_state.get("query") or ""),
                        verified_claims=writer_state["verified_claims"],
                        rejected_claims=writer_state.get("rejected_claims") or [],
                        retrieved_docs=writer_state.get("retrieved_docs") or [],
                        answer=writer_state.get("answer"),
                        answer_status=writer_state.get("answer_status"),
                        evidence_spans=writer_state.get("evidence_spans") or [],
                        statistical_audit=writer_state.get("statistical_audit"),
                        code_verification=writer_state.get("code_verification"),
                        answer_language=writer_state.get("answer_language"),
                    )
                    result["final_report"] = safe_md
                    result["written_report_data"] = safe_data
                    report_data = safe_data
                    approved_refs = [
                        ref for ref in report_data.get("citation_references", [])
                        if isinstance(ref, dict) and ref.get("evidence_id") in approved_ids
                    ]
                    audit["post_writer_fallback"] = {"triggered": True, "reason": prose_reason}

            result["answer"] = report_data["answer"]
            result["short_answer"] = report_data["answer"]
            if report_data["status"] in {"NOT_FOUND_IN_DOCUMENT", "CLARIFICATION_NEEDED", "INSUFFICIENT_EVIDENCE"}:
                result["answer_status"] = report_data["status"]
            if report_data["status"] == "INSUFFICIENT_EVIDENCE":
                result["confidence"] = "low"
        audit["references"] = approved_refs
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
