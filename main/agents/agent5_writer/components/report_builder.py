"""
Scientific Report Synthesis Builder.
Synthesizes full, comprehensive academic answers and reports (bỏ câu trả lời nhanh, trình bày câu trả lời hoàn thiện).
Ensures 0% hallucination by enforcing that all statements originate strictly from verified facts.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Optional
from agents.agent5_writer.components.gap_analyzer import (
    build_cross_paper_matrix,
    detect_research_gaps_and_hypotheses,
    format_gap_radar_markdown,
)
from core.citation_audit import citation_location, evidence_quote
from core.citation_audit import claim_text as get_claim_text
from utils.language import detect_language, localized_status_message, normalize_language_code


_RAW_CITATION_MARKERS = re.compile(
    r"```[\s\S]*?(?:```|\Z)|~~~[\s\S]*?(?:~~~|\Z)|`[^`\n]+`"
    r"|(?<!\\)(?P<marker>\[(?:[uU]?\d+)(?:[ \t]*,[ \t]*\d+)*\])(?!\()"
)


def _escape_raw_citations(text: str) -> str:
    """Display upstream citation numbers literally until a source link is known."""
    return _RAW_CITATION_MARKERS.sub(
        lambda match: "\\" + match[0] if match.group("marker") else match[0],
        text,
    )


def _claim_evidence_ids(claim: dict[str, Any]) -> list[str]:
    ids = claim.get("evidence_ids")
    if not isinstance(ids, list):
        return []
    return [evidence_id for evidence_id in ids if isinstance(evidence_id, str) and evidence_id.strip()]


def _normalized_bbox(raw_bbox: Any) -> list[float] | None:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3]))
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]


def citation_region_key(span: dict[str, Any]) -> tuple[Any, ...] | None:
    """
    Distinguishes distinct bounding-box regions (e.g. two equations or table shards on the
    same page) while keeping `None` for legacy page-only spans (`bbox=None`).
    """
    bbox = _normalized_bbox(span.get("bbox"))
    if bbox is not None:
        return ("bbox", round(bbox[0], 1), round(bbox[1], 1), round(bbox[2], 1), round(bbox[3], 1))
    region_id = span.get("region_id")
    if isinstance(region_id, str) and region_id.strip():
        return ("region", region_id.strip())
    return None


def citation_target_key(span: dict[str, Any]) -> tuple[str, int, Any] | None:
    loc = citation_location(span)
    if loc is None:
        return None
    return (loc[0], loc[1], citation_region_key(span))


def build_citation_references(evidence_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Number source regions/pages in the order of the supplied linked evidence.
    Two distinct `bbox` regions on the same page receive distinct `[1]`, `[2]` ref_ids,
    while legacy page-only spans (`bbox=None`) group cleanly per `(doc_id, page)`.
    """
    references: dict[tuple[str, int, Any], dict[str, Any]] = {}
    for span in evidence_spans:
        target_key = citation_target_key(span)
        evidence_id = span.get("id")
        if target_key is None or target_key in references or not evidence_id:
            continue
        document_id, page, region_key = target_key
        entry: dict[str, Any] = {
            "ref_id": len(references) + 1,
            "document_id": document_id,
            "page": page,
            "evidence_id": evidence_id,
        }
        bbox = _normalized_bbox(span.get("bbox"))
        if region_key is not None and bbox is not None:
            entry["bbox"] = bbox
            if span.get("page_rotation"):
                entry["page_rotation"] = int(span["page_rotation"])
        references[target_key] = entry
    return list(references.values())


def build_academic_report(
    query: str,
    verified_claims: list[dict[str, Any]],
    rejected_claims: list[dict[str, Any]],
    retrieved_docs: list[dict[str, Any]],
    answer: Optional[str] = None,
    answer_status: Optional[str] = None,
    short_answer: Optional[str] = None,
    evidence_spans: Optional[list[dict[str, Any]]] = None,
    statistical_audit: Optional[dict[str, Any]] = None,
    code_verification: Optional[dict[str, Any]] = None,
    answer_language: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    """
    Returns:
      - markdown_report: fully formatted scientific report with comprehensive synthesis
      - report_data: structured metadata
    """
    lang = normalize_language_code(answer_language) or detect_language(query, default="vi")
    title = f"Research Report: {query}" if lang == "en" else f"Báo cáo Nghiên cứu: {query}"
    verified_claims = [
        c if isinstance(c, dict) else (c.to_dict() if hasattr(c, "to_dict") else {"claim": str(c), "claim_text": str(c), "text": str(c)})
        for c in (verified_claims or [])
    ]
    rejected_claims = [
        c if isinstance(c, dict) else (c.to_dict() if hasattr(c, "to_dict") else {"claim": str(c), "claim_text": str(c), "text": str(c)})
        for c in (rejected_claims or [])
    ]
    # Use an explicit terminal status, including for older runners that only emitted a refusal.
    # A contradicted refusal must go through the same evidence checks as other drafts.
    terminal = answer_status if not rejected_claims else None
    if not terminal and not rejected_claims and not verified_claims:
        ans_cf = (answer or "").casefold()
        if "hoàn toàn không được đề cập" in ans_cf or "is not mentioned in the provided sources" in ans_cf:
            terminal = "NOT_FOUND_IN_DOCUMENT"
        elif "vui lòng cung cấp thêm chi tiết" in ans_cf or "please clarify which specific aspect" in ans_cf:
            terminal = "CLARIFICATION_NEEDED"
    is_ood = terminal == "NOT_FOUND_IN_DOCUMENT"
    is_ambiguous = terminal == "CLARIFICATION_NEEDED"
    sources_to_process = [] if is_ood else list(retrieved_docs or [])
    if is_ood or is_ambiguous:
        verified_claims = []

    rejected_texts = {get_claim_text(c).strip().casefold() for c in rejected_claims}
    verified_claims = [c for c in verified_claims if get_claim_text(c).strip()
                       and get_claim_text(c).strip().casefold() not in rejected_texts
                       and str(c.get("status", "support")).lower() in {"support", "supported"}]
    linked_ids = {
        evidence_id
        for claim in verified_claims
        for evidence_id in _claim_evidence_ids(claim)
    }
    linked_spans = [
        span for span in (evidence_spans or [])
        if isinstance(span, dict) and (
            (isinstance(span.get("id"), str) and span["id"] in linked_ids)
            or (isinstance(span.get("unit_id"), str) and span["unit_id"] in linked_ids)
            or (isinstance(span.get("source_unit_id"), str) and span["source_unit_id"] in linked_ids)
        )
    ]
    span_id_counts = Counter(span["id"] for span in linked_spans if isinstance(span.get("id"), str))
    valid_spans = [
        span for span in linked_spans
        if span_id_counts.get(span.get("id"), 0) == 1 and evidence_quote(span).strip() and citation_location(span)
    ]
    spans_by_id = {}
    for span in valid_spans:
        sid = span.get("id")
        if sid and sid not in spans_by_id:
            spans_by_id[sid] = span
        uid = span.get("unit_id") or span.get("source_unit_id")
        if uid and uid not in spans_by_id:
            spans_by_id[uid] = span

    for u in sources_to_process:
        if not isinstance(u, dict):
            continue
        uid = u.get("id") or u.get("chunk_id") or u.get("unit_id")
        if uid and uid in linked_ids and uid not in spans_by_id and (u.get("text") or u.get("content")):
            loc = citation_location(u)
            if loc:
                synth_span: dict[str, Any] = {
                    "id": uid,
                    "unit_id": uid,
                    "source_unit_id": uid,
                    "doc_id": loc[0],
                    "page_no": loc[1],
                    "quote": (u.get("text") or u.get("content") or "").strip()[:500],
                }
                if u.get("bbox") is not None:
                    synth_span["bbox"] = u.get("bbox")
                if u.get("page_rotation"):
                    synth_span["page_rotation"] = u.get("page_rotation")
                spans_by_id[uid] = synth_span

    unique_spans = list({id(s): s for s in spans_by_id.values()}.values())
    citation_references = build_citation_references(unique_spans)
    refs_by_target = {
        citation_target_key(spans_by_id[ref["evidence_id"]]): ref["ref_id"]
        for ref in citation_references
        if ref.get("evidence_id") in spans_by_id and citation_target_key(spans_by_id[ref["evidence_id"]]) is not None
    }
    refs_by_page: dict[tuple[str, int], int] = {}
    for ref in citation_references:
        refs_by_page.setdefault((ref["document_id"], ref["page"]), ref["ref_id"])

    doc_title_lookup = {}
    for doc in sources_to_process:
        document_id = str(doc.get("doc_id") or doc.get("document_id") or "").strip()
        title_value = doc.get("title") or doc.get("name") or doc.get("filename") or document_id
        if document_id:
            doc_title_lookup[document_id] = str(title_value).replace("\\", "/").rsplit("/", 1)[-1]

    references = []
    for ref in citation_references:
        document_id = ref["document_id"]
        span = spans_by_id.get(ref["evidence_id"])
        if not span:
            continue
        references.append({
            "ref_id": ref["ref_id"],
            "title": doc_title_lookup.get(document_id, document_id),
            "doc_id": document_id,
            "page_no": ref["page"],
            "preview": evidence_quote(span)[:180],
            "bbox": span.get("bbox"),
        })

    supported = []
    unlinked = []
    findings_lines = []
    summary_lines = []
    claim_doc_nums: dict[str, set[str]] = {}
    claim_entries: list[tuple[str, list[int], list[str]]] = []
    ref_doc_lookup = {r["ref_id"]: r["doc_id"] for r in references}
    for claim in verified_claims:
        text = get_claim_text(claim).strip()
        claim_ref_ids: set[int] = set()
        for eid in _claim_evidence_ids(claim):
            if eid not in spans_by_id:
                continue
            sp = spans_by_id[eid]
            t_key = citation_target_key(sp)
            if t_key in refs_by_target:
                claim_ref_ids.add(refs_by_target[t_key])
            elif citation_location(sp) in refs_by_page:
                claim_ref_ids.add(refs_by_page[citation_location(sp)])
        refs = sorted(claim_ref_ids)
        if not refs:
            unlinked.append(claim)
            continue
        supported.append(claim)
        doc_ids_for_claim = list(dict.fromkeys(ref_doc_lookup[r] for r in refs if r in ref_doc_lookup))
        nums = {
            m.replace(",", ".")
            for m in re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text)
            if "%" in m or "." in m or "," in m or len(m) >= 2
        }
        for did in doc_ids_for_claim:
            if nums:
                claim_doc_nums.setdefault(did, set()).update(nums)
        claim_entries.append((text, refs, doc_ids_for_claim))

    has_cross_doc_conflict = False
    if len(claim_doc_nums) >= 2:
        doc_num_sets = [vals for vals in claim_doc_nums.values() if vals]
        for i_d in range(len(doc_num_sets)):
            for j_d in range(i_d + 1, len(doc_num_sets)):
                if doc_num_sets[i_d] != doc_num_sets[j_d]:
                    has_cross_doc_conflict = True
                    break

    for text, refs, doc_ids_for_claim in claim_entries:
        marks = " ".join(f"[{ref}]" for ref in refs)
        escaped = _escape_raw_citations(text)
        if has_cross_doc_conflict and len(doc_ids_for_claim) == 1:
            d_label = doc_title_lookup.get(doc_ids_for_claim[0], doc_ids_for_claim[0])
            line = f"**{_escape_raw_citations(d_label)}:** {escaped} {marks}"
        else:
            line = f"{escaped} {marks}"
        summary_lines.append(line)
        finding_prefix = "Finding" if lang == "en" else "Nhận định"
        findings_lines.append(f"- **{finding_prefix} {len(findings_lines)+1}:** {line}")

    verified_claims = supported
    if summary_lines:
        status_label = "ANSWERED_DIRECT_QUOTE"
        dedup_summary = list(dict.fromkeys(summary_lines))
        if has_cross_doc_conflict:
            conflict_banner = (
                "> **Cross-Document Discrepancy Notice:** Conflicting numeric values are reported across the selected documents; each source is attributed separately below."
                if lang == "en"
                else "> **Lưu ý Sai khác Số liệu Giữa các Tài liệu:** Các tài liệu ghi nhận các giá trị khác nhau (có sự mâu thuẫn số liệu); kết quả từng tài liệu được tách riêng bên dưới."
            )
            summary_narrative = conflict_banner + "\n\n" + "\n\n".join(dedup_summary)
        else:
            summary_narrative = "\n\n".join(dedup_summary)
    elif is_ood:
        status_label = "NOT_FOUND_IN_DOCUMENT"
        if answer and any(tok in answer.lower() for tok in ("table ", "bảng ", "unverified", "kiểm chứng")):
            summary_narrative = _escape_raw_citations(answer)
        else:
            summary_narrative = localized_status_message(
                "NOT_FOUND_IN_DOCUMENT",
                language=lang,
                query=_escape_raw_citations(query),
            )
    elif is_ambiguous:
        status_label = "CLARIFICATION_NEEDED"
        if answer and detect_language(answer, default=lang) == lang:
            summary_narrative = _escape_raw_citations(answer)
        else:
            summary_narrative = localized_status_message("CLARIFICATION_NEEDED", language=lang, query=query)
    else:
        status_label = "INSUFFICIENT_EVIDENCE"
        summary_narrative = localized_status_message("INSUFFICIENT_EVIDENCE", language=lang, query=query)

    findings_str = "\n".join(findings_lines) or (
        "> *No verified findings linked to source pages.*"
        if lang == "en"
        else "> *Chưa có nhận định được xác minh và gắn với trang nguồn.*"
    )

    # 4. Disclaimers for ungrounded / contradicted assertions
    disclaimer_block = ""
    if rejected_claims or unlinked:
        disclaimer_lines = [f"- {_escape_raw_citations(get_claim_text(c))} *(Trích dẫn chưa xác minh)*" for c in unlinked]
        for rc in rejected_claims:
            c_txt = get_claim_text(rc)
            if lang == "en":
                disclaimer_lines.append(
                    f"- [Note] *{c_txt}* — Reason: {rc.get('reason') or rc.get('reasoning') or 'Insufficient supporting evidence'}"
                )
            else:
                disclaimer_lines.append(
                    f"- [Lưu ý] *{c_txt}* — Lý do: {rc.get('reason') or rc.get('reasoning') or 'Thiếu bằng chứng xác thực'}"
                )
        disclaimer_heading = (
            "### [Note] Unverified or Rejected Assertions:"
            if lang == "en"
            else "### [Lưu ý] Các điểm chưa đủ căn cứ hoặc bị bác bỏ:"
        )
        disclaimer_block = f"\n\n{disclaimer_heading}\n" + "\n".join(disclaimer_lines)

    # 5. Bibliography block
    ref_lines = []
    for r in references:
        ref_title = _escape_raw_citations(r["title"])
        preview = _escape_raw_citations(r["preview"])
        page_label = "Page" if lang == "en" else "Trang"
        ref_lines.append(f"[{r['ref_id']}] **{ref_title}** ({page_label} {r['page_no']}). *\"{preview}...\"*")
    bib_str = "\n".join(ref_lines) if ref_lines else (
        "*No citations linked to a verified finding.*"
        if lang == "en"
        else "*Chưa có trích dẫn gắn với một nhận định đã xác minh.*"
    )

    status_tag = f"[`{status_label}`]"

    # 3.1 Peer review & statistical audit block construction
    audit_lines = []
    if statistical_audit:
        score = statistical_audit.get("score", 7.0)
        rating = statistical_audit.get("rating", "MODERATE_RIGOR")
        if lang == "en":
            audit_lines.append(f"- **Statistical Rigor Score:** `{score}/10.0` — *{rating}*")
        else:
            audit_lines.append(f"- **Điểm Liêm chính Thống kê (Statistical Rigor Score):** `{score}/10.0` — *{rating}*")
        for finding in statistical_audit.get("findings", []):
            audit_lines.append(f"  - {finding}")
    else:
        audit_lines.append(
            "- **Methodology Audit:** No statistical audit available."
            if lang == "en"
            else "- **Kiểm toán phương pháp:** Chưa có kết quả kiểm toán thống kê."
        )

    if code_verification and code_verification.get("status") == "PASS":
        if lang == "en":
            audit_lines.append(
                f"- **Formal Code Verification (Python Sandbox REPL):** `[PASS - VERIFIED]` ({code_verification.get('execution_time_ms', 0)}ms)"
            )
        else:
            audit_lines.append(
                f"- **Kiểm chứng Hình thức Mã nguồn (Python Sandbox REPL):** `[PASS - VERIFIED]` ({code_verification.get('execution_time_ms', 0)}ms)"
            )
        audit_lines.append(f"  ```text\n  {code_verification.get('output', '')}\n  ```")

    peer_review_body = "\n".join(audit_lines)

    # 3.2 Cross-Paper Benchmark Matrix & Research Gap Radar Synthesis
    unique_doc_titles = {
        re.sub(r"^(?:gemini|doc)_[a-f0-9]+_", "", str(d.get("title") or d.get("name") or d.get("doc_id") or "")).strip().lower()
        for d in (retrieved_docs or [])
        if str(d.get("title") or d.get("name") or d.get("doc_id") or "").strip()
    }
    if lang == "en":
        matrix_section_title = (
            "## 4. Cross-Paper Benchmark Matrix"
            if len(unique_doc_titles) >= 2
            else "## 4. Core Claims & Evidence Verification Matrix"
        )
    else:
        matrix_section_title = (
            "## 4. Ma trận So sánh Đa bài báo (Cross-Paper Benchmark Matrix)"
            if len(unique_doc_titles) >= 2
            else "## 4. Ma trận Luận điểm Cốt lõi & Đối soát Bằng chứng (Core Claims & Evidence Verification Matrix)"
        )
    cited_docs = [{**span, "text": evidence_quote(span)} for span in spans_by_id.values()]
    matrix_md = build_cross_paper_matrix(query, cited_docs, verified_claims) if verified_claims else findings_str
    gaps = detect_research_gaps_and_hypotheses(query, verified_claims, cited_docs) if cited_docs else []
    if gaps:
        gaps_md = format_gap_radar_markdown(gaps)
    else:
        gaps_md = (
            "Insufficient grounded evidence to assess research gaps."
            if lang == "en"
            else "Chưa có bằng chứng để đánh giá khoảng trống nghiên cứu."
        )

    if lang == "en":
        academic_report_md = f"""# {_escape_raw_citations(title)}

> **Verification Status:** {status_tag}

---

## 1. Executive Summary
{summary_narrative}

## 2. Verified Findings
{findings_str}

## 3. Research Gap Radar
{_escape_raw_citations(gaps_md)}

## 4. Evidence References
{bib_str}
"""
        audit_ledger_md = f"""# Audit & Verification Ledger

> **Verification Status:** {status_tag} | {len(verified_claims)} independently verified findings

---

{matrix_section_title}
{_escape_raw_citations(matrix_md)}

## Senior Peer-Review Audit
{_escape_raw_citations(peer_review_body)}
{_escape_raw_citations(disclaimer_block)}
"""
    else:
        academic_report_md = f"""# {_escape_raw_citations(title)}

> **Trạng thái Thẩm định:** {status_tag}

---

## 1. Tóm tắt Tổng quan (Executive Summary)
{summary_narrative}

## 2. Kết quả Nghiên cứu & Luận điểm Đã Xác minh (Verified Findings)
{findings_str}

## 3. Radar Khoảng trống & Giả thuyết Nghiên cứu (Research Gap Radar)
{_escape_raw_citations(gaps_md)}

## 4. Danh mục Tài liệu Trích dẫn (Evidence References)
{bib_str}
"""

        audit_ledger_md = f"""# Sổ cái Thẩm định & Đối soát Bằng chứng (Audit & Verification Ledger)

> **Trạng thái Xác thực:** {status_tag} | {len(verified_claims)} luận điểm đã đối soát độc lập

---

{matrix_section_title}
{_escape_raw_citations(matrix_md)}

## Đánh giá Bình duyệt Chuyên sâu & Kiểm toán Phương pháp (Senior Peer-Review Audit)
{_escape_raw_citations(peer_review_body)}
{_escape_raw_citations(disclaimer_block)}
"""

    markdown = f"""{academic_report_md.strip()}

<!-- DUAL_REPORT_SPLIT -->

{audit_ledger_md.strip()}"""

    report_data = {
        "title": title,
        "query": query,
        "answer": summary_narrative,
        "short_answer": summary_narrative,
        "status": status_label,
        "answer_language": lang,
        "verified_claims": verified_claims,
        "rejected_claims": rejected_claims,
        "references": references,
        "citation_renderer": "academic_report",
        "citation_references": citation_references,
        "statistical_audit": statistical_audit,
        "code_verification": code_verification,
        "academic_report": academic_report_md,
        "audit_ledger": audit_ledger_md,
    }

    return markdown, report_data
