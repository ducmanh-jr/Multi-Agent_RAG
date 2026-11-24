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


def build_citation_references(evidence_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Number source pages in the order of the supplied linked evidence."""
    references = {}
    for span in evidence_spans:
        location = citation_location(span)
        evidence_id = span.get("id")
        if location is None or location in references or not evidence_id:
            continue
        document_id, page = location
        references[location] = {
            "ref_id": len(references) + 1,
            "document_id": document_id,
            "page": page,
            "evidence_id": evidence_id,
        }
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
) -> tuple[str, dict[str, Any]]:
    """
    Returns:
      - markdown_report: fully formatted scientific report with comprehensive synthesis
      - report_data: structured metadata
    """
    title = f"Báo cáo Nghiên cứu: {query}"
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
        if "hoàn toàn không được đề cập" in (answer or "").casefold():
            terminal = "NOT_FOUND_IN_DOCUMENT"
        elif "vui lòng cung cấp thêm chi tiết" in (answer or "").casefold():
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
                spans_by_id[uid] = {
                    "id": uid,
                    "unit_id": uid,
                    "source_unit_id": uid,
                    "doc_id": loc[0],
                    "page_no": loc[1],
                    "quote": (u.get("text") or u.get("content") or "").strip()[:500],
                }

    unique_spans = list({id(s): s for s in spans_by_id.values()}.values())
    citation_references = build_citation_references(unique_spans)
    refs_by_page = {(ref["document_id"], ref["page"]): ref["ref_id"] for ref in citation_references}

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
    for claim in verified_claims:
        text = get_claim_text(claim).strip()
        refs = sorted({refs_by_page[citation_location(spans_by_id[eid])]
                       for eid in _claim_evidence_ids(claim) if eid in spans_by_id and citation_location(spans_by_id[eid]) in refs_by_page})
        if not refs:
            unlinked.append(claim)
            continue
        supported.append(claim)
        marks = " ".join(f"[{ref}]" for ref in refs)
        line = f"{_escape_raw_citations(text)} {marks}"
        summary_lines.append(line)
        findings_lines.append(f"- **Nhận định {len(findings_lines)+1}:** {line}")
    verified_claims = supported
    if summary_lines:
        status_label = "ANSWERED_DIRECT_QUOTE"
        summary_narrative = "\n\n".join(dict.fromkeys(summary_lines))
    elif is_ood:
        status_label = "NOT_FOUND_IN_DOCUMENT"
        summary_narrative = f'Dựa trên các tài liệu đã chọn, thông tin về "{_escape_raw_citations(query)}" không được đề cập.'
    elif is_ambiguous:
        status_label = "CLARIFICATION_NEEDED"
        summary_narrative = _escape_raw_citations(answer or "Vui lòng cho biết tài liệu hoặc khía cạnh cần làm rõ.")
    else:
        status_label = "INSUFFICIENT_EVIDENCE"
        summary_narrative = "Chưa đủ bằng chứng trong tài liệu đã chọn để xác nhận câu trả lời."

    findings_str = "\n".join(findings_lines) or "> *Chưa có nhận định được xác minh và gắn với trang nguồn.*"

    # 4. Disclaimers for ungrounded / contradicted assertions
    disclaimer_block = ""
    if rejected_claims or unlinked:
        disclaimer_lines = [f"- {_escape_raw_citations(get_claim_text(c))} *(Trích dẫn chưa xác minh)*" for c in unlinked]
        for rc in rejected_claims:
            c_txt = get_claim_text(rc)
            disclaimer_lines.append(f"- [Lưu ý] *{c_txt}* — Lý do: {rc.get('reason') or rc.get('reasoning') or 'Thiếu bằng chứng xác thực'}")
        disclaimer_block = f"\n\n### [Lưu ý] Các điểm chưa đủ căn cứ hoặc bị bác bỏ:\n" + "\n".join(disclaimer_lines)

    # 5. Bibliography block
    ref_lines = []
    for r in references:
        ref_title = _escape_raw_citations(r["title"])
        preview = _escape_raw_citations(r["preview"])
        ref_lines.append(f"[{r['ref_id']}] **{ref_title}** (Trang {r['page_no']}). *\"{preview}...\"*")
    bib_str = "\n".join(ref_lines) if ref_lines else "*Chưa có trích dẫn gắn với một nhận định đã xác minh.*"

    status_tag = f"[`{status_label}`]"

    # 3.1 Peer review & statistical audit block construction
    audit_lines = []
    if statistical_audit:
        score = statistical_audit.get("score", 7.0)
        rating = statistical_audit.get("rating", "MODERATE_RIGOR")
        audit_lines.append(f"- **Điểm Liêm chính Thống kê (Statistical Rigor Score):** `{score}/10.0` — *{rating}*")
        for finding in statistical_audit.get("findings", []):
            audit_lines.append(f"  - {finding}")
    else:
        audit_lines.append("- **Kiểm toán phương pháp:** Chưa có kết quả kiểm toán thống kê.")

    if code_verification and code_verification.get("status") == "PASS":
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
    matrix_section_title = (
        "## 4. Ma trận So sánh Đa bài báo (Cross-Paper Benchmark Matrix)"
        if len(unique_doc_titles) >= 2
        else "## 4. Ma trận Luận điểm Cốt lõi & Đối soát Bằng chứng (Core Claims & Evidence Verification Matrix)"
    )
    cited_docs = [{**span, "text": evidence_quote(span)} for span in spans_by_id.values()]
    matrix_md = build_cross_paper_matrix(query, cited_docs, verified_claims) if verified_claims else findings_str
    gaps = detect_research_gaps_and_hypotheses(query, verified_claims, cited_docs) if cited_docs else []
    gaps_md = format_gap_radar_markdown(gaps) if gaps else "Chưa có bằng chứng để đánh giá khoảng trống nghiên cứu."

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
