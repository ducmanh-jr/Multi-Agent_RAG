"""Select figures from the current PDFs and ground visual answers in their original crops."""

import math
import re
import unicodedata
from pathlib import Path

from agents.agent5_writer.components.report_builder import build_citation_references
from core.claim_semantics import (
    extract_directed_entity_pairs,
    extract_entity_quantity_bindings,
)
from core.figure_reader import read_figure
from core.pdf_figures import extract_pdf_figures, figure_reference, render_pdf_figure
from core.pdf_ingest import compute_file_sha256, get_cached_page_ocr
from core.state import AgentState, AtomicClaim, EvidenceSpan, SourceUnit
from core.verifier import check_numerical_and_date_consistency, extract_numbers_and_years
from utils.bilingual import expand_bilingual_tokens
from utils.logger import logger


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold().replace("đ", "d"))
                   if not unicodedata.combining(c))


_FIG_NUM = r"[a-z]?\d+(?:[.-]\d+)*(?:[a-z])?"
_NUMBERED = re.compile(rf"\b(?:fig(?:ure)?s?\.?|hinh|bieu do|so do)\s*(?:so\s*)?({_FIG_NUM})\b")
_VISUAL = re.compile(
    r"\b(?:figures?|fig\.|pictures?|images?|plots?|charts?|diagrams?|graphs?|photographs?|"
    r"bieu do|do thi|so do|hinh|anh minh hoa|buc anh|hinh chup|cot nao|which (?:bar|column)|tallest bar|shortest bar|truc tung|truc hoanh)\b"
)
_GENERIC = set("""
    figure figures fig hinh anh minh hoa bieu do thi chart diagram image picture plot graph paper page trang
    the of and how what which is are does did this that these from for show me please explain describe used
    trong tai lieu bai bao cho biet ve nao cua va thi la gi hay toi trich xuat xem co cac nhung o tren
    duoi voi lien quan the hien ket qua gia tri nhan xet bao nhieu mau cao thap nhat ben phai trai
""".split())


def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", _fold(text))
            if len(word) >= 2 and not word.isdigit() and word not in _GENERIC}


def _select(query: str, catalog: list[dict]) -> dict:
    normalized = _fold(query)
    # These common Vietnamese phrases contain "hình" but also occur in ordinary text questions.
    visual_query = re.sub(r"\b(?:mo|tinh|dien|loai|doi|tao|cau) hinh\b", " ", normalized)
    visual_query = re.sub(r"\b(?:graph neural networks?|graph theory|image (?:classification|recognition|enhancement|processing)|xu ly hinh anh)\b", " ", visual_query)
    numbers = []
    for match in _NUMBERED.finditer(visual_query):
        numbers.append(match[1])
        rest = visual_query[match.end():]
        while more := re.match(rf"\s*(?:,|/|&|\band\b|\bva\b)\s*({_FIG_NUM})\b", rest):
            numbers.append(more[1])
            rest = rest[more.end():]
    numbers = list(dict.fromkeys(numbers))
    photo = re.search(r"\bảnh\b(?!\s+hưởng)", query.casefold()) and "xu ly hinh anh" not in normalized
    explicit = bool(numbers or photo or _VISUAL.search(visual_query))
    pages = {int(p) for p in re.findall(r"\b(?:page|trang)\s+(\d{1,5})\b", normalized)}
    if pages:
        catalog = [f for f in catalog if f["page_no"] in pages]
    terms = _tokens(query)
    translated = set().union(*(_tokens(t) for t in expand_bilingual_tokens(query, set())))

    def scores(fig):
        caption = _tokens(fig.get("caption", ""))
        labels = _tokens(fig.get("embedded_text", ""))
        ocr_extra = _tokens(fig.get("ocr_text", ""))
        if not ocr_extra and fig.get("kind") == "scan" and fig.get("source_path"):
            try:
                sha = compute_file_sha256(fig["source_path"])
                cached = get_cached_page_ocr(sha, int(fig.get("page_no", 1)))
                if cached and cached.get("text"):
                    ocr_extra = _tokens(cached["text"])
            except Exception:
                ocr_extra = set()
        all_labels = labels | ocr_extra
        title = _tokens(fig.get("doc_id", ""))
        return (len(terms & title), 3 * len(terms & caption) + 2 * len(terms & all_labels)
                + len(translated & (caption | all_labels)))

    ranked = sorted(catalog, key=lambda f: scores(f) if numbers else scores(f)[::-1], reverse=True)
    selected = []
    status = "matched"
    if numbers:
        for number in numbers:
            matches = [f for f in ranked if number in _NUMBERED.findall(_fold(f.get("label", "")))]
            if not matches:
                status = "missing"
                break
            if len(matches) > 1 and scores(matches[0]) == scores(matches[1]):
                status = "ambiguous"
                break
            selected.append(matches[0])
    elif explicit:
        hits = [f for f in ranked if scores(f)[1] > 0]
        if hits:
            selected = hits[:2]
        elif len(ranked) == 1:
            selected = ranked
        elif ranked and re.search(r"\b(?:list|all|cac hinh|liet ke|nhung hinh|hinh nao)\b", normalized):
            selected = ranked[:3]
        else:
            status = "ambiguous" if ranked else "missing"
    else:
        # Ordinary text questions may use a figure's actual caption, labels, or full-page scan OCR as evidence.
        selected = [f for f in ranked if scores(f)[1] >= 5][:2]
        if not selected:
            scans = [f for f in ranked if f.get("kind") == "scan" and (scores(f)[1] >= 1 or len(ranked) <= 2)]
            if scans:
                selected = scans[:2]
    return {
        "explicit": explicit, "status": status,
        "figures": selected[:3] if status == "matched" else [],
        "choices": list(dict.fromkeys(f'{f.get("doc_id")} · {f.get("label") or "hình"} · trang {f["page_no"]}'
                                      for f in ranked))[:5] if status == "ambiguous" else [],
    }


def figure_context(state: AgentState) -> dict:
    return (state.get("document_schema") or {}).get("figure_evidence") or {}


def cross_check_figure_reading(fig: dict) -> dict:
    """
    Cross-checks pixel/vision observations against extractable PDF text (caption + embedded labels +
    body reference + OCR) on the full tuple `(entity/series, value, unit, direction)`:
      1. Detects swapped entity/series values (e.g., source `A = 81%, B = 92%` vs vision `A = 92%, B = 81%`)
         and marks `"entity_value_conflict"` instead of `"corroborated"`.
      2. Separates directly printed numbers (`printed_values`) from axis-estimated values (`axis_estimated_values`).
    """
    reading = fig.get("reading") or {}
    method = reading.get("method") or "unavailable"
    visual_text = (reading.get("text") or "").strip()
    structured = reading.get("structured") if isinstance(reading.get("structured"), dict) else {}
    printed_vals = list(structured.get("printed_values") or [])
    axis_est_vals = list(structured.get("axis_estimated_values") or [])

    if method not in {"vision", "ocr"} or not visual_text:
        return {
            "status": "not_applicable",
            "unverified_numbers": [],
            "printed_values": printed_vals,
            "axis_estimated_values": axis_est_vals,
        }

    nums = extract_numbers_and_years(visual_text)
    if not nums:
        return {
            "status": "no_numeric_claims",
            "unverified_numbers": [],
            "printed_values": printed_vals,
            "axis_estimated_values": axis_est_vals,
        }

    pdf_parts = [
        fig.get("caption") or "",
        fig.get("embedded_text") or "",
        fig.get("ocr_text") or "",
        (fig.get("reference") or {}).get("text") or "",
    ]
    pdf_text = "\n".join(p for p in pdf_parts if p).strip()
    if not pdf_text:
        return {
            "status": "visual_only_numbers",
            "unverified_numbers": nums,
            "printed_values": printed_vals,
            "axis_estimated_values": axis_est_vals,
            "value_source": "axis_estimated" if axis_est_vals else "visual_only",
        }

    # 1. Verify entity–series–value–unit tuple bindings & comparative direction (catches A/B swap!)
    vis_bindings = extract_entity_quantity_bindings(visual_text)
    pdf_bindings = extract_entity_quantity_bindings(pdf_text)
    conflicts: list[str] = []
    if vis_bindings and pdf_bindings:
        for vb in vis_bindings:
            same_ent = [pb for pb in pdf_bindings if pb["entity_key"] == vb["entity_key"]]
            if same_ent:
                # Entity exists in PDF reference/labels: at least one binding for that entity must match value & unit!
                if not any(
                    math.isclose(vb["value"], pb["value"], abs_tol=0.05)
                    and (not vb["unit"] or not pb["unit"] or vb["unit"] == pb["unit"])
                    for pb in same_ent
                ):
                    conflicts.append(
                        f"{vb['entity']}={vb['value']}{vb['unit']} (PDF={same_ent[0]['value']}{same_ent[0]['unit']})"
                    )

    # Also check directed comparative reversals (e.g. "Method A exceeds Method B" when PDF has A=81%, B=92%)
    vis_pairs = extract_directed_entity_pairs(visual_text)
    pdf_pairs = extract_directed_entity_pairs(pdf_text)
    for i in range(len(pdf_bindings)):
        for j in range(i + 1, len(pdf_bindings)):
            b1, b2 = pdf_bindings[i], pdf_bindings[j]
            if b1["unit"] == b2["unit"] and b1["entity_key"] != b2["entity_key"] and not math.isclose(b1["value"], b2["value"], abs_tol=1e-9):
                hi, lo = (b1["entity_key"], b2["entity_key"]) if b1["value"] > b2["value"] else (b2["entity_key"], b1["entity_key"])
                if (hi, lo) not in pdf_pairs:
                    pdf_pairs.append((hi, lo))
    for v_hi, v_lo in vis_pairs:
        if (v_lo, v_hi) in pdf_pairs:
            conflicts.append(f"comparative_swap:{v_hi}>{v_lo}")

    if conflicts:
        return {
            "status": "entity_value_conflict",
            "conflicts": conflicts,
            "unverified_numbers": conflicts,
            "printed_values": printed_vals,
            "axis_estimated_values": axis_est_vals,
        }

    num_ok, date_ok, missing = check_numerical_and_date_consistency(visual_text, pdf_text)
    # Any axis-estimated values not present in PDF text must remain flagged as estimated, never corroborated
    if axis_est_vals:
        est_nums = extract_numbers_and_years(" ".join(axis_est_vals))
        uncorroborated_est = [n for n in est_nums if not check_numerical_and_date_consistency(n, pdf_text)[0]]
        if uncorroborated_est:
            return {
                "status": "axis_estimated_values",
                "unverified_numbers": uncorroborated_est,
                "printed_values": printed_vals,
                "axis_estimated_values": axis_est_vals,
                "value_source": "axis_estimated",
            }

    if num_ok and date_ok:
        return {
            "status": "corroborated",
            "unverified_numbers": [],
            "printed_values": printed_vals,
            "axis_estimated_values": axis_est_vals,
            "value_source": "printed_direct",
        }
    return {
        "status": "visual_only_numbers",
        "unverified_numbers": missing,
        "printed_values": printed_vals,
        "axis_estimated_values": axis_est_vals,
        "value_source": "visual_only",
    }


def add_figure_evidence(state: AgentState, update: dict, mode: str) -> dict:
    query = state.get("query", "")
    paths = state.get("pdf_paths") or [state.get("pdf_path")]
    paths = list(dict.fromkeys(str(p) for p in paths if p and Path(p).suffix.lower() == ".pdf" and Path(p).is_file()))
    if not paths:
        return update

    context = figure_context(state)
    if not context:
        catalog = []
        for path in paths:
            try:
                catalog.extend(extract_pdf_figures(path))
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("Figure index unavailable for %s (%s)", Path(path).name, type(exc).__name__)
        context = _select(query, catalog)
        for fig in context["figures"]:
            # Text-only agent citations identify the PDF by title. Duplicate titles need an exact figure-unit link.
            fig["unique_title"] = len({f["source_path"] for f in catalog if f["doc_id"] == fig["doc_id"]}) == 1
        if not context["explicit"] and not context["figures"]:
            return update
        # If the PDF has zero embedded raster/vector figures (`not catalog`) AND the retrieved text
        # units already contain the requested figure label/caption in plain text (e.g. text-rendered
        # "Figure 2: ..."), fall through to standard text-grounded synthesis instead of blocking.
        if context["status"] == "missing" and not catalog:
            req_nums = _NUMBERED.findall(_fold(query))
            unit_blob = _fold(" ".join(str(u.get("text") or "") for u in (update.get("source_units") or [])))
            if req_nums and all(n in _NUMBERED.findall(unit_blob) for n in req_nums):
                context = {**context, "explicit": False}
                return {
                    **update,
                    "document_schema": {**(update.get("document_schema") or {}), "figure_evidence": context},
                }
        for idx, original in enumerate(context["figures"]):
            fig = dict(original)
            try:
                fig["reference"] = figure_reference(fig, _tokens(query))
            except (KeyError, OSError, ValueError, RuntimeError):
                fig["reference"] = None
            if context["explicit"]:
                try:
                    png = render_pdf_figure(fig["source_path"], fig["figure_id"])
                    fig["reading"] = read_figure(png, query, mode)
                except (KeyError, OSError, ValueError, RuntimeError) as exc:
                    logger.warning("Figure reading unavailable (%s)", type(exc).__name__)
                    fig["reading"] = {"method": "unavailable", "text": ""}
                fig["cross_check"] = cross_check_figure_reading(fig)
            context["figures"][idx] = fig

    # Only original PDF text enters the quote verifier. OCR and model observations stay attributed.
    units = []
    for fig in context["figures"]:
        for field in ("caption", "embedded_text"):
            text = (fig.get(field) or "").strip()[:1800]
            if text:
                bbox = fig.get("caption_bbox") if field == "caption" else fig["bbox"]
                unit = SourceUnit(f'{fig["figure_id"]}:{field}', text, fig["doc_id"], fig["page_no"], list(bbox or fig["bbox"]))
                units.append({**unit.to_dict(), "figure_id": fig["figure_id"], "source_path": fig["source_path"]})
        if ref := fig.get("reference"):
            unit = SourceUnit(f'{fig["figure_id"]}:reference', ref["text"], fig["doc_id"], ref["page_no"], ref["bbox"])
            units.append({**unit.to_dict(), "figure_id": fig["figure_id"], "source_path": fig["source_path"]})
    return {
        **update,
        "document_schema": {**(update.get("document_schema") or {}), "figure_evidence": context},
        "source_units": [*units, *(update.get("source_units") or [])],
        "reasoning_trace": [*(update.get("reasoning_trace") or []), {
            "agent": "reader", "action": "figure_reading", "status": context["status"],
            "figures": [{
                "figure_id": f["figure_id"],
                "page": f["page_no"],
                "method": (f.get("reading") or {}).get("method", "pdf_text"),
                "cross_check": (f.get("cross_check") or {}).get("status", "not_applicable"),
            } for f in context["figures"]],
        }],
    }


def _plain(text: str) -> str:
    # Keep PDF text / OCR and model output in the text of the report, including in Markdown exports.
    return re.sub(r"([\\\x60*_{}\[\]<>#!|])", r"\\\1", " ".join(text.split()))


def draft_visual_answer(state: AgentState) -> dict:
    context = figure_context(state)
    blocks, claims, spans = [], [], []
    if context["status"] == "ambiguous":
        names = "\n".join(f"- {_plain(name)}" for name in context["choices"])
        blocks.append(f"Bạn muốn xem hình nào? Hãy cho biết tên tài liệu, số hình hoặc trang.\n\n{names}")
    elif context["status"] == "missing":
        blocks.append("Chưa định vị được hình theo yêu cầu trong PDF đã chọn. Bạn có thể chỉ rõ số trang hoặc tên hình.")

    for fig in context["figures"]:
        label = fig.get("label") or ("Trang scan" if fig.get("kind") == "scan" else "Hình trong PDF")
        heading = f'**{_plain(label)}** · {_plain(fig["doc_id"])} · trang {fig["page_no"]}'
        lines = [heading]
        for unit in state.get("source_units") or []:
            if unit.get("figure_id") != fig["figure_id"]:
                continue
            span = EvidenceSpan(id=f'ev:{unit["id"]}', quote=unit["text"], source_unit_id=unit["id"],
                                doc_id=unit["doc_id"], page_no=unit["page_no"], bbox=unit["bbox"])
            spans.append(span.to_dict())
            claims.append(AtomicClaim(id=f'claim:{unit["id"]}', text=unit["text"], evidence_ids=[span.id]).to_dict())
            prefix = "Chú thích" if unit["id"].endswith(":caption") else "Chữ trong hình"
            if unit["id"].endswith(":reference"):
                prefix = f'Tác giả giải thích ở trang {unit["page_no"]}'
            line = f'{prefix}: {_plain(unit["text"])}'
            if unit["id"].endswith(":reference"):
                lines.insert(1, line)
            else:
                lines.append(line)
        reading = fig.get("reading") or {}
        method = reading.get("method")
        prefix = {"ocr": "Chữ nhận dạng từ ảnh (OCR)", "vision": "Diễn giải của mô hình thị giác",
                  "geometry": "So sánh trực tiếp trên ảnh"}.get(method)
        if prefix and reading.get("text"):
            line = f'{prefix}: {_plain(reading["text"])}'
            if method == "ocr" and fig.get("reference"):
                lines.append(line)
            else:
                lines.insert(1, line)
        cross_check = fig.get("cross_check") or {}
        if cross_check.get("status") == "entity_value_conflict" and cross_check.get("conflicts"):
            conflict_str = "; ".join(str(c) for c in cross_check["conflicts"][:6])
            lines.append(
                f"Cảnh báo mâu thuẫn nhãn–trị số trên hình: kết quả đọc ảnh ({_plain(conflict_str)}) "
                "bị đảo nhãn hoặc lệch với văn bản nguồn PDF — không dùng trị số bị đảo này làm kết luận."
            )
        elif cross_check.get("status") in {"visual_only_numbers", "axis_estimated_values"} and cross_check.get("unverified_numbers"):
            unverified = ", ".join(str(n) for n in cross_check["unverified_numbers"][:8])
            lines.append(
                f"Lưu ý đối chiếu ảnh–chữ: các giá trị ({_plain(unverified)}) chỉ đọc/ước lượng từ pixel ảnh, "
                "chưa có trong phần chữ gốc/chú thích PDF — vui lòng kiểm tra trực tiếp trên hình bên dưới."
            )
        if method not in {"geometry", "vision"}:
            lines.append("Có thể đối chiếu ảnh gốc bên dưới. Những chi tiết đồ họa vượt quá phần chữ đọc được cần được kiểm tra trên hình.")
        blocks.append("\n\n".join(lines))
    return {
        "answer": "\n\n".join(blocks), "claims": claims, "evidence_spans": spans,
        "critic_action": "approve", "answer_status": "CLARIFICATION_NEEDED" if context["status"] == "ambiguous" else "FIGURE_RESPONSE",
        "reasoning_trace": [{"agent": "summarizer", "action": "summarized_figure_evidence"}],
    }


def check_visual_answer(state: AgentState) -> dict:
    """Verify literal PDF claims against their exact figure unit. Pixel interpretations remain separate."""
    units = {u["id"]: u for u in state.get("source_units") or [] if u.get("figure_id")}
    spans = {s["id"]: s for s in state.get("evidence_spans") or []}
    verified, rejected = [], []
    for claim in state.get("claims") or []:
        span = spans.get(claim["evidence_ids"][0], {})
        unit = units.get(span.get("source_unit_id"), {})
        supported = (unit.get("text") == claim["text"] == span.get("quote")
                     and (unit.get("doc_id"), unit.get("page_no")) == (span.get("doc_id"), span.get("page_no")))
        verdict = {**claim, "status": "SUPPORT" if supported else "NOT_ENOUGH_INFO",
                   "reason": "Đã đối chiếu với văn bản trên trang nguồn của PDF." if supported else "Chưa xác nhận được đoạn chữ gốc."}
        (verified if supported else rejected).append(verdict)
    return {
        "fact_check_label": "support" if verified and not rejected else "not_enough_info",
        "verified_claims": verified, "rejected_claims": rejected, "evidence_gaps": [],
        "reasoning_trace": [{"agent": "fact_checker", "action": "checked_figure_source_text", "verified": len(verified)}],
    }


def write_visual_answer(state: AgentState) -> dict:
    linked = {eid for claim in state.get("verified_claims") or [] for eid in claim.get("evidence_ids", [])}
    refs = build_citation_references([s for s in state.get("evidence_spans") or [] if s["id"] in linked])
    report = state["answer"]
    if refs:
        report += "\n\nNguồn PDF · " + " · ".join(f'[{r["ref_id"]}] (Trang {r["page"]})' for r in refs)
    # A verified caption does not establish the accuracy of a pixel interpretation.
    return {"final_report": report, "confidence": "low",
            "written_report_data": {"citation_renderer": "figure_report", "citation_references": refs}}


def _illustrates_answer(figure: dict, spans: list, query_terms: set[str]) -> bool:
    caption_terms = _tokens(figure["caption"] + " " + figure["embedded_text"])
    number = _NUMBERED.findall(_fold(figure["label"]))
    for span in spans:
        if span.get("doc_id") != figure["doc_id"]:
            continue
        if span.get("source_unit_id") in {f'{figure["figure_id"]}:{field}' for field in ("caption", "embedded_text")}:
            return True
        if not figure.get("unique_title"):
            continue
        quote = span.get("quote") or span.get("text") or ""
        if number and number[0] in _NUMBERED.findall(_fold(quote)):
            return True
        if len(caption_terms & query_terms & _tokens(quote)) >= 2:
            return True
    return False


def attach_answer_figures(state: AgentState, result: dict) -> dict:
    context = figure_context(state)
    if not context:
        return result
    figures = context["figures"]
    if not context["explicit"]:
        if state.get("answer_status") in {"NOT_FOUND_IN_DOCUMENT", "CLARIFICATION_NEEDED"}:
            return result
        query = state.get("query", "")
        terms = _tokens(query)
        for translation in expand_bilingual_tokens(query, set()):
            terms.update(_tokens(translation))
        spans = state.get("evidence_spans") or []
        figures = [f for f in figures if _illustrates_answer(f, spans, terms)]
    if not figures:
        return result
    attachments = [{key: f[key] for key in ("figure_id", "source_path", "page_no", "caption", "label")} for f in figures]
    return {**result, "reasoning_trace": [*(result.get("reasoning_trace") or []),
                                          {"node": "writer_node", "action": "answer_figures", "figures": attachments}]}
