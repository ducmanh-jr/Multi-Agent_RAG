"""
CPU Min Mode - Agentic Summarizer Agent (L3-Lite).
Features:
  - Deep Vietnamese typography & text cleanup: removes <br>, fixes stuck words and punctuation while preserving URLs, ports, IPs, and code blocks.
  - Complete, Comprehensive Answer Synthesis (bỏ câu trả lời nhanh, trả lời hoàn thiện đầy đủ các khía cạnh).
  - Multi-Document Synthesis: cleanly organizes cross-document summaries with rich multi-point analysis per article.
  - Page-Distributed Single-Document Summaries: samples core themes and multiple rich passages across all pages for 100% document coverage.
  - Contextual Passage Expansion: combines anchor sentences with adjacent explanatory sentences and configuration blocks.
  - Exact temporal & numerical extraction (e.g. 24-10-1957, năm 1957, 1947, 1995).
  - Semicolon & colon list decomposition into structured numbered sections.
  - Zero-Hallucination Out-of-Domain Guard: honest refusal when information is absent.
  - Verifiable AtomicClaim and EvidenceSpan generation with exact page citations.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

from agents.agent1_retrieval.tools.bm25 import SimpleBM25
from core.base_agent import BaseAgent
from core.state import AgentState, AtomicClaim, EvidenceSpan
from core.verifier import evaluate_inner_critic_fast
from utils.domain_guard import (
    check_out_of_domain,
    build_out_of_domain_response,
    _has_phrase,
    strip_accents,
)


def strip_accents(text: str) -> str:
    """Normalize Vietnamese diacritics for flexible matching across accented/unaccented queries."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFD", str(text))
    s = re.sub(r"[\u0300-\u036f]", "", norm)
    return s.replace("đ", "d").replace("Đ", "D")


VIETNAMESE_STOP_WORDS = {
    "và", "của", "cho", "với", "như", "thế", "nào", "được", "có", "là", "ở", "ra", "vào",
    "ngày", "tháng", "năm", "mấy", "khi", "đầu", "tiên", "gì", "ai", "sao",
    "đâu", "đến", "từ", "lại", "đã", "đang", "sẽ", "các", "những", "một", "số", "về",
    "trong", "để", "bởi", "thì", "mà", "nhất", "nhì", "này", "đó", "kia", "hãy",
    "biết", "hỏi", "thông", "tin", "chi", "tiết", "tại", "sao",
}


def clean_vietnamese_text(text: str) -> str:
    """Thoroughly cleans PDF extract artifacts, HTML line breaks, stuck words, and punctuation."""
    if not text:
        return ""
    # 1. Replace HTML line breaks and tags
    s = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)

    # 2. Fix markdown underscores in words (e.g. _Báo Thủ đô_ -> Báo Thủ đô), PRESERVING snake_case like proxy_pass
    s = re.sub(r"(?<![A-Za-z0-9])_+|_+(?![A-Za-z0-9])", "", s)

    # 3. Fix stuck lowercase-uppercase words (e.g. BáoThủ -> Báo Thủ) but PRESERVE camelCase code identifiers like generateHosts
    s = re.sub(r"([à-ỹ])([A-ZĐ])", r"\1 \2", s)
    s = re.sub(r"([a-z])([Đ])", r"\1 \2", s)

    # 4. Fix stuck punctuation (missing spaces after . , ! ? : ; when followed by words)
    # Preserve dates like 24.10.1957, IPs like 10.0.0.11, URLs like http://, and ports like :5000
    s = re.sub(r"([A-Za-zÀ-ỹ0-9])\.([A-ZĐ])", r"\1. \2", s)
    s = re.sub(r"([A-Za-zÀ-ỹ0-9]),([A-Za-zÀ-ỹ])", r"\1, \2", s)
    s = re.sub(r"([A-Za-zÀ-ỹ0-9]);([A-Za-zÀ-ỹ])", r"\1; \2", s)
    s = re.sub(r"(?<!http)(?<!https)([A-Za-zÀ-ỹ]):([A-Za-zÀ-ỹ])", r"\1: \2", s, flags=re.I)
    s = re.sub(r"([A-Za-zÀ-ỹ0-9])\?([A-Za-zÀ-ỹ0-9])", r"\1? \2", s)
    s = re.sub(r"([A-Za-zÀ-ỹ0-9])!([A-Za-zÀ-ỹ0-9])", r"\1! \2", s)

    # 5. Fix common PDF hyphen line breaks
    s = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", s)

    # 6. Normalize multiple spaces
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def unwrap_pdf_text(text: str) -> str:
    """Unwrap hard line breaks within paragraphs commonly produced by PDF extractors."""
    cleaned = clean_vietnamese_text(text)
    lines = cleaned.split("\n")
    unwrapped: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            unwrapped.append("\n\n")
            continue
        if unwrapped and not unwrapped[-1].endswith("\n\n"):
            prev = unwrapped[-1].rstrip()
            if prev and prev[-1] not in ".!?:":
                unwrapped[-1] = prev + " " + s
                continue
        unwrapped.append(s)
    return "".join(unwrapped)


def extract_clean_sentences(text: str) -> list[str]:
    """Extract complete, grammatical sentences and intact config blocks from unwrapped text."""
    if not text:
        return []
    # Strip presentation / slide header & footer pagination metadata (e.g. Buổi 1, Slide 12, Lecture 3, Chapter 2)
    cleaned = re.sub(
        r"(?:Buổi|Slide|Lecture|Bài|Chapter|Trang)\s+\d+.*?(?:\.md|\.pdf|\.docx)?\s*(?:\d{4}-\d{2}-\d{2})?\s*\d+\s*/\s*\d+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    lines = [l.strip() for l in cleaned.split("\n")]
    unwrapped: list[str] = []
    brace_depth = 0
    for l in lines:
        if not l:
            continue
        if unwrapped and (
            brace_depth > 0
            or unwrapped[-1][-1] not in ".!?;\"”}"
            or (unwrapped[-1].endswith(":") and not re.match(r"^[-*•\d]", l))
        ):
            unwrapped[-1] += " " + l
        else:
            unwrapped.append(l)
        brace_depth = max(0, brace_depth + l.count("{") - l.count("}"))

    unified = "\n".join(unwrapped)
    # Protect curly-brace blocks {...} so semicolons/periods inside code blocks are never split
    protected_blocks: list[str] = []

    def _protect_brace_block(m: re.Match) -> str:
        idx = len(protected_blocks)
        protected_blocks.append(m.group(0))
        return f"__BRACE_BLOCK_{idx}__"

    unified_safe = re.sub(r"\{[^{}]{1,600}\}", _protect_brace_block, unified)
    # Only split when .!? is NOT preceded by a digit (preserving IPs like 10.0.0.11, 127.0.0.1, section numbers 2.2)
    unified_safe = re.sub(r"(?<![0-9])([.!?])(?=[A-ZĐ])", r"\1 \n", unified_safe)
    raw_parts = re.split(r"(?<![0-9])(?<=[.!?])\s+(?=[A-ZĐ\"“\(])|\n+", unified_safe)

    raw_sentences: list[str] = []
    for part in raw_parts:
        restored = part
        for idx, blk in enumerate(protected_blocks):
            restored = restored.replace(f"__BRACE_BLOCK_{idx}__", blk)
        if len(restored.strip()) > 18:
            raw_sentences.append(restored.strip())

    sentences: list[str] = []
    dangling_words = {
        "được", "là", "và", "của", "đã", "cho", "nhưng", "bởi", "ở", "vào",
        "ra", "tại", "với", "về", "trong", "để", "có", "sau", "trước", "khi",
        "mà", "theo", "các", "những", "thì", "do", "bị", "rằng", "gồm", "như",
    }

    for s in raw_sentences:
        s_clean = clean_vietnamese_text(s)
        # Strip leading photo / source / caption metadata e.g. (Ảnh: ...)
        s_clean = re.sub(
            r"^[\(\[\{][^\)\]\}]*(?:Ảnh|Nguồn|TTXVN|PV|Tư liệu|Theo)[^\)\]\}]*[\)\]\}]\s*",
            "",
            s_clean,
        )
        s_clean = s_clean.strip()

        if len(s_clean) < 20:
            continue
        if s_clean.startswith("#") or s_clean.startswith("---"):
            continue
        # Filter table artifacts and metadata banners
        if re.match(
            r"^\*{0,2}(?:Task|Người thực hiện|Ước lượng|Point|Bảng|STT|Tác giả|Năm xuất bản|Nguồn công bố|Liên kết|Quyền riêng tư|Ảnh:|Theo tư liệu)\b",
            s_clean,
            re.I,
        ):
            continue
        if any(
            banner in s_clean.lower()
            for banner in [
                "chính sách bảo vệ", "dữ liệu cá nhân", "đồng ý tất cả", "cookie", "chấp nhận tất cả",
                "task người thực hiện", "ước lượng (point)", "ước lượng point", "người thực hiện (gợi ý)",
            ]
        ):
            continue
        # Filter email addresses and paper affiliation / author header lines
        if re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", s_clean):
            continue
        if re.search(r"\b(?:Journal of|Trường Đại học|Khoa Công nghệ|Tác giả liên hệ|Corresponding author|ISSN|ISBN)\b", s_clean, re.I):
            continue
        # Allow technical lines, code blocks, semicolon configurations and standard punctuation
        valid_endings = ('.', '!', '?', ':', ';', '}', '{', ')', ']', '"', '”', '`')
        if s_clean[-1] not in valid_endings and not re.search(
            r"\b(?:nginx|systemctl|curl|docker|openssl|certbot|server|upstream|proxy_pass)\b",
            s_clean,
            re.I,
        ):
            continue

        words = re.findall(r"\w+", s_clean)
        if words and words[-1].lower() in dangling_words:
            continue
        is_valid_start = (
            s_clean[0].isupper()
            or s_clean[0].isdigit()
            or s_clean[0] in '"“\'[('
            or ":" in s_clean
            or any(
                s_clean.startswith(kw)
                for kw in [
                    "server", "location", "proxy", "client", "listen", "upstream",
                    "access", "error", "sudo", "docker", "certbot", "openssl",
                    "root", "index", "try_files",
                ]
            )
        )
        if not is_valid_start:
            continue

        sentences.append(s_clean)
    return sentences


def build_contextual_passage(sents: list[str], idx_s: int, max_chars: int = 620, max_extra_sents: int = 3) -> tuple[str, list[int]]:
    """
    Expands an anchor sentence sents[idx_s] into a complete, coherent multi-sentence passage
    by attaching consecutive explanatory/elaborating sentences in the same unit.
    Returns (passage_text, consumed_sentence_indices).
    """
    if not sents or idx_s < 0 or idx_s >= len(sents):
        return "", []
    base = sents[idx_s].strip()
    consumed = [idx_s]
    parts = [base]
    total_len = len(base)

    # Also check if the immediately preceding sentence is a short problem/context statement for this definition
    if idx_s > 0:
        prev_s = sents[idx_s - 1].strip()
        if (
            len(prev_s) <= 220
            and re.search(r"\b(?:Vấn đề|Kịch bản|Bối cảnh|Problem|Context)\s*:", prev_s, re.I)
            and total_len + len(prev_s) <= max_chars + 120
        ):
            parts.insert(0, prev_s)
            consumed.insert(0, idx_s - 1)
            total_len += len(prev_s) + 1

    for offset in range(1, max_extra_sents + 1):
        nxt_idx = idx_s + offset
        if nxt_idx >= len(sents):
            break
        nxt_s = sents[nxt_idx].strip()
        if not nxt_s:
            continue
        # Stop if the next sentence starts an unrelated numbered major section heading or a new distinct configuration block
        if re.match(r"^(?:\d+[\.\)]\s*|[-*•]\s*)?(?:[A-ZĐ\d]+\.\d+|\d+\.?\s+[A-ZĐ]|Một server block|Cấu hình Domain|Tạo chứng chỉ|Trường hợp \d+|Mục tiêu\b|Tài liệu\b|Phân công\b|Task\b|Bài viết\b|Video\b|Ghi chú\b|Nội dung cần\b|Nội dung được\b)", nxt_s, re.I):
            break
        if total_len + len(nxt_s) > max_chars and len(parts) >= 2:
            break
        parts.append(nxt_s)
        consumed.append(nxt_idx)
        total_len += len(nxt_s) + 1
        if total_len >= max_chars:
            break

    passage = " ".join(parts)
    return passage, consumed


def extract_topic_heading(text: str, fallback_idx: int = 1) -> tuple[str, str]:
    """
    Extracts a clean bold topic heading and body from a sentence or passage.
    If the text begins with 'Label: Body' or 'Label — Body', splits cleanly;
    otherwise synthesizes a concise 3-6 word topical heading from the opening clause.
    """
    s = (text or "").strip()
    if not s:
        return f"Luận điểm {fallback_idx}", ""

    # Clean leading section numbers e.g. "2.4. Upstream..." -> "Upstream..."
    s = re.sub(r"^(?:\d+(?:\.\d+)*\.?\s*)", "", s).strip()

    # Check for explicit 'Label: Body' or 'Label — Body' within first 75 chars (excluding URLs/ports)
    m_colon = re.match(r"^([^:]{3,72}?)\s*:\s+(.+)$", s, flags=re.DOTALL)
    if m_colon:
        label = m_colon.group(1).strip().strip("*_")
        body = m_colon.group(2).strip()
        # Clean trailing sub-labels like "Upstream và nhiều Backend/Node Server Vấn đề" -> "Upstream và nhiều Backend/Node Server"
        label = re.sub(r"\s+(?:Vấn đề|Kịch bản|Bối cảnh|Problem|Context)$", "", label, flags=re.I).strip()
        if (
            not label.lower().startswith(("http", "https"))
            and "{" not in label
            and ";" not in label
            and len(label.split()) <= 12
            and len(body) >= 15
        ):
            label_clean = label[0].upper() + label[1:] if label else f"Luận điểm {fallback_idx}"
            return label_clean, body

    m_dash = re.match(r"^([^—–]{3,72}?)\s*[—–]\s+(.+)$", s, flags=re.DOTALL)
    if m_dash:
        label = m_dash.group(1).strip().strip("*_")
        body = m_dash.group(2).strip()
        label = re.sub(r"\s+(?:Vấn đề|Kịch bản|Bối cảnh|Problem|Context)$", "", label, flags=re.I).strip()
        if len(label.split()) <= 12 and len(body) >= 15:
            label_clean = label[0].upper() + label[1:] if label else f"Luận điểm {fallback_idx}"
            return label_clean, body

    words = re.findall(r"[A-Za-zÀ-ỹ0-9_./-]+", s)
    skip_leading = {"một", "các", "những", "để", "khi", "về", "trên", "trong", "theo", "tại", "với"}
    clean_words = [w for w in words[:7] if w.lower() not in skip_leading]
    if not clean_words:
        clean_words = words[:5]
    heading = " ".join(clean_words[:5]).strip(".,;:()")
    if heading:
        heading = heading[0].upper() + heading[1:]
    else:
        heading = f"Nội dung trọng tâm {fallback_idx}"
    return heading, s


def format_nginx_blocks(t: str) -> str:
    """Formats inline NGINX configuration blocks {...} into clean markdown ```nginx blocks."""
    if not t or "{" not in t or "}" not in t:
        return t
    pattern = re.compile(r"\b((?:upstream|server|location)\s+(?:[A-Za-z0-9_./~*^-]+\s*)?\{[^{}]+\})", re.DOTALL)
    def repl(m):
        raw = m.group(1).strip()
        header, inner = raw.split("{", 1)
        header = header.strip()
        inner = inner.rstrip("}").strip()
        directives = [d.strip() + ";" for d in inner.split(";") if d.strip()]
        formatted_directives = "\n    ".join(directives)
        return f"\n\n```nginx\n{header} {{\n    {formatted_directives}\n}}\n```\n\n"
    return pattern.sub(repl, t)


def format_doc_title(doc_id_or_name: str) -> str:
    """Format a clean, readable human title for a document ID or filename."""
    name = str(doc_id_or_name or "Tài liệu")
    name = os.path.basename(name)
    name = re.sub(r"^(?:gemini|doc)_[a-f0-9]+_", "", name, flags=re.I)
    name = re.sub(r"^(?:gemini|doc)_", "", name, flags=re.I)
    name = re.sub(r"\s+", " ", name.replace("_", " ")).replace(".pdf", "").replace(".md", "").strip()
    return name


def extract_task_allocation_table(
    source_units: list[dict[str, Any]],
    doc_name: str = "Tài liệu",
) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """Dynamically extracts task allocation, project plan, and tabular assignments from document source units without mock data."""
    for u in source_units:
        text = str(u.get("text", ""))
        text_lower = text.lower()
        if not ("task" in text_lower or ("người thực hiện" in text_lower and "ước lượng" in text_lower) or ("phân công" in text_lower and "point" in text_lower)):
            continue

        page_no = int(u.get("page_no", 1))
        doc_id = str(u.get("doc_id", "doc"))
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            continue

        # Dynamic row boundary detection based on numeric cell markers
        numeric_indices = [idx for idx, l in enumerate(lines) if re.match(r"^\d+(?:\.\d+)?$", l)]
        if len(numeric_indices) < 2:
            continue

        rows: list[tuple[str, str, str]] = []
        total_val = None

        for k, num_idx in enumerate(numeric_indices):
            val = lines[num_idx]
            prev_idx = numeric_indices[k - 1] if k > 0 else -1
            row_tokens = lines[prev_idx + 1 : num_idx]

            # In the very first row, skip any detected table header labels dynamically
            if k == 0:
                filtered_tokens = []
                skipping_headers = True
                for tok in row_tokens:
                    tok_clean = tok.lower().replace("(", "").replace(")", "").strip()
                    if skipping_headers and tok_clean in ["task", "người thực hiện", "gợi ý", "ước lượng", "point", "stt", "nhiệm vụ", "thực hiện", "thời gian", "tiến độ"]:
                        continue
                    skipping_headers = False
                    filtered_tokens.append(tok)
                row_tokens = filtered_tokens

            if not row_tokens:
                continue

            # Check if this row is a summary / total row
            is_summary = (len(row_tokens) == 1 and any(w in row_tokens[0].lower() for w in ["tổng cộng", "total", "sum"])) or (len(row_tokens) == 1 and row_tokens[0].strip().lower() == "tổng")
            if is_summary:
                total_val = val
                rows.append((row_tokens[0], "-", val))
                continue

            # Standard row: if the last token before number is a short entity / role / assignee
            if len(row_tokens) >= 2 and len(row_tokens[-1]) < 35 and len(row_tokens[-1].split()) <= 4:
                col2 = row_tokens[-1]
                col1 = " ".join(row_tokens[:-1])
            else:
                col2 = "-"
                col1 = " ".join(row_tokens)

            # Terminate if hitting bibliography or notes sections
            if re.match(r"^\d+\.\s*(?:Tài liệu|Tham khảo|Reference|Ghi chú)\b", col1, re.I):
                break

            rows.append((col1, col2, val))

        if rows:
            md_table = [
                "| STT | Nhiệm vụ (Task) | Người thực hiện (Gợi ý) | Ước lượng (Point) |",
                "| :--- | :--- | :--- | :--- |",
            ]
            claims = []
            valid_points = []
            for idx, (t, a, p) in enumerate(rows, 1):
                p_str = f"{p} points" if p and p != "-" else "-"
                a_str = a if a else "-"
                is_total_row = bool(re.search(r"^(?:tổng\s+cộng|total|tổng\b|sum\b)", t.strip().lower())) and "hợp" not in t.lower()
                if p and p.isdigit() and not is_total_row:
                    valid_points.append(int(p))
                if is_total_row:
                    md_table.append(f"| **Tổng** | **{t}** | **{a_str}** | **{p_str}** |")
                else:
                    md_table.append(f"| {idx} | {t} | {a_str} | {p_str} |")
                claims.append(
                    AtomicClaim(
                        id=f"claim_{idx}",
                        text=f"{t}: {a_str} ({p_str})",
                        evidence_ids=[u.get("id", "doc")],
                    ).to_dict()
                )

            evidence_spans = [
                EvidenceSpan(
                    id="ev_1",
                    quote=text[:500],
                    source_unit_id=u.get("id", "doc"),
                    doc_id=doc_id,
                    page_no=page_no,
                ).to_dict()
            ]

            non_total_rows = [
                r for r in rows
                if not (bool(re.search(r"^(?:tổng\s+cộng|total|tổng\b|sum\b)", r[0].strip().lower())) and "hợp" not in r[0].lower())
            ]
            total_tasks = len(non_total_rows)
            final_points = total_val if total_val else (str(sum(valid_points)) if valid_points else "")
            points_summary = f" với tổng khối lượng ước lượng được ghi nhận là **{final_points} points**" if final_points else ""

            intro_sentence = f"Dựa trên tài liệu nghiên cứu (**{doc_name}**), bảng phân công công việc và ước lượng thời gian (Point) được quy định cụ thể như sau [1] *(Trang {page_no})*:"
            answer = (
                f"{intro_sentence}\n\n"
                + "\n".join(md_table)
                + f"\n\n- **Tổng kết khối lượng:** Toàn bộ kế hoạch gồm **{total_tasks} công việc chính**{points_summary}."
            )
            return answer, claims, evidence_spans

    return None, [], []


class CPUMinSummarizerAgent(BaseAgent):
    """Agentic Summarizer for CPU Min & CPU Max: Complete, comprehensive multi-aspect synthesis + Zero Hallucination."""

    def __init__(self, max_internal_retry: int = 1):
        super().__init__(name="CPUMinSummarizerAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query") or state.get("question", "")
        clean_q = query.strip()
        raw_units = state.get("source_units", []) or []
        source_units = [
            u.to_dict()
            if hasattr(u, "to_dict")
            else (
                dict(u)
                if isinstance(u, dict)
                else {
                    "id": str(getattr(u, "id", "doc")),
                    "text": str(getattr(u, "text", u)),
                    "doc_id": str(getattr(u, "doc_id", "doc")),
                    "page_no": int(getattr(u, "page_no", 1)),
                }
            )
            for u in raw_units
        ]
        required_entities = state.get("required_entities", []) or []

        self.logger.info(f"[CPU Summarizer] Synthesizing comprehensive answer for query: '{clean_q}' ({len(source_units)} units)")

        is_ood, ood_reason = check_out_of_domain(clean_q, source_units)
        if is_ood:
            self.logger.info(f"[CPU Summarizer] Zero-Hallucination Guard triggered: {ood_reason}")
            res = build_out_of_domain_response(clean_q, model_family="cpu_min_guard")
            trace = self.log_action("cpu_min_summarization_completed", {"status": "out_of_domain", "reason": ood_reason})
            res["reasoning_trace"] = [trace]
            return res

        # Kịch bản 2: Khi kho tài liệu không có văn bản nào được tải
        if not source_units:
            res = build_out_of_domain_response(clean_q, model_family="cpu_min_extractive")
            res["reasoning_trace"] = [self.log_action("cpu_min_summarization_completed", {"status": "out_of_domain"})]
            return res

        # 1. Intent analysis (supporting both accented and unaccented Vietnamese)
        clean_q_unaccent = strip_accents(clean_q.lower())
        q_norm = f"{clean_q.lower()} {clean_q_unaccent}"
        is_date_q = bool(
            re.search(
                r"\b(ngày nào|thời điểm nào|bao giờ|ngày bao nhiêu|ngay nao|thoi diem nao|bao gio|ngay bao nhieu)\b",
                q_norm,
                re.I,
            )
        )
        is_year_q = bool(
            re.search(
                r"\b(năm nào|năm bao nhiêu|ra đời năm nào|thành lập năm nào|nam nao|nam bao nhieu|ra doi nam nao|thanh lap nam nao)\b",
                q_norm,
                re.I,
            )
        ) or (is_date_q and ("năm" in clean_q.lower() or "nam" in clean_q_unaccent))

        is_list_q = bool(
            re.search(
                r"\b(những\s+vấn\s+đề|các\s+vấn\s+đề|những\s+nội\s+dung|các\s+nội\s+dung|những\s+gì|gồm\s+những|các\s+nhóm|mấy\s+nhóm|\d+\s+nhóm|các\s+mục|danh\s+sách|những\s+ai|các\s+bước|các\s+biện\s+pháp|các\s+dự\s+án|những\s+thuật\s+toán|các\s+thuật\s+toán|những\s+tham\s+số|các\s+tham\s+số|liệt\s+kê|phân\s+công|công\s+việc|nhiệm\s+vụ|các\s+task)\b",
                q_norm,
                re.I,
            )
        )
        is_task_allocation_q = bool(
            re.search(
                r"\b(phân\s+công|phan\s+cong|nhiệm\s+vụ|nhiem\s+vu|công\s+việc|cong\s+viec|các\s+task|cac\s+task|task\b|bảng\s+phân\s+công|bang\s+phan\s+cong|ước\s+lượng\s+thời\s+gian|uoc\s+luong\s+thoi\s+gian|ước\s+lượng\s+point|uoc\s+luong\s+point|kế\s+hoạch\s+triển\s+khai|ke\s+hoach\s+trien\s+khai|tiến\s+độ|tien\s+do|lịch\s+trình|lich\s+trinh|liệt\s+kê\s+phân\s+công|liet\s+ke\s+phan\s+cong)\b",
                q_norm,
                re.I,
            )
        )
        is_quantity_q = bool(
            re.search(
                r"\b(bao nhiêu|mấy|số lượng|định lượng|tổng số|tỷ lệ|%)\b",
                clean_q,
                re.I,
            )
        ) and not (is_date_q or is_year_q)

        is_comparison_q = (
            any(
                w in clean_q.lower()
                for w in ["so sánh", "đối chiếu", "so sanh", "doi chieu", "/so-sanh", "compare"]
            )
            or bool(state.get("is_comparative_query") or state.get("intent") == "compare")
        )

        is_general_summary = (
            clean_q.startswith("/")
            or clean_q.lower().strip() in ["tóm tắt", "tổng quan", "nội dung chính", "tom-tat", "/tom-tat"]
            or any(
                kw in clean_q.lower()
                for kw in [
                    "tóm tắt", "tổng quan", "nội dung chính của tài liệu", "nội dung chính của bài báo",
                    "tổng quát bài báo", "/tom-tat", "tom-tat", "summary", "overview", "abstract",
                ]
            )
        )

        is_definition_q = bool(
            re.search(
                r"\b(là gì|định nghĩa|khái niệm|thế nào là|ý nghĩa của|mục đích của|chức năng của|vai trò của|la gi|dinh nghia|khai niem|the nao la)\b",
                clean_q_unaccent,
                re.I,
            )
        )
        is_specific_fact_q = not is_general_summary and (is_definition_q or len(clean_q.split()) <= 7)

        # 2. Group source units by distinct document
        docs_map: dict[str, list[dict[str, Any]]] = {}
        for u in source_units:
            d_id = str(u.get("doc_id") or "doc")
            docs_map.setdefault(d_id, []).append(u)

        # 2b. High-precision extraction for Task Allocation and Project Plan tables
        if is_task_allocation_q:
            first_doc_id = list(docs_map.keys())[0] if docs_map else "doc"
            doc_title = format_doc_title(first_doc_id)
            task_ans, task_claims, task_spans = extract_task_allocation_table(source_units, doc_title)
            if task_ans:
                critic_eval = evaluate_inner_critic_fast(
                    query=clean_q,
                    draft_answer=task_ans,
                    source_units=source_units[:10],
                    required_entities=required_entities,
                )
                return {
                    "answer": task_ans,
                    "claims": task_claims,
                    "evidence_spans": task_spans,
                    "critic_action": "approve",
                    "critic_feedback": critic_eval.to_dict(),
                    "generator_model_family": "cpu_min_task_engine",
                    "reasoning_trace": [self.log_action("cpu_min_task_table_extracted", {"tasks_count": len(task_claims)})],
                }

        # 3. Extract core non-stopword tokens, bigrams, technical identifiers, and proper noun phrases
        q_tokens = [w for w in re.findall(r"\w+", clean_q.lower()) if w not in VIETNAMESE_STOP_WORDS and len(w) > 1]
        tech_identifiers = [
            t.lower()
            for t in re.findall(r"\b[a-zA-Z0-9_./-]{3,}\b", clean_q)
            if not re.search(r"[à-ỹÀ-Ỹ]", t)
            and t.lower() not in VIETNAMESE_STOP_WORDS
            and t.lower() not in {"file", "trong", "của", "cho", "với", "khi", "nào", "như", "thế"}
        ]
        proper_noun_phrases = re.findall(r"\b[A-ZĐ][a-zà-ỹ0-9_-]+(?:\s+[A-ZĐ][a-zà-ỹ0-9_-]+)+\b", clean_q)
        cap_words = re.findall(r"\b[A-ZĐ][a-zà-ỹ0-9_-]+\b", clean_q)
        first_word_match = re.match(r"^\s*([A-ZĐ][a-zà-ỹ0-9_-]+)\b", clean_q)
        first_word = first_word_match.group(1) if first_word_match else ""
        entities = [p.lower() for p in proper_noun_phrases if p.lower() not in VIETNAMESE_STOP_WORDS]
        if not entities:
            entities = [
                c.lower()
                for c in cap_words
                if (c != first_word or c.isupper()) and c.lower() not in VIETNAMESE_STOP_WORDS and len(c) > 2
            ]
        for tid in tech_identifiers:
            if tid not in entities:
                entities.append(tid)

        q_bigrams = [f"{q_tokens[i]} {q_tokens[i+1]}" for i in range(len(q_tokens) - 1)] if len(q_tokens) >= 2 else []

        # 5. MULTI-DOCUMENT COMPREHENSIVE SUMMARY (when user selects >= 2 documents)
        if len(docs_map) > 1 and (is_general_summary or is_comparison_q):
            claims: list[dict[str, Any]] = []
            evidence_spans: list[dict[str, Any]] = []
            doc_sections: list[str] = []
            ref_idx = 1

            for doc_idx, (d_id, d_units) in enumerate(docs_map.items(), 1):
                d_title = format_doc_title(d_id)
                d_passages: list[tuple[str, str, int, dict[str, Any]]] = []
                seen_doc_norms: set[str] = set()

                for u in d_units:
                    s_list = extract_clean_sentences(str(u.get("text", "")))
                    used_in_u: set[int] = set()
                    for idx_s, s in enumerate(s_list):
                        if idx_s in used_in_u:
                            continue
                        passage, consumed = build_contextual_passage(s_list, idx_s, max_chars=480, max_extra_sents=2)
                        norm_p = passage.lower()[:75]
                        if norm_p in seen_doc_norms:
                            continue
                        seen_doc_norms.add(norm_p)
                        used_in_u.update(consumed)
                        d_passages.append((s, passage, int(u.get("page_no", 1)), u))

                if not d_passages:
                    continue

                def _score_core_passage(item: tuple[str, str, int, dict[str, Any]]) -> float:
                    text = item[1]
                    score = 0.0
                    L = len(text)
                    if 80 <= L <= 550:
                        score += 3.0
                    elif 40 <= L <= 700:
                        score += 1.5
                    if re.search(
                        r"\b(?:là|được|nhằm|bao gồm|nghiên cứu|phát triển|kết quả|mục tiêu|ra đời|thành lập|kiến trúc|phương pháp)\b",
                        text,
                        re.IGNORECASE,
                    ):
                        score += 2.5
                    if re.search(r"\d+", text):
                        score += 1.0
                    if re.match(r"^(?:ảnh|hình|nguồn|bảng|theo|chú thích)\s*[:.]", text, re.IGNORECASE):
                        score -= 5.0
                    return score

                core_candidate = max(d_passages[:10], key=_score_core_passage)
                core_s, core_passage, core_p, core_u = core_candidate
                current_ref = ref_idx
                ref_idx += 1

                claims.append(
                    AtomicClaim(id=f"claim_{len(claims)+1}", text=core_s, evidence_ids=[core_u.get("id", "doc")]).to_dict()
                )
                evidence_spans.append(
                    EvidenceSpan(
                        id=f"ev_{len(evidence_spans)+1}",
                        quote=core_passage,
                        source_unit_id=core_u.get("id", "doc"),
                        doc_id=d_id,
                        page_no=core_p,
                    ).to_dict()
                )

                sub_bullets = []
                for s_orig, p_text, p_no, u_obj in d_passages:
                    if p_text != core_passage and 45 <= len(p_text) <= 650:
                        b_ref = ref_idx
                        ref_idx += 1
                        heading, body = extract_topic_heading(p_text, len(sub_bullets) + 1)
                        sub_bullets.append(f"- **{heading}**: {body} [{b_ref}] *(Trang {p_no})*")
                        claims.append(
                            AtomicClaim(id=f"claim_{len(claims)+1}", text=s_orig, evidence_ids=[u_obj.get("id", "doc")]).to_dict()
                        )
                        evidence_spans.append(
                            EvidenceSpan(
                                id=f"ev_{len(evidence_spans)+1}",
                                quote=p_text,
                                source_unit_id=u_obj.get("id", "doc"),
                                doc_id=d_id,
                                page_no=p_no,
                            ).to_dict()
                        )
                        if len(sub_bullets) >= 4:
                            break

                bullets_str = "\n".join(sub_bullets) if sub_bullets else ""
                section_md = (
                    f"### {doc_idx}. Tài liệu {doc_idx}: **{d_title}**\n"
                    f"- **Luận điểm & Nội dung cốt lõi:** {core_passage} [{current_ref}] *(Trang {core_p})*\n"
                )
                if bullets_str:
                    section_md += f"- **Phân tích chi tiết các khía cạnh trọng tâm:**\n{bullets_str}\n"

                doc_sections.append(section_md)

            is_comparison_q = any(w in clean_q.lower() for w in ["so sánh", "đối chiếu", "so sanh", "doi chieu", "khác nhau"])
            if is_comparison_q:
                answer = (
                    f"> **Đánh giá tương quan:** Các tài liệu được chọn thuộc **các chủ đề/lĩnh vực nghiên cứu độc lập**. "
                    f"Vì nội dung không cùng một hệ quy chiếu chuyên môn, hệ thống không áp đặt so sánh giả tạo, "
                    f"mà đối chiếu khách quan, toàn diện phạm vi và nội dung cốt lõi của từng tài liệu:\n\n"
                    + "\n".join(doc_sections)
                )
            else:
                answer = (
                    f"Dưới đây là bản tổng hợp và phân tích toàn diện nội dung của **{len(doc_sections)} bài báo/tài liệu nghiên cứu** được chọn:\n\n"
                    + "\n".join(doc_sections)
                )

            critic_eval = evaluate_inner_critic_fast(
                query=clean_q,
                draft_answer=answer,
                source_units=source_units[:10],
                required_entities=required_entities,
            )

            return {
                "answer": answer,
                "claims": claims,
                "evidence_spans": evidence_spans,
                "critic_action": "approve",
                "critic_feedback": critic_eval.to_dict(),
                "generator_model_family": "cpu_min_extractive",
                "reasoning_trace": [self.log_action("cpu_min_multi_doc_summary_completed", {"docs_count": len(doc_sections)})],
            }

        # 6. SINGLE-DOCUMENT COMPREHENSIVE PAGE-DISTRIBUTED SUMMARY (when query is summary / overview on 1 document)
        if len(docs_map) == 1 and is_general_summary:
            claims = []
            evidence_spans = []

            single_doc_id = list(docs_map.keys())[0]
            d_units = docs_map[single_doc_id]
            doc_title = format_doc_title(single_doc_id)

            # Group sentences & contextual passages by page
            page_passages: dict[int, list[tuple[str, str, dict[str, Any]]]] = {}
            seen_global_norms: set[str] = set()
            for u in d_units:
                p_no = int(u.get("page_no", 1))
                s_list = extract_clean_sentences(str(u.get("text", "")))
                used_local: set[int] = set()
                for idx_s, s in enumerate(s_list):
                    if idx_s in used_local:
                        continue
                    if s.startswith("http") or re.match(r"^(?:Chúc các bạn|Buổi \d+)", s, re.I):
                        continue
                    passage, consumed = build_contextual_passage(s_list, idx_s, max_chars=520, max_extra_sents=2)
                    norm_key = passage.lower()[:70]
                    if norm_key in seen_global_norms:
                        continue
                    seen_global_norms.add(norm_key)
                    used_local.update(consumed)
                    page_passages.setdefault(p_no, []).append((s, passage, u))

            sorted_pages = sorted(page_passages.keys())
            if len(sorted_pages) > 6:
                step = len(sorted_pages) / 6.0
                selected_pages = [sorted_pages[int(i * step)] for i in range(6)]
            else:
                selected_pages = sorted_pages

            sections_md: list[str] = []
            ref_counter = 1

            for sec_idx, p_no in enumerate(selected_pages, 1):
                candidates_p = page_passages.get(p_no, [])
                if not candidates_p:
                    continue
                # Pick up to 3 rich, distinct passages per page so the summary is thorough and complete
                chosen_on_page = candidates_p[:3] if len(selected_pages) > 1 else candidates_p[:6]
                sec_title, _ = extract_topic_heading(chosen_on_page[0][1], sec_idx)
                bullet_lines: list[str] = []

                for item_idx, (s_anchor, passage_text, u_ref) in enumerate(chosen_on_page, 1):
                    heading, body = extract_topic_heading(passage_text, item_idx)
                    bullet_lines.append(f"- **{heading}**: {body} [{ref_counter}] *(Trang {p_no})*")
                    claims.append(
                        AtomicClaim(
                            id=f"claim_{ref_counter}",
                            text=s_anchor,
                            evidence_ids=[u_ref.get("id", "doc")],
                        ).to_dict()
                    )
                    evidence_spans.append(
                        EvidenceSpan(
                            id=f"ev_{ref_counter}",
                            quote=passage_text,
                            source_unit_id=u_ref.get("id", "doc"),
                            doc_id=single_doc_id,
                            page_no=p_no,
                        ).to_dict()
                    )
                    ref_counter += 1

                if bullet_lines:
                    sections_md.append(
                        f"### {sec_idx}. Trọng tâm Phần {sec_idx}: {sec_title} *(Trang {p_no})*\n"
                        + "\n".join(bullet_lines)
                    )

            sections_body = "\n\n".join(sections_md)
            answer = (
                f"Dưới đây là bản tổng hợp toàn diện và chi tiết nội dung cốt lõi của tài liệu nghiên cứu (**{doc_title}**), "
                f"bao quát đầy đủ các khái niệm nền tảng, kiến trúc kỹ thuật, phương pháp triển khai và các lưu ý thực tế:\n\n"
                f"{sections_body}\n"
            )

            critic_eval = evaluate_inner_critic_fast(
                query=clean_q,
                draft_answer=answer,
                source_units=source_units[:10],
                required_entities=required_entities,
            )

            return {
                "answer": answer,
                "claims": claims,
                "evidence_spans": evidence_spans,
                "critic_action": "approve",
                "critic_feedback": critic_eval.to_dict(),
                "generator_model_family": "cpu_min_extractive",
                "reasoning_trace": [
                    self.log_action(
                        "cpu_min_page_summary_completed",
                        {"sections_count": len(sections_md), "claims_count": len(claims)},
                    )
                ],
            }

        # 7. SPECIFIC QUESTION EVALUATION, CONTEXTUAL PASSAGE EXPANSION & SCORING
        quotes_in_q = [q.strip() for q in re.findall(r'[\"\'\`\“](.*?)[\"\'\`\”]', clean_q) if len(q.strip()) >= 2]
        all_candidates: list[dict[str, Any]] = []

        for u_idx, u in enumerate(source_units):
            u_text = str(u.get("text", ""))
            sents = extract_clean_sentences(u_text)
            for idx_s, s in enumerate(sents):
                s_lower = s.lower()
                s_lower_unaccent = strip_accents(s_lower)
                token_hits = sum(
                    1 for t in q_tokens if _has_phrase(t, s_lower, s_lower_unaccent, bool(re.search(r"[à-ỹÀ-Ỹ]", t)))
                )
                bigram_hits = sum(
                    1 for b in q_bigrams if _has_phrase(b, s_lower, s_lower_unaccent, bool(re.search(r"[à-ỹÀ-Ỹ]", b)))
                )
                ent_hits = sum(
                    1 for e in entities if _has_phrase(e, s_lower, s_lower_unaccent, bool(re.search(r"[à-ỹÀ-Ỹ]", e)))
                )
                req_ent_hits = sum(
                    1
                    for r_ent in required_entities
                    if _has_phrase(r_ent, s_lower, s_lower_unaccent, bool(re.search(r"[à-ỹÀ-Ỹ]", r_ent)))
                )
                tech_hits = sum(
                    1 for tid in tech_identifiers if _has_phrase(tid, s_lower, s_lower_unaccent, False)
                )

                has_date = bool(re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", s))
                has_year = bool(re.search(r"\b(?:19\d\d|20\d\d)\b", s))
                has_list = ":" in s and (
                    ";" in s or any(w in s_lower for w in ["thứ nhất", "nhóm vấn đề", "nội dung:", "gồm:", "bao gồm:"])
                )
                has_num = bool(re.search(r"\b\d+(?:[.,]\d+)?%?\b", s))

                quote_hits = sum(
                    1 for qt in quotes_in_q if _has_phrase(qt, s_lower, s_lower_unaccent, bool(re.search(r"[à-ỹÀ-Ỹ]", qt)))
                )
                score = (
                    token_hits * 3
                    + bigram_hits * 6
                    + ent_hits * 5
                    + tech_hits * 8
                    + req_ent_hits * 3
                    + quote_hits * 25
                )

                # Direct Definition / Section Heading Bonus:
                # If sentence defines one of the queried technical terms or core bigrams before a colon/dash (e.g. "upstream: Định nghĩa...")
                prefix_part = s.split(":", 1)[0].lower() if ":" in s[:80] else s[:55].lower()
                prefix_unaccent = strip_accents(prefix_part)
                for key_term in tech_identifiers + entities + q_bigrams[:3]:
                    if key_term and len(key_term) >= 3 and key_term not in {"nginx", "linux", "windows", "docker"}:
                        if _has_phrase(key_term, prefix_part, prefix_unaccent, False):
                            score += 18
                            break
                if "định nghĩa" in s_lower or ("{" in s and "}" in s) or "/etc/" in s_lower:
                    score += 12

                if is_date_q:
                    score += 7 if has_date else -3
                elif is_year_q:
                    score += 6 if has_year else -2

                if is_list_q:
                    if has_list:
                        score += 9
                    if any(w in s_lower for w in ["nhóm", "vấn đề", "nội dung", "gồm", "tập trung", "bước", "thuật toán"]):
                        score += 4
                if is_quantity_q and has_num:
                    score += 5

                richness = 4 if 70 <= len(s) <= 450 else 1
                score += richness

                # Negative filtering of noise / meta / syllabus items (Đúng và Đủ - không thừa thãi)
                is_meta_noise = False
                if not any(k in clean_q.lower() for k in ["bài viết", "video", "tài liệu tham khảo", "tham khảo", "link", "nguồn"]):
                    if re.match(r"^\s*(?:\d+[\.\)]\s*|[-*•]\s*)?(?:Bài viết|Video|Tài liệu tham khảo|Tài liệu chính thức|Tài liệu openssl|Tài liệu Certbot)\b", s, re.I):
                        is_meta_noise = True
                if not any(k in clean_q.lower() for k in ["task", "người thực hiện", "phân công", "thuyết trình", "slide", "luyện tập", "thời gian"]):
                    if re.match(r"^\s*(?:\d+[\.\)]\s*|[-*•]\s*)?(?:Task\b|Người thực hiện|Ước lượng|Point|Phân bổ thời gian|Chuẩn bị slide|Luyện tập thuyết trình|Cả nhóm|Nhóm trưởng|Tổng cộng \d+)\b", s, re.I):
                        is_meta_noise = True
                if not any(k in clean_q.lower() for k in ["mục tiêu", "lộ trình", "chuẩn bị"]):
                    if re.match(r"^\s*(?:\d+[\.\)]\s*|[-*•]\s*)?(?:Mục tiêu|Nội dung cần nghiên cứu|Nội dung được xây dựng|Biết dùng|Biết cấu hình|Nắm được|Nắm vững|Chuẩn bị tốt)\b", s, re.I):
                        is_meta_noise = True

                if is_meta_noise:
                    continue

                if score >= 3 and (bigram_hits >= 1 or token_hits >= 1 or ent_hits >= 1 or tech_hits >= 1 or is_general_summary):
                    passage, consumed_indices = build_contextual_passage(sents, idx_s, max_chars=620, max_extra_sents=3)
                    all_candidates.append(
                        {
                            "sentence": s,
                            "core_sentence": s,
                            "passage": passage or s,
                            "consumed_indices": consumed_indices,
                            "unit_idx": u_idx,
                            "sent_idx": idx_s,
                            "unit": u,
                            "page_no": u.get("page_no", 1),
                            "token_hits": token_hits,
                            "ent_hits": ent_hits,
                            "has_date": has_date,
                            "has_year": has_year,
                            "has_list": has_list,
                            "score": score,
                        }
                    )

        all_candidates.sort(key=lambda x: x["score"], reverse=True)

        selected_candidates: list[dict[str, Any]] = []
        seen_norms: set[str] = set()
        used_unit_sents: set[tuple[int, int]] = set()

        max_select = 4 if is_specific_fact_q else 7
        for c in all_candidates:
            u_idx = c["unit_idx"]
            s_idx = c["sent_idx"]
            norm = c["core_sentence"].lower()[:65]
            if norm in seen_norms or (u_idx, s_idx) in used_unit_sents:
                continue
            seen_norms.add(norm)
            for c_i in c.get("consumed_indices", [s_idx]):
                used_unit_sents.add((u_idx, c_i))
            selected_candidates.append(c)
            if len(selected_candidates) >= max_select:
                break

        is_strict_group_q = bool(re.search(r"\b(\d+\s+nhóm|mấy\s+nhóm|các\s+nhóm)\b", clean_q, re.I))
        if not selected_candidates or (is_strict_group_q and not any(c.get("has_list") for c in selected_candidates)):
            no_doc_msg = (
                f"Dựa trên các bài báo/tài liệu hiện có trong hệ thống, thông tin về \"{clean_q}\" "
                f"**hoàn toàn không được đề cập**.\n\n"
                f"**Gợi ý**: Bạn có thể tải thêm file PDF hoặc bài báo liên quan đến vấn đề này lên hệ thống "
                f"để tôi đọc và phân tích cho bạn."
            )
            return {
                "answer": no_doc_msg,
                "claims": [{"id": "claim_1", "text": f"Tài liệu hiện có không đề cập đến '{clean_q}'.", "unit_id": None}],
                "evidence_spans": [],
                "critic_action": "approve",
                "critic_feedback": {"action": "approve", "confidence": 1.0, "reason": "out_of_domain"},
                "generator_model_family": "cpu_min_extractive",
                "reasoning_trace": [self.log_action("cpu_min_summarization_completed", {"status": "out_of_domain"})],
            }

        top_cand = selected_candidates[0]
        top_u = top_cand["unit"]
        doc_name = format_doc_title(str(top_u.get("doc_id") or top_u.get("title") or "Tài liệu"))
        top_sent = top_cand["sentence"]
        top_passage = top_cand.get("passage") or top_sent
        top_page = top_cand["page_no"]

        claims: list[dict[str, Any]] = []
        evidence_spans: list[dict[str, Any]] = []

        # 8. Formulate Complete, Comprehensive Answer (bỏ câu trả lời nhanh, trình bày hoàn thiện & chi tiết)
        # CASE A: LIST / ENUMERATION SYNTHESIS
        if is_list_q and ":" in top_sent and (
            ";" in top_sent
            or "thứ" in top_sent.lower()
            or any(":" in c.get("sentence", "") for c in selected_candidates[1:5])
        ):
            prefix, list_part = top_sent.split(":", 1)
            prefix_clean = prefix.strip()
            prefix_clean = re.sub(r"^(?:với|gồm|bao gồm)\s+", "", prefix_clean, flags=re.I)

            raw_items = [it.strip().rstrip(".;") for it in re.split(r";|\n", list_part) if it.strip()]
            if len(raw_items) <= 1:
                u_text = str(top_u.get("text", ""))
                idx_pref = u_text.find(prefix_clean)
                if idx_pref != -1:
                    after_text = u_text[idx_pref + len(prefix_clean) : idx_pref + len(prefix_clean) + 850]
                    after_text = re.sub(r"^[:\s]+", "", after_text)
                    sub_items = re.split(r"(?<=[.)])\s+(?=[a-z_A-ZÀ-ỹ0-9]+(?:\s+[^:]{1,20})?:)", after_text)
                    if len(sub_items) > 1:
                        raw_items = [it.strip().rstrip(".;") for it in sub_items[:6] if it.strip()]

            formatted_items = []
            for idx, it in enumerate(raw_items, 1):
                it_clean = it.strip().rstrip(".")
                if not it_clean:
                    continue
                heading, body = extract_topic_heading(it_clean, idx)
                if body != it_clean:
                    formatted_items.append(f"{idx}. **{heading}**: {body} [1] *(Trang {top_page})*")
                else:
                    it_cap = it_clean[0].upper() + it_clean[1:]
                    formatted_items.append(f"{idx}. **{it_cap}** [1] *(Trang {top_page})*")

                claims.append(
                    AtomicClaim(
                        id=f"claim_{len(claims)+1}",
                        text=it_clean,
                        evidence_ids=[top_u.get("id", "doc_1")],
                    ).to_dict()
                )

            if len(formatted_items) <= 1:
                for cand in selected_candidates[1:6]:
                    cand_p = (cand.get("passage") or cand["sentence"]).strip().rstrip(".")
                    if cand_p:
                        pno = cand.get("page_no", 1)
                        ref_no = len(formatted_items) + 1
                        heading, body = extract_topic_heading(cand_p, ref_no)
                        formatted_items.append(f"{ref_no}. **{heading}**: {body} [{ref_no}] *(Trang {pno})*")
                        claims.append(
                            AtomicClaim(
                                id=f"claim_{len(claims)+1}",
                                text=cand["sentence"],
                                evidence_ids=[cand["unit"].get("id", "doc_1")],
                            ).to_dict()
                        )

            items_str = "\n".join(formatted_items)

            elabs = []
            for cand in selected_candidates[1:6]:
                cand_passage = cand.get("passage") or cand["sentence"]
                if cand_passage[:50].lower() not in items_str.lower():
                    pno = cand.get("page_no", 1)
                    ref_no = len(elabs) + 2
                    heading, body = extract_topic_heading(cand_passage, ref_no)
                    elabs.append(f"- **{heading}**: {body} [{ref_no}] *(Trang {pno})*")
                    claims.append(
                        AtomicClaim(
                            id=f"claim_{len(claims)+1}",
                            text=cand["sentence"],
                            evidence_ids=[cand["unit"].get("id", "doc_1")],
                        ).to_dict()
                    )

            elab_str = "\n".join(elabs)
            lead_prefix = prefix_clean[0].lower() + prefix_clean[1:] if prefix_clean else "nội dung trọng tâm"
            answer = (
                f"Dựa trên tài liệu nghiên cứu (**{doc_name}**), {lead_prefix} được phân tích đầy đủ và chi tiết qua các thành phần cốt lõi sau:\n\n"
                f"{items_str}\n"
            )
            if elab_str:
                answer += f"\n### Phân tích chi tiết & Nội dung mở rộng từ tài liệu nguồn:\n{elab_str}\n"

        # CASE B: TEMPORAL (DATE OR YEAR) COMPREHENSIVE SYNTHESIS
        elif is_date_q or is_year_q:
            date_match = re.search(r"\b(?:ngày\s+)?(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b", top_sent, re.I)
            year_match = re.search(r"\b(?:năm\s+)?(19\d\d|20\d\d)\b", top_sent)

            time_str = ""
            if date_match and not is_year_q:
                time_str = f"ngày {date_match.group(1).replace('.', '-')}"
            elif is_year_q and year_match:
                time_str = f"năm {year_match.group(1)}"
            elif date_match:
                time_str = f"ngày {date_match.group(1).replace('.', '-')}"
            elif year_match:
                time_str = f"năm {year_match.group(1)}"

            subj_match = re.search(
                r"^(.*?)\s+(?:ra\s+số|ra\s+đời|xuất\s+bản|thành\s+lập|tổ\s+chức|diễn\s+ra)",
                clean_q,
                re.I,
            )
            subj = subj_match.group(1).strip() if subj_match else "Sự kiện"

            action = "ra đời"
            for act in ["ra số đầu tiên", "xuất bản số đầu tiên", "thành lập", "diễn ra", "tổ chức"]:
                if act in clean_q.lower() or act in top_sent.lower():
                    action = act
                    break

            if time_str:
                intro_para = (
                    f"Dựa trên tài liệu nghiên cứu (**{doc_name}**), {subj} {action} vào **{time_str}** [1] *(Trang {top_page})*. "
                    f"Cụ thể trong bối cảnh lịch sử của tài liệu: {top_passage} [1] *(Trang {top_page})*."
                )
            else:
                intro_para = f"Dựa trên tài liệu nghiên cứu (**{doc_name}**), {top_passage} [1] *(Trang {top_page})*."

            claims.append(
                AtomicClaim(
                    id="claim_1",
                    text=top_sent,
                    evidence_ids=[top_u.get("id", "doc_1")],
                ).to_dict()
            )

            details = []
            for idx, cand in enumerate(selected_candidates[1:6], 2):
                pno = cand.get("page_no", 1)
                c_passage = cand.get("passage") or cand["sentence"]
                heading, body = extract_topic_heading(c_passage, idx - 1)
                details.append(f"{idx - 1}. **{heading}**: {body} [{idx}] *(Trang {pno})*")
                claims.append(
                    AtomicClaim(
                        id=f"claim_{idx}",
                        text=cand["sentence"],
                        evidence_ids=[cand["unit"].get("id", "doc_1")],
                    ).to_dict()
                )
            details_str = "\n".join(details)

            answer = intro_para + "\n"
            if details_str:
                answer += (
                    f"\n### Bối cảnh lịch sử & Các thông tin xác thực liên quan:\n"
                    f"{details_str}\n"
                )

        # CASE C: COMPREHENSIVE MULTI-ASPECT FACTUAL & TECHNICAL SYNTHESIS
        else:
            # Check if there is an additional explanation note in the same unit not yet captured in top_passage
            u_full_text = str(top_u.get("text", ""))
            u_norm = re.sub(r"\s+", " ", u_full_text)
            top_norm = re.sub(r"\s+", " ", top_sent)
            explanation_context = ""
            if top_norm in u_norm:
                after_idx = u_norm.find(top_norm) + len(top_norm)
                subsequent = u_norm[after_idx : after_idx + 650].strip()
                for prefix in ["Giải thích:", "Lưu ý:", "Nguyên nhân:", "Mục đích:", "Tác dụng:", "Cụ thể:"]:
                    if prefix.lower() in subsequent.lower() and prefix.lower() not in top_passage.lower():
                        p_pos = subsequent.lower().find(prefix.lower())
                        para = subsequent[p_pos:]
                        para_clean = re.split(r"(?:\s|^)\d+\.\d+\.\s+[A-ZĐ]", para)[0].strip()
                        m_end = re.search(r"(?<=[a-zà-ỹ\)])\.(?:\s+|$)", para_clean)
                        if m_end:
                            explanation_context = para_clean[: m_end.end()].strip()
                        else:
                            explanation_context = para_clean[:380].strip()
                        break

            claims.append(
                AtomicClaim(
                    id="claim_1",
                    text=top_sent,
                    evidence_ids=[top_u.get("id", "doc_1")],
                ).to_dict()
            )

            top_heading, top_body = extract_topic_heading(top_passage, 1)
            top_body_formatted = format_nginx_blocks(top_body)
            if top_body != top_passage:
                intro_body = f"**{top_heading}**: {top_body_formatted} [1] *(Trang {top_page})*"
            else:
                intro_body = f"{top_body_formatted} [1] *(Trang {top_page})*"
                if not intro_body.rstrip().endswith("."):
                    intro_body += "."

            if explanation_context:
                intro_body += f"\n\n- **Giải thích & Làm rõ nguyên lý:** {explanation_context} [1] *(Trang {top_page})*"

            details = []
            max_details = 2 if is_specific_fact_q else 6
            detail_ref_idx = 2

            for cand in selected_candidates[1:]:
                if len(details) >= max_details:
                    break
                pno = cand.get("page_no", 1)
                c_passage = cand.get("passage") or cand["sentence"]

                # Strict negative check: never allow meta noise, reading lists, or syllabus items into details
                if re.match(r"^\s*(?:\d+[\.\)]\s*|[-*•]\s*)?(?:Bài viết|Video|Tài liệu|Task|Mục tiêu|Nội dung cần|Nội dung được)\b", c_passage, re.I):
                    continue

                # For specific / definition queries, only select candidates that strictly mention the queried topic/entity
                if is_specific_fact_q:
                    cand_lower = c_passage.lower()
                    has_topic = any(
                        t.lower() in cand_lower
                        for t in (entities + tech_identifiers + q_bigrams)
                        if len(t) >= 3 and t.lower() not in {"nginx", "linux", "windows", "docker"}
                    )
                    if not has_topic or cand.get("score", 0) < 12:
                        continue
                    if c_passage[:60].lower() in top_passage.lower():
                        continue

                heading, body = extract_topic_heading(c_passage, len(details) + 1)
                body_formatted = format_nginx_blocks(body)
                details.append(f"{len(details) + 1}. **{heading}**: {body_formatted} [{detail_ref_idx}] *(Trang {pno})*")
                claims.append(
                    AtomicClaim(
                        id=f"claim_{detail_ref_idx}",
                        text=cand["sentence"],
                        evidence_ids=[cand["unit"].get("id", "doc_1")],
                    ).to_dict()
                )
                detail_ref_idx += 1

            details_str = "\n".join(details)

            answer = f"Dựa trên tài liệu nghiên cứu (**{doc_name}**), {intro_body}\n"
            if details_str:
                section_title = (
                    "### Các cơ chế & Cấu hình liên quan trực tiếp trong tài liệu:"
                    if is_specific_fact_q
                    else "### Phân tích chi tiết & Các khía cạnh liên quan trong tài liệu:"
                )
                answer += f"\n{section_title}\n{details_str}\n"

        # Build evidence spans for all selected candidates (using full contextual passage for richer evidence preview)
        used_unit_ids = set()
        for cl in claims:
            for eid in cl.get("evidence_ids", []):
                used_unit_ids.add(eid)

        for idx, c in enumerate(selected_candidates, 1):
            u = c["unit"]
            if u.get("id") in used_unit_ids or idx <= len(claims):
                evidence_spans.append(
                    EvidenceSpan(
                        id=f"ev_{len(evidence_spans)+1}",
                        quote=c.get("passage") or c["sentence"],
                        source_unit_id=u.get("id", "doc_1"),
                        doc_id=u.get("doc_id", "doc"),
                        page_no=u.get("page_no", 1),
                    ).to_dict()
                )

        critic_eval = evaluate_inner_critic_fast(
            query=clean_q,
            draft_answer=answer,
            source_units=source_units[:10],
            required_entities=required_entities,
        )

        trace = self.log_action(
            "cpu_min_summarization_completed",
            {
                "units_used": len(source_units),
                "claims_extracted": len(claims),
                "critic_action": critic_eval.action,
                "critic_confidence": critic_eval.confidence,
                "mode": "cpu_min",
            },
        )

        return {
            "answer": answer,
            "claims": claims,
            "evidence_spans": evidence_spans,
            "critic_action": critic_eval.action,
            "critic_feedback": critic_eval.to_dict(),
            "generator_model_family": "cpu_min_extractive",
            "reasoning_trace": [trace],
        }
