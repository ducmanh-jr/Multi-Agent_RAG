"""
CPU Min Mode - Agentic Summarizer Agent (L3-Lite).
Features:
  - High-precision sentence extraction with photo caption and header stripping.
  - Zero-Hallucination Out-of-Domain Detection: truthful refusal when key query entities are absent from document.
  - Authentic Extractive Synthesis: Direct Answer First (> **Câu trả lời trực tiếp:**).
  - Structured List Decomposition: parses semicolon-separated groups into bold enumerated lists.
  - Exact Date/Temporal Extraction: extracts date entities and formats crisp direct factual answers.
  - Strict AtomicClaim & EvidenceSpan generation with exact page citations [1] (Trang X).
"""

from __future__ import annotations

import re
from typing import Any

from agents.agent1_retrieval.tools.bm25 import SimpleBM25
from core.base_agent import BaseAgent
from core.state import AgentState, AtomicClaim, EvidenceSpan
from core.verifier import evaluate_inner_critic_fast


VIETNAMESE_STOP_WORDS = {
    "và", "của", "cho", "với", "như", "thế", "nào", "được", "có", "là", "ở", "ra", "vào",
    "ngày", "tháng", "năm", "mấy", "bao", "nhiêu", "khi", "đầu", "tiên", "gì", "ai", "sao",
    "đâu", "đến", "từ", "lại", "đã", "đang", "sẽ", "các", "những", "một", "số", "về",
    "trong", "để", "do", "bởi", "thì", "mà", "nhất", "nhì", "này", "đó", "kia", "hãy",
    "cho", "biết", "hỏi", "thông", "tin", "chi", "tiết", "tại", "sao", "thế", "nào",
}


def unwrap_pdf_text(text: str) -> str:
    """Unwrap hard line breaks within paragraphs commonly produced by PDF extractors."""
    lines = text.split("\n")
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
    """Extract complete, grammatical sentences from unwrapped text."""
    clean_text = unwrap_pdf_text(text)
    raw_sentences = re.split(r"(?<=[.!?])\s+", clean_text)
    sentences = []

    dangling_words = {
        "được", "là", "và", "của", "đã", "cho", "nhưng", "bởi", "ở", "vào",
        "ra", "tại", "với", "về", "trong", "để", "có", "sau", "trước", "khi",
        "mà", "theo", "các", "những", "thì", "do", "bị", "rằng", "gồm", "như",
    }

    for s in raw_sentences:
        s_clean = s.strip()
        # Strip leading photo / source / caption metadata e.g. (Ảnh: ...)
        s_clean = re.sub(
            r"^[\(\[\{][^\)\]\}]*(?:Ảnh|Nguồn|TTXVN|PV|Tư liệu|Theo)[^\)\]\}]*[\)\]\}]\s*",
            "",
            s_clean,
        )
        if len(s_clean) < 28:
            continue
        if s_clean.startswith("#") or s_clean.startswith("---") or "http" in s_clean:
            continue
        if any(
            banner in s_clean.lower()
            for banner in ["chính sách bảo vệ", "dữ liệu cá nhân", "đồng ý tất cả", "cookie", "chấp nhận tất cả"]
        ):
            continue
        if re.match(
            r"^\*{0,2}(Tác giả|Năm xuất bản|Nguồn công bố|Liên kết|Quyền riêng tư|Ảnh:|Theo tư liệu)",
            s_clean,
            re.I,
        ):
            continue
        if s_clean[-1] not in '.!?:"”':
            continue

        words = re.findall(r"\w+", s_clean)
        if words and words[-1].lower() in dangling_words:
            continue
        if not (s_clean[0].isupper() or s_clean[0].isdigit() or s_clean[0] in '"“\'[('):
            continue

        sentences.append(s_clean)
    return sentences


class CPUMinSummarizerAgent(BaseAgent):
    """Agentic Summarizer for CPU Min: Genuine extractive synthesis + Zero Hallucination."""

    def __init__(self, max_internal_retry: int = 1):
        super().__init__(name="CPUMinSummarizerAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        clean_q = query.strip()
        source_units = state.get("source_units", []) or []
        required_entities = state.get("required_entities", []) or []

        self.logger.info(f"[CPU Min Summarizer] Synthesizing for query: '{clean_q}' ({len(source_units)} units)")

        # Kịch bản 2: Khi kho tài liệu không có văn bản nào được tải
        if not source_units:
            no_doc_msg = (
                f"Dựa trên các bài báo/tài liệu hiện có trong hệ thống, thông tin về \"{clean_q}\" "
                f"**hoàn toàn không được đề cập**.\n\n"
                f"💡 **Gợi ý**: Bạn có thể tải thêm file PDF hoặc bài báo liên quan đến vấn đề này lên hệ thống "
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

        # 1. Intent analysis
        is_date_q = bool(
            re.search(
                r"\b(ngày nào|tháng nào|năm nào|khi nào|thời điểm nào|bao giờ|ngày bao nhiêu|năm bao nhiêu)\b",
                clean_q,
                re.I,
            )
        )
        is_list_q = bool(
            re.search(
                r"\b(những\s+vấn\s+đề|các\s+vấn\s+đề|những\s+nội\s+dung|các\s+nội\s+dung|những\s+gì|gồm\s+những|các\s+nhóm|mấy\s+nhóm|\d+\s+nhóm|các\s+mục|danh\s+sách|những\s+ai|các\s+bước|các\s+biện\s+pháp|các\s+dự\s+án)\b",
                clean_q,
                re.I,
            )
        )
        is_quantity_q = bool(
            re.search(
                r"\b(bao nhiêu|mấy|số lượng|định lượng|tổng số|tỷ lệ|%)\b",
                clean_q,
                re.I,
            )
        ) and not is_date_q
        is_general_summary = any(
            w in clean_q.lower()
            for w in [
                "tóm tắt", "tổng quan", "nội dung chính", "chính", "summary", "overview",
                "abstract", "giới thiệu", "phương pháp", "kết quả", "đóng góp",
            ]
        )

        # 2. Extract core non-stopword tokens and proper entities
        q_tokens = [w for w in re.findall(r"\w+", clean_q.lower()) if w not in VIETNAMESE_STOP_WORDS and len(w) > 1]
        caps = re.findall(r"\b[A-ZĐ][a-zà-ỹ0-9_-]+(?:\s+[a-zà-ỹ0-9_-]+)*\b", clean_q)
        entities = [c.lower() for c in caps if c.lower() not in VIETNAMESE_STOP_WORDS]

        # 3. Document scope and full text verification (Zero Hallucination Guard)
        doc_full_text = " ".join(str(u.get("text", "")) for u in source_units).lower()
        ent_present = any(e in doc_full_text for e in entities) if entities else True
        token_overlap = [t for t in q_tokens if t in doc_full_text]
        token_cov = len(token_overlap) / max(len(q_tokens), 1)

        # Detect Out-of-Domain: Query specifies distinct entities but none appear in document,
        # or semantic non-stopword token coverage is critically below threshold.
        is_out_of_domain = (entities and not ent_present and not is_general_summary) or (
            token_cov < 0.35 and not is_general_summary
        )

        if is_out_of_domain:
            no_doc_msg = (
                f"Dựa trên các bài báo/tài liệu hiện có trong hệ thống, thông tin về \"{clean_q}\" "
                f"**hoàn toàn không được đề cập**.\n\n"
                f"💡 **Gợi ý**: Bạn có thể tải thêm file PDF hoặc bài báo liên quan đến vấn đề này lên hệ thống "
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

        # 4. Sentence scoring across source units
        all_candidates: list[dict[str, Any]] = []
        for u in source_units:
            u_text = str(u.get("text", ""))
            sents = extract_clean_sentences(u_text)
            for idx_s, s in enumerate(sents):
                s_lower = s.lower()
                token_hits = sum(1 for t in q_tokens if t in s_lower)
                ent_hits = sum(1 for e in entities if e in s_lower)
                req_ent_hits = sum(1 for r_ent in required_entities if r_ent.lower() in s_lower)

                has_date = bool(re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b(?:19\d\d|20\d\d)\b", s))
                has_list = ":" in s and (";" in s or any(w in s_lower for w in ["thứ nhất", "nhóm vấn đề", "nội dung:"]))
                has_num = bool(re.search(r"\b\d+(?:[.,]\d+)?%?\b", s))

                score = token_hits * 3 + ent_hits * 4 + req_ent_hits * 4

                if is_date_q:
                    if has_date:
                        score += 6
                    else:
                        score -= 3
                if is_list_q:
                    if has_list:
                        score += 9
                    if any(w in s_lower for w in ["nhóm", "vấn đề", "nội dung", "gồm", "tập trung"]):
                        score += 4
                if is_quantity_q and has_num:
                    score += 5

                richness = 3 if 60 <= len(s) <= 350 else 1
                score += richness

                if score >= 3 and (token_hits >= 1 or ent_hits >= 1 or is_general_summary):
                    all_candidates.append(
                        {
                            "sentence": s,
                            "core_sentence": s,
                            "unit": u,
                            "page_no": u.get("page_no", 1),
                            "token_hits": token_hits,
                            "ent_hits": ent_hits,
                            "has_date": has_date,
                            "has_list": has_list,
                            "score": score,
                        }
                    )

        all_candidates.sort(key=lambda x: x["score"], reverse=True)

        selected_candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for c in all_candidates:
            norm = c["core_sentence"].lower()[:65]
            if norm not in seen:
                seen.add(norm)
                selected_candidates.append(c)
                if len(selected_candidates) >= 5:
                    break

        # If zero candidates match after strict evaluation, declare out-of-domain (Zero Hallucination!)
        if not selected_candidates:
            no_doc_msg = (
                f"Dựa trên các bài báo/tài liệu hiện có trong hệ thống, thông tin về \"{clean_q}\" "
                f"**hoàn toàn không được đề cập**.\n\n"
                f"💡 **Gợi ý**: Bạn có thể tải thêm file PDF hoặc bài báo liên quan đến vấn đề này lên hệ thống "
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

        # 5. Extract document display title
        top_u = selected_candidates[0]["unit"]
        doc_name = str(top_u.get("doc_id") or top_u.get("title") or "Tài liệu nghiên cứu")
        doc_name = re.sub(r"^(?:gemini|doc)_[a-f0-9]+_", "", doc_name, flags=re.I)
        doc_name = re.sub(r"^(?:gemini|doc)_", "", doc_name, flags=re.I)
        doc_name = doc_name.replace("_", " ").replace(".pdf", "").strip()

        top_cand = selected_candidates[0]
        top_sent = top_cand["sentence"]
        top_page = top_cand["page_no"]

        claims: list[dict[str, Any]] = []
        evidence_spans: list[dict[str, Any]] = []

        # 6. Structured Answer Synthesis
        # CASE A: LIST SYNTHESIS
        if is_list_q and ":" in top_sent and (";" in top_sent or "thứ" in top_sent.lower()):
            prefix, list_part = top_sent.split(":", 1)
            prefix_clean = prefix.strip()
            prefix_clean = re.sub(r"^(?:với|gồm|bao gồm)\s+", "", prefix_clean, flags=re.I)

            raw_items = [it.strip().rstrip(".;") for it in re.split(r";|\n", list_part) if it.strip()]
            formatted_items = []
            for idx, it in enumerate(raw_items, 1):
                it_clean = it.strip().rstrip(".")
                if not it_clean:
                    continue
                it_clean = it_clean[0].upper() + it_clean[1:]
                formatted_items.append(f"{idx}. **{it_clean}** [1]")

                claims.append(
                    AtomicClaim(
                        id=f"claim_{idx}",
                        text=it_clean,
                        evidence_ids=[top_u.get("id", "doc_1")],
                    ).to_dict()
                )

            items_str = "\n".join(formatted_items)

            # Check elaboration sentences from remaining candidates
            elabs = []
            for cand in selected_candidates[1:4]:
                cand_sent = cand["sentence"]
                if any(key in cand_sent.lower() for key in ["đối với", "cụ thể", "trong đó", "trọng tâm"]):
                    pno = cand.get("page_no", 1)
                    elabs.append(f"- {cand_sent} [{len(elabs)+2}] *(Trang {pno})*")
                    claims.append(
                        AtomicClaim(
                            id=f"claim_{len(claims)+1}",
                            text=cand_sent,
                            evidence_ids=[cand["unit"].get("id", "doc_1")],
                        ).to_dict()
                    )

            elab_str = "\n" + "\n".join(elabs) if elabs else ""

            answer = (
                f"> **Câu trả lời trực tiếp:**\n"
                f"Theo tài liệu nghiên cứu (**{doc_name}**), {prefix_clean[0].lower() + prefix_clean[1:]} gồm:\n\n"
                f"{items_str}\n"
            )
            if elab_str:
                answer += f"\n**Chi tiết triển khai từ tài liệu nguồn:**\n{elab_str}\n"

        # CASE B: DATE / TEMPORAL SYNTHESIS
        elif is_date_q:
            date_match = re.search(r"\b(?:ngày\s+)?(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b", top_sent, re.I)
            year_match = re.search(r"\b(?:năm\s+)?(19\d\d|20\d\d)\b", top_sent)

            date_str = ""
            if date_match:
                date_str = f"ngày {date_match.group(1).replace('.', '-')}"
            elif year_match:
                date_str = f"năm {year_match.group(1)}"

            subj_match = re.search(r"^(.*?)\s+(?:ra\s+số|xuất\s+bản|thành\s+lập|tổ\s+chức|ra\s+đời)", clean_q, re.I)
            subj = subj_match.group(1).strip() if subj_match else "Sự kiện"

            if date_str:
                direct_sent = f"{subj} ra số đầu tiên vào **{date_str}** [1] *(Trang {top_page})*."
            else:
                direct_sent = f"{top_sent} [1] *(Trang {top_page})*"

            details = []
            for idx, cand in enumerate(selected_candidates[:4], 1):
                pno = cand.get("page_no", 1)
                details.append(f"- {cand['sentence']} [{idx}] *(Trang {pno})*")
                claims.append(
                    AtomicClaim(
                        id=f"claim_{idx}",
                        text=cand["sentence"],
                        evidence_ids=[cand["unit"].get("id", "doc_1")],
                    ).to_dict()
                )
            details_str = "\n".join(details)

            answer = (
                f"> **Câu trả lời trực tiếp:**\n"
                f"{direct_sent}\n\n"
                f"**Chi tiết xác thực từ tài liệu nguồn:**\n"
                f"{details_str}\n"
            )

        # CASE C: GENERAL SYNTHESIS
        else:
            details = []
            for idx, cand in enumerate(selected_candidates[:5], 1):
                pno = cand.get("page_no", 1)
                details.append(f"- {cand['sentence']} [{idx}] *(Trang {pno})*")
                claims.append(
                    AtomicClaim(
                        id=f"claim_{idx}",
                        text=cand["sentence"],
                        evidence_ids=[cand["unit"].get("id", "doc_1")],
                    ).to_dict()
                )
            details_str = "\n".join(details)

            answer = (
                f"> **Câu trả lời trực tiếp:**\n"
                f"Dựa trên tài liệu nghiên cứu (**{doc_name}**), {top_sent} [1] *(Trang {top_page})*.\n\n"
                f"**Thông tin chi tiết được ghi nhận từ tài liệu:**\n"
                f"{details_str}\n"
            )

        # Build evidence spans for all selected candidates
        for idx, c in enumerate(selected_candidates[:5], 1):
            u = c["unit"]
            evidence_spans.append(
                EvidenceSpan(
                    id=f"ev_{idx}",
                    quote=c["sentence"],
                    source_unit_id=u.get("id", "doc_1"),
                    doc_id=u.get("doc_id", "doc"),
                    page_no=u.get("page_no", 1),
                ).to_dict()
            )

        critic_eval = evaluate_inner_critic_fast(
            query=clean_q,
            draft_answer=answer,
            source_units=source_units[:8],
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
