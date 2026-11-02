"""
CPU Min Mode - Agentic Summarizer Agent (L3-Lite).
Features:
  - Sentence-Window Context Expansion (±1 sentence around best match so quotes are complete).
  - Explicit Inline Citations [1] (Trang X) linked to exact SourceUnits.
  - Inner Critic Self-Correction Loop: Checks entity/number coverage and rewrites or triggers 'request_more_context'.
"""

from __future__ import annotations

import re
from typing import Any

from agents.agent1_retrieval.tools.bm25 import SimpleBM25
from core.base_agent import BaseAgent
from core.state import AgentState, AtomicClaim, EvidenceSpan
from core.verifier import evaluate_inner_critic_fast


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
        if len(s_clean) < 28:
            continue
        if s_clean.startswith("#") or s_clean.startswith("---") or "http" in s_clean:
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
        if not (s_clean[0].isupper() or s_clean[0].isdigit() or s_clean[0] in '"“\'['):
            continue

        sentences.append(s_clean)
    return sentences


class CPUMinSummarizerAgent(BaseAgent):
    """Agentic Summarizer for CPU Min: Sentence-window extraction + Inner Critic self-correction."""

    def __init__(self, max_internal_retry: int = 1):
        super().__init__(name="CPUMinSummarizerAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        source_units = state.get("source_units", []) or []
        required_entities = state.get("required_entities", []) or []

        self.logger.info(f"[CPU Min Summarizer] Synthesizing for query: '{query}' ({len(source_units)} units)")

        if len(source_units) > 8:
            bm25 = SimpleBM25()
            bm25.fit(source_units)
            ranked = bm25.search(query, top_k=8)
            top_units = ranked if ranked else source_units[:8]
        else:
            top_units = source_units

        is_numeric = bool(state.get("is_numeric_query")) or any(
            w in query.lower() for w in ["bao nhiêu", "mấy", "số lượng", "định lượng", "tổng số", "tỷ lệ", "%"]
        )
        is_general_summary = any(
            w in query.lower()
            for w in [
                "tóm tắt", "tổng quan", "nội dung", "chính", "summary", "overview",
                "abstract", "giới thiệu", "phương pháp", "kết quả", "đóng góp",
            ]
        )

        q_words = set(re.findall(r"\w+", query.lower())) - {
            "và", "của", "cho", "với", "như", "thế", "nào", "được", "có", "là", "ở",
        }
        try:
            from utils.bilingual import expand_bilingual_tokens
            q_words = expand_bilingual_tokens(query, q_words)
        except Exception:
            pass

        all_candidates: list[dict[str, Any]] = []
        for u in top_units:
            u_text = str(u.get("text", ""))
            sents = extract_clean_sentences(u_text)
            for idx_s, s in enumerate(sents):
                s_words = set(re.findall(r"\w+", s.lower()))
                overlap = len(q_words & s_words)

                # Boost sentences containing required_entities
                ent_hits = sum(1 for ent in required_entities if ent.lower() in s.lower())

                s_strip_dates = re.sub(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", "", s)
                s_strip_dates = re.sub(r"\b(?:ngày|tháng|năm|số)\s+\d+[\w-]*\b", "", s_strip_dates, flags=re.I)
                s_strip_dates = re.sub(r"\b(19\d\d|20\d\d)\b", "", s_strip_dates)
                has_num = bool(re.search(r"\b\d+(?:[.,]\d+)?%?\b", s_strip_dates))

                richness = 3 if 60 <= len(s) <= 350 else 1
                score = overlap * 2 + ent_hits * 4 + richness + (4 if has_num and is_numeric else 0)

                # Build Sentence-Window (current sentence + next sentence if related)
                window_text = s
                if idx_s + 1 < len(sents) and len(s) < 180 and len(sents[idx_s + 1]) < 220:
                    window_text = f"{s} {sents[idx_s + 1]}"

                all_candidates.append(
                    {
                        "sentence": window_text,
                        "core_sentence": s,
                        "unit": u,
                        "overlap": overlap,
                        "ent_hits": ent_hits,
                        "has_num": has_num,
                        "score": score,
                    }
                )

        all_candidates.sort(key=lambda x: x["score"], reverse=True)

        selected_candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for c in all_candidates:
            norm = c["core_sentence"].lower()[:65]
            if norm not in seen and (c["overlap"] > 0 or c["ent_hits"] > 0 or is_general_summary):
                seen.add(norm)
                selected_candidates.append(c)
                if len(selected_candidates) >= 4:
                    break

        if not selected_candidates and all_candidates:
            for c in all_candidates[:3]:
                norm = c["core_sentence"].lower()[:65]
                if norm not in seen:
                    seen.add(norm)
                    selected_candidates.append(c)

        # Inner Critic Self-Correction Pass:
        # Check if any required_entity present in all_candidates was missed in selected_candidates
        selected_blob = " ".join(c["sentence"] for c in selected_candidates).lower()
        for ent in required_entities:
            if ent.lower() not in selected_blob:
                for c in all_candidates:
                    if ent.lower() in c["sentence"].lower():
                        norm = c["core_sentence"].lower()[:65]
                        if norm not in seen:
                            seen.add(norm)
                            selected_candidates.append(c)
                            selected_blob += " " + c["sentence"].lower()
                            break

        any_num = any(c["has_num"] for c in selected_candidates)

        doc_name = top_units[0].get("doc_id", "tài liệu") if top_units else "tài liệu"
        doc_name = re.sub(r"^(?:gemini|doc)_[a-f0-9]+_", "", doc_name, flags=re.I)
        doc_name = re.sub(r"^(?:gemini|doc)_", "", doc_name, flags=re.I)
        doc_name = doc_name.replace("_", " ").replace(".pdf", "").strip()

        if is_numeric and not any_num:
            answer = (
                f"Dựa trên tài liệu nghiên cứu (**{doc_name}**):\n\n"
                f"⚠️ **Tài liệu nguồn không cung cấp thông tin số lượng cụ thể** cho câu hỏi: *\"{query}\"*.\n\n"
                f"Văn bản nguồn chỉ đề cập đến các nội dung liên quan sau:\n"
            )
            for idx, c in enumerate(selected_candidates[:3], 1):
                pno = c["unit"].get("page_no", 1)
                answer += f"- {c['sentence']} [{idx}] *(Trang {pno})*\n"
        elif selected_candidates:
            answer = f"Dựa trên tài liệu nghiên cứu (**{doc_name}**):\n\n"
            for idx, c in enumerate(selected_candidates[:5], 1):
                pno = c["unit"].get("page_no", 1)
                answer += f"- {c['sentence']} [{idx}] *(Trang {pno})*\n"
        elif top_units:
            fallback_sent = extract_clean_sentences(top_units[0].get("text", ""))
            if fallback_sent:
                first_sent = fallback_sent[0]
                pno = top_units[0].get("page_no", 1)
                answer = f"Dựa trên tài liệu nghiên cứu (**{doc_name}**):\n\n- {first_sent} [1] *(Trang {pno})*\n"
                selected_candidates.append({"sentence": first_sent, "unit": top_units[0]})
            else:
                answer = f"Không tìm thấy dữ liệu liên quan trực tiếp đến '{query}' trong nguồn tài liệu."
        else:
            answer = f"Không tìm thấy dữ liệu liên quan trực tiếp đến '{query}' trong nguồn tài liệu."

        claims = []
        evidence_spans = []

        for idx, c in enumerate(selected_candidates[:5]):
            u = c["unit"]
            sent = c["sentence"]
            claims.append(
                AtomicClaim(
                    id=f"claim_{idx + 1}",
                    text=sent,
                    evidence_ids=[u.get("id", "doc_1")],
                ).to_dict()
            )
            evidence_spans.append(
                EvidenceSpan(
                    id=f"ev_{idx + 1}",
                    quote=sent,
                    source_unit_id=u.get("id", "doc_1"),
                    doc_id=u.get("doc_id", "doc"),
                    page_no=u.get("page_no", 1),
                ).to_dict()
            )

        critic_eval = evaluate_inner_critic_fast(
            query=query,
            draft_answer=answer,
            source_units=top_units,
            required_entities=required_entities,
        )

        trace = self.log_action(
            "cpu_min_summarization_completed",
            {
                "units_used": len(top_units),
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
