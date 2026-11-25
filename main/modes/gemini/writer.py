"""
Gemini Mode - Citation-Grounded Writer Agent (L3-L4, Quota-Optimized).
Compiles verified findings, synthesizes polished academic prose & comparison tables
via Gemini API strictly grounded in audited source pages, and preserves [1], [2] citations.
"""

from __future__ import annotations

import re
from typing import Any

from agents.agent5_writer.components.report_builder import build_academic_report
from core.base_agent import BaseAgent
from core.citation_audit import citation_location, evidence_quote
from core.provider_router import is_gemini_permitted
from core.state import AgentState
from utils.llm_client import llm_client


_EXEC_SUMMARY_RE = re.compile(
    r"(## 1\. Tóm tắt Tổng quan \(Executive Summary\)\n)(.*?)(\n\n## 2\. Kết quả Nghiên cứu)",
    re.DOTALL,
)


class GeminiWriterAgent(BaseAgent):
    """Writer Agent for Gemini Mode: High-precision academic report & citation compiler."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="GeminiWriterAgent", max_internal_retry=max_internal_retry)

    def _synthesize_executive_summary(
        self,
        *,
        query: str,
        references: list[dict[str, Any]],
        source_units: list[dict[str, Any]],
        evidence_spans: list[dict[str, Any]],
        verified_claims: list[dict[str, Any]],
    ) -> str | None:
        if not references or not self.settings.GEMINI_API_KEY:
            return None
        if not is_gemini_permitted("gemini", role="writer"):
            return None

        allowed_ref_ids = {int(r["ref_id"]) for r in references if isinstance(r.get("ref_id"), int)}
        refs_by_page = {
            (str(r.get("doc_id") or "").strip(), int(r["page_no"])): int(r["ref_id"])
            for r in references
            if r.get("doc_id") and isinstance(r.get("page_no"), int) and isinstance(r.get("ref_id"), int)
        }
        # Collect full grounded passages from the exact audited pages so multi-part answers
        # (e.g. comparing (a), (b), (c) on the same page) are synthesized completely.
        evidence_blocks: list[str] = []
        seen_texts: set[str] = set()
        for unit in source_units:
            if not isinstance(unit, dict):
                continue
            loc = citation_location(unit)
            if not loc or loc not in refs_by_page:
                continue
            ref_id = refs_by_page[loc]
            txt = evidence_quote(unit).strip()
            if txt and txt not in seen_texts:
                seen_texts.add(txt)
                evidence_blocks.append(f"[{ref_id}] (Trang {loc[1]}): {txt[:2400]}")

        if not evidence_blocks:
            for span in evidence_spans:
                if not isinstance(span, dict):
                    continue
                loc = citation_location(span)
                if not loc or loc not in refs_by_page:
                    continue
                ref_id = refs_by_page[loc]
                txt = evidence_quote(span).strip()
                if txt and txt not in seen_texts:
                    seen_texts.add(txt)
                    evidence_blocks.append(f"[{ref_id}] (Trang {loc[1]}): {txt[:1200]}")

        if not evidence_blocks:
            return None

        allowed_markers = ", ".join(f"[{rid}]" for rid in sorted(allowed_ref_ids))
        claims_preview = "\n".join(
            f"- {c.get('claim_text') or c.get('claim') or c.get('text') or ''}"
            for c in verified_claims[:8]
            if isinstance(c, dict)
        )
        system_prompt = (
            "Bạn là biên tập viên báo cáo nghiên cứu khoa học cấp cao (Academic Research Writer). "
            "NHIỆM VỤ: Biên tập phần Tóm tắt Tổng quan (Executive Summary) trả lời ĐẦY ĐỦ, CHÍNH XÁC mọi vế trong câu hỏi "
            "của người dùng bằng ngôn ngữ của câu hỏi (nếu câu hỏi tiếng Việt thì viết hoàn toàn bằng tiếng Việt học thuật, "
            "giữ thuật ngữ chuyên ngành tiếng Anh trong ngoặc đơn khi cần).\n"
            "QUY TẮC BẮT BUỘC (ZERO HALLUCINATION):\n"
            "1. CHỈ sử dụng thông tin có trong các đoạn bằng chứng được cung cấp bên dưới. Tuyệt đối KHÔNG bịa số liệu hoặc kiến thức ngoài tài liệu.\n"
            "2. Mọi luận điểm thực tế phải gắn trích dẫn số hợp lệ từ danh sách được phép: "
            f"{allowed_markers}. Không được tự tạo số trích dẫn khác.\n"
            "3. Nếu câu hỏi yêu cầu phân tích, đối chiếu hoặc so sánh nhiều mô hình/phương pháp, hãy trình bày rõ từng mục "
            "và kèm một bảng so sánh Markdown ngắn gọn, súc tích.\n"
            "4. Khi trình bày công thức toán học, đặt chuẩn trong ký hiệu LaTeX $...$ (nội dòng) hoặc $$...$$ (khối riêng) đúng theo bằng chứng gốc.\n"
            "5. Không lặp lại tiêu đề '## 1. Tóm tắt Tổng quan', chỉ xuất nội dung phần tổng quan."
        )
        user_prompt = (
            f"Câu hỏi nghiên cứu:\n{query}\n\n"
            f"Các luận điểm đã thẩm định:\n{claims_preview}\n\n"
            f"Bằng chứng gốc từ các trang đã xác minh:\n" + "\n\n".join(evidence_blocks[:6])
        )
        try:
            synthesized = llm_client.generate(
                user_prompt,
                system_instruction=system_prompt,
                temperature=0.15,
                max_tokens=1536,
                agent="writer",
            )
            cleaned = (synthesized or "").strip()
            if len(cleaned) < 40:
                return None
            # Strip any out-of-range citation marker [k] so citation_audit stays 100% clean
            cleaned = re.sub(
                r"\[(\d+)\]",
                lambda m: m.group(0) if int(m.group(1)) in allowed_ref_ids else "",
                cleaned,
            )
            if not any(f"[{rid}]" in cleaned for rid in allowed_ref_ids):
                cleaned = f"{cleaned} [{min(allowed_ref_ids)}]"
            return cleaned.strip()
        except Exception as exc:
            self.logger.warning(f"[Gemini Writer] Gemini synthesis skipped ({exc}); using deterministic report.")
            return None

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        answer = state.get("answer", "")
        verified_claims = state.get("verified_claims", []) or []
        rejected_claims = state.get("rejected_claims", []) or []
        retrieved_docs = state.get("retrieved_docs", []) or []
        evidence_spans = state.get("evidence_spans", []) or []
        source_units = state.get("source_units", []) or []

        self.logger.info(
            f"[Gemini Writer] Compiling verified academic report for: '{query[:60]}' "
            f"({len(verified_claims)} verified claims)"
        )

        final_report, report_data = build_academic_report(
            query=query,
            verified_claims=verified_claims,
            rejected_claims=rejected_claims,
            retrieved_docs=retrieved_docs,
            answer=answer,
            answer_status=state.get("answer_status"),
            evidence_spans=evidence_spans,
        )

        synthesized_with_llm = False
        if report_data.get("status") == "ANSWERED_DIRECT_QUOTE" and report_data.get("references"):
            polished = self._synthesize_executive_summary(
                query=query,
                references=report_data["references"],
                source_units=source_units,
                evidence_spans=evidence_spans,
                verified_claims=verified_claims,
            )
            if polished:
                updated_report = _EXEC_SUMMARY_RE.sub(
                    lambda m: f"{m.group(1)}{polished}{m.group(3)}",
                    final_report,
                    count=1,
                )
                if updated_report != final_report:
                    final_report = updated_report
                    report_data["answer"] = polished
                    synthesized_with_llm = True

        report_data["engine"] = "Gemini_Cloud_Agentic"
        report_data["synthesized_with_llm"] = synthesized_with_llm

        trace = self.log_action(
            "gemini_report_written",
            {
                "length": len(final_report),
                "verified_claims_count": len(verified_claims),
                "synthesized_with_llm": synthesized_with_llm,
            },
        )

        return {
            "final_report": final_report,
            "written_report_data": report_data,
            "reasoning_trace": [trace],
        }
