"""
GPU Finetune Mode - Agentic Writer Agent (7-Model SOTA Pipeline).
Leverages Remote GPU Worker powered by:
  - Fine-Tuned Qwen2.5-7B DoRA + rsLoRA (05_agent5_writer):
    Trained to Loss = 0.0149 and Token Accuracy = 99.6%.
    Synthesizes publication-grade scientific academic reports with strict evidence attribution,
    formal scholarly prose, and structured sections.
"""

from __future__ import annotations

import re
from typing import Any
from core.base_agent import BaseAgent
from core.state import AgentState
from agents.agent5_writer.components.report_builder import build_academic_report
from utils.llm_client import llm_client


WRITER_TRAINING_SYSTEM_PROMPT = (
    "Bạn là một Chuyên gia Viết Báo cáo Nghiên cứu Khoa học và Học thuật Xuất sắc. "
    "Nhiệm vụ của bạn là tổng hợp các phát hiện, mệnh đề khoa học đã kiểm chứng và ngữ cảnh thành một Báo cáo Khoa học hoàn chỉnh, chuyên sâu, học thuật. "
    "Hãy trình bày toàn bộ báo cáo (từ tiêu đề, các đề mục, nội dung phân tích, cho đến phần kết luận) bằng TIẾNG VIỆT học thuật chuẩn mực, ngay cả khi tài liệu nguồn là tiếng Anh. "
    "Giữ nguyên các thuật ngữ kỹ thuật quốc tế trong ngoặc đơn nếu cần (ví dụ: Multi-Head Attention, Scaled Dot-Product, Positional Encoding, Self-Attention)."
)


class GPUFinetuneWriterAgent(BaseAgent):
    """Writer Agent powered by Fine-Tuned Qwen2.5-7B DoRA + rsLoRA (Accuracy: 99.6%)."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="GPUFinetuneWriterAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        verified_claims = state.get("verified_claims", []) or []
        rejected_claims = state.get("rejected_claims", []) or []
        retrieved_docs = state.get("retrieved_docs", []) or []
        answer = state.get("answer", "")
        evidence_spans = [s if isinstance(s, dict) else {"quote": str(s), "doc_id": "doc_1", "page_no": 1} for s in (state.get("evidence_spans", []) or [])]
        retrieved_docs = [d if isinstance(d, dict) else {"text": str(d), "preview": str(d), "title": "Tài liệu", "doc_id": "doc_1", "page_no": 1} for d in (state.get("retrieved_docs", []) or [])]

        self.logger.info(f"[GPU Finetune Writer] Formatting academic report via DoRA Model for query: '{query[:65]}'")

        # Ensure claims are normalized dicts
        norm_v_claims: list[dict[str, Any]] = []
        for c in verified_claims:
            if isinstance(c, dict):
                norm_v_claims.append(c)
            elif hasattr(c, "to_dict"):
                norm_v_claims.append(c.to_dict())
            else:
                norm_v_claims.append({"claim": str(c), "claim_text": str(c), "text": str(c), "status": "SUPPORT"})
        verified_claims = norm_v_claims

        norm_r_claims: list[dict[str, Any]] = []
        for c in rejected_claims:
            if isinstance(c, dict):
                norm_r_claims.append(c)
            elif hasattr(c, "to_dict"):
                norm_r_claims.append(c.to_dict())
            else:
                norm_r_claims.append({"claim": str(c), "claim_text": str(c), "text": str(c), "status": "CONTRADICT"})
        rejected_claims = norm_r_claims

        evidence_status = str(state.get("evidence_status") or "")
        answer_status = str(state.get("answer_status") or "")
        is_out_of_domain = bool(state.get("is_out_of_domain", False))
        primary_intent = str(state.get("primary_intent") or "")
        is_refusal = (
            primary_intent != "MIXED_QA"
            and (
                is_out_of_domain
                or answer_status == "NOT_FOUND_IN_DOCUMENT"
                or any(kw in str(answer).lower() for kw in ["hoàn toàn không được đề cập", "không được đề cập", "chưa xuất hiện trong tài liệu", "không tìm thấy"])
                or (evidence_status == "NOT_FOUND" and not verified_claims)
            )
        )

        # Honest Refusal Fast-Path: If no evidence was found in the target document,
        # produce an authoritative, concise, academic refusal immediately without wasting 2048 tokens or hanging remote API!
        if is_refusal:
            if state.get("final_report") and "OUT_OF_BOUNDS_PAGE" in str(state["final_report"]):
                refusal_report = state["final_report"]
            else:
                specific_reason = str(state.get("answer") or "")
                if "không tồn tại trang" in specific_reason.lower() or "chỉ có" in specific_reason.lower() or "ngoài phạm vi" in specific_reason.lower():
                    refusal_report = (
                        f"# Báo cáo Thẩm định Học thuật (Academic Audit Report)\n\n"
                        f"> **Trạng thái Thẩm định:** [`OUT_OF_BOUNDS_PAGE` | `HONEST_REFUSAL`]\n\n"
                        f"### 1. Tóm tắt Kết luận\n"
                        f"{specific_reason}\n\n"
                        f"### 2. Chi tiết Thẩm định\n"
                        f"- **Yêu cầu đối chiếu:** *\"{query}\"*\n"
                        f"- **Kết luận xác minh:** Số trang yêu cầu nằm ngoài phạm vi số trang thực tế của tài liệu mục tiêu. "
                        f"Hệ thống từ chối tạo sinh nội dung suy đoán để tuân thủ nghiêm ngặt chính sách bảo vệ trung thực học thuật (Zero-Hallucination Policy).\n"
                    )
                else:
                    refusal_report = (
                        f"# Báo cáo Thẩm định Học thuật (Academic Audit Report)\n\n"
                        f"> **Trạng thái Thẩm định:** [`NOT_FOUND_IN_DOCUMENT` | `HONEST_REFUSAL`]\n\n"
                        f"### 1. Tóm tắt Kết luận\n"
                        f"Dựa trên việc rà soát và đối chiếu toàn diện tài liệu mục tiêu, hệ thống xác nhận **không tìm thấy** "
                        f"bất kỳ thông tin hoặc bằng chứng thực nghiệm nào liên quan đến yêu cầu: *\"{query}\"*.\n\n"
                        f"### 2. Chi tiết Thẩm định\n"
                        f"- **Mệnh đề kiểm tra:** Không có cơ sở đối chiếu trong tài liệu mục tiêu.\n"
                        f"- **Kết luận xác minh:** Từ chối phản hồi suy đoán nhằm tuân thủ nghiêm ngặt chính sách bảo vệ trung thực học thuật (Zero-Hallucination Policy).\n"
                    )
            trace = self.log_action("gpu_finetune_honest_refusal_fast_path", {"length": len(refusal_report), "model": "honest_refusal_fast_path"})
            return {
                "final_report": refusal_report,
                "answer": refusal_report,
                "citations": [],
                "reasoning_trace": [trace],
            }

        final_report = ""
        report_data: dict[str, Any] = {}

        # 1. Attempt generation via fine-tuned DoRA model on Remote GPU Worker
        if self.settings.REMOTE_INFERENCE_URL:
            try:
                v_claims_text = "\n".join([f"- {c.get('claim_text') or c.get('text') or str(c)}" for c in verified_claims])
                is_vi_query = bool(re.search(r"[à-ỹÀ-Ỹ]", query))
                prompt = (
                    f"# Đề tài / Câu hỏi nghiên cứu: {query}\n\n"
                    f"## Tóm tắt Tổng quan (Executive Summary):\n{answer}\n\n"
                    f"## Các Mệnh đề Khoa học Đã Thẩm định (Verified Scientific Claims):\n{v_claims_text}\n\n"
                    f"Hãy soạn thảo Báo cáo Nghiên cứu Khoa học hoàn chỉnh bằng TIẾNG VIỆT học thuật, súc tích, bám sát các sự thật đã được kiểm chứng ở trên, chia cấu trúc rõ ràng và gắn trích dẫn chính xác [1], [2]."
                )

                res = llm_client.call_kaggle_specialist(
                    endpoint="api/writer",
                    payload={
                        "query": query,
                        "answer": answer,
                        "claims": verified_claims,
                        "adapter": "writer",
                        "language": "vi" if is_vi_query else "en",
                        "max_tokens": 1100,
                    },
                    fallback_prompt=prompt,
                    fallback_system=WRITER_TRAINING_SYSTEM_PROMPT,
                    adapter="writer",
                    timeout=max(75.0, float(self.settings.REMOTE_TIMEOUT_SECONDS or 90.0)),
                )
                if isinstance(res, dict) and isinstance(res.get("usage"), dict):
                    wu = res["usage"]
                    self.logger.info(
                        f"[GPU Finetune Writer] engine={wu.get('engine')} "
                        f"tokens={wu.get('completion_tokens')} latency={wu.get('latency_sec')}s "
                        f"speed={wu.get('tok_per_sec')} tok/s"
                    )
                if isinstance(res, dict) and res.get("report"):
                    rep = res["report"]
                    # If query is in Vietnamese, ensure output report contains substantial Vietnamese
                    if len(re.findall(r"[à-ỹÀ-Ỹ]", rep)) >= 25 or not re.search(r"[à-ỹÀ-Ỹ]", query):
                        final_report = rep
                        report_data = res.get("metadata", {})
                        self.logger.info(f"[GPU Finetune Writer] DoRA Writer produced {len(final_report)} chars report in Vietnamese.")
                    else:
                        self.logger.info("[GPU Finetune Writer] Remote DoRA writer returned non-Vietnamese report; reconstructing with Vietnamese Academic Report Builder.")
            except Exception as e:
                self.logger.warning(f"[GPU Finetune Writer] Remote DoRA writer fallback to local report builder ({e})")

        # 2. Local Academic Report Builder fallback
        if not final_report:
            final_report, report_data = build_academic_report(
                query=query,
                verified_claims=verified_claims,
                rejected_claims=rejected_claims,
                retrieved_docs=retrieved_docs,
                answer=answer,
                evidence_spans=evidence_spans,
            )

        # 3. Ensure references metadata is populated for audit and citation tracking
        if not report_data.get("references"):
            _, fallback_data = build_academic_report(
                query=query,
                verified_claims=verified_claims,
                rejected_claims=rejected_claims,
                retrieved_docs=retrieved_docs,
                answer=answer,
                evidence_spans=evidence_spans,
            )
            if fallback_data.get("references"):
                report_data["references"] = fallback_data["references"]

        # 4. If final_report lacks citation markers, append Evidence References section
        if report_data.get("references") and not re.search(r"\[\d+\]", final_report):
            ref_lines = []
            for i, r in enumerate(report_data["references"], 1):
                if isinstance(r, dict):
                    ref_id = r.get("ref_id", i)
                    title = r.get("title", "Tài liệu")
                    page_no = r.get("page_no", 1)
                    preview = r.get("preview", "")
                    ref_lines.append(f"[{ref_id}] **{title}** (Trang {page_no}). *\"{preview}...\"*")
                else:
                    ref_lines.append(f"[{i}] **Tài liệu** (Trang 1). *\"{str(r)}...\"*")
            final_report += "\n\n## 4. Danh mục Tài liệu Trích dẫn (Evidence References)\n" + "\n".join(ref_lines)

        trace = self.log_action("gpu_finetune_report_written", {"length": len(final_report), "model": "dora_rs_lora"})

        # Extract structured citations from final_report and state
        existing_citations = list(state.get("citations") or [])
        cited_markers = set(re.findall(r"\[(\d+)\]", final_report))
        for m in sorted(cited_markers):
            existing_citations.append({"id": f"ref_{m}", "source": f"Citation [{m}]"})
        page_markers = set(re.findall(r"(?:Page|Trang)\s*(\d+)", final_report, re.I))
        for p in sorted(page_markers):
            existing_citations.append({"id": f"page_{p}", "source": f"Page {p}"})

        return {
            "final_report": final_report,
            "written_report_data": report_data,
            "citations": existing_citations,
            "reasoning_trace": [trace],
        }
