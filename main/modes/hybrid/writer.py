"""
Hybrid Mode - Citation-Grounded Writer Agent (L4 SOTA Hierarchy).
Prioritizes:
  1. Priority 1 (Cao nhất): GPU Finetune Qwen2.5-7B DoRA + rsLoRA Academic Writer (Accuracy 99.6%)
  2. Priority 2: High-Performance Local Academic Report Builder with Research Gap Radar
  3. Priority 3 (Cuối cùng): Gemini Academic Writer Fallback
"""

from __future__ import annotations

from typing import Any

from agents.agent5_writer.components.report_builder import build_academic_report
from core.base_agent import BaseAgent
from core.provider_router import provider_router
from core.state import AgentState
from modes.gpu_finetune.writer import GPUFinetuneWriterAgent
from modes.gemini.writer import GeminiWriterAgent
from utils.local_vulkan_setup import probe_all_capabilities


class HybridWriterAgent(BaseAgent):
    """Policy-driven Hybrid Writer routing between Gemini, Kaggle Qwen DoRA Writer, and Local Report Builder."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="HybridWriterAgent", max_internal_retry=max_internal_retry)
        self._gpu_finetune_writer = GPUFinetuneWriterAgent(max_internal_retry=max_internal_retry)
        self._gemini_writer = GeminiWriterAgent(max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query") or state.get("question", "")
        verified_claims = state.get("verified_claims", []) or []
        rejected_claims = state.get("rejected_claims", []) or []
        retrieved_docs = state.get("retrieved_docs", []) or []
        answer = state.get("answer", "")
        evidence_spans = state.get("evidence_spans", []) or []
        gen_family = state.get("generator_model_family", "auto")
        ver_family = state.get("verifier_model_family", "cross_model_guard")

        caps = probe_all_capabilities()
        chain = provider_router.resolve_chain("writer", state=state, caps=caps)
        final_report = ""
        report_data: dict[str, Any] = {}
        writer_backend = "local_report_builder"

        for key in chain:
            if key == "gemini":
                try:
                    res = self._gemini_writer.run_in_mode(state, "gemini")
                    if res.get("final_report") and len(res["final_report"].strip()) > 60:
                        final_report = res["final_report"]
                        report_data = res.get("written_report_data", {})
                        writer_backend = report_data.get("engine", "gemini_writer")
                        break
                except Exception as e:
                    self.logger.warning(f"[Hybrid Writer] Gemini Writer failed ({e}), falling back...")
            elif key == "gpu_finetune":
                try:
                    res = self._gpu_finetune_writer.run_in_mode(state, "gpu_finetune")
                    if res.get("final_report") and len(res["final_report"].strip()) > 100:
                        final_report = res["final_report"]
                        report_data = res.get("written_report_data", {})
                        writer_backend = report_data.get("engine", "gpu_finetune_dora_writer")
                        break
                except Exception as e:
                    self.logger.warning(f"[Hybrid Writer] GPU Finetune Writer failed ({e}), falling back...")
            elif key == "cpu_max":
                final_report, report_data = build_academic_report(
                    query=query,
                    verified_claims=verified_claims,
                    rejected_claims=rejected_claims,
                    retrieved_docs=retrieved_docs,
                    answer=answer,
                    answer_status=state.get("answer_status"),
                    evidence_spans=evidence_spans,
                )
                writer_backend = "local_report_builder"
                break

        # Guaranteed offline local fallback
        if not final_report:
            final_report, report_data = build_academic_report(
                query=query,
                verified_claims=verified_claims,
                rejected_claims=rejected_claims,
                retrieved_docs=retrieved_docs,
                answer=answer,
                answer_status=state.get("answer_status"),
                evidence_spans=evidence_spans,
            )
            writer_backend = "local_report_builder"
            writer_backend = "local_academic_report_builder"

        # Priority 3 (Cuối cùng): Gemini Writer fallback nếu cả 2 trên không tạo được báo cáo
        if not final_report and caps.get("gemini", {}).get("online"):
            try:
                res = self._gemini_writer.run_in_mode(state, "gemini")
                final_report = res.get("final_report", "")
                report_data = res.get("written_report_data", {})
                writer_backend = "gemini_writer_fallback"
            except Exception as e:
                self.logger.warning(f"[Hybrid Writer] Gemini Writer fallback failed ({e})")

        report_data["engine"] = f"Hybrid_{writer_backend}"
        report_data["generator_model_family"] = gen_family
        report_data["verifier_model_family"] = ver_family

        trace = self.log_action(
            f"Ký duyệt báo cáo học thuật ({writer_backend})",
            {
                "report_length": len(final_report),
                "writer_backend": writer_backend,
                "generator_model_family": gen_family,
                "verifier_model_family": ver_family,
                "verified_claims_count": len(verified_claims),
            },
        )

        return {
            "final_report": final_report,
            "written_report_data": report_data,
            "reasoning_trace": [trace],
        }
