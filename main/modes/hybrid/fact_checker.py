"""
Hybrid Mode - Cross-Model Adversarial Fact Checker Agent (L4 SOTA Hierarchy).
Prioritizes:
  1. Priority 1 (Cao nhất): GPU Finetune BGE-Reranker-v2-m3 NLI Head (Macro F1 = 1.0) + Numeric/Date Guard
  2. Priority 2: Local CPU Max MiniCheck NLI + Deterministic Numerical & Date Guard (100% offline)
  3. Priority 3: GPU Max 32B CoT Judge (nếu cụm vLLM 32B online)
  4. Priority 4 (Cuối cùng): Gemini Batch NLI API (Last Resort Fallback)
"""

from __future__ import annotations

from typing import Any

from core.base_agent import BaseAgent
from core.state import AgentState
from core.verifier import review_abstention
from modes.cpu_max.fact_checker import CPUMaxFactCheckerAgent
from modes.gemini.fact_checker import GeminiFactCheckerAgent
from modes.gpu_finetune.fact_checker import GPUFinetuneFactCheckerAgent
from modes.gpu_max.fact_checker import GPUMaxFactCheckerAgent
from utils.local_vulkan_setup import probe_all_capabilities


class HybridFactCheckerAgent(BaseAgent):
    """Flagship Hybrid Fact-Checker: SOTA BGE NLI Head First -> Local CPU -> Gemini Last Fallback."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="HybridFactCheckerAgent", max_internal_retry=max_internal_retry)
        self._gpu_finetune_fc = GPUFinetuneFactCheckerAgent(max_internal_retry=max_internal_retry)
        self._cpu_max_fc = CPUMaxFactCheckerAgent(max_internal_retry=max_internal_retry)
        self._gpu_max_fc = GPUMaxFactCheckerAgent(max_internal_retry=max_internal_retry)
        self._gemini_fc = GeminiFactCheckerAgent(max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        refusal = review_abstention(state)
        if refusal is not None:
            return refusal

        gen_family = str(state.get("generator_model_family") or "unknown").lower()
        caps = probe_all_capabilities()

        selected_verifier = "cpu_max_minicheck"
        out: dict[str, Any] = {}

        # Priority 1 (Cao nhất): Fine-Tuned BGE-Reranker-v2-m3 NLI Head on GPU Worker (Macro F1 = 1.0)
        if caps.get("gpu_finetune", {}).get("online") or caps.get("gpu_ft", {}).get("online"):
            try:
                selected_verifier = "gpu_finetune_bge_nli_head"
                out = self._gpu_finetune_fc.run_in_mode(state, "gpu_finetune")
            except Exception as exc:
                self.logger.warning(f"[Hybrid FactChecker] GPU Finetune NLI failed ({exc}), cascading...")

        # Priority 2: Local CPU Max MiniCheck NLI + Deterministic Numerical Guard (100% offline)
        if not out or "fact_check_label" not in out:
            try:
                selected_verifier = "local_cpu_minicheck_nli"
                out = self._cpu_max_fc.run_in_mode(state, "cpu_max")
                out["verifier_model_family"] = "local_minicheck_nli"
            except Exception as exc:
                self.logger.warning(f"[Hybrid FactChecker] Local CPU verifier failed ({exc}), cascading...")

        # Priority 3: GPU Max 32B CoT Judge nếu online
        if (not out or "fact_check_label" not in out) and caps.get("kaggle_max", {}).get("online") and caps.get("kaggle_max", {}).get("has_vllm_32b"):
            try:
                selected_verifier = "gpu_max_32b_cot_judge"
                out = self._gpu_max_fc.run_in_mode(state, "gpu_max")
            except Exception as exc:
                self.logger.warning(f"[Hybrid FactChecker] GPU Max judge failed ({exc}), cascading...")

        # Priority 4 (Ưu tiên CUỐI CÙNG): Gemini Batch NLI API (Chỉ khi GPU & Local CPU đều offline)
        if (not out or "fact_check_label" not in out) and caps.get("gemini", {}).get("online"):
            try:
                selected_verifier = "gemini_batch_nli_fallback"
                out = self._gemini_fc.run_in_mode(state, "gemini")
            except Exception as exc:
                self.logger.warning(f"[Hybrid FactChecker] Gemini verifier fallback failed ({exc})")

        # Emergency fallback: deterministic local verification
        if not out or "fact_check_label" not in out:
            out = self._cpu_max_fc.run_in_mode(state, "cpu_max")
            selected_verifier = "deterministic_emergency"

        ver_family = str(out.get("verifier_model_family") or selected_verifier)
        trace = self.log_action(
            f"Đối soát sự thật ({ver_family}) bảo vệ độ trung thực học thuật",
            {
                "generator_model_family": gen_family,
                "verifier_model_family": ver_family,
                "fact_check_label": out.get("fact_check_label", "support"),
                "verified_count": len(out.get("verified_claims", [])),
                "rejected_count": len(out.get("rejected_claims", [])),
            },
        )
        existing_traces = list(out.get("reasoning_trace") or [])
        existing_traces.append(trace)
        out["reasoning_trace"] = existing_traces
        return out
