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
from core.provider_router import provider_router
from core.state import AgentState
from core.verifier import review_abstention
from modes.cpu_max.fact_checker import CPUMaxFactCheckerAgent
from modes.gemini.fact_checker import GeminiFactCheckerAgent
from modes.gpu_finetune.fact_checker import GPUFinetuneFactCheckerAgent
from modes.gpu_max.fact_checker import GPUMaxFactCheckerAgent
from utils.local_vulkan_setup import probe_all_capabilities


class HybridFactCheckerAgent(BaseAgent):
    """Policy-driven Hybrid Fact-Checker: Local/Kaggle BGE NLI Head ALWAYS prioritized over Gemini."""

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
        chain = provider_router.resolve_chain("fact_checker", state=state, caps=caps)

        fc_runners = {
            "gpu_finetune": ("gpu_finetune_bge_nli_head", self._gpu_finetune_fc, "gpu_finetune"),
            "cpu_max": ("local_cpu_minicheck_nli", self._cpu_max_fc, "cpu_max"),
            "gpu_max": ("gpu_max_32b_cot_judge", self._gpu_max_fc, "gpu_max"),
            "gemini": ("gemini_batch_nli_fallback", self._gemini_fc, "gemini"),
        }

        selected_verifier = "cpu_max_minicheck"
        out: dict[str, Any] = {}

        for idx, key in enumerate(chain):
            entry = fc_runners.get(key)
            if not entry:
                continue
            # Do not stop at cpu_max verbatim-only fallback if no local NLI model is loaded
            # and a downstream LLM/NLI verifier (gpu_max or gemini) is available in the chain.
            if (
                key == "cpu_max"
                and getattr(self._cpu_max_fc.judge, "classifier", None) is None
                and any(later in {"gpu_max", "gemini"} for later in chain[idx + 1 :])
            ):
                continue
            label, runner, inf_mode = entry
            try:
                candidate = runner.run_in_mode(state, inf_mode)
                if candidate and "fact_check_label" in candidate:
                    selected_verifier = label
                    out = candidate
                    if key == "cpu_max":
                        out.setdefault("verifier_model_family", "local_minicheck_nli")
                    break
            except Exception as exc:
                self.logger.warning(f"[Hybrid FactChecker] Backend '{key}' failed ({exc}), cascading...")

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
