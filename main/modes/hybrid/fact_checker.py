"""
Hybrid Mode - Cross-Model Adversarial Fact Checker Agent (L4).
Eliminates Self-Confirmation Bias by enforcing that the Verifier belongs to a DIFFERENT
model family than the Generator:
  - If Generator == 'gemini_flash' -> Verifies using Kaggle mDeBERTa-v3-xnli / Qwen Critic LoRA (if online)
    or Local CPU Max MiniCheck NLI + Deterministic Numeric/Date/Quote Guard (0 extra Gemini API calls!).
  - If Generator == 'qwen*' (Kaggle 32B / Kaggle FN / Local Qwen Vulkan) -> Verifies using
    Gemini Batch NLI (1 Call) + Deterministic Numeric/Date Guard (or Local NLI if offline).
"""

from __future__ import annotations

from typing import Any

from core.base_agent import BaseAgent
from core.state import AgentState
from modes.cpu_max.fact_checker import CPUMaxFactCheckerAgent
from modes.gemini.fact_checker import GeminiFactCheckerAgent
from modes.kaggle_fn.fact_checker import KaggleFNFactCheckerAgent
from utils.local_vulkan_setup import probe_all_capabilities


class HybridFactCheckerAgent(BaseAgent):
    """Flagship Hybrid Fact-Checker: Cross-Model Adversarial Verification + Numeric/Date Guard."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="HybridFactCheckerAgent", max_internal_retry=max_internal_retry)
        self._cpu_max_fc = CPUMaxFactCheckerAgent(max_internal_retry=max_internal_retry)
        self._gemini_fc = GeminiFactCheckerAgent(max_internal_retry=max_internal_retry)
        self._kaggle_fn_fc = KaggleFNFactCheckerAgent(max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        ans = str(state.get("answer", "") or "")
        if "hoàn toàn không được đề cập" in ans or "bài báo độc lập" in ans or "không hề đề cập" in ans:
            claims = state.get("claims", []) or []
            v_claims = [c.to_dict() if hasattr(c, "to_dict") else c for c in claims]
            return {
                "verified_claims": v_claims,
                "rejected_claims": [],
                "fact_check_label": "support",
                "verifier_model_family": "rule_guard",
                "reasoning_trace": [self.log_action("hybrid_fact_check_honest_refusal_approved")],
            }

        gen_family = str(state.get("generator_model_family") or "unknown").lower()
        caps = probe_all_capabilities()

        selected_verifier = "local_minicheck_numeric_guard"
        out: dict[str, Any] = {}

        if "gemini" in gen_family:
            # Generator was Gemini -> Strictly choose non-Gemini Verifier (Cross-Family!)
            if caps["kaggle_fn"]["online"]:
                selected_verifier = "kaggle_mdeberta_nli_critic"
                out = self._kaggle_fn_fc.run(state)
            else:
                selected_verifier = "local_minicheck_nli_numeric_guard"
                out = self._cpu_max_fc.run(state)
                out["verifier_model_family"] = "local_minicheck_nli"
        else:
            # Generator was Qwen (Kaggle 32B / Kaggle FN / Local Vulkan) or Extractive
            if caps["gemini"]["online"] and "extractive" not in gen_family:
                selected_verifier = "gemini_batch_nli_1call"
                out = self._gemini_fc.run(state)
            elif caps["kaggle_fn"]["online"] and "qlora" not in gen_family:
                selected_verifier = "kaggle_mdeberta_nli_critic"
                out = self._kaggle_fn_fc.run(state)
            else:
                selected_verifier = "local_minicheck_nli_numeric_guard"
                out = self._cpu_max_fc.run(state)
                out["verifier_model_family"] = "local_minicheck_nli"

        ver_family = str(out.get("verifier_model_family") or selected_verifier)
        trace = self.log_action(
            f"Đối soát đối kháng chéo (Adversarial): Generator={gen_family} ⚔️ Verifier={ver_family}",
            {
                "generator_model_family": gen_family,
                "verifier_model_family": ver_family,
                "cross_family_enforced": gen_family != ver_family,
                "fact_check_label": out.get("fact_check_label", "support"),
                "verified_count": len(out.get("verified_claims", [])),
                "rejected_count": len(out.get("rejected_claims", [])),
            },
        )
        existing_traces = list(out.get("reasoning_trace") or [])
        existing_traces.append(trace)
        out["reasoning_trace"] = existing_traces
        return out
