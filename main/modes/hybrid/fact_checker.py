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

import re
from typing import Any

from core.base_agent import BaseAgent
from core.state import AgentState
from modes.cpu_max.fact_checker import CPUMaxFactCheckerAgent
from modes.gemini.fact_checker import GeminiFactCheckerAgent
from modes.gpu_ft.fact_checker import GPUFTFactCheckerAgent as KaggleFNFactCheckerAgent
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
        is_explicit_refusal = any(
            pat in ans.lower()
            for pat in [
                "hoàn toàn không được đề cập",
                "hoàn toàn không đề cập",
                "không tìm thấy thông tin",
                "tài liệu hiện tại không cung cấp đủ thông tin",
            ]
        ) or bool(state.get("is_out_of_domain"))

        # Negative Verification: If the agent claims information is absent, verify that context indeed lacks it
        if is_explicit_refusal:
            query = state.get("query") or state.get("question", "")
            source_units = state.get("source_units", []) or state.get("retrieved_docs", []) or []
            full_text = " ".join(str(u.get("text", "")).lower() for u in source_units)
            
            # Extract key technical entities from question
            q_entities = [
                e.lower()
                for e in re.findall(r"\b[A-Za-z0-9_-]{4,}\b", query)
                if e.lower() not in {"trong", "tai", "lieu", "trang", "theo", "chinh", "xac", "khong"}
            ]
            
            # If ANY significant entity is prominently found in retrieved context, then refusal is FALSE!
            found_entities = [e for e in q_entities if e in full_text] if (full_text and q_entities) else []
            if len(found_entities) >= 2 and len(q_entities) >= 2:
                # The document actually mentions these entities! Agent falsely refused!
                self.logger.warning(f"[Hybrid FactChecker] False refusal detected: {found_entities} are in context!")
                return {
                    "verified_claims": [],
                    "rejected_claims": [{"claim": ans, "reason": f"Tài liệu thực tế CÓ chứa thông tin về: {found_entities}"}],
                    "fact_check_label": "contradict",
                    "verifier_model_family": "negative_verification_guard",
                    "reasoning_trace": [self.log_action("hybrid_fact_check_false_refusal_detected")],
                }
            
            claims = state.get("claims", []) or []
            v_claims = [c.to_dict() if hasattr(c, "to_dict") else c for c in claims]
            return {
                "verified_claims": v_claims,
                "rejected_claims": [],
                "fact_check_label": "support",
                "verifier_model_family": "negative_verification_guard",
                "reasoning_trace": [self.log_action("hybrid_fact_check_verified_honest_refusal")],
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
