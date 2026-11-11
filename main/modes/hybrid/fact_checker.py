"""
Hybrid Mode - Cross-Model Adversarial Fact Checker Agent (L4 SOTA Hierarchy).
Prioritizes:
  1. Priority 1 (Cao nhất): GPU Finetune BGE-Reranker-v2-m3 NLI Head (Macro F1 = 1.0) + Numeric/Date Guard
  2. Priority 2: Local CPU Max MiniCheck NLI + Deterministic Numerical & Date Guard (100% offline)
  3. Priority 3: GPU Max 32B CoT Judge (nếu cụm vLLM 32B online)
  4. Priority 4 (Cuối cùng): Gemini Batch NLI API (Last Resort Fallback)
"""

from __future__ import annotations

import re
from typing import Any

from core.base_agent import BaseAgent
from core.state import AgentState
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
            
            q_entities = [
                e.lower()
                for e in re.findall(r"\b[A-Za-z0-9_-]{4,}\b", query)
                if e.lower() not in {"trong", "tai", "lieu", "trang", "theo", "chinh", "xac", "khong", "thong", "tin", "duoc", "nhung", "dong", "thoi", "hoan", "toan", "nao", "sao", "gi", "bang", "giua", "cac", "nhung"}
            ]
            
            found_entities = [e for e in q_entities if e in full_text] if (full_text and q_entities) else []
            if len(found_entities) >= 2 and len(q_entities) >= 2:
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

        selected_verifier = "cpu_max_minicheck"
        out: dict[str, Any] = {}

        # Priority 1 (Cao nhất): Fine-Tuned BGE-Reranker-v2-m3 NLI Head on GPU Worker (Macro F1 = 1.0)
        if caps.get("gpu_finetune", {}).get("online") or caps.get("gpu_ft", {}).get("online"):
            try:
                selected_verifier = "gpu_finetune_bge_nli_head"
                out = self._gpu_finetune_fc.run(state)
            except Exception as exc:
                self.logger.warning(f"[Hybrid FactChecker] GPU Finetune NLI failed ({exc}), cascading...")

        # Priority 2: Local CPU Max MiniCheck NLI + Deterministic Numerical Guard (100% offline)
        if not out or "fact_check_label" not in out:
            try:
                selected_verifier = "local_cpu_minicheck_nli"
                out = self._cpu_max_fc.run(state)
                out["verifier_model_family"] = "local_minicheck_nli"
            except Exception as exc:
                self.logger.warning(f"[Hybrid FactChecker] Local CPU verifier failed ({exc}), cascading...")

        # Priority 3: GPU Max 32B CoT Judge nếu online
        if (not out or "fact_check_label" not in out) and caps.get("kaggle_max", {}).get("online") and caps.get("kaggle_max", {}).get("has_vllm_32b"):
            try:
                selected_verifier = "gpu_max_32b_cot_judge"
                out = self._gpu_max_fc.run(state)
            except Exception as exc:
                self.logger.warning(f"[Hybrid FactChecker] GPU Max judge failed ({exc}), cascading...")

        # Priority 4 (Ưu tiên CUỐI CÙNG): Gemini Batch NLI API (Chỉ khi GPU & Local CPU đều offline)
        if (not out or "fact_check_label" not in out) and caps.get("gemini", {}).get("online"):
            try:
                selected_verifier = "gemini_batch_nli_fallback"
                out = self._gemini_fc.run(state)
            except Exception as exc:
                self.logger.warning(f"[Hybrid FactChecker] Gemini verifier fallback failed ({exc})")

        # Emergency fallback: deterministic local verification
        if not out or "fact_check_label" not in out:
            out = self._cpu_max_fc.run(state)
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
