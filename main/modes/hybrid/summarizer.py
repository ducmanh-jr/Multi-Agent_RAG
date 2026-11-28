"""
Hybrid Mode - Adaptive Multi-Model Summarizer Agent (L4 SOTA Hierarchy).
Dynamically routes synthesis to the strongest live backend based on SOTA performance:
  1. Priority 1 (Cao nhất): GPU Finetune Dual-QLoRA (Generator + Critic on Dual T4)
  2. Priority 2: GPU Max 32B Reasoning (Qwen3-32B-AWQ on Dual T4) if online
  3. Priority 3: Local CPU Max (Qwen2.5-7B on Radeon 780M Vulkan / Local Synthesizer)
  4. Priority 4 (Cuối cùng): Gemini 2.5 Flash API (Last Resort Fallback)
"""

from __future__ import annotations

from typing import Any

from core.base_agent import BaseAgent
from core.provider_router import provider_router
from core.state import AgentState
from modes.cpu_max.summarizer import CPUMaxSummarizerAgent
from modes.gemini.summarizer import GeminiSummarizerAgent
from modes.gpu_finetune.summarizer import GPUFinetuneSummarizerAgent
from modes.gpu_max.summarizer import GPUMaxSummarizerAgent
from utils.local_vulkan_setup import probe_all_capabilities


class HybridSummarizerAgent(BaseAgent):
    """Policy-driven Hybrid Summarizer routing between Gemini 3.5 Flash, Kaggle Qwen 7B/8B, GPU Max, and Local CPU."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="HybridSummarizerAgent", max_internal_retry=max_internal_retry)
        self._gpu_finetune = GPUFinetuneSummarizerAgent(max_internal_retry=max_internal_retry)
        self._gpu_max = GPUMaxSummarizerAgent(max_internal_retry=max_internal_retry)
        self._cpu_max = CPUMaxSummarizerAgent(max_internal_retry=max_internal_retry)
        self._gemini = GeminiSummarizerAgent(max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        complexity = str(state.get("complexity") or "MEDIUM").upper()
        caps = probe_all_capabilities()
        chain = provider_router.resolve_chain("generator", state=state, caps=caps)

        backend_runners = {
            "gemini": ("gemini_primary", self._gemini, "gemini"),
            "gpu_finetune": ("gpu_finetune_dual_qlora", self._gpu_finetune, "gpu_finetune"),
            "gpu_max": ("gpu_max_32b", self._gpu_max, "gpu_max"),
            "cpu_max": ("cpu_max_local", self._cpu_max, "cpu_max"),
        }

        selected_backend = "cpu_max"
        out: dict[str, Any] = {}
        degraded_backup: tuple[str, dict[str, Any]] | None = None

        for idx, key in enumerate(chain):
            entry = backend_runners.get(key)
            if not entry:
                continue
            label, runner, inf_mode = entry
            try:
                candidate = runner.run_in_mode(state, inf_mode)
                if candidate and candidate.get("answer"):
                    fam = str(candidate.get("generator_model_family") or "").lower()
                    is_degraded_extractive = (
                        key in {"gemini", "gpu_finetune"}
                        and any(tok in fam for tok in ("extractive", "cpu_fallback"))
                    )
                    has_downstream_llm = any(later in {"gpu_finetune", "gpu_max", "gemini"} for later in chain[idx + 1 :])
                    if is_degraded_extractive and has_downstream_llm:
                        if degraded_backup is None:
                            degraded_backup = (f"{label}_extractive_backup", candidate)
                        self.logger.warning(
                            "[Hybrid Summarizer] Backend '%s' degraded to '%s'; cascading to next LLM in %s...",
                            key, fam, chain[idx + 1 :],
                        )
                        continue
                    selected_backend = label
                    out = candidate
                    break
            except Exception as exc:
                self.logger.warning(f"[Hybrid Summarizer] Backend '{key}' failed ({exc}), cascading...")

        # Use degraded extractive backup if all downstream LLMs also failed
        if not out.get("answer") and degraded_backup is not None:
            selected_backend, out = degraded_backup

        # Emergency local fallback if every backend in chain failed
        if not out.get("answer"):
            out = self._cpu_max.run_in_mode(state, "cpu_max")
            selected_backend = "cpu_max_emergency"

        trace = self.log_action(
            f"Điều phối mô hình ({selected_backend}) tổng hợp luận điểm & Inner Critic",
            {
                "selected_generator_backend": selected_backend,
                "provider_chain": chain,
                "generator_model_family": out.get("generator_model_family", selected_backend),
                "complexity": complexity,
                "critic_action": out.get("critic_action", "lower_confidence"),
            },
        )
        existing_traces = list(out.get("reasoning_trace") or [])
        existing_traces.append(trace)
        out["reasoning_trace"] = existing_traces
        return out
