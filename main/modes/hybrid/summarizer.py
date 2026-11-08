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
from core.state import AgentState
from modes.cpu_max.summarizer import CPUMaxSummarizerAgent
from modes.gemini.summarizer import GeminiSummarizerAgent
from modes.gpu_finetune.summarizer import GPUFinetuneSummarizerAgent
from modes.gpu_max.summarizer import GPUMaxSummarizerAgent
from utils.local_vulkan_setup import probe_all_capabilities


class HybridSummarizerAgent(BaseAgent):
    """Flagship Hybrid Summarizer: SOTA GPU Finetune First -> Local CPU -> Gemini Last Fallback."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="HybridSummarizerAgent", max_internal_retry=max_internal_retry)
        self._gpu_finetune = GPUFinetuneSummarizerAgent(max_internal_retry=max_internal_retry)
        self._gpu_max = GPUMaxSummarizerAgent(max_internal_retry=max_internal_retry)
        self._cpu_max = CPUMaxSummarizerAgent(max_internal_retry=max_internal_retry)
        self._gemini = GeminiSummarizerAgent(max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        complexity = str(state.get("complexity") or "MEDIUM").upper()
        caps = probe_all_capabilities()

        selected_backend = "cpu_max"
        out: dict[str, Any] = {}

        # Priority 1: GPU Finetune Dual-QLoRA (Generator + Critic) on Kaggle GPU Worker
        if caps.get("gpu_finetune", {}).get("online") or caps.get("gpu_ft", {}).get("online"):
            try:
                selected_backend = "gpu_finetune_dual_qlora"
                out = self._gpu_finetune.run(state)
            except Exception as exc:
                self.logger.warning(f"[Hybrid Summarizer] GPU Finetune failed ({exc}), cascading...")

        # Priority 2: GPU Max 32B Reasoning if available
        if not out.get("answer") and caps.get("kaggle_max", {}).get("online") and caps.get("kaggle_max", {}).get("has_vllm_32b"):
            try:
                selected_backend = "gpu_max_32b"
                out = self._gpu_max.run(state)
            except Exception as exc:
                self.logger.warning(f"[Hybrid Summarizer] GPU Max 32B failed ({exc}), cascading...")

        # Priority 3: Local CPU Max (Radeon 780M Vulkan Ollama 7B or Local Extractive Synthesizer)
        if not out.get("answer"):
            try:
                selected_backend = "cpu_max_local"
                out = self._cpu_max.run(state)
            except Exception as exc:
                self.logger.warning(f"[Hybrid Summarizer] Local CPU Max failed ({exc}), cascading to Gemini fallback...")

        # Priority 4 (Ưu tiên CUỐI CÙNG): Gemini 2.5 Flash API (chỉ khi GPU & Local CPU đều không khả dụng)
        if not out.get("answer") and caps.get("gemini", {}).get("online"):
            try:
                selected_backend = "gemini_2.5_flash_fallback"
                out = self._gemini.run(state)
            except Exception as exc:
                self.logger.warning(f"[Hybrid Summarizer] Gemini fallback failed ({exc})")

        # Emergency fallback if all failed
        if not out.get("answer"):
            out = self._cpu_max.run(state)
            selected_backend = "cpu_max_emergency"

        trace = self.log_action(
            f"Điều phối mô hình ({selected_backend}) tổng hợp luận điểm & Inner Critic",
            {
                "selected_generator_backend": selected_backend,
                "generator_model_family": out.get("generator_model_family", selected_backend),
                "complexity": complexity,
                "critic_action": out.get("critic_action", "approve"),
            },
        )
        existing_traces = list(out.get("reasoning_trace") or [])
        existing_traces.append(trace)
        out["reasoning_trace"] = existing_traces
        return out
