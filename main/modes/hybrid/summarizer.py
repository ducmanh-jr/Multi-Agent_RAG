"""
Hybrid Mode - Adaptive Multi-Model Summarizer Agent (L4).
Dynamically routes synthesis to the strongest live backend based on query complexity
and real-time capability probing:
  1. Kaggle Max (Qwen3-32B-AWQ on Dual T4) if online & has_vllm_32b
  2. Kaggle FN (Generator + Critic QLoRA on Dual T4) if Kaggle Worker online
  3. Local CPU Max (Qwen2.5-7B on Radeon 780M Vulkan) for LOW complexity queries when Ollama is online
  4. Gemini 2.5 Flash (1 Single-Call Synthesis + Self-Critic) when Gemini API is online
  5. CPU Max Extractive Synthesizer (100% offline deterministic guarantee)
"""

from __future__ import annotations

from typing import Any

from core.base_agent import BaseAgent
from core.state import AgentState
from modes.cpu_max.summarizer import CPUMaxSummarizerAgent
from modes.gemini.summarizer import GeminiSummarizerAgent
from modes.kaggle_fn.summarizer import KaggleFNSummarizerAgent
from modes.kaggle_max.summarizer import KaggleMaxSummarizerAgent
from utils.local_vulkan_setup import probe_all_capabilities


class HybridSummarizerAgent(BaseAgent):
    """Flagship Hybrid Summarizer: Dynamic Best-Available Generator Router."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="HybridSummarizerAgent", max_internal_retry=max_internal_retry)
        self._kaggle_max = KaggleMaxSummarizerAgent(max_internal_retry=max_internal_retry)
        self._kaggle_fn = KaggleFNSummarizerAgent(max_internal_retry=max_internal_retry)
        self._gemini = GeminiSummarizerAgent(max_internal_retry=max_internal_retry)
        self._cpu_max = CPUMaxSummarizerAgent(max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        complexity = str(state.get("complexity") or "MEDIUM").upper()
        caps = probe_all_capabilities()

        selected_backend = "cpu_max"
        out: dict[str, Any] = {}

        # Priority 1: Kaggle Max 32B Reasoning if live
        if caps["kaggle_max"]["online"] and caps["kaggle_max"].get("has_vllm_32b"):
            try:
                selected_backend = "kaggle_max_32b"
                out = self._kaggle_max.run(state)
            except Exception as exc:
                self.logger.warning(f"[Hybrid Summarizer] Kaggle Max 32B failed ({exc}), cascading...")

        # Priority 2: Kaggle FN Multi-LoRA (Generator + Critic) if live
        if not out.get("answer") and caps["kaggle_fn"]["online"]:
            try:
                selected_backend = "kaggle_fn_multilora"
                out = self._kaggle_fn.run(state)
            except Exception as exc:
                self.logger.warning(f"[Hybrid Summarizer] Kaggle FN failed ({exc}), cascading...")

        # Priority 3: Local Radeon 780M Vulkan (Ollama 7B) if query is LOW complexity and Ollama is online
        if not out.get("answer") and complexity == "LOW" and caps["cpu_max"]["ollama_online"]:
            try:
                selected_backend = "local_radeon_780m_vulkan"
                out = self._cpu_max.run(state)
            except Exception as exc:
                self.logger.warning(f"[Hybrid Summarizer] Local Vulkan failed ({exc}), cascading...")

        # Priority 4: Gemini 2.5 Flash (1 Call) if API key & quota available
        if not out.get("answer") and caps["gemini"]["online"]:
            try:
                selected_backend = "gemini_2.5_flash"
                out = self._gemini.run(state)
            except Exception as exc:
                self.logger.warning(f"[Hybrid Summarizer] Gemini Flash failed ({exc}), cascading...")

        # Priority 5: Local CPU Max (Radeon 780M Vulkan or Extractive Synthesizer)
        if not out.get("answer"):
            selected_backend = "cpu_max_local"
            out = self._cpu_max.run(state)

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
