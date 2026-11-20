"""
Hybrid Mode - Adaptive Multi-Model Summarizer Agent (Policy-Compliant Hierarchy).
Centralized via ModelGateway (ZERO Gemini):
  1. Priority 1: GPU Finetune Dual-QLoRA (Generator + Critic on Dual T4)
  2. Priority 2: GPU Max 32B Reasoning (Qwen3-32B-AWQ on Dual T4) if online
  3. Priority 3: Local CPU Max (Qwen2.5-7B on Radeon 780M Vulkan / Local Synthesizer)
"""

from __future__ import annotations

from typing import Any

from core.base_agent import BaseAgent
from core.model_gateway import model_gateway
from core.state import AgentState
from modes.cpu_max.summarizer import CPUMaxSummarizerAgent
from modes.gpu_finetune.summarizer import GPUFinetuneSummarizerAgent
from modes.gpu_max.summarizer import GPUMaxSummarizerAgent
from utils.local_vulkan_setup import probe_all_capabilities


class HybridSummarizerAgent(BaseAgent):
    """Hybrid Summarizer: GPU Finetune -> GPU Max -> Local CPU (Zero Gemini)."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="HybridSummarizerAgent", max_internal_retry=max_internal_retry)
        self._gpu_finetune = GPUFinetuneSummarizerAgent(max_internal_retry=max_internal_retry)
        self._gpu_max = GPUMaxSummarizerAgent(max_internal_retry=max_internal_retry)
        self._cpu_max = CPUMaxSummarizerAgent(max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        complexity = str(state.get("complexity") or "MEDIUM").upper()
        caps = probe_all_capabilities()
        backend_chain = model_gateway.get_hybrid_backend_chain(caps, stage="summarizer")

        selected_backend = "cpu_max"
        out: dict[str, Any] = {}

        for backend in backend_chain:
            if out.get("processing_status") == "VERIFICATION_FAILED" or out.get("critic_action") == "verification_failed":
                break
            if out.get("answer"):
                break
            if backend == "gpu_finetune":
                try:
                    selected_backend = "gpu_finetune_dual_qlora"
                    out = self._gpu_finetune.run(state)
                except Exception as exc:
                    self.logger.warning(f"[Hybrid Summarizer] GPU Finetune failed ({exc}), cascading...")
            elif backend == "gpu_max":
                try:
                    selected_backend = "gpu_max_32b"
                    out = self._gpu_max.run(state)
                except Exception as exc:
                    self.logger.warning(f"[Hybrid Summarizer] GPU Max 32B failed ({exc}), cascading...")
            elif backend == "cpu_max":
                try:
                    selected_backend = "cpu_max_local"
                    out = self._cpu_max.run(state)
                except Exception as exc:
                    self.logger.warning(f"[Hybrid Summarizer] Local CPU Max failed ({exc}). Zero Gemini fallback allowed.")

        # Emergency local fallback ONLY if all candidates were offline/empty (never override VERIFICATION_FAILED!)
        if (
            not out.get("answer")
            and out.get("processing_status") != "VERIFICATION_FAILED"
            and out.get("critic_action") != "verification_failed"
        ):
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
