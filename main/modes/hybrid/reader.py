"""
Hybrid Mode - Adaptive Reader Agent (Policy-Compliant Hierarchy).
Centralized via ModelGateway (ZERO Gemini):
  1. Priority 1: GPU Finetune LayoutLMv3 Base 2D Document Layout Analysis
  2. Priority 2: High-Speed Local C-Engine PyMuPDF SourceUnit & Markdown Extractor (100% offline)
"""

from __future__ import annotations

from typing import Any
from core.base_agent import BaseAgent
from core.model_gateway import model_gateway
from core.state import AgentState
from modes.cpu_max.reader import CPUMaxReaderAgent
from modes.gpu_finetune.reader import GPUFinetuneReaderAgent
from utils.local_vulkan_setup import probe_all_capabilities


class HybridReaderAgent(BaseAgent):
    """Hybrid Reader: GPU Finetune LayoutLMv3 -> Local PyMuPDF (Zero Gemini)."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="HybridReaderAgent", max_internal_retry=max_internal_retry)
        self._gpu_finetune_reader = GPUFinetuneReaderAgent(max_internal_retry=max_internal_retry)
        self._cpu_max_reader = CPUMaxReaderAgent(max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        caps = probe_all_capabilities()
        backend_chain = model_gateway.get_hybrid_backend_chain(caps, stage="reader")

        # Priority 1: GPU Finetune LayoutLMv3 Base if online
        if "gpu_finetune" in backend_chain:
            try:
                res = self._gpu_finetune_reader.run(state)
                if res.get("source_units"):
                    return res
            except Exception as e:
                self.logger.warning(f"[Hybrid Reader] GPU Finetune LayoutLMv3 failed ({e}), cascading to Local CPU...")

        # Priority 2: High-Speed Local C-Engine PyMuPDF (no Gemini fallback)
        return self._cpu_max_reader.run(state)
