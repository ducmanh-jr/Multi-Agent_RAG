"""
Hybrid Mode - Adaptive Reader Agent (L4 SOTA Hierarchy).
Prioritizes:
  1. Priority 1 (Cao nhất): GPU Finetune LayoutLMv3 Base 2D Document Layout Analysis (Tables & Equations)
  2. Priority 2: High-Speed Local C-Engine PyMuPDF SourceUnit & Markdown Extractor (100% offline)
  3. Priority 3 (Cuối cùng): Gemini Multimodal Vision Reader Fallback
"""

from __future__ import annotations

from typing import Any
from core.base_agent import BaseAgent
from core.state import AgentState
from modes.cpu_max.reader import CPUMaxReaderAgent
from modes.gpu_finetune.reader import GPUFinetuneReaderAgent
from modes.gemini.reader import GeminiReaderAgent
from utils.local_vulkan_setup import probe_all_capabilities


class HybridReaderAgent(BaseAgent):
    """Flagship Hybrid Reader: LayoutLMv3 First -> PyMuPDF Local -> Gemini Multimodal Fallback."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="HybridReaderAgent", max_internal_retry=max_internal_retry)
        self._gpu_finetune_reader = GPUFinetuneReaderAgent(max_internal_retry=max_internal_retry)
        self._cpu_max_reader = CPUMaxReaderAgent(max_internal_retry=max_internal_retry)
        self._gemini_reader = GeminiReaderAgent(max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        caps = probe_all_capabilities()

        # Priority 1: GPU Finetune LayoutLMv3 Base if online
        if caps.get("gpu_finetune", {}).get("online") or caps.get("gpu_ft", {}).get("online"):
            try:
                res = self._gpu_finetune_reader.run(state)
                if res.get("source_units"):
                    return res
            except Exception as e:
                self.logger.warning(f"[Hybrid Reader] GPU Finetune LayoutLMv3 failed ({e}), cascading...")

        # Priority 2: High-Speed Local C-Engine PyMuPDF
        try:
            res = self._cpu_max_reader.run(state)
            if res.get("source_units"):
                return res
        except Exception as e:
            self.logger.warning(f"[Hybrid Reader] Local PyMuPDF failed ({e}), cascading to Gemini...")

        # Priority 3 (Cuối cùng): Gemini Vision Reader API
        if caps.get("gemini", {}).get("online"):
            try:
                return self._gemini_reader.run(state)
            except Exception as e:
                self.logger.warning(f"[Hybrid Reader] Gemini Reader fallback failed: {e}")

        # Emergency local fallback
        return self._cpu_max_reader.run(state)
