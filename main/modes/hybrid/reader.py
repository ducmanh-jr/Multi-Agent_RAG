"""
Hybrid Mode - Adaptive Reader Agent (L4).
Runs local C-Speed PyMuPDF SourceUnit extraction + automatic Markdown Table & Equation extraction
on top hit PDF pages (zero network overhead, zero API calls).
"""

from __future__ import annotations

from typing import Any
from modes.cpu_max.reader import CPUMaxReaderAgent
from core.state import AgentState


class HybridReaderAgent(CPUMaxReaderAgent):
    """Flagship Hybrid Reader Agent: Local PyMuPDF + Structural Table Extractor."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(max_internal_retry=max_internal_retry)
        self.name = "HybridReaderAgent"

    def run(self, state: AgentState) -> dict[str, Any]:
        return super().run(state)
