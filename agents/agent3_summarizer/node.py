"""
Agent 3: Summarizer Agent (Generator + Critic Architecture).
Combines:
  1. Generator: Extracts atomic claims and verbatim evidence spans from SourceUnits.
  2. Critic: Self-audits the draft to remove ungrounded assertions.
"""

from __future__ import annotations

from typing import Any
from core.base_agent import BaseAgent
from core.state import AgentState
from agents.agent3_summarizer.components.generator import generate_summary_and_claims
from agents.agent3_summarizer.components.critic import critique_summary_and_claims


class SummarizerAgent(BaseAgent):
    """Summarizer Agent producing grounded answers and atomic propositions."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="SummarizerAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        source_units = state.get("source_units", []) or []

        self.logger.info(f"SummarizerAgent running for: '{query}' with {len(source_units)} source units.")

        # 1. Generator Pass
        raw_answer, evidence_spans, raw_claims = generate_summary_and_claims(query, source_units)

        # 2. Critic Pass
        refined_answer, filtered_claims, warnings = critique_summary_and_claims(
            raw_answer, raw_claims, evidence_spans
        )

        trace = self.log_action(
            "summarization_and_critique_completed",
            {
                "extracted_claims_count": len(raw_claims),
                "approved_claims_count": len(filtered_claims),
                "evidence_spans_count": len(evidence_spans),
                "warnings": warnings,
            },
        )

        return {
            "answer": refined_answer,
            "evidence_spans": evidence_spans,
            "claims": filtered_claims,
            "reasoning_trace": [trace],
        }
