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
from agents.agent3_summarizer.components.generator import (
    generate_summary_and_claims,
    pack_context_with_coverage,
)
from agents.agent3_summarizer.components.critic import evaluate_critic_contract


class SummarizerAgent(BaseAgent):
    """Summarizer Agent producing grounded answers, context coverage, and fail-closed Critic decisions."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="SummarizerAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        source_units = state.get("source_units", []) or []

        self.logger.info(f"SummarizerAgent running for: '{query}' with {len(source_units)} source units.")

        # 0. Compute context coverage for downstream Evidence assessment
        _, _, context_coverage = pack_context_with_coverage(source_units, max_tokens=3800)

        # 1. Generator Pass
        raw_answer, evidence_spans, raw_claims = generate_summary_and_claims(query, source_units)

        # 2. Fail-Closed Critic Pass
        critic_res = evaluate_critic_contract(raw_answer, raw_claims, evidence_spans)

        trace = self.log_action(
            "summarization_and_critique_completed",
            {
                "extracted_claims_count": len(raw_claims),
                "approved_claims_count": len(critic_res["claims"]),
                "evidence_spans_count": len(evidence_spans),
                "critic_action": critic_res["critic_action"],
                "processing_status": critic_res["processing_status"],
                "context_coverage": context_coverage,
                "warnings": critic_res["warnings"],
            },
        )

        out: dict[str, Any] = {
            "answer": critic_res["answer"],
            "evidence_spans": evidence_spans,
            "claims": critic_res["claims"],
            "critic_action": critic_res["critic_action"],
            "critic_feedback": critic_res["critic_feedback"],
            "processing_status": critic_res["processing_status"],
            "error_stage": critic_res["error_stage"],
            "error_code": critic_res["error_code"],
            "error_message": critic_res["error_message"],
            "confidence": critic_res["confidence"],
            "context_coverage": context_coverage,
            "reasoning_trace": [trace],
        }
        return out
