"""Deterministic source-text verifier for the cpu_min execution mode."""

from typing import Any

from core.base_agent import BaseAgent
from core.state import AgentState
from core.verifier import review_claims


class CPUMinFactCheckerAgent(BaseAgent):
    def __init__(self, max_internal_retry: int = 1):
        super().__init__(name="CPUMinFactCheckerAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        result = review_claims(state)
        result["verifier_model_family"] = "deterministic_guard"
        result["tools_used"] = ["deterministic_numeric_math_guard", "verbatim_quote_verifier"]
        result.setdefault("reasoning_trace", []).append(
            self.log_action("cpu_min_fact_check_completed", {"label": result["fact_check_label"],
                            "verified_count": len(result["verified_claims"]), "rejected_count": len(result["rejected_claims"])}))
        return result
