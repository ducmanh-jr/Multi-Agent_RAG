"""GPU specialist NLI verifier for the gpu_ft execution mode."""

from typing import Any

from core.base_agent import BaseAgent
from core.state import AgentState
from core.verifier import review_claims, specialist_nli
from utils.llm_client import llm_client


class GPUFTFactCheckerAgent(BaseAgent):
    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="GPUFTFactCheckerAgent", max_internal_retry=max_internal_retry)

    def _review(self, pairs):
        return specialist_nli(pairs, llm_client, finetuned=False)

    def run(self, state: AgentState) -> dict[str, Any]:
        result = review_claims(state, self._review if self.settings.REMOTE_INFERENCE_URL else None)
        result["verifier_model_family"] = "critic_lora"
        result.setdefault("reasoning_trace", []).append(
            self.log_action("gpu_ft_fact_check_completed", {"label": result["fact_check_label"],
                            "verified_count": len(result["verified_claims"]), "rejected_count": len(result["rejected_claims"])}))
        return result
