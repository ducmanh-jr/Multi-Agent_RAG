"""GPU Max batch NLI verifier for the gpu_max execution mode."""

from typing import Any

from core.base_agent import BaseAgent
from core.state import AgentState
from core.verifier import nli_prompt, review_claims
from utils.llm_client import llm_client


class GPUMaxFactCheckerAgent(BaseAgent):
    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="GPUMaxFactCheckerAgent", max_internal_retry=max_internal_retry)

    def _review(self, pairs):
        response = llm_client.call_gpu_max_32b(
            prompt=nli_prompt(pairs), enable_thinking=False, temperature=0.1, max_tokens=2048, json_mode=True)
        data = response.get("parsed") if isinstance(response, dict) else None
        if isinstance(data, dict):
            res = data.get("results")
            return res if isinstance(res, list) else []
        return data if isinstance(data, list) else []

    def run(self, state: AgentState) -> dict[str, Any]:
        result = review_claims(state, self._review if self.settings.REMOTE_INFERENCE_URL else None)
        result["verifier_model_family"] = "qwen3_32b_awq"
        result.setdefault("reasoning_trace", []).append(
            self.log_action("gpu_max_fact_check_completed", {"label": result["fact_check_label"],
                            "verified_count": len(result["verified_claims"]), "rejected_count": len(result["rejected_claims"])}))
        return result
