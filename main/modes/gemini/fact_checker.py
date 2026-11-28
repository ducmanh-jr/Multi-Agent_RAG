"""Gemini batch NLI verifier for the gemini execution mode."""

from typing import Any

from core.base_agent import BaseAgent
from core.state import AgentState
import json
import re

from core.verifier import nli_prompt, review_claims
from utils.llm_client import llm_client


class GeminiFactCheckerAgent(BaseAgent):
    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="GeminiFactCheckerAgent", max_internal_retry=max_internal_retry)

    def _review(self, pairs):
        raw = llm_client._call_gemini(prompt=nli_prompt(pairs), temperature=0.1, max_tokens=2048, json_mode=True)
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(), flags=re.I)
        try:
            data = json.loads(clean)
        except (TypeError, ValueError):
            try:
                from utils.json_repair import extract_and_parse_json

                data = extract_and_parse_json(clean)
            except Exception:
                return []
        if isinstance(data, dict):
            res = data.get("results")
            return res if isinstance(res, list) else []
        return data if isinstance(data, list) else []

    def run(self, state: AgentState) -> dict[str, Any]:
        result = review_claims(state, self._review)
        result["verifier_model_family"] = "gemini_flash"
        result.setdefault("reasoning_trace", []).append(
            self.log_action("gemini_fact_check_completed", {"label": result["fact_check_label"],
                            "verified_count": len(result["verified_claims"]), "rejected_count": len(result["rejected_claims"])}))
        return result
