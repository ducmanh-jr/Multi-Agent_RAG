"""
Gemini Mode - Citation-Grounded Writer Agent (M1 Locked Architecture).
Enforces:
  1. Never feeds raw `source_units` to the Writer LLM.
  2. Orders approved `verified_claims` by relevance (`claim_order` JSON permutation) while
     leaving citation numbering and prose rendering to the deterministic `build_academic_report`
     and `write_with_citation_audit` post-Writer gate.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agents.agent5_writer.components.report_builder import build_academic_report
from core.base_agent import BaseAgent
from core.citation_audit import claim_text
from core.provider_router import is_gemini_permitted
from core.state import AgentState
from utils.llm_client import llm_client

GEMINI_WRITER_ORDER_SYSTEM_PROMPT = (
    "Order all supplied verified findings by relevance to the user's question. "
    "Return ONLY valid JSON with key 'claim_order', a permutation of the supplied integer indices. "
    "Do not add, rewrite, summarize, or drop any finding."
)


class GeminiWriterAgent(BaseAgent):
    """Writer Agent for Gemini Mode: Claim-ordering + deterministic span-level citation compiler."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="GeminiWriterAgent", max_internal_retry=max_internal_retry)

    def _order_verified_claims(
        self,
        *,
        query: str,
        verified_claims: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool]:
        if not self.settings.GEMINI_API_KEY or not is_gemini_permitted("gemini", role="writer"):
            return verified_claims, False

        candidates = [
            {"index": idx, "claim_text": claim_text(claim)}
            for idx, claim in enumerate(verified_claims)
            if isinstance(claim, dict) and claim_text(claim).strip() and (claim.get("evidence_ids") or [])
        ]
        if len(candidates) <= 1:
            return verified_claims, False

        indices = {item["index"] for item in candidates}
        user_prompt = json.dumps({"query": query, "claims": candidates}, ensure_ascii=False)
        try:
            raw = llm_client.generate(
                user_prompt,
                system_instruction=GEMINI_WRITER_ORDER_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=256,
                agent="writer",
            )
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(), flags=re.I)
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if not match:
                return verified_claims, False
            parsed = json.loads(match.group(0))
            order = parsed.get("claim_order") if isinstance(parsed, dict) else None
            if (
                isinstance(order, list)
                and len(order) == len(indices)
                and all(type(i) is int for i in order)
                and set(order) == indices
            ):
                remaining = [c for i, c in enumerate(verified_claims) if i not in indices]
                ordered = [verified_claims[i] for i in order] + remaining
                return ordered, True
        except Exception as exc:
            self.logger.warning(f"[Gemini Writer] Claim ordering skipped ({exc}); keeping verified finding order.")
        return verified_claims, False

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        answer = state.get("answer", "")
        verified_claims = list(state.get("verified_claims", []) or [])
        rejected_claims = state.get("rejected_claims", []) or []
        retrieved_docs = state.get("retrieved_docs", []) or []
        evidence_spans = state.get("evidence_spans", []) or []

        self.logger.info(
            f"[Gemini Writer] Compiling verified academic report for: '{query[:60]}' "
            f"({len(verified_claims)} verified claims)"
        )

        ordered_claims, ordered_with_llm = self._order_verified_claims(
            query=query,
            verified_claims=verified_claims,
        )

        final_report, report_data = build_academic_report(
            query=query,
            verified_claims=ordered_claims,
            rejected_claims=rejected_claims,
            retrieved_docs=retrieved_docs,
            answer=answer,
            answer_status=state.get("answer_status"),
            evidence_spans=evidence_spans,
            statistical_audit=state.get("statistical_audit"),
            code_verification=state.get("code_verification"),
            answer_language=state.get("answer_language"),
        )

        report_data["engine"] = "Gemini_Cloud_Agentic"
        report_data["synthesized_with_llm"] = False
        report_data["ordered_with_llm"] = ordered_with_llm

        trace = self.log_action(
            "gemini_report_written",
            {
                "length": len(final_report),
                "verified_claims_count": len(report_data.get("verified_claims") or []),
                "synthesized_with_llm": False,
                "ordered_with_llm": ordered_with_llm,
            },
        )

        return {
            "final_report": final_report,
            "written_report_data": report_data,
            "reasoning_trace": [trace],
        }
