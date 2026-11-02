"""
CPU Max Mode - Agentic Summarizer Agent (L3).
Two-Pass Generator + Inner Critic Self-Correction on AMD Radeon 780M Vulkan (qwen2.5:7b-instruct / qwen3:8b):
  1. Generator Pass: Synthesizes citation-grounded answer [1], [2] and extracts atomic claims + evidence spans.
  2. Inner Critic Pass: Audits hallucination & entity coverage; triggers an internal rewrite if needed
     or sets critic_action='request_more_context' to re-trigger Retrieval.
"""

from __future__ import annotations

from typing import Any

from agents.agent3_summarizer.components.critic import critique_summary_and_claims
from agents.agent3_summarizer.components.generator import generate_summary_and_claims
from core.base_agent import BaseAgent
from core.schemas import SummarizerCriticOutput
from core.state import AgentState
from core.verifier import (
    TRAINING_LAB_CRITIC_SYSTEM_PROMPT,
    ensure_inline_citations,
    evaluate_inner_critic_fast,
)
from utils.llm_client import llm_client
from utils.local_vulkan_setup import probe_ollama_models


class CPUMaxSummarizerAgent(BaseAgent):
    """Summarizer Agent for CPU Max: Dual-pass Generator + Inner Critic with Radeon 780M Vulkan support."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="CPUMaxSummarizerAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        source_units = state.get("source_units", []) or []
        required_entities = state.get("required_entities", []) or []

        self.logger.info(f"[CPU Max Summarizer] Generator-Critic loop for '{query[:65]}' ({len(source_units)} units).")

        # 1. Generator Pass (calls local Ollama 7B Vulkan if running, else structured context synthesizer)
        raw_answer, evidence_spans, raw_claims = generate_summary_and_claims(query, source_units)

        # 2. Inner Critic Evaluation
        ollama_info = probe_ollama_models()
        critic_output = evaluate_inner_critic_fast(
            query=query,
            draft_answer=raw_answer,
            source_units=source_units,
            required_entities=required_entities,
        )

        # If Ollama 7B is online and fast critic flagged rewrite, run an LLM Critic + Rewrite pass
        if ollama_info["online"] and not critic_output.is_supported and critic_output.action == "rewrite":
            context_str = "\n\n".join(
                f"[{idx + 1}] (ID: {u.get('id')}, Trang {u.get('page_no', 1)}): {u.get('text', '')}"
                for idx, u in enumerate(source_units[:8])
            )
            critic_prompt = (
                f"Context:\n\"\"\"\n{context_str}\n\"\"\"\n\n"
                f"Question: {query}\n\n"
                f"Generated Summary:\n{raw_answer}"
            )
            try:
                critic_json = llm_client.generate_json(
                    prompt=critic_prompt,
                    system_instruction=TRAINING_LAB_CRITIC_SYSTEM_PROMPT,
                    temperature=0.1,
                    agent="summarizer",
                )
                if isinstance(critic_json, dict) and "action" in critic_json:
                    critic_output = SummarizerCriticOutput.from_dict(critic_json)
            except Exception as exc:
                self.logger.info(f"[CPU Max Critic] Fallback to deterministic critic: {exc}")

            if critic_output.action == "rewrite" and (
                critic_output.unsupported_claims or critic_output.missing_key_points
            ):
                rewrite_query = (
                    f"{query}\n[Yêu cầu sửa lỗi từ Critic: Loại bỏ ý thiếu căn cứ {critic_output.unsupported_claims[:2]}; "
                    f"Bổ sung ý còn thiếu {critic_output.missing_key_points[:2]}]"
                )
                raw_answer, evidence_spans, raw_claims = generate_summary_and_claims(rewrite_query, source_units)

        # 3. Claim-level filtering & inline citation injection
        refined_answer, filtered_claims, warnings = critique_summary_and_claims(
            raw_answer, raw_claims, evidence_spans
        )
        refined_answer = ensure_inline_citations(refined_answer, source_units, evidence_spans)

        trace = self.log_action(
            "cpu_max_summarization_completed",
            {
                "ollama_vulkan_active": ollama_info["online"],
                "model_used": ollama_info["cpu_max_model"] if ollama_info["online"] else "extractive_synthesizer",
                "extracted_claims_count": len(raw_claims),
                "approved_claims_count": len(filtered_claims),
                "evidence_spans_count": len(evidence_spans),
                "critic_action": critic_output.action,
                "warnings": warnings,
            },
        )

        return {
            "answer": refined_answer,
            "evidence_spans": evidence_spans,
            "claims": filtered_claims,
            "critic_action": critic_output.action,
            "critic_feedback": critic_output.to_dict(),
            "generator_model_family": "qwen_local_vulkan" if ollama_info["online"] else "cpu_max_extractive",
            "reasoning_trace": [trace],
        }
