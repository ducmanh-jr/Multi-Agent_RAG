"""
Kaggle FN Mode - Agentic Summarizer Agent (L3 Multi-LoRA Specialist).
Implements the full Generator ('generator') -> Critic ('critic') -> Action self-correction loop
trained in training_lab/agent3_summarizer/Code/node.py:
  - Uses exact TRAINING_LAB_GENERATOR_SYSTEM_PROMPT and TRAINING_LAB_CRITIC_SYSTEM_PROMPT.
  - Calls /api/summarize/generator (adapter='generator') and /api/summarize/critic (adapter='critic').
  - Executes up to max_internal_retry=2 internal rewrite loops when Critic returns action='rewrite'.
  - Propagates critic_action='request_more_context' to LangGraph Orchestrator when context is insufficient.
"""

from __future__ import annotations

from typing import Any

from agents.agent3_summarizer.components.generator import (
    generate_summary_and_claims,
    pack_context_by_token_budget,
)
from core.base_agent import BaseAgent
from core.schemas import SummarizerCriticOutput, SummarizerGeneratorOutput
from core.state import AgentState, EvidenceSpan
from core.verifier import (
    TRAINING_LAB_CRITIC_SYSTEM_PROMPT,
    TRAINING_LAB_GENERATOR_SYSTEM_PROMPT,
    ensure_inline_citations,
    find_best_supporting_unit,
    split_answer_into_atomic_claims,
)
from utils.llm_client import llm_client


class KaggleFNSummarizerAgent(BaseAgent):
    """Summarizer Agent executing the fine-tuned Generator + Critic QLoRA loop on Kaggle GPU."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="KaggleFNSummarizerAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        source_units = state.get("source_units", []) or []

        self.logger.info(
            f"[Kaggle FN Summarizer] Running Multi-LoRA Generator -> Critic loop for: '{query[:65]}' "
            f"({len(source_units)} units)"
        )

        context_str, _ = pack_context_by_token_budget(source_units, max_tokens=3800)

        current_answer = ""
        current_spans_raw: list[Any] = []
        critic_out = SummarizerCriticOutput()
        feedback_prompt_addition = ""
        internal_attempts = 0

        for attempt in range(self.max_internal_retry):
            internal_attempts = attempt + 1
            gen_user_prompt = f'Context:\n"""\n{context_str}\n"""\n\nCâu hỏi: {query}'
            if feedback_prompt_addition:
                gen_user_prompt += f"\n\n{feedback_prompt_addition}"

            try:
                gen_dict = llm_client.call_kaggle_specialist(
                    endpoint="api/summarize/generator",
                    payload={
                        "context": context_str,
                        "question": query if not feedback_prompt_addition else f"{query}\n{feedback_prompt_addition}",
                        "adapter": "generator",
                    },
                    fallback_prompt=gen_user_prompt,
                    fallback_system=TRAINING_LAB_GENERATOR_SYSTEM_PROMPT,
                    adapter="generator",
                )
                gen_parsed = SummarizerGeneratorOutput.from_dict(gen_dict)
                if gen_parsed.answer:
                    current_answer = gen_parsed.answer
                    current_spans_raw = gen_parsed.evidence_spans

                # Run Fine-Tuned Critic ('critic' adapter)
                critic_user_prompt = (
                    f'Context:\n"""\n{context_str}\n"""\n\n'
                    f"Question: {query}\n\n"
                    f"Generated Summary:\n{current_answer}"
                )
                critic_dict = llm_client.call_kaggle_specialist(
                    endpoint="api/summarize/critic",
                    payload={
                        "context": context_str,
                        "question": query,
                        "generated_summary": current_answer,
                        "adapter": "critic",
                    },
                    fallback_prompt=critic_user_prompt,
                    fallback_system=TRAINING_LAB_CRITIC_SYSTEM_PROMPT,
                    adapter="critic",
                )
                critic_out = SummarizerCriticOutput.from_dict(critic_dict)
            except Exception as e:
                self.logger.warning(f"[Kaggle FN Summarizer] Remote tunnel offline ({e}), using local grounded synthesis.")
                break

            self.logger.info(
                f"[Kaggle FN Critic] Pass {internal_attempts}/{self.max_internal_retry}: "
                f"is_supported={critic_out.is_supported}, action={critic_out.action}, conf={critic_out.confidence}"
            )

            if critic_out.is_supported or critic_out.action == "approve":
                break
            if critic_out.action in ("request_more_context", "lower_confidence"):
                break
            if critic_out.action == "rewrite" and attempt + 1 < self.max_internal_retry:
                feedback_prompt_addition = (
                    "[YÊU CẦU SỬA LỖI TỪ CRITIC AGENT]:\n"
                    f"- Loại bỏ ngay các thông tin thiếu căn cứ: {critic_out.unsupported_claims}\n"
                    f"- Bổ sung các ý chính bị bỏ sót từ Context: {critic_out.missing_key_points}"
                )

        if not current_answer:
            current_answer, fb_spans, _ = generate_summary_and_claims(query, source_units)
            current_spans_raw = fb_spans

        # Build structured EvidenceSpans & AtomicClaims
        evidence_spans: list[dict[str, Any]] = []
        for idx, sp in enumerate(current_spans_raw[:6]):
            quote_str = sp.get("quote", "") if isinstance(sp, dict) else str(sp)
            if not quote_str.strip():
                continue
            best_u = find_best_supporting_unit(quote_str, source_units)
            evidence_spans.append(
                EvidenceSpan(
                    id=f"ev_{idx + 1}",
                    quote=quote_str,
                    unit_id=best_u.get("id", "u1") if best_u else "u1",
                    doc_id=best_u.get("doc_id", "doc") if best_u else "doc",
                    page_no=best_u.get("page_no", 1) if best_u else 1,
                ).to_dict()
            )

        if not evidence_spans and source_units:
            for idx, u in enumerate(source_units[:3]):
                evidence_spans.append(
                    EvidenceSpan(
                        id=f"ev_{idx + 1}",
                        quote=str(u.get("text", ""))[:220],
                        unit_id=u.get("id", f"u{idx + 1}"),
                        doc_id=u.get("doc_id", "doc"),
                        page_no=u.get("page_no", 1),
                    ).to_dict()
                )

        claims = split_answer_into_atomic_claims(current_answer, source_units)
        current_answer = ensure_inline_citations(current_answer, source_units, evidence_spans)

        trace = self.log_action(
            "kaggle_fn_summarization_completed",
            {
                "generator_adapter": "generator",
                "critic_adapter": "critic",
                "internal_attempts": internal_attempts,
                "critic_action": critic_out.action,
                "critic_confidence": critic_out.confidence,
                "claims_count": len(claims),
            },
        )

        return {
            "answer": current_answer,
            "claims": claims,
            "evidence_spans": evidence_spans,
            "critic_action": critic_out.action,
            "critic_feedback": critic_out.to_dict(),
            "generator_model_family": "qwen2.5_7b_qlora",
            "reasoning_trace": [trace],
        }
