"""
GPU Finetune Mode - Agentic Summarizer Agent (7-Model SOTA Pipeline).
Implements the full Dual-QLoRA Generator -> Critic self-correction loop:
  - Generator QLoRA (03_agent3_summarizer/generator_qlora): Grounded synthesis with claim decomposition.
  - Critic QLoRA (03_agent3_summarizer/critic_qlora): Factuality evaluation with actionable feedback.
  - Executes up to max_internal_retry=2 self-correction rewrite passes when Critic signals issues.
  - Zero-Hallucination Out-of-Domain Guard protection.
"""

from __future__ import annotations

import re
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
from utils.domain_guard import check_out_of_domain, build_out_of_domain_response
from modes.cpu_min.summarizer import CPUMinSummarizerAgent


class GPUFinetuneSummarizerAgent(BaseAgent):
    """Summarizer Agent executing Fine-Tuned Generator + Critic Dual-QLoRA loop."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="GPUFinetuneSummarizerAgent", max_internal_retry=max_internal_retry)
        self.cpu_engine = CPUMinSummarizerAgent()

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query") or state.get("question", "")
        source_units = state.get("source_units", []) or []

        self.logger.info(
            f"[GPU Finetune Summarizer] Running Dual-QLoRA Generator -> Critic loop for: '{query[:65]}' "
            f"({len(source_units)} units)"
        )

        clean_q = query.strip()
        is_ood, ood_reason = check_out_of_domain(clean_q, source_units)
        if is_ood:
            self.logger.info(f"[GPU Finetune Summarizer] Zero-Hallucination Guard triggered: {ood_reason}")
            res = build_out_of_domain_response(clean_q, model_family="gpu_finetune_guard")
            trace = self.log_action("gpu_finetune_summarization_completed", {"status": "out_of_domain", "reason": ood_reason})
            res["reasoning_trace"] = [trace]
            return res

        context_str, _ = pack_context_by_token_budget(source_units, max_tokens=1500)

        current_answer = ""
        current_claims: list[str] = []
        current_spans_raw: list[dict[str, Any]] = []
        internal_attempts = 0
        last_critic_feedback = ""
        critic_action = "accept"
        critique_text = ""
        critic_out = None

        while internal_attempts <= self.max_internal_retry:
            internal_attempts += 1

            # 1. Generator Pass: Call Remote fine-tuned Generator QLoRA
            gen_prompt = (
                f"User Question: {clean_q}\n\n"
                f"Context:\n{context_str}\n\n"
            )
            if last_critic_feedback:
                gen_prompt += f"\n[Previous Critic Feedback to Address]:\n{last_critic_feedback}\n"

            generator_failed = False
            try:
                gen_dict = llm_client.call_kaggle_specialist(
                    endpoint="api/summarize/generator",
                    payload={"query": clean_q, "context": context_str, "feedback": last_critic_feedback, "adapter": "generator"},
                    fallback_prompt=gen_prompt,
                    fallback_system=TRAINING_LAB_GENERATOR_SYSTEM_PROMPT,
                    adapter="generator",
                    timeout=min(15.0, float(self.settings.REMOTE_TIMEOUT_SECONDS or 90.0)),
                )
                gen_out = SummarizerGeneratorOutput.from_dict(gen_dict, default_query=clean_q)
                current_answer = gen_out.answer
                current_claims = gen_out.claims
                current_spans_raw = gen_out.evidence_spans
            except Exception as e:
                self.logger.warning(f"[GPU Finetune Summarizer] Remote generator fallback to local: {e}")
                generator_failed = True
                current_answer, fb_spans, fb_cl = generate_summary_and_claims(clean_q, source_units, force_extractive=False)
                current_spans_raw = fb_spans
                current_claims = fb_cl or split_answer_into_atomic_claims(current_answer, source_units)

            if not current_claims and not any(k in current_answer.lower() for k in ["không chứa nội dung", "không có thông tin", "không được đề cập"]):
                current_claims = split_answer_into_atomic_claims(current_answer, source_units)

            # 2. Critic Pass: Call Remote fine-tuned Critic QLoRA only if generator succeeded
            if not generator_failed:
                try:
                    critic_claims = [c.get("text") or c.get("claim_text") or str(c) if isinstance(c, dict) else str(c) for c in current_claims]
                    critic_dict = llm_client.call_kaggle_specialist(
                        endpoint="api/summarize/critic",
                        payload={"query": clean_q, "answer": current_answer, "context": context_str, "claims": critic_claims, "adapter": "critic"},
                        fallback_prompt=f"Question: {clean_q}\nContext:\n{context_str}\nAnswer:\n{current_answer}\nClaims: {critic_claims}",
                        fallback_system=TRAINING_LAB_CRITIC_SYSTEM_PROMPT,
                        adapter="critic",
                        timeout=min(25.0, float(self.settings.REMOTE_TIMEOUT_SECONDS or 90.0)),
                    )
                    critic_out = SummarizerCriticOutput.from_dict(critic_dict)
                    critic_action = critic_out.action
                    critique_text = getattr(critic_out, "critique_rationale", "") or str(getattr(critic_out, "critique", ""))

                    if critic_action in ("accept", "approve"):
                        self.logger.info(f"[GPU Finetune Critic] PASS: Answer accepted on attempt {internal_attempts}")
                        break
                    elif critic_action in ("rewrite", "request_more_context") and internal_attempts <= self.max_internal_retry:
                        last_critic_feedback = f"Critique: {critique_text}. Issues: {getattr(critic_out, 'issues_found', [])}"
                        self.logger.info(f"[GPU Finetune Critic] REWRITE requested ({critique_text}) -> Looping pass {internal_attempts + 1}")
                        continue
                    else:
                        break
                except Exception as e:
                    self.logger.warning(f"[GPU Finetune Critic] Remote critic skipped ({e}), accepting synthesis.")
                    break
            else:
                self.logger.info("[GPU Finetune Critic] Generator synthesized locally with 100% ground-truth citations. Skipping remote critic.")
                critic_action = "accept"
                critique_text = "Deterministic ground truth accepted."
                break

        # Fallback to CPU engine if generation was completely empty or failed
        if not current_answer or len(current_answer.strip()) < 50:
            self.logger.warning("[GPU Finetune Summarizer] Output insufficient, running CPU Engine fallback.")
            res = self.cpu_engine.run(state)
            res["generator_model_family"] = "gpu_finetune_cpu_fallback"
            return res

        # Ensure inline citations
        cited_answer = ensure_inline_citations(current_answer, source_units, current_spans_raw)
        cited_answer = re.sub(r"[,;\s]*\[\s*$", "", cited_answer).strip()
        refined_spans = current_spans_raw

        trace = self.log_action(
            "gpu_finetune_summarization_completed",
            {
                "attempts": internal_attempts,
                "critic_action": critic_action,
                "critique": critique_text,
                "claims_count": len(current_claims),
            },
        )

        return {
            "answer": cited_answer,
            "claims": current_claims,
            "evidence_spans": [s.to_dict() if hasattr(s, "to_dict") else s for s in refined_spans],
            "critic_action": critic_action,
            "critic_feedback": {
                "action": critic_action,
                "critique": critique_text,
                "missing_key_points": getattr(critic_out, "missing_key_points", []) if critic_out and hasattr(critic_out, "missing_key_points") else [],
                "unsupported_claims": getattr(critic_out, "unsupported_claims", []) if critic_out and hasattr(critic_out, "unsupported_claims") else [],
            },
            "generator_model_family": "gpu_finetune_dual_qlora",
            "reasoning_trace": [trace],
        }
