"""
CPU Max Mode - Agentic Summarizer Agent (L3).
Maximal Performance Dual-Path Architecture:
  1. High-Performance Local Vulkan Path (when Ollama 7B/8B is online):
     - Generator Pass: Synthesizes complete, comprehensive citation-grounded answer [1], [2] via Radeon 780M Vulkan.
     - Inner Critic Pass: Evaluates hallucination and triggers self-correction rewrite if needed.
  2. High-Precision Pure-CPU Engine (when Ollama is offline):
     - Zero-Latency deterministic academic synthesis (Zero Hallucination Guard).
     - Complete, multi-aspect contextual passage synthesis (no brief quick-answer stub).
     - Exact date/year extraction, colon/semicolon list decomposition, intact code/config blocks.
     - Distributed page-sampling for single-doc summaries & multi-document synthesis.
     - Unaccented Vietnamese query support with diacritic normalization.
"""

from __future__ import annotations

import os
import re
from typing import Any

from core.base_agent import BaseAgent
from core.state import AgentState
from modes.cpu_min.summarizer import CPUMinSummarizerAgent
from utils.local_vulkan_setup import probe_ollama_models
from utils.domain_guard import check_out_of_domain, build_out_of_domain_response


class CPUMaxSummarizerAgent(BaseAgent):
    """Summarizer Agent for CPU Max: Dual-path Vulkan LLM + High-Precision Comprehensive CPU Engine."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="CPUMaxSummarizerAgent", max_internal_retry=max_internal_retry)
        self.cpu_engine = CPUMinSummarizerAgent()

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        source_units = state.get("source_units", []) or []
        required_entities = state.get("required_entities", []) or []

        self.logger.info(f"[CPU Max Summarizer] Synthesizing comprehensive answer for '{query[:65]}' ({len(source_units)} units).")

        clean_q = (query or "").strip()
        is_ood, ood_reason = check_out_of_domain(clean_q, source_units)
        if is_ood:
            self.logger.info(f"[CPU Max Summarizer] Zero-Hallucination Guard triggered: {ood_reason}")
            res = build_out_of_domain_response(clean_q, model_family="cpu_max_academic_engine")
            trace = self.log_action("cpu_max_summarization_completed", {"status": "out_of_domain", "reason": ood_reason})
            res["reasoning_trace"] = [trace]
            return res

        # Probe Ollama with adequate timeout for local server
        from agents.agent3_summarizer.components.generator import pack_context_with_coverage
        _, _, context_coverage = pack_context_with_coverage(source_units, max_tokens=3800)
        ollama_info = probe_ollama_models(timeout=2.0)

        # Path A: Local Ollama 7B Vulkan LLM is online (with fail-closed Critic contract)
        if ollama_info.get("online"):
            try:
                from agents.agent3_summarizer.components.critic import (
                    build_verification_failed_state,
                    evaluate_critic_contract,
                )
                from agents.agent3_summarizer.components.generator import generate_summary_and_claims
                from core.verifier import ensure_inline_citations

                raw_answer, evidence_spans, raw_claims = generate_summary_and_claims(query, source_units)
                if raw_answer:
                    raw_answer = re.sub(r"^>\s*\*\*Câu trả lời trực tiếp:\*\*\s*", "", raw_answer.strip(), flags=re.IGNORECASE).strip()

                # If generator returned fallback stub, overly short answer (< 200 chars), or empty claims, use comprehensive CPU engine
                if not raw_answer or len(raw_answer.strip()) < 200 or "chế độ dự phòng" in raw_answer or not raw_claims:
                    self.logger.warning(
                        "[CPU Max Summarizer] Local LLM produced brief/stub answer, seamlessly engaging CPU High-Precision Comprehensive engine."
                    )
                    res = self.cpu_engine.run(state)
                    res["generator_model_family"] = "cpu_max_academic_engine"
                    res.setdefault("context_coverage", context_coverage)
                    return res

                critic_res = evaluate_critic_contract(raw_answer, raw_claims, evidence_spans)
                if critic_res["processing_status"] == "VERIFICATION_FAILED":
                    fail_trace = self.log_action(
                        "cpu_max_critic_verification_failed",
                        {"error": critic_res["error_message"]},
                    )
                    return build_verification_failed_state(
                        answer=raw_answer,
                        reason=critic_res["error_message"] or "Critic verification failed",
                        context_coverage=context_coverage,
                        model_family="qwen_local_vulkan",
                        trace=fail_trace,
                    )

                refined_answer = ensure_inline_citations(critic_res["answer"], source_units, evidence_spans)
                refined_answer = re.sub(r"^>\s*\*\*Câu trả lời trực tiếp:\*\*\s*", "", refined_answer.strip(), flags=re.IGNORECASE).strip()

                trace = self.log_action(
                    "cpu_max_summarization_completed",
                    {
                        "ollama_vulkan_active": True,
                        "model_used": ollama_info.get("cpu_max_model", "qwen2.5:7b-instruct"),
                        "claims_count": len(critic_res["claims"]),
                        "evidence_spans_count": len(evidence_spans),
                        "critic_action": critic_res["critic_action"],
                    },
                )
                return {
                    "answer": refined_answer,
                    "evidence_spans": evidence_spans,
                    "claims": critic_res["claims"],
                    "critic_action": critic_res["critic_action"],
                    "critic_feedback": critic_res["critic_feedback"],
                    "processing_status": "SUCCESS",
                    "confidence": critic_res["confidence"],
                    "context_coverage": context_coverage,
                    "generator_model_family": "qwen_local_vulkan",
                    "reasoning_trace": [trace],
                }
            except Exception as e:
                self.logger.warning(f"[CPU Max Summarizer] Local LLM generator failed ({e}), falling back to CPU engine.")

        # Path B: Pure-CPU High-Precision Comprehensive Engine (Fast, 0s latency, zero hallucinations)
        res = self.cpu_engine.run(state)
        res["generator_model_family"] = "cpu_max_academic_engine"
        res.setdefault("context_coverage", context_coverage)
        return res
