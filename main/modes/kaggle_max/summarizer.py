"""
Kaggle Max Mode - Agentic 32B Reasoning Summarizer Agent (L3-L4 on Dual Tesla T4 TP=2).
Deploys Qwen/Qwen3-32B-AWQ (or QwQ-32B-AWQ) with /think Chain-of-Thought reasoning:
  - Streams <think>...</think> reasoning steps into thinking_trace & reasoning_trace for UI display.
  - Synthesizes deeply reasoned answers with [1], [2] citations and self-critique.
"""

from __future__ import annotations

import re
from typing import Any

from agents.agent3_summarizer.components.generator import (
    generate_summary_and_claims,
    pack_context_by_token_budget,
)
from core.base_agent import BaseAgent
from core.schemas import SummarizerCriticOutput
from core.state import AgentState, AtomicClaim, EvidenceSpan
from core.verifier import (
    ensure_inline_citations,
    evaluate_inner_critic_fast,
    find_best_supporting_unit,
    split_answer_into_atomic_claims,
)
from utils.llm_client import llm_client
from utils.domain_guard import check_out_of_domain, build_out_of_domain_response
from modes.cpu_min.summarizer import CPUMinSummarizerAgent


class KaggleMaxSummarizerAgent(BaseAgent):
    """Summarizer Agent powered by Kaggle 32B AWQ Reasoning Model (Qwen3-32B-AWQ / QwQ-32B-AWQ) on Dual T4."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="KaggleMaxSummarizerAgent", max_internal_retry=max_internal_retry)
        self.cpu_engine = CPUMinSummarizerAgent()

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query") or state.get("question", "")
        source_units = state.get("source_units", []) or []
        required_entities = state.get("required_entities", []) or []

        clean_q = query.strip()
        is_ood, ood_reason = check_out_of_domain(clean_q, source_units)
        if is_ood:
            self.logger.info(f"[Kaggle Max Summarizer] Zero-Hallucination Guard triggered: {ood_reason}")
            res = build_out_of_domain_response(clean_q, model_family="qwen3_32b_awq")
            res["thinking_trace"] = f"<think>Phát hiện câu hỏi nằm ngoài phạm vi tài liệu ({ood_reason}). Kích hoạt phản hồi Zero-Hallucination Honest Refusal.</think>"
            trace = self.log_action("kaggle_max_summarization_completed", {"status": "out_of_domain", "reason": ood_reason})
            res["reasoning_trace"] = [trace]
            return res

        self.logger.info(f"[Kaggle Max Summarizer] 32B /think CoT reasoning on Dual T4 for: '{query[:65]}'")

        context_str, _ = pack_context_by_token_budget(source_units, max_tokens=6000)

        prompt = f"""Bạn là Đại Mô Hình Suy Luận Nghiên Cứu Khoa Học 32B (Qwen3-32B-AWQ / QwQ-32B-AWQ chạy Tensor Parallelism TP=2 trên Kaggle Dual Tesla T4).
Hãy suy luận từng bước (Chain-of-Thought ngắn gọn dưới 600 từ trong thẻ <think>...</think>), đối chiếu chéo các đoạn tài liệu và số liệu, sau đó xuất kết quả JSON chuẩn xác:

Tài liệu nguồn:
{context_str}

Câu hỏi nghiên cứu: "{query}"

YÊU CẦU TRẢ VỀ ĐÚNG ĐỊNH DẠNG JSON:
{{
  "answer": "Câu trả lời học thuật chuyên sâu, có gắn trích dẫn [1], [2] tương ứng tài liệu nguồn...",
  "evidence_spans": [
    "Câu trích dẫn nguyên văn từ tài liệu..."
  ],
  "claims": [
    {{"text": "Mệnh đề nguyên tử độc lập", "unit_id": "Unit ID"}}
  ],
  "is_supported": true,
  "missing_key_points": [],
  "critic_action": "approve"
}}"""

        answer = ""
        thinking_trace = ""
        raw_claims: list[Any] = []
        raw_spans: list[Any] = []
        critic_out = SummarizerCriticOutput()

        if self.settings.REMOTE_INFERENCE_URL:
            try:
                res_32b = llm_client.call_kaggle_max_32b(
                    prompt=prompt,
                    enable_thinking=True,  # Deep /think CoT mode for 32B synthesis!
                    temperature=0.2,
                    max_tokens=2560,
                    json_mode=True,
                )
                thinking_trace = str(res_32b.get("thinking") or "")
                parsed = res_32b.get("parsed") or {}
                if isinstance(parsed, dict) and parsed.get("answer"):
                    answer = str(parsed.get("answer", "")).strip()
                    raw_spans = parsed.get("evidence_spans", []) or []
                    raw_claims = parsed.get("claims", []) or []
                    critic_out = SummarizerCriticOutput.from_dict(
                        {
                            "is_supported": parsed.get("is_supported", True),
                            "missing_key_points": parsed.get("missing_key_points", []),
                            "action": parsed.get("critic_action", "approve"),
                        }
                    )
                elif res_32b.get("text"):
                    answer = str(res_32b["text"]).strip()
            except Exception as e:
                self.logger.warning(f"[Kaggle Max Summarizer] Remote 32B GPU error: {e}")

        if not answer or len(answer.strip()) < 50 or "chế độ dự phòng" in answer:
            self.logger.warning("[Kaggle Max Summarizer] Remote 32B GPU offline/timeout, falling back to High-Precision CPU Engine.")
            res = self.cpu_engine.run(state)
            res["generator_model_family"] = "kaggle_max_cpu_fallback"
            return res

        evidence_spans: list[dict[str, Any]] = []
        for idx, sp in enumerate(raw_spans[:6]):
            q_str = sp.get("quote", "") if isinstance(sp, dict) else str(sp)
            if not q_str.strip():
                continue
            best_u = find_best_supporting_unit(q_str, source_units)
            evidence_spans.append(
                EvidenceSpan(
                    id=f"ev_{idx + 1}",
                    quote=q_str,
                    unit_id=best_u.get("id", "u1") if best_u else "u1",
                    doc_id=best_u.get("doc_id", "doc") if best_u else "doc",
                    page_no=best_u.get("page_no", 1) if best_u else 1,
                ).to_dict()
            )

        claims = split_answer_into_atomic_claims(answer, source_units, existing_claims=raw_claims)
        answer = ensure_inline_citations(answer, source_units, evidence_spans)

        trace = self.log_action(
            "kaggle_max_summarization_completed",
            {
                "model": "Qwen3-32B-AWQ (TP=2 Dual T4)",
                "thinking_tokens_chars": len(thinking_trace),
                "thinking_preview": thinking_trace[:400] if thinking_trace else "",
                "critic_action": critic_out.action,
                "claims_count": len(claims),
            },
        )

        return {
            "answer": answer,
            "claims": claims,
            "evidence_spans": evidence_spans,
            "thinking_trace": thinking_trace,
            "critic_action": critic_out.action,
            "critic_feedback": critic_out.to_dict(),
            "generator_model_family": "qwen3_32b_awq",
            "reasoning_trace": [trace],
        }
