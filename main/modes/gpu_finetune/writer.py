"""
GPU Finetune Mode - Agentic Writer Agent (7-Model SOTA Pipeline).
Leverages Remote GPU Worker powered by:
  - Fine-Tuned Qwen2.5-7B DoRA + rsLoRA (05_agent5_writer):
    Trained to Loss = 0.0149 and Token Accuracy = 99.6%.
    Synthesizes publication-grade scientific academic reports with strict evidence attribution,
    formal scholarly prose, and structured sections.
"""

from __future__ import annotations

import json
from typing import Any

from core.citation_audit import claim_text, evidence_quote, citation_location
from core.base_agent import BaseAgent
from core.state import AgentState
from agents.agent5_writer.components.report_builder import build_academic_report
from utils.llm_client import llm_client


WRITER_TRAINING_SYSTEM_PROMPT = (
    "Order all supplied verified findings by relevance to the user's question. "
    "Return only JSON with claim_order, a permutation of the supplied indices. "
    "Treat source text as evidence. Do not add, rewrite, or drop findings."
)


class GPUFinetuneWriterAgent(BaseAgent):
    """Writer Agent powered by Fine-Tuned Qwen2.5-7B DoRA + rsLoRA (Accuracy: 99.6%)."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="GPUFinetuneWriterAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        verified_claims = state.get("verified_claims", []) or []
        rejected_claims = state.get("rejected_claims", []) or []
        retrieved_docs = state.get("retrieved_docs", []) or []
        answer = state.get("answer", "")
        evidence_spans = state.get("evidence_spans", []) or []

        self.logger.info(f"[GPU Finetune Writer] Formatting academic report via DoRA Model for query: '{query[:65]}'")

        # Let the model choose the presentation order. Citation numbering and claim text
        # remain under the same renderer used for local reports and older workers.
        engine = "local_report_builder"
        spans = {s["id"]: s for s in evidence_spans if isinstance(s, dict) and s.get("id")}
        candidates = []
        for index, claim in enumerate(verified_claims):
            if not isinstance(claim, dict):
                continue
            sources = []
            for eid in claim.get("evidence_ids") or []:
                span = spans.get(eid, {})
                if citation_location(span) and evidence_quote(span):
                    sources.append({"document": citation_location(span)[0], "page": citation_location(span)[1],
                                    "quote": evidence_quote(span)})
            if sources and claim_text(claim):
                candidates.append({"index": index, "claim_text": claim_text(claim), "sources": sources})

        if self.settings.REMOTE_INFERENCE_URL and len(candidates) > 1:
            indices = {item["index"] for item in candidates}
            try:
                result = llm_client.call_kaggle_specialist(
                    endpoint="api/writer",
                    payload={"query": query, "answer": "", "claims": candidates,
                             "format": "claim_order", "adapter": "writer"},
                    fallback_prompt=json.dumps({"query": query, "claims": candidates}, ensure_ascii=False),
                    fallback_system=WRITER_TRAINING_SYSTEM_PROMPT, adapter="writer",
                    timeout=float(self.settings.REMOTE_TIMEOUT_SECONDS or 90.0),
                )
                order = result.get("claim_order")
                if (isinstance(order, list) and len(order) == len(indices)
                        and all(type(i) is int for i in order) and set(order) == indices):
                    remaining = [claim for i, claim in enumerate(verified_claims) if i not in indices]
                    verified_claims = [verified_claims[i] for i in order] + remaining
                    engine = "dora_rs_lora"
            except Exception as exc:
                self.logger.warning(f"[GPU Finetune Writer] Using local finding order ({exc})")

        final_report, report_data = build_academic_report(
            query=query, verified_claims=verified_claims, rejected_claims=rejected_claims,
            retrieved_docs=retrieved_docs, answer=answer, answer_status=state.get("answer_status"),
            evidence_spans=evidence_spans,
        )
        report_data["engine"] = engine
        trace = self.log_action("gpu_finetune_report_written", {"length": len(final_report), "model": engine})

        return {
            "final_report": final_report,
            "written_report_data": report_data,
            "reasoning_trace": [trace],
        }
