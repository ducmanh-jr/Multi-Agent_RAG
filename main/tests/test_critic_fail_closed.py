"""
Automated End-to-End Fail-Closed Critic Verification Suite (Đợt 4 — Criterion A).

Verifies:
  1. Critic exception -> processing_status='VERIFICATION_FAILED', error_stage='summarizer_critic',
     critic_action='verification_failed', confidence='low', empty approved claims (never fail-open).
  2. Critic malformed/invalid JSON schema -> fail-closed identically.
  3. Real LangGraph execution: routes summarizer_node -> escalation_node (bypasses fact_checker
     and never calls writer as verified), outputting an explicit unverified warning report.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agents.agent3_summarizer.components.critic import (
    critique_summary_and_claims,
    evaluate_critic_contract,
)
from agents.agent3_summarizer.node import SummarizerAgent
from core.graph import build_research_rag_graph


def test_critic_exception_fails_closed_never_approves_claims():
    claims = [{"id": "claim_1", "text": "Model achieves 95% accuracy."}]
    spans = [{"id": "ev_1", "quote": "Model achieves 95% accuracy.", "doc_id": "d1", "page_no": 1}]

    with patch(
        "agents.agent3_summarizer.components.critic.llm_client.generate_json",
        side_effect=RuntimeError("LLM connection timeout"),
    ):
        res = evaluate_critic_contract("Draft answer", claims, spans)
        assert res["processing_status"] == "VERIFICATION_FAILED"
        assert res["error_stage"] == "summarizer_critic"
        assert res["critic_action"] == "verification_failed"
        assert res["confidence"] == "low"
        assert res["claims"] == []

        # Backward-compatible tuple wrapper must also fail-closed (empty claims)
        _, filtered, warnings = critique_summary_and_claims("Draft answer", claims, spans)
        assert filtered == []
        assert any("VERIFICATION_FAILED" in w for w in warnings)


def test_critic_invalid_schema_fails_closed():
    claims = [{"id": "claim_1", "text": "Transformers use self-attention."}]
    spans = [{"id": "ev_1", "quote": "Transformers use self-attention.", "doc_id": "d1", "page_no": 1}]

    # Missing required list fields or malformed JSON parser error
    for bad_payload in (
        {"error": "Failed to parse JSON response", "raw_text": "not json"},
        {"approved_claim_ids": "claim_1", "critic_warnings": []},  # not a list
        {"unexpected_key": 123},
    ):
        with patch(
            "agents.agent3_summarizer.components.critic.llm_client.generate_json",
            return_value=bad_payload,
        ):
            res = evaluate_critic_contract("Draft answer", claims, spans)
            assert res["processing_status"] == "VERIFICATION_FAILED"
            assert res["error_stage"] == "summarizer_critic"
            assert res["critic_action"] == "verification_failed"
            assert res["confidence"] == "low"
            assert res["claims"] == []


def test_critic_fail_closed_end_to_end_on_real_langgraph():
    retrieval = MagicMock()
    retrieval.run.return_value = {
        "retrieved_docs": [{"id": "d1", "doc_id": "d1", "page_no": 1, "text": "Attention is all you need."}],
        "reasoning_trace": [{"node": "MockRetrieval", "action": "ok"}],
    }
    reader = MagicMock()
    reader.run.return_value = {
        "source_units": [{"id": "u1", "doc_id": "d1", "page_no": 1, "text": "Attention is all you need."}],
        "reasoning_trace": [{"node": "MockReader", "action": "ok"}],
    }
    fact_checker = MagicMock()
    writer = MagicMock()

    real_summarizer = SummarizerAgent()

    graph = build_research_rag_graph(
        mode="cpu_min",
        agent_overrides={
            "retrieval": retrieval,
            "reader": reader,
            "summarizer": real_summarizer,
            "fact_checker": fact_checker,
            "writer": writer,
        },
    )

    with patch(
        "agents.agent3_summarizer.node.generate_summary_and_claims",
        return_value=(
            "Draft unverified answer.",
            [{"id": "ev_1", "quote": "Attention is all you need.", "doc_id": "d1", "page_no": 1}],
            [{"id": "claim_1", "text": "Attention is all you need."}],
        ),
    ), patch(
        "agents.agent3_summarizer.components.critic.llm_client.generate_json",
        side_effect=RuntimeError("Critic backend crashed"),
    ):
        result = graph.invoke(
            {"query": "Tóm tắt nội dung tài liệu PDF này", "pdf_path": "paper.pdf"}
        )

    assert result["processing_status"] == "VERIFICATION_FAILED"
    assert result["error_stage"] == "summarizer_critic"
    assert result["critic_action"] == "verification_failed"
    assert result["confidence"] == "low"
    assert "CHƯA ĐƯỢC XÁC MINH" in result["final_report"]
    # Must NEVER invoke fact_checker or writer as a verified report
    fact_checker.run.assert_not_called()
    writer.run.assert_not_called()
