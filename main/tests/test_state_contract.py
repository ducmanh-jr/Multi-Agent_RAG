"""
Automated AgentState & Schema Contract Verification Suite (Đợt 2).

Verifies:
  A. Backward compatibility with legacy AgentState fields, legacy agent payloads,
     and LangGraph StateGraph compilation.
  B. `reasoning_trace` Annotated[..., operator.add] reducer accumulation without overwrite,
     and `SourceUnit.provenance` multi-source preservation + strict validation.
  C. Orthogonality of the 4 independent state groups:
     InputStatus, QueryResolutionStatus, ProcessingStatus, EvidenceStatus (including NOT_APPLICABLE).
"""

from __future__ import annotations

import operator
from typing import Any, get_args, get_type_hints

import pytest
from langgraph.graph import END, START, StateGraph

from core.schemas import (
    ContextCoverage,
    RouterPlanOutput,
    SummarizerCriticOutput,
    VALID_EVIDENCE_STATUSES,
    VALID_INPUT_STATUSES,
    VALID_PRIMARY_INTENTS,
    VALID_PROCESSING_STATUSES,
    VALID_QUERY_RESOLUTION_STATUSES,
    map_evidence_to_legacy_fact_check_label,
)
from core.state import (
    AgentState,
    SourceUnit,
    compute_content_hash,
    create_initial_state,
    normalize_provenance_entry,
    validate_state_contract,
)


def test_four_independent_status_groups_defined_completely():
    assert VALID_INPUT_STATUSES == frozenset({"OK", "NO_DOCUMENT_SELECTED"})
    assert VALID_QUERY_RESOLUTION_STATUSES == frozenset(
        {"CLEAR", "AMBIGUOUS", "CLARIFICATION_REQUIRED"}
    )
    assert VALID_PROCESSING_STATUSES == frozenset(
        {
            "PENDING",
            "RUNNING",
            "SUCCESS",
            "RETRIEVAL_FAILED",
            "EXTRACTION_FAILED",
            "PARSING_FAILED",
            "VERIFICATION_FAILED",
            "LLM_FAILED",
            "TOOL_FAILED",
            "TIMEOUT",
            "CANCELLED",
        }
    )
    assert VALID_EVIDENCE_STATUSES == frozenset(
        {
            "NOT_APPLICABLE",
            "FOUND",
            "PARTIAL",
            "NOT_FOUND",
            "CONFLICTING_EVIDENCE",
        }
    )
    assert "GENERAL_QA" in VALID_PRIMARY_INTENTS
    assert "TASK_ASSISTED_RESPONSE" in VALID_PRIMARY_INTENTS
    assert "TASK_EXECUTION" in VALID_PRIMARY_INTENTS


def test_status_orthogonality_invariants():
    # 1. No document selected is NOT a retrieval failure
    no_doc_state = create_initial_state(
        "Tóm tắt nội dung PDF",
        primary_intent="DOCUMENT_QA",
        input_status="NO_DOCUMENT_SELECTED",
        query_resolution="CLEAR",
        processing_status="SUCCESS",
        evidence_status="NOT_APPLICABLE",
    )
    assert validate_state_contract(no_doc_state) == []
    assert no_doc_state["input_status"] == "NO_DOCUMENT_SELECTED"
    assert no_doc_state["processing_status"] == "SUCCESS"

    # 2. GENERAL_QA uses NOT_APPLICABLE, never forced to NOT_FOUND
    general_state = create_initial_state(
        "Python là gì?",
        primary_intent="GENERAL_QA",
        input_status="OK",
        query_resolution="CLEAR",
        processing_status="SUCCESS",
        evidence_status="NOT_APPLICABLE",
    )
    assert validate_state_contract(general_state) == []
    assert general_state["evidence_status"] == "NOT_APPLICABLE"
    assert map_evidence_to_legacy_fact_check_label(general_state["evidence_status"]) == "support"

    # 3. NOT_FOUND and CONFLICTING_EVIDENCE are distinct
    not_found_state = create_initial_state(
        "Bài báo có nói về Mars không?",
        primary_intent="DOCUMENT_QA",
        processing_status="SUCCESS",
        evidence_status="NOT_FOUND",
    )
    conflict_state = create_initial_state(
        "Độ chính xác là bao nhiêu?",
        primary_intent="DOCUMENT_QA",
        processing_status="SUCCESS",
        evidence_status="CONFLICTING_EVIDENCE",
    )
    assert validate_state_contract(not_found_state) == []
    assert validate_state_contract(conflict_state) == []
    assert map_evidence_to_legacy_fact_check_label("NOT_FOUND") == "not_enough_info"
    assert map_evidence_to_legacy_fact_check_label("CONFLICTING_EVIDENCE") == "contradict"

    # 4. Ambiguous query is NOT a processing failure
    ambiguous_state = create_initial_state(
        "Kết quả thế nào?",
        primary_intent="CLARIFICATION_REQUIRED",
        query_resolution="AMBIGUOUS",
        processing_status="SUCCESS",
        evidence_status="NOT_APPLICABLE",
    )
    assert validate_state_contract(ambiguous_state) == []

    # Invalid statuses must be caught by validator
    bad_state = {
        "primary_intent": "UNKNOWN_INTENT",
        "input_status": "MISSING",
        "query_resolution": "WEIRD",
        "processing_status": "CRASHED",
        "evidence_status": "MAYBE",
    }
    errors = validate_state_contract(bad_state)
    assert len(errors) == 5


def test_agent_state_preserves_all_legacy_fields_and_accepts_legacy_payloads():
    hints = get_type_hints(AgentState, include_extras=True)

    required_legacy_fields = {
        "query",
        "intent",
        "complexity",
        "required_entities",
        "is_numeric_query",
        "is_comparative_query",
        "pdf_path",
        "pdf_paths",
        "perspectives",
        "sub_queries",
        "retrieved_docs",
        "seen_chunks",
        "retrieval_strategy",
        "alpha",
        "top_k",
        "document_schema",
        "source_units",
        "answer",
        "evidence_spans",
        "claims",
        "critic_action",
        "critic_feedback",
        "thinking_trace",
        "generator_model_family",
        "verified_claims",
        "rejected_claims",
        "evidence_gaps",
        "fact_check_label",
        "statistical_audit",
        "code_verification",
        "verifier_model_family",
        "final_report",
        "short_answer",
        "answer_status",
        "written_report_data",
        "retry_count",
        "max_reflection_loops",
        "chat_history",
        "query_with_context",
        "factcheck_calls",
        "confidence",
        "no_new_evidence",
        "reasoning_trace",
    }
    missing = required_legacy_fields - set(hints.keys())
    assert not missing, f"Legacy AgentState fields missing: {missing}"

    # Verify existing agents can run on a legacy state dict without new fields
    from agents.agent2_reader.node import ReaderAgent

    legacy_payload: AgentState = {
        "query": "What is LoRA?",
        "pdf_path": None,
        "retrieved_docs": [
            {"id": "doc_1", "doc_id": "doc_1", "page_no": 2, "text": "LoRA freezes pretrained weights."}
        ],
        "retry_count": 0,
        "reasoning_trace": [],
    }
    reader_out = ReaderAgent().run(legacy_payload)
    assert len(reader_out["source_units"]) == 1
    assert reader_out["source_units"][0]["provenance"] == [
        {"doc_id": "doc_1", "page_no": 2, "unit_id": "doc_1_unit_0", "character_offset": 0}
    ]


def test_reasoning_trace_reducer_accumulates_without_overwrite():
    hints = get_type_hints(AgentState, include_extras=True)
    trace_type = hints["reasoning_trace"]
    meta = getattr(trace_type, "__metadata__", ())
    assert operator.add in meta, "reasoning_trace must use Annotated[..., operator.add]"

    def node_a(state: AgentState) -> dict[str, Any]:
        return {
            "processing_status": "RUNNING",
            "reasoning_trace": [{"node": "node_a", "step": 1}],
        }

    def node_b(state: AgentState) -> dict[str, Any]:
        return {
            "processing_status": "SUCCESS",
            "reasoning_trace": [{"node": "node_b", "step": 2}],
        }

    wf = StateGraph(AgentState)
    wf.add_node("node_a", node_a)
    wf.add_node("node_b", node_b)
    wf.add_edge(START, "node_a")
    wf.add_edge("node_a", "node_b")
    wf.add_edge("node_b", END)
    compiled = wf.compile()

    result = compiled.invoke({"query": "test reducer", "reasoning_trace": [{"node": "init", "step": 0}]})
    assert [t["node"] for t in result["reasoning_trace"]] == ["init", "node_a", "node_b"]
    assert result["processing_status"] == "SUCCESS"


def test_source_unit_multi_source_provenance_and_validation():
    shared_text = "Transformers rely entirely on self-attention mechanisms."
    u1 = SourceUnit(id="u1", text=shared_text, doc_id="doc_1", page_no=4, character_offset=10)
    u2 = SourceUnit(id="u2", text=shared_text, doc_id="doc_2", page_no=7, character_offset=50)

    assert compute_content_hash(u1.text) == compute_content_hash(u2.text)
    assert u1.provenance == [{"doc_id": "doc_1", "page_no": 4, "unit_id": "u1", "character_offset": 10}]

    # Merge preserves both sources and ignores duplicates
    u1.merge_provenance(u2)
    u1.merge_provenance(u2)  # idempotent
    assert len(u1.provenance) == 2
    assert {"doc_id": "doc_1", "page_no": 4, "unit_id": "u1", "character_offset": 10} in u1.provenance
    assert {"doc_id": "doc_2", "page_no": 7, "unit_id": "u2", "character_offset": 50} in u1.provenance

    # to_dict() preserves full provenance list
    d = u1.to_dict()
    assert len(d["provenance"]) == 2

    # Invalid provenance entries must raise ValueError explicitly
    invalid_entries: list[Any] = [
        "not_a_dict",
        {"page_no": 1},  # missing doc_id
        {"doc_id": "   ", "page_no": 1},  # blank doc_id
        {"doc_id": "doc_1"},  # missing page_no
        {"doc_id": "doc_1", "page_no": -1},  # negative page_no
        {"doc_id": "doc_1", "page_no": True},  # bool page_no
        {"doc_id": "doc_1", "page_no": "4"},  # non-int page_no
        {"doc_id": "doc_1", "page_no": 1, "character_offset": -5},  # negative offset
    ]
    for bad in invalid_entries:
        with pytest.raises(ValueError):
            normalize_provenance_entry(bad)


def test_router_plan_and_critic_and_context_coverage_schemas():
    # Legacy RouterPlanOutput dict without new fields still parses cleanly
    legacy_plan = RouterPlanOutput.from_dict(
        {"intent": "compare", "complexity": "HIGH", "sub_queries": ["q1", "q2"]},
        default_query="q1",
    )
    assert legacy_plan.intent == "compare"
    assert legacy_plan.primary_intent == "DOCUMENT_QA"
    assert legacy_plan.query_resolution == "CLEAR"
    assert legacy_plan.input_status == "OK"

    # Critic schema accepts new fail-closed 'verification_failed' action
    critic = SummarizerCriticOutput.from_dict(
        {"is_supported": False, "action": "verification_failed", "confidence": 0.0}
    )
    assert critic.action == "verification_failed"

    # ContextCoverage tracks included vs omitted units accurately
    cov = ContextCoverage.from_dict(
        {"total_units": 20, "included_units": 8, "omitted_unit_ids": ["u9", "u10"]}
    )
    assert cov.omitted_units == 12
    assert cov.truncated is True
    assert cov.to_dict()["omitted_unit_ids"] == ["u9", "u10"]
