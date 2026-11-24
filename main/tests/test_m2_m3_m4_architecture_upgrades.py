"""
M2, M3, and M4 Architectural Upgrade Tests:
  - M2: Unified Control Plane (LangGraph + Supervisor <-> Research DAG manifest <-> TaskScheduler <-> EvidenceStore).
  - M3: Supervised Agent Autonomy (`AgentDecision` contract & `supervisor_evaluate_decision` gatekeeper,
        including proactive Reader/Retrieval proposals routed through Supervisor).
  - M4: End-to-End Claim Lineage (`Claim -> EvidenceSpan -> SourceUnit -> Document`) and
        Multi-Document Source Quality Audit (duplicate/preprint content hash detection, provenance pass rate).
"""

from __future__ import annotations

import importlib
from typing import Any

from core.config import settings
from core.coordination import (
    AgentDecision,
    propose_agent_decision,
    supervisor_evaluate_decision,
)
from core.evidence_store import EvidenceStore
from core.research_dag import build_research_dag_plan, execute_full_research_dag
from core.supervisor import supervisor_planner


def test_m2_unified_control_plane_progressive_evidence_store_and_dag_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "CORPUS_PATH", str(tmp_path / "empty.jsonl"))
    graph_mod = importlib.import_module("core.graph")

    observed_schemas: dict[str, dict[str, Any]] = {}

    def fake_retrieval(state):
        return {
            "retrieved_docs": [
                {"id": "u_alpha", "doc_id": "paper1.pdf", "page_no": 3, "text": "Transformer achieves 28.4 BLEU on WMT14."}
            ],
            "seen_chunks": [
                {"id": "u_alpha", "doc_id": "paper1.pdf", "page_no": 3, "text": "Transformer achieves 28.4 BLEU on WMT14."}
            ],
        }

    def fake_reader(state):
        return {
            "source_units": [
                {
                    "id": "u_alpha",
                    "doc_id": "paper1.pdf",
                    "page_no": 3,
                    "doi": "10.5555/3295222.3295349",
                    "text": "Transformer achieves 28.4 BLEU on WMT14.",
                }
            ]
        }

    def fake_summarizer(state):
        # Capture schema right after Reader to verify progressive EvidenceStore population
        observed_schemas["after_reader"] = dict(state.get("document_schema") or {})
        return {
            "answer": "Transformer achieves 28.4 BLEU on WMT14.",
            "claims": [{"id": "c_bleu", "text": "Transformer achieves 28.4 BLEU on WMT14.", "evidence_ids": ["sp_bleu"]}],
            "evidence_spans": [
                {
                    "id": "sp_bleu",
                    "source_unit_id": "u_alpha",
                    "doc_id": "paper1.pdf",
                    "page_no": 3,
                    "quote": "28.4 BLEU on WMT14",
                }
            ],
        }

    def fake_checker(state):
        observed_schemas["after_summarizer"] = dict(state.get("document_schema") or {})
        return {
            "fact_check_label": "support",
            "verified_claims": [
                {"id": "c_bleu", "text": "Transformer achieves 28.4 BLEU on WMT14.", "status": "SUPPORT", "evidence_ids": ["sp_bleu"]}
            ],
            "rejected_claims": [],
            "evidence_gaps": [],
        }

    runners = {
        "retrieval": fake_retrieval,
        "reader": fake_reader,
        "summarizer": fake_summarizer,
        "fact_checker": fake_checker,
        "writer": lambda s: {"final_report": s.get("answer", "")},
    }
    monkeypatch.setattr(graph_mod, "get_agent_runners", lambda mode: runners)
    monkeypatch.setattr(graph_mod.ScoutAgent, "run", lambda self, state: {})

    app = graph_mod.build_research_rag_graph("cpu_min", auto_scout=False)
    final_state = app.invoke({"query": "What BLEU score does Transformer achieve?", "pdf_paths": ["paper1.pdf"]})

    # 1. Verify progressive EvidenceStore population after Reader and Summarizer
    assert observed_schemas["after_reader"]["evidence_store"]["summary"]["source_count"] == 1
    assert observed_schemas["after_summarizer"]["evidence_store"]["summary"]["span_count"] == 1
    assert observed_schemas["after_summarizer"]["evidence_store"]["summary"]["verified_span_count"] == 1

    # 2. Verify Control Plane metadata and final EvidenceStore + DAG state
    dag_manifest = final_state["document_schema"]["research_dag"]
    assert dag_manifest["control_plane"] == "langgraph_supervisor"
    assert dag_manifest["last_verify_outcome"] == "PASSED"

    # 3. Verify compatibility DAG executor also populates EvidenceStore under same schema contract
    plan = supervisor_planner.plan_query("What BLEU score does Transformer achieve?")
    compat_state, _ = execute_full_research_dag(
        build_research_dag_plan(plan),
        {"query": "What BLEU score does Transformer achieve?"},
        retrieve_fn=fake_retrieval,
        synthesize_fn=fake_summarizer,
        verify_fn=fake_checker,
        write_fn=lambda s: {"final_report": s.get("answer", "")},
    )
    assert "evidence_store" in compat_state["document_schema"]
    assert compat_state["document_schema"]["evidence_store"]["summary"]["verified_claim_count"] == 1


def test_m3_supervised_agent_decision_contract_and_gatekeeper_policies():
    # 1. Canonical and directional alias construction
    decision = AgentDecision(
        source_agent="reader",
        status="completed",
        next_action="request_more_evidence",
        target_agent="retrieval",
        reason="Missing Table 3 experimental results",
        evidence_ids=["ev_001", "ev_002"],
        suggested_queries=["Table 3 ablation accuracy"],
        round_index=0,
    )
    d_dict = decision.to_dict()
    assert d_dict["decision_id"].startswith("dec_")
    assert d_dict["source_agent"] == "reader"
    assert d_dict["action"] == "request_more_evidence"
    assert d_dict["target_agent"] == "retrieval"
    assert d_dict["evidence_refs"] == ["ev_001", "ev_002"]
    assert "Table 3 ablation accuracy" in d_dict["parameters"]["suggested_queries"]
    assert len(d_dict["signature"]) == 64

    # 2. Supervisor approves valid Reader -> Retrieval proposal within budget
    handoff, audit = supervisor_evaluate_decision(decision, {"retry_count": 0, "max_reflection_loops": 2})
    assert handoff is not None
    assert handoff.sender == "reader"
    assert handoff.recipient == "retrieval"
    assert audit is not None and audit["approved"] is True
    assert audit["rejection_reason"] is None

    # 3. Supervisor blocks when reflection budget is exhausted
    blocked_handoff, blocked_audit = supervisor_evaluate_decision(
        decision, {"retry_count": 2, "max_reflection_loops": 2}
    )
    assert blocked_handoff is None
    assert blocked_audit is not None and blocked_audit["approved"] is False
    assert blocked_audit["rejection_reason"] == "reflection_budget_exhausted"

    # 4. Supervisor blocks unpermitted target policy (e.g. retrieval -> summarizer)
    illegal = AgentDecision(
        source_agent="retrieval",
        action="revise_draft",
        target_agent="summarizer",
        reason="Illegal target for retrieval",
    )
    illegal_handoff, illegal_audit = supervisor_evaluate_decision(
        illegal, {"retry_count": 0, "max_reflection_loops": 2}
    )
    assert illegal_handoff is None
    assert illegal_audit is not None and illegal_audit["approved"] is False
    assert str(illegal_audit["rejection_reason"]).startswith("policy_target_not_permitted")

    # 5. Supervisor blocks duplicate signature anti-loop
    dup_state = {
        "retry_count": 0,
        "max_reflection_loops": 2,
        "reasoning_trace": [handoff.as_trace()],
    }
    dup_handoff, dup_audit = supervisor_evaluate_decision(decision, dup_state)
    assert dup_handoff is None
    assert dup_audit is not None and dup_audit["rejection_reason"] == "duplicate_handoff_signature"


def test_m3_reader_proactive_agent_decision_routes_through_langgraph_supervisor(monkeypatch, tmp_path):
    """
    End-to-end LangGraph verification of Supervised Agent Autonomy:
    On pass 1, Reader detects a missing referenced table and emits an AgentDecision
    requesting Retrieval for 'Table 4 latency breakdown'. Supervisor approves it,
    routes back to Retrieval -> Reader (pass 2) -> Summarizer -> Fact Checker -> Writer.
    """
    monkeypatch.setattr(settings, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "CORPUS_PATH", str(tmp_path / "empty.jsonl"))
    graph_mod = importlib.import_module("core.graph")

    reader_calls = {"count": 0}
    retrieval_queries: list[str] = []

    def fake_retrieval(state):
        q = state.get("query", "")
        retrieval_queries.append(q)
        chunk = {
            "id": f"u_{len(retrieval_queries)}",
            "doc_id": "sys.pdf",
            "page_no": len(retrieval_queries),
            "text": f"Table 4 shows p99 latency is 14.2 ms for {q}.",
        }
        return {"retrieved_docs": [chunk], "seen_chunks": [chunk]}

    def fake_reader(state):
        reader_calls["count"] += 1
        units = list(state.get("retrieved_docs") or [])
        if reader_calls["count"] == 1:
            return {
                "source_units": units,
                "agent_decision": {
                    "action": "request_more_evidence",
                    "target_agent": "retrieval",
                    "reason": "Missing Table 4 referenced on page 1",
                    "suggested_queries": ["Table 4 latency breakdown"],
                },
            }
        return {"source_units": units, "agent_decision": None}

    def fake_summarizer(state):
        units = state.get("source_units") or []
        u0 = units[0]
        return {
            "answer": u0["text"],
            "claims": [{"id": "c_lat", "text": u0["text"], "evidence_ids": ["sp_lat"]}],
            "evidence_spans": [
                {"id": "sp_lat", "source_unit_id": u0["id"], "doc_id": u0["doc_id"], "page_no": u0["page_no"], "quote": u0["text"]}
            ],
        }

    def fake_checker(state):
        return {
            "fact_check_label": "support",
            "verified_claims": [{"id": "c_lat", "text": state.get("answer", ""), "status": "SUPPORT", "evidence_ids": ["sp_lat"]}],
            "rejected_claims": [],
            "evidence_gaps": [],
        }

    runners = {
        "retrieval": fake_retrieval,
        "reader": fake_reader,
        "summarizer": fake_summarizer,
        "fact_checker": fake_checker,
        "writer": lambda s: {"final_report": s.get("answer", "")},
    }
    monkeypatch.setattr(graph_mod, "get_agent_runners", lambda mode: runners)
    monkeypatch.setattr(graph_mod.ScoutAgent, "run", lambda self, state: {})

    app = graph_mod.build_research_rag_graph("cpu_min", auto_scout=False)
    result = app.invoke({
        "query": "What is the p99 latency in Table 4?",
        "pdf_paths": ["sys.pdf"],
        "max_reflection_loops": 2,
    })
    assert reader_calls["count"] == 2
    assert len(retrieval_queries) == 2
    assert result["retry_count"] == 1
    handoff_events = [e for e in result["reasoning_trace"] if e.get("action") == "agent_handoff" and e.get("sender") == "reader"]
    assert len(handoff_events) == 1
    assert handoff_events[0]["recipient"] == "retrieval"


def test_m4_claim_to_source_lineage_and_multi_document_duplicate_quality_audit():
    state = {
        "source_units": [
            {
                "id": "u_arxiv",
                "doc_id": "arxiv_2401_0001v1.pdf",
                "page_no": 5,
                "doi": "10.48550/arXiv.2401.0001",
                "text": "Our proposed Graph-RAG architecture improves Recall@10 by 18.4% over baseline BM25.",
                "provenance": [
                    {"doc_id": "arxiv_2401_0001v1.pdf", "page_no": 5, "unit_id": "u_arxiv"},
                    {"doc_id": "acl_2025_camera_ready.pdf", "page_no": 6, "unit_id": "u_acl"},
                ],
            },
            {
                # Preprint vs published duplicate passage across distinct doc_id
                "id": "u_acl",
                "doc_id": "acl_2025_camera_ready.pdf",
                "page_no": 6,
                "doi": "10.18653/v1/2025.acl-long.42",
                "text": "Our proposed Graph-RAG architecture   improves Recall@10 by 18.4% over baseline BM25.",
            },
        ],
        "evidence_spans": [
            {
                "id": "sp_valid",
                "source_unit_id": "u_arxiv",
                "doc_id": "arxiv_2401_0001v1.pdf",
                "page_no": 5,
                "quote": "improves Recall@10 by 18.4% over baseline BM25",
            },
            {
                "id": "sp_broken",
                "source_unit_id": "u_arxiv",
                "doc_id": "arxiv_2401_0001v1.pdf",
                "page_no": 5,
                "quote": "fabricated quote not in any document",
            },
        ],
        "verified_claims": [
            {
                "id": "claim_recall",
                "text": "Graph-RAG improves Recall@10 by 18.4% over baseline BM25.",
                "status": "SUPPORT",
                "evidence_ids": ["sp_valid"],
            }
        ],
        "rejected_claims": [
            {
                "id": "claim_unverified",
                "text": "Unsupported claim tied to fabricated quote.",
                "status": "NOT_ENOUGH_INFO",
                "evidence_ids": ["sp_broken"],
            }
        ],
        "evidence_gaps": [],
    }

    store = EvidenceStore.from_state(state)
    payload = store.to_dict()

    # 1. Verify Claim -> EvidenceSpan -> SourceUnit -> Document lineage
    lineage_valid = store.trace_claim_lineage("claim_recall")
    assert lineage_valid is not None
    assert lineage_valid["lineage_complete"] is True
    assert lineage_valid["supporting_documents"] == ["arxiv_2401_0001v1.pdf"]
    chain_entry = lineage_valid["evidence_chain"][0]
    assert chain_entry["span_id"] == "sp_valid"
    assert chain_entry["source_unit_id"] == "u_arxiv"
    assert chain_entry["doc_id"] == "arxiv_2401_0001v1.pdf"
    assert chain_entry["page_no"] == 5
    assert chain_entry["doi"] == "10.48550/arXiv.2401.0001"
    assert chain_entry["context_hash"]
    assert chain_entry["content_hash"]

    lineage_broken = store.trace_claim_lineage("claim_unverified")
    assert lineage_broken is not None
    assert lineage_broken["lineage_complete"] is False

    # 2. Verify Multi-Document Source Quality Audit (preprint vs published duplicate detection)
    quality = payload["source_quality"]
    assert quality["unique_document_count"] == 2
    assert quality["units_with_doi"] == 2
    assert quality["multi_source_corroborated_units"] == 1
    assert quality["has_cross_document_duplicates"] is True
    assert len(quality["duplicate_content_groups"]) == 1
    assert quality["duplicate_content_groups"][0]["documents"] == [
        "acl_2025_camera_ready.pdf",
        "arxiv_2401_0001v1.pdf",
    ]
    assert quality["provenance_pass_rate"] == 0.5
    assert quality["verified_lineage_rate"] == 1.0
