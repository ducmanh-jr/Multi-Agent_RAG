"""
Research V2 — Step 1 Correctness Regression Suite:
1. Checkpoint trace accumulation & continuous vs interrupted-after-2-handoffs + resume parity.
2. Strict Citation F1: penalizes out-of-range markers ([999]), separates answerable vs unanswerable,
   and supports gold_sources (doc_id, page_no) recall denominator.
3. Scout duplicate PDF detection by resolved path AND file SHA-256.
4. Run manifest truthful `parallel_enabled` and observed DAG parallel branch counts across modes.
5. Research DAG `max_tasks` enforcement in both initial planning and replanning (inheriting manifest budget).
6. Hardened Supervisor `AgentDecision` validation (dict & object parity, sender/action/status/round checks,
   and Supervisor-recomputed canonical anti-loop signatures).
7. Ablation study stratified sampling (>4 items) and fair paired handoff OFF vs ON runner.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import Any

import fitz
import pytest

from api.research_runs import ResearchRequest, ResearchRunService
from api.run_manifest import get_run_manifest
from core import db, graph as graph_module
from core.config import RunConfig, settings
from core.coordination import (
    AgentDecision,
    evaluate_agent_handoff,
    supervisor_evaluate_decision,
)
from core.graph import build_research_rag_graph, resolve_next_node_after_checkpoint
from core.research_dag import (
    ResearchDAGBudget,
    build_research_dag_plan,
    replan_research_dag_for_gaps,
)
from core.schemas import RouterPlanOutput
from eval.ablation_study import run_paired_handoff_ablation, sample_stratified_questions
from eval.strict_metrics import compute_strict_citation_f1


def _make_pdf(path: Path, pages: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open() as doc:
        for text in pages:
            page = doc.new_page(width=595, height=842)
            page.insert_text((72, 72), text)
        doc.save(str(path))
    return path


def _strip_runtime_timing(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("action") in {"run_manifest", "run_resumed_from_checkpoint", "citation_audit", "answer_figures"}:
            continue
        item = {k: v for k, v in ev.items() if k not in {"duration", "duration_sec", "timestamp", "elapsed_ms"}}
        cleaned.append(item)
    return cleaned


def test_step1_checkpoint_continuous_vs_interrupted_after_two_handoffs_and_resume(
    monkeypatch, research_db, tmp_path
):
    pdf = _make_pdf(tmp_path / "handoff_parity.pdf", ["Latency in Table 4 is 11.4 ms for Hybrid mode."])
    db.save_document("doc_hp", pdf.name, str(pdf), pdf.stat().st_size)

    def make_runners(crash_after_two_handoffs: dict[str, bool]):
        def fake_retrieval(state):
            r = int(state.get("retry_count") or 0)
            chunk = {
                "id": f"u_r{r}",
                "doc_id": "handoff_parity",
                "page_no": 1,
                "text": "Latency in Table 4 is 11.4 ms for Hybrid mode." if r >= 1 else "Introductory overview text.",
            }
            return {
                "retrieved_docs": [chunk],
                "reasoning_trace": [{"node": "Retrieval", "action": f"ret_step_r{r}"}],
            }

        def fake_reader(state):
            r = int(state.get("retry_count") or 0)
            return {
                "source_units": list(state.get("retrieved_docs") or []),
                "reasoning_trace": [{"node": "Reader", "action": f"read_step_r{r}"}],
            }

        def fake_summarizer(state):
            r = int(state.get("retry_count") or 0)
            if r == 2 and crash_after_two_handoffs["armed"]:
                crash_after_two_handoffs["armed"] = False
                raise RuntimeError("Simulated crash right after 2nd handoff")
            u = (state.get("source_units") or [{"id": "u_r0", "doc_id": "handoff_parity", "page_no": 1, "text": "Intro"}])[0]
            txt = "Latency is 11.4 ms." if r >= 2 else "Preliminary unverified draft."
            return {
                "answer": txt,
                "claims": [{"id": f"c_r{r}", "text": txt, "evidence_ids": [f"sp_r{r}"]}],
                "evidence_spans": [
                    {
                        "id": f"sp_r{r}",
                        "source_unit_id": u["id"],
                        "doc_id": "handoff_parity",
                        "page_no": 1,
                        "quote": u["text"],
                    }
                ],
                "reasoning_trace": [{"node": "Summarizer", "action": f"sum_step_r{r}"}],
            }

        def fake_checker(state):
            r = int(state.get("retry_count") or 0)
            if r == 0:
                return {
                    "fact_check_label": "not_enough_info",
                    "verified_claims": [],
                    "rejected_claims": [],
                    "evidence_gaps": [{"missing_aspect": "Table 4 latency", "suggested_query": "Table 4 latency ms"}],
                    "reasoning_trace": [{"node": "FactChecker", "action": "fc_step_r0"}],
                }
            if r == 1:
                return {
                    "fact_check_label": "contradict",
                    "verified_claims": [],
                    "rejected_claims": [{"claim_text": "Preliminary unverified draft.", "status": "CONTRADICT", "reason": "Use exact 11.4 ms"}],
                    "evidence_gaps": [],
                    "reasoning_trace": [{"node": "FactChecker", "action": "fc_step_r1"}],
                }
            return {
                "fact_check_label": "support",
                "verified_claims": [{**c, "status": "SUPPORT"} for c in (state.get("claims") or [])],
                "rejected_claims": [],
                "evidence_gaps": [],
                "reasoning_trace": [{"node": "FactChecker", "action": "fc_step_r2"}],
            }

        def fake_writer(state):
            return {
                "final_report": f"Final: {state.get('answer', '')}",
                "reasoning_trace": [{"node": "Writer", "action": "write_done"}],
            }

        return {
            "retrieval": fake_retrieval,
            "reader": fake_reader,
            "summarizer": fake_summarizer,
            "fact_checker": fake_checker,
            "writer": fake_writer,
        }

    monkeypatch.setattr(graph_module.ScoutAgent, "run", lambda self, state: {"reasoning_trace": [{"action": "bypass_discovery"}]})
    monkeypatch.setattr(graph_module, "add_figure_evidence", lambda inputs, update, mode: update)

    service = ResearchRunService()

    async def _run_both():
        # 1. Continuous run
        monkeypatch.setattr(graph_module, "get_agent_runners", lambda mode: make_runners({"armed": False}))
        cont_run = await service.start(
            ResearchRequest(
                query="What is the latency in Table 4?",
                config=RunConfig.capture(),
                mode="cpu_min",
                document_ids=("doc_hp",),
                auto_scout=False,
                max_reflection_loops=3,
                parallel_retrieval=False,
            )
        )
        await asyncio.wait_for(cont_run.result(), timeout=15)
        cont_rec = db.get_run(cont_run.record["id"])
        cont_ckpts = db.list_run_checkpoints(cont_run.record["id"])

        # 2. Interrupted run after 2 handoffs -> resume
        crash_flag = {"armed": True}
        monkeypatch.setattr(graph_module, "get_agent_runners", lambda mode: make_runners(crash_flag))
        interrupted = await service.start(
            ResearchRequest(
                query="What is the latency in Table 4?",
                config=RunConfig.capture(),
                mode="cpu_min",
                document_ids=("doc_hp",),
                auto_scout=False,
                max_reflection_loops=3,
                parallel_retrieval=False,
            )
        )
        await asyncio.wait_for(interrupted.result(), timeout=15)
        failed_rec = db.get_run(interrupted.record["id"])
        assert failed_rec is not None and failed_rec["status"] == "failed"
        assert "Simulated crash right after 2nd handoff" in (failed_rec.get("error") or "")

        latest_ckpt = db.get_latest_run_checkpoint(interrupted.record["id"])
        assert latest_ckpt is not None
        assert latest_ckpt["node_name"] in {"fact_checker_node", "supervisor_node"}
        ckpt_state = latest_ckpt["state"]
        assert ckpt_state["retry_count"] == 2
        ckpt_handoffs = [e for e in ckpt_state["reasoning_trace"] if e.get("action") == "agent_handoff"]
        # Both prior handoffs MUST be present in the checkpoint state trace (never overwritten by last node!)
        assert len(ckpt_handoffs) == 2

        next_node = resolve_next_node_after_checkpoint(latest_ckpt["node_name"], ckpt_state)
        assert next_node == "summarizer_node"

        resumed = await service.resume(interrupted.record["id"], config=RunConfig.capture(), mode="cpu_min")
        await asyncio.wait_for(resumed.result(), timeout=15)
        resumed_rec = db.get_run(resumed.record["id"])
        resumed_ckpts = db.list_run_checkpoints(resumed.record["id"])
        return cont_rec, cont_ckpts, latest_ckpt, resumed_rec, resumed_ckpts

    cont_rec, cont_ckpts, latest_ckpt, resumed_rec, resumed_ckpts = asyncio.run(_run_both())

    assert cont_rec["status"] == "completed"
    assert resumed_rec["status"] == "completed"

    cont_trace = _strip_runtime_timing(cont_rec["reasoning_trace"])
    resumed_trace = _strip_runtime_timing(resumed_rec["reasoning_trace"])
    assert resumed_trace == cont_trace

    # Verify final checkpoint trace also matches continuous run trace with zero duplicates
    final_cont_ckpt_trace = _strip_runtime_timing(cont_ckpts[-1]["state"]["reasoning_trace"])
    final_resumed_ckpt_trace = _strip_runtime_timing(resumed_ckpts[-1]["state"]["reasoning_trace"])
    assert final_resumed_ckpt_trace == final_cont_ckpt_trace

    cont_sigs = [e["signature"] for e in cont_trace if e.get("action") == "agent_handoff"]
    resumed_sigs = [e["signature"] for e in resumed_trace if e.get("action") == "agent_handoff"]
    assert len(cont_sigs) == 2
    assert resumed_sigs == cont_sigs

    # Anti-loop MUST still block repeating the 1st handoff using resumed checkpoint state
    first_handoff = next(e for e in latest_ckpt["state"]["reasoning_trace"] if e.get("action") == "agent_handoff")
    dup_decision = AgentDecision(
        source_agent=first_handoff["sender"],
        action="request_more_evidence",
        target_agent=first_handoff["recipient"],
        reason=first_handoff["reason"],
        focus=first_handoff["focus"],
        round_index=2,
    )
    blocked_h, blocked_audit = supervisor_evaluate_decision(
        dup_decision,
        {**latest_ckpt["state"], "max_reflection_loops": 5},
        expected_sender=first_handoff["sender"],
    )
    assert blocked_h is None
    assert blocked_audit is not None and blocked_audit["rejection_reason"] == "duplicate_handoff_signature"


def test_step1_strict_citation_f1_penalizes_out_of_range_separates_abstention_and_gold_sources():
    evidence = [{"doc_id": "paper_a.pdf", "page_no": 2, "text": "Transformer achieves 28.4 BLEU on WMT14."}]

    # 1. Exact valid citation
    assert compute_strict_citation_f1("Achieves 28.4 BLEU [1].", evidence, ["28.4 BLEU"]) == (1.0, 1.0, 1.0)

    # 2. Out-of-range citation [999] MUST penalize precision: P=0.5, R=1.0, F1=0.6667
    p, r, f1 = compute_strict_citation_f1("Achieves 28.4 BLEU [1][999].", evidence, ["28.4 BLEU"])
    assert p == 0.5
    assert r == 1.0
    assert f1 == pytest.approx(0.6667, abs=1e-4)

    # 3. Only out-of-range citations -> 0.0
    assert compute_strict_citation_f1("Claim [0][999].", evidence, ["28.4 BLEU"]) == (0.0, 0.0, 0.0)

    # 4. Answerable question with empty evidence and un-cited assertion MUST NOT get 1.0!
    assert compute_strict_citation_f1("Transformer achieves 28.4 BLEU.", [], ["28.4 BLEU"], is_unanswerable=False) == (0.0, 0.0, 0.0)
    assert compute_strict_citation_f1(
        "Unrelated uncited text.",
        [{"doc_id": "paper_a.pdf", "page_no": 1, "text": "Copyright notice only."}],
        ["28.4 BLEU"],
        is_unanswerable=False,
    ) == (0.0, 0.0, 0.0)

    # 5. Unanswerable question scored separately — fabricated uncited answer MUST get 0.0, NEVER 1.0!
    assert compute_strict_citation_f1("Không có thông tin trong tài liệu.", [], [], is_unanswerable=True) == (1.0, 1.0, 1.0)
    assert compute_strict_citation_f1("The result is 12345 ms.", [], [], is_unanswerable=True) == (0.0, 0.0, 0.0)
    assert compute_strict_citation_f1("Không có thông tin [1].", evidence, [], is_unanswerable=True) == (0.0, 0.0, 0.0)

    # 6. gold_sources penalizes retrieval miss in Recall even when cited chunk is valid
    p_gs, r_gs, f1_gs = compute_strict_citation_f1(
        "Achieves 28.4 BLEU [1].",
        evidence,
        ["28.4 BLEU"],
        gold_sources=[("paper_a.pdf", 2), ("paper_a.pdf", 5)],
    )
    assert p_gs == 1.0
    assert r_gs == 0.5
    assert f1_gs == pytest.approx(0.6667, abs=1e-4)


def test_step1_scout_duplicate_pdf_by_path_and_sha256_marks_no_new_documents(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "CORPUS_PATH", str(tmp_path / "empty.jsonl"))

    orig_pdf = _make_pdf(tmp_path / "base.pdf", ["Baseline measurement is 42.0 ms."])
    same_bytes_copy = tmp_path / "copy_identical_sha.pdf"
    same_bytes_copy.write_bytes(orig_pdf.read_bytes())
    brand_new_pdf = _make_pdf(tmp_path / "new_paper.pdf", ["Follow-up measurement is 19.5 ms."])

    for label, scout_returned_path, expect_new in [
        ("same_path_unnormalized", str(tmp_path / "." / "base.pdf"), False),
        ("different_path_identical_sha256", str(same_bytes_copy), False),
        ("genuinely_new_pdf", str(brand_new_pdf), True),
    ]:
        calls = {"ret": 0}

        def fake_scout_run(self, state, _ret_path=scout_returned_path):
            selected = state.get("pdf_paths") or ([state["pdf_path"]] if state.get("pdf_path") else [])
            if selected:
                return {"pdf_path": selected[0], "pdf_paths": selected, "reasoning_trace": [{"action": "bypass_discovery"}]}
            return {
                "scout_decision": "success",
                "pdf_path": _ret_path,
                "pdf_paths": [_ret_path],
                "reasoning_trace": [{"action": "paper_discovered_and_ingested", "path": _ret_path}],
            }

        def fake_retrieval(state, _calls=calls):
            _calls["ret"] += 1
            if _calls["ret"] == 1:
                return {
                    "retrieved_docs": [{"id": "u1", "doc_id": "base", "page_no": 1, "text": "Baseline measurement is 42.0 ms."}],
                    "agent_decision": {
                        "action": "discover_additional_literature",
                        "target_agent": "scout",
                        "reason": f"Need external doc ({label})",
                        "suggested_queries": ["external follow-up measurement"],
                    },
                }
            return {
                "retrieved_docs": [{"id": "u2", "doc_id": "new_paper", "page_no": 1, "text": "Follow-up measurement is 19.5 ms."}],
                "agent_decision": None,
            }

        runners = {
            "retrieval": fake_retrieval,
            "reader": lambda s: {"source_units": list(s.get("retrieved_docs") or [])},
            "summarizer": lambda s: {
                "answer": "Baseline 42.0 ms.",
                "claims": [{"id": "c1", "text": "Baseline 42.0 ms.", "evidence_ids": ["sp1"]}],
                "evidence_spans": [{"id": "sp1", "source_unit_id": "u1", "doc_id": "base", "page_no": 1, "quote": "Baseline measurement is 42.0 ms."}],
            },
            "fact_checker": lambda s: {"fact_check_label": "support", "verified_claims": s["claims"], "rejected_claims": []},
            "writer": lambda s: {"final_report": s["answer"]},
        }
        monkeypatch.setattr(graph_module.ScoutAgent, "run", fake_scout_run)
        monkeypatch.setattr(graph_module, "get_agent_runners", lambda mode, _r=runners: _r)
        monkeypatch.setattr(graph_module, "add_figure_evidence", lambda inputs, update, mode: update)

        app = build_research_rag_graph("cpu_min", auto_scout=False, allow_scout_handoff=True)
        out = app.invoke({"query": "Find external follow-up measurement", "pdf_paths": [str(orig_pdf)], "max_reflection_loops": 2})

        scout_outcomes = [
            e for e in out["reasoning_trace"]
            if e.get("action") == "handoff_outcome_recorded" and e.get("recipient") == "scout"
        ]
        assert len(scout_outcomes) == 1
        if expect_new:
            assert scout_outcomes[0]["outcome_status"] == "discovered_new_documents"
            assert scout_outcomes[0]["new_documents_count"] == 1
            assert len(out["pdf_paths"]) == 2
        else:
            assert scout_outcomes[0]["outcome_status"] == "no_new_documents"
            assert scout_outcomes[0]["new_documents_count"] == 0
            assert len(out["pdf_paths"]) == 1
            assert out.get("no_new_evidence") is True


def test_step1_run_manifest_reports_parallel_enabled_and_observed_branches_in_hybrid(
    monkeypatch, research_db, tmp_path
):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    monkeypatch.setattr(settings, "REMOTE_INFERENCE_URL", "")
    pdf = _make_pdf(tmp_path / "parallel_hybrid.pdf", ["Model A achieves 91% F1 while Model B achieves 88% F1."])
    db.save_document("doc_ph", pdf.name, str(pdf), pdf.stat().st_size)

    def fake_retrieval(state):
        q = state.get("query", "")
        return {
            "retrieved_docs": [
                {"id": f"c_{abs(hash(q)) % 1000}", "doc_id": "parallel_hybrid", "page_no": 1, "text": f"Match for {q}: Model A achieves 91% F1."}
            ]
        }

    runners = {
        "retrieval": fake_retrieval,
        "reader": lambda s: {"source_units": list(s.get("retrieved_docs") or [])},
        "summarizer": lambda s: {
            "answer": "Model A achieves 91% F1.",
            "claims": [{"id": "c1", "text": "Model A achieves 91% F1.", "evidence_ids": ["sp1"]}],
            "evidence_spans": [{"id": "sp1", "source_unit_id": s["source_units"][0]["id"], "doc_id": "parallel_hybrid", "page_no": 1, "quote": s["source_units"][0]["text"]}],
        },
        "fact_checker": lambda s: {"fact_check_label": "support", "verified_claims": s["claims"], "rejected_claims": []},
        "writer": lambda s: {"final_report": s["answer"]},
    }
    monkeypatch.setattr(graph_module, "get_agent_runners", lambda mode: runners)
    monkeypatch.setattr(graph_module.ScoutAgent, "run", lambda self, state: {"reasoning_trace": [{"action": "bypass_discovery"}]})
    monkeypatch.setattr(graph_module, "add_figure_evidence", lambda inputs, update, mode: update)

    service = ResearchRunService()

    async def _run():
        run = await service.start(
            ResearchRequest(
                query="Compare Model A and Model B performance metrics",
                config=RunConfig.capture(),
                mode="hybrid",
                document_ids=("doc_ph",),
                auto_scout=False,
                parallel_retrieval=True,
                max_reflection_loops=0,
            )
        )
        await asyncio.wait_for(run.result(), timeout=15)
        return db.get_run(run.record["id"])

    rec = asyncio.run(_run())
    manifest = get_run_manifest(rec)
    assert manifest is not None
    assert manifest["configuration"]["retrieval"]["parallel_requested"] is True
    assert manifest["configuration"]["retrieval"]["parallel_enabled"] is True
    assert manifest["configuration"]["retrieval"]["parallel_mode"] == "research_dag"
    assert manifest["observed"]["parallel_execution_observed"] is True
    assert len(manifest["observed"]["parallel_dag_executions"]) >= 1
    assert manifest["observed"]["parallel_dag_executions"][0]["branch_count"] >= 2


def test_step1_dag_max_tasks_enforced_in_initial_plan_and_replan_without_repassing_budget():
    plan3 = RouterPlanOutput(
        intent="compare",
        complexity="HIGH",
        sub_queries=["Query A", "Query B", "Query C"],
    )

    # 1. Impossible budget < 5 raises ValueError
    with pytest.raises(ValueError, match="max_tasks"):
        build_research_dag_plan(plan3, budget=ResearchDAGBudget(max_tasks=4))

    # 2. max_tasks=5 with 3 sub-queries MUST create exactly 5 tasks (1 investigate + 4 downstream)
    dag5 = build_research_dag_plan(plan3, budget=ResearchDAGBudget(max_tasks=5, max_parallel_subqueries=3))
    manifest5 = dag5.to_manifest()
    assert manifest5["total_tasks"] == 5
    assert len(manifest5["nodes"]) == 5
    assert sum(1 for n in manifest5["nodes"] if n["kind"] == "investigate") == 1

    # Replan WITHOUT passing `budget` MUST inherit max_tasks=5 from manifest["budget"], NOT default 16!
    replanned5 = replan_research_dag_for_gaps(
        manifest5,
        [{"missing_aspect": "Latency", "suggested_query": "latency breakdown"}],
    )
    assert replanned5["total_tasks"] == 5
    assert replanned5["added_gap_tasks"] == 0
    assert replanned5["budget_exhausted"] is True

    # 3. max_tasks=6 with 3 sub-queries -> 2 investigate + 4 downstream = 6
    dag6 = build_research_dag_plan(plan3, budget=ResearchDAGBudget(max_tasks=6, max_parallel_subqueries=3))
    assert dag6.to_manifest()["total_tasks"] == 6

    # 4. Tight budget max_tasks=8 across initial plan + two replans WITHOUT re-passing budget
    dag8 = build_research_dag_plan(
        RouterPlanOutput(intent="factoid", complexity="LOW", sub_queries=["Single Q"]),
        budget=ResearchDAGBudget(max_tasks=8, max_parallel_subqueries=3),
    )
    m8 = dag8.to_manifest()
    assert m8["total_tasks"] == 5
    r1 = replan_research_dag_for_gaps(
        m8,
        [
            {"missing_aspect": "A", "suggested_query": "Gap query 1"},
            {"missing_aspect": "B", "suggested_query": "Gap query 2"},
            {"missing_aspect": "C", "suggested_query": "Gap query 3"},
        ],
    )
    assert r1["total_tasks"] <= 8
    r2 = replan_research_dag_for_gaps(
        r1,
        [{"missing_aspect": "D", "suggested_query": "Gap query 4"}],
    )
    assert r2["total_tasks"] <= 8
    assert r2["budget_exhausted"] is True


def test_step1_supervisor_validates_sender_action_status_round_and_recomputes_canonical_signature():
    base_state = {"retry_count": 0, "max_reflection_loops": 3, "reasoning_trace": []}

    # 1. Spoofed source_agent on AgentDecision object
    spoof_obj = AgentDecision(
        source_agent="writer",
        action="revise_draft",
        target_agent="summarizer",
        reason="Spoofed sender",
    )
    h, audit = evaluate_agent_handoff("reader", {**base_state, "agent_decision": spoof_obj})
    assert h is None
    assert audit is not None
    assert audit["approved"] is False
    assert audit["decision_state"] == "REJECTED"
    assert audit["rejection_reason"] == "source_agent_mismatch:writer!=reader"

    # 2. Spoofed source_agent on dict payload
    h_d, audit_d = evaluate_agent_handoff(
        "reader",
        {**base_state, "agent_decision": {"source_agent": "writer", "action": "request_more_evidence", "target_agent": "retrieval"}},
    )
    assert h_d is None
    assert audit_d is not None and audit_d["rejection_reason"] == "source_agent_mismatch:writer!=reader"

    # 3. Invalid action rejected
    h_act, audit_act = evaluate_agent_handoff(
        "reader",
        {**base_state, "agent_decision": {"action": "drop_database", "target_agent": "retrieval"}},
    )
    assert h_act is None
    assert audit_act is not None and audit_act["rejection_reason"] == "invalid_decision_action:drop_database"

    # 4. Invalid status rejected
    h_st, audit_st = evaluate_agent_handoff(
        "reader",
        {**base_state, "agent_decision": {"action": "request_more_evidence", "target_agent": "retrieval", "status": "hacked"}},
    )
    assert h_st is None
    assert audit_st is not None and audit_st["rejection_reason"] == "invalid_decision_status:hacked"

    # 5. Stale round_index rejected when budget still remains
    stale_obj = AgentDecision(
        source_agent="reader",
        action="request_more_evidence",
        target_agent="retrieval",
        reason="Missing Table 4",
        focus=["Table 4"],
        round_index=0,
    )
    h_rnd, audit_rnd = supervisor_evaluate_decision(
        stale_obj,
        {"retry_count": 1, "max_reflection_loops": 3, "reasoning_trace": []},
        expected_sender="reader",
    )
    assert h_rnd is None
    assert audit_rnd is not None and audit_rnd["rejection_reason"] == "round_index_mismatch:0!=1"

    # 6. Forged signature MUST NOT bypass duplicate anti-loop check (both object and dict!)
    first_h, first_audit = supervisor_evaluate_decision(stale_obj, base_state, expected_sender="reader")
    assert first_h is not None and first_audit is not None
    assert first_audit["decision_state"] == "APPROVED"
    canonical_sig = first_h.as_trace()["signature"]
    assert first_audit["signature"] == canonical_sig

    forged_obj = AgentDecision(
        source_agent="reader",
        action="request_more_evidence",
        target_agent="retrieval",
        reason="Missing Table 4",
        focus=["Table 4"],
        round_index=0,
        signature="forged_signature_bypass_attempt_12345",
    )
    dup_h, dup_audit = supervisor_evaluate_decision(
        forged_obj,
        {**base_state, "reasoning_trace": [first_h.as_trace()]},
        expected_sender="reader",
    )
    assert dup_h is None
    assert dup_audit is not None
    assert dup_audit["rejection_reason"] == "duplicate_handoff_signature"
    assert dup_audit["signature"] == canonical_sig

    dup_dict_h, dup_dict_audit = evaluate_agent_handoff(
        "reader",
        {
            **base_state,
            "reasoning_trace": [first_h.as_trace()],
            "agent_decision": {
                "action": "request_more_evidence",
                "target_agent": "retrieval",
                "reason": "Missing Table 4",
                "focus": ["Table 4"],
                "signature": "another_forged_signature_99999",
            },
        },
    )
    assert dup_dict_h is None
    assert dup_dict_audit is not None
    assert dup_dict_audit["rejection_reason"] == "duplicate_handoff_signature"
    assert dup_dict_audit["signature"] == canonical_sig


def test_step1_ablation_stratified_sampling_and_paired_handoff_runner(monkeypatch, tmp_path):
    # 1. Stratified sampling MUST return up to num_samples > 4 cleanly
    synthetic = [
        {"id": f"{cat}_{i}", "category": cat, "question": f"Q {cat} {i}"}
        for cat in ("single_hop", "multi_hop", "table_comparison", "adversarial_trap")
        for i in range(5)
    ]
    sampled = sample_stratified_questions(synthetic, 12)
    assert len(sampled) == 12
    counts: dict[str, int] = {}
    for item in sampled:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    assert counts == {"single_hop": 3, "multi_hop": 3, "table_comparison": 3, "adversarial_trap": 3}

    # 2. Fair paired ablation (handoff OFF vs ON) on identical graph with a handoff-improved item & sufficient item
    pdf = _make_pdf(tmp_path / "ablation_doc.pdf", ["Table 4 accuracy is 94.2%."])
    monkeypatch.setattr(settings, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "CORPUS_PATH", str(tmp_path / "empty.jsonl"))

    def fake_retrieval(state):
        r = int(state.get("retry_count") or 0)
        q = state.get("query", "")
        if "Table 4" in q and r == 0:
            return {"retrieved_docs": [{"id": "u_intro", "doc_id": "ablation_doc", "page_no": 1, "text": "See Table 4 for accuracy."}]}
        return {"retrieved_docs": [{"id": "u_tab4", "doc_id": "ablation_doc", "page_no": 1, "text": "Table 4 accuracy is 94.2%."}]}

    def fake_reader(state):
        units = list(state.get("retrieved_docs") or [])
        r = int(state.get("retry_count") or 0)
        if units and "See Table 4" in units[0]["text"] and r == 0:
            return {
                "source_units": units,
                "agent_decision": {
                    "action": "request_more_evidence",
                    "target_agent": "retrieval",
                    "reason": "Missing Table 4",
                    "focus": ["Table 4 accuracy"],
                },
            }
        return {"source_units": units, "agent_decision": None}

    def fake_summarizer(state):
        units = state.get("source_units") or []
        u0 = units[0]
        if "94.2%" in u0["text"]:
            ans = "Table 4 accuracy is 94.2% [1]."
        else:
            ans = "See Table 4 for details [1]."
        return {
            "answer": ans,
            "claims": [{"id": "c1", "text": ans, "evidence_ids": ["sp1"]}],
            "evidence_spans": [{"id": "sp1", "source_unit_id": u0["id"], "doc_id": "ablation_doc", "page_no": 1, "quote": u0["text"]}],
        }

    runners = {
        "retrieval": fake_retrieval,
        "reader": fake_reader,
        "summarizer": fake_summarizer,
        "fact_checker": lambda s: {"fact_check_label": "support", "verified_claims": s["claims"], "rejected_claims": []},
        "writer": lambda s: {"final_report": s["answer"]},
    }
    monkeypatch.setattr(graph_module, "get_agent_runners", lambda mode: runners)
    monkeypatch.setattr(graph_module.ScoutAgent, "run", lambda self, state: {"reasoning_trace": [{"action": "bypass_discovery"}]})
    monkeypatch.setattr(graph_module, "add_figure_evidence", lambda inputs, update, mode: update)

    report = run_paired_handoff_ablation(
        [
            {"id": "q_sufficient", "category": "single_hop", "question": "What is the headline accuracy?", "pdf_paths": [str(pdf)], "gold_keywords": ["94.2%"]},
            {"id": "q_needs_handoff", "category": "table_comparison", "question": "What is the accuracy in Table 4?", "pdf_paths": [str(pdf)], "gold_keywords": ["94.2%"]},
            {"id": "q_trap", "category": "adversarial_trap", "question": "What is Mars orbital speed?", "pdf_paths": [str(pdf)], "is_unanswerable": True, "gold_keywords": []},
        ],
        mode="cpu_min",
        repeats=2,
    )
    assert report["total_questions"] == 3
    # Unanswerable traps MUST be excluded from Citation F1 average denominator
    assert report["citation_f1_scored_count"] == 2
    assert report["unanswerable_count"] == 1
    assert report["order_alternated"] is True
    assert report["graph_precompiled"] is True
    assert report["repeats"] == 2
    assert report["outcomes"]["improved"] == 1
    assert report["outcomes"]["degraded"] == 0
    assert report["outcomes"]["handoff_triggered"] == 1
    assert report["outcomes"]["handoff_yield"] == 1.0
    assert report["outcomes"]["reader_handoff_yield"] == 1.0
    assert report["handoff_on"]["composite"] > report["handoff_off"]["composite"]
    assert "latency_no_handoff_p50_s" in report["handoff_on"]
    assert "latency_with_handoff_p50_s" in report["handoff_on"]
    assert report["per_question"][2]["on"]["citation_f1"] is None
