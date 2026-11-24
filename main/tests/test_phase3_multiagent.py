"""Regression & capability tests for Phase 3 Multi-Agent Orchestration upgrades:
1. N-document parallel ComparisonPlan (3+ PDFs) with barrier-gated review and multi-source citation audit.
2. Multi-mode ComparisonPlan unlocked beyond cpu_min (e.g. gemini/hybrid).
3. Safe coroutine execution (run_coro_sync) inside an already-running event loop.
4. LLM-assisted SupervisorPlanner decomposition with fail-open fallback.
5. Bounded Fact-Checker -> Scout handoff for iterative literature discovery when auto_scout=True.
"""

from __future__ import annotations

import asyncio
import json
import fitz
import pytest

from core.comparison_plan import ComparisonPlan, run_coro_sync
from core.config import RunConfig, settings
from core.coordination import decide_handoff
from core.graph import build_research_rag_graph
from core.supervisor import supervisor_planner
from core.tools import clear_pdf_cache
from utils.llm_client import llm_client


def _make_pdf(path, text: str) -> str:
    with fitz.open() as doc:
        page = doc.new_page(width=500, height=400)
        page.insert_text((40, 80), text, fontsize=11)
        doc.save(str(path))
    return str(path)


def test_three_document_parallel_comparison_plan(tmp_path):
    clear_pdf_cache()
    p1 = _make_pdf(tmp_path / "alpha-trial.pdf", "Alpha irrigation trial ran for 14 days with drip sensors.")
    p2 = _make_pdf(tmp_path / "beta-trial.pdf", "Beta irrigation trial ran for 28 days with mist nozzles.")
    p3 = _make_pdf(tmp_path / "gamma-trial.pdf", "Gamma irrigation trial ran for 42 days with flood canals.")
    paths = [p1, p2, p3]

    plan = supervisor_planner.plan_document_comparison({"pdf_paths": paths, "is_comparative_query": True})
    assert isinstance(plan, ComparisonPlan)
    assert len(plan.paths) == 3

    config = RunConfig.capture(corpus_path=str(tmp_path / "missing.jsonl"))
    result = build_research_rag_graph("cpu_min", run_config=config, comparison_plan=True).invoke({
        "query": "Compare the duration in days between Alpha, Beta, and Gamma irrigation trials.",
        "pdf_paths": paths,
        "pdf_path": p1,
        "max_reflection_loops": 1,
    })

    schedules = [e["details"] for e in result["reasoning_trace"] if e.get("action") == "comparison_task_schedule"]
    assert len(schedules) >= 2
    first_schedule = schedules[0]
    assert first_schedule["status"] == "COMPLETED"
    assert set(first_schedule["required_sources"]) == {"alpha-trial", "beta-trial", "gamma-trial"}
    task_ids = [t["id"] for t in first_schedule["tasks"]]
    assert task_ids == ["retrieve-1", "retrieve-2", "retrieve-3", "read-1", "read-2", "read-3", "collect_evidence"]
    assert all(t["status"] == "COMPLETED" for t in first_schedule["tasks"])


def test_comparison_plan_unlocked_for_gemini_mode(tmp_path, monkeypatch):
    clear_pdf_cache()
    p1 = _make_pdf(tmp_path / "model-a.pdf", "Model A achieved 91.2% accuracy on the benchmark.")
    p2 = _make_pdf(tmp_path / "model-b.pdf", "Model B achieved 85.4% accuracy on the benchmark.")

    monkeypatch.setattr(settings, "GEMINI_DENSE_ENABLED", False)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-key")

    def fake_gemini(prompt="", system_instruction="", **kwargs):
        sys_l = (system_instruction or "").lower()
        p_l = (prompt or "").lower()
        if "evaluate each claim" in sys_l or "evaluate each claim" in p_l:
            return json.dumps({
                "results": [
                    {
                        "claim_id": "claim_1",
                        "status": "SUPPORT",
                        "quote": "Model A achieved 91.2% accuracy on the benchmark.",
                        "rationale": "Khớp trực tiếp nguồn Model A.",
                    },
                    {
                        "claim_id": "claim_2",
                        "status": "SUPPORT",
                        "quote": "Model B achieved 85.4% accuracy on the benchmark.",
                        "rationale": "Khớp trực tiếp nguồn Model B.",
                    },
                ]
            })
        if "critic" in sys_l or "chuyên gia kiểm định" in sys_l:
            return json.dumps({
                "is_supported": True,
                "Unsupported_claims": [],
                "Missing_key_points": [],
                "action": "approve",
                "confidence": 0.95,
            })
        return json.dumps({
            "answer": "Model A achieved 91.2% accuracy on the benchmark. [1] Model B achieved 85.4% accuracy on the benchmark. [2]",
            "claims": [
                {"id": "claim_1", "text": "Model A achieved 91.2% accuracy on the benchmark."},
                {"id": "claim_2", "text": "Model B achieved 85.4% accuracy on the benchmark."},
            ],
            "evidence_spans": [
                "Model A achieved 91.2% accuracy on the benchmark.",
                "Model B achieved 85.4% accuracy on the benchmark.",
            ],
        })

    monkeypatch.setattr(llm_client, "_call_gemini", fake_gemini)
    config = RunConfig.capture(corpus_path=str(tmp_path / "missing.jsonl"))
    result = build_research_rag_graph("gemini", run_config=config, comparison_plan=True).invoke({
        "query": "Compare Model A and Model B accuracy.",
        "pdf_paths": [p1, p2],
        "pdf_path": p1,
        "max_reflection_loops": 1,
    })

    schedules = [e["details"] for e in result["reasoning_trace"] if e.get("action") == "comparison_task_schedule"]
    assert len(schedules) >= 2
    assert schedules[0]["status"] == "COMPLETED"
    assert set(schedules[0]["required_sources"]) == {"model-a", "model-b"}
    assert schedules[-1]["stage"] == "review_and_audit"
    assert schedules[-1]["status"] == "COMPLETED"
    assert "91.2%" in result["final_report"] and "85.4%" in result["final_report"]


def test_run_coro_sync_works_inside_running_event_loop():
    async def outer():
        async def inner():
            await asyncio.sleep(0)
            return "nested_ok"
        return run_coro_sync(inner())

    assert asyncio.run(outer()) == "nested_ok"


def test_supervisor_llm_refinement_and_fail_open_fallback():
    def good_llm(prompt: str) -> str:
        return json.dumps({
            "intent": "compare",
            "complexity": "HIGH",
            "sub_queries": ["Qwen2.5-7B latency on T4", "Llama-3-8B latency on T4"],
            "required_entities": ["Qwen2.5-7B", "Llama-3-8B", "T4"],
            "is_comparative_query": True,
            "is_numeric_query": True,
            "alpha": 0.45,
            "top_k": 8,
        })

    plan = supervisor_planner.plan_query_with_llm(
        "Which model has lower latency on T4?",
        llm_fn=good_llm,
    )
    assert plan.intent == "compare"
    assert plan.is_comparative_query is True
    assert "Qwen2.5-7B latency on T4" in plan.sub_queries
    assert "Qwen2.5-7B" in plan.required_entities

    # Malformed/crashing LLM must fail open to deterministic plan
    def broken_llm(prompt: str) -> str:
        raise RuntimeError("provider down")

    fallback = supervisor_planner.plan_query_with_llm(
        "Compare Model A vs Model B accuracy",
        llm_fn=broken_llm,
    )
    assert fallback.intent == "compare"
    assert fallback.sub_queries


def test_fact_checker_handoff_to_scout_when_local_evidence_exhausted(tmp_path, monkeypatch):
    clear_pdf_cache()
    base_state = {
        "query": "What is the yield of Solar-X?",
        "retry_count": 0,
        "max_reflection_loops": 2,
        "fact_check_label": "not_enough_info",
        "rejected_claims": [{"claim_id": "c1", "text": "Solar-X yields 50%.", "status": "NOT_ENOUGH_INFO"}],
        "evidence_gaps": [{"missing_aspect": "Solar-X yield", "suggested_query": "Solar-X photovoltaic yield"}],
        "no_new_evidence": True,
        "source_units": [{"id": "u1", "text": "Unrelated text."}],
        "retrieved_docs": [{"id": "u1", "text": "Unrelated text."}],
        "reasoning_trace": [],
    }

    # Strictly disabled when allow_scout_handoff is False
    assert decide_handoff("fact_checker", {**base_state, "allow_scout_handoff": False}) is None

    # Emits bounded Scout handoff when allow_scout_handoff is True
    handoff = decide_handoff("fact_checker", {**base_state, "allow_scout_handoff": True})
    assert handoff is not None
    assert handoff.sender == "fact_checker"
    assert handoff.recipient == "scout"
    assert handoff.reason == "discover_additional_literature"

    # End-to-end graph check: Scout receives handoff, appends discovered PDF, and Retrieval runs on both
    p_orig = _make_pdf(tmp_path / "initial.pdf", "Photovoltaic panel design overview without field yield data.")
    p_found = _make_pdf(tmp_path / "solar-panel.pdf", "Photovoltaic panel yield reached 48% in field tests.")

    from agents.agent0_scout.node import ScoutAgent
    from modes.cpu_min.fact_checker import CPUMinFactCheckerAgent
    from modes.cpu_min.retrieval import CPUMinRetrievalAgent

    scout_queries = []
    original_run = ScoutAgent.run

    def fake_scout_run(self, state):
        if not state.get("pdf_paths") and not state.get("pdf_path"):
            scout_queries.append(state.get("query"))
            return {
                "scout_decision": "success",
                "pdf_path": p_found,
                "pdf_paths": [p_found],
                "reasoning_trace": [{"agent": "LiteratureScout", "action": "paper_discovered_and_ingested"}],
            }
        return original_run(self, state)

    monkeypatch.setattr(ScoutAgent, "run", fake_scout_run)

    orig_ret = CPUMinRetrievalAgent.run
    ret_calls = 0

    def tracked_ret(self, state, **kw):
        nonlocal ret_calls
        ret_calls += 1
        out = orig_ret(self, state, **kw)
        if ret_calls == 1:
            out["no_new_evidence"] = True
        return out

    monkeypatch.setattr(CPUMinRetrievalAgent, "run", tracked_ret)

    orig_fc = CPUMinFactCheckerAgent.run
    fc_calls = 0

    def tracked_fc(self, state):
        nonlocal fc_calls
        fc_calls += 1
        if fc_calls == 1:
            return {
                "fact_check_label": "not_enough_info",
                "verified_claims": [],
                "rejected_claims": [{"claim_id": "c1", "text": "Missing field yield.", "status": "NOT_ENOUGH_INFO"}],
                "evidence_gaps": [{"missing_aspect": "field yield", "suggested_query": "photovoltaic panel field yield"}],
                "reasoning_trace": [{"agent": "fact_checker", "action": "insufficient_local_evidence"}],
            }
        return orig_fc(self, state)

    monkeypatch.setattr(CPUMinFactCheckerAgent, "run", tracked_fc)

    config = RunConfig.capture(corpus_path=str(tmp_path / "missing.jsonl"))
    res = build_research_rag_graph("cpu_min", run_config=config, auto_scout=True).invoke({
        "query": "What is the photovoltaic panel yield in field tests?",
        "pdf_path": p_orig,
        "pdf_paths": [p_orig],
        "max_reflection_loops": 2,
    })

    assert scout_queries == ["photovoltaic panel field yield"]
    assert p_orig in res["pdf_paths"] and p_found in res["pdf_paths"]
    assert any(
        e.get("action") == "agent_handoff_received" and e.get("agent") == "scout"
        for e in res["reasoning_trace"]
    )
