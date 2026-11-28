"""Regression & integration tests for Phase 2.7, Phase 3, and Phase 4 upgrades:
  - BaseAgent bounded tool loop & `run_task`
  - Controlled parallel Hub execution (`parallel_dispatch_node` via `TaskScheduler`)
  - Async-safe `RunnableLambda` graph nodes (`ainvoke` + `astream` non-blocking + sync `invoke` parity)
  - Role-based `max_tokens` caps (`config`, `llm_client.generate_json`, `kaggle_finetune_server`)
  - Authenticated `POST /api/config/remote-url/register` & `kaggle/run_kaggle.py` auto-registration + `hybrid_sota_base`
  - `disabled_agent_handoffs` per-agent autonomy ablation & held-out 40-pair NLI harness
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import fitz
import pytest

MAIN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MAIN_DIR.parent
for p in (str(MAIN_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _create_pdf(path: Path, pages: list[str]) -> str:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return str(path)


# ---------------------------------------------------------------------------
# 1. BaseAgent Bounded Tool Loop & `run_task`
# ---------------------------------------------------------------------------


def test_base_agent_bounded_tool_loop_and_run_task():
    from core.agent_protocol import AgentTask
    from core.base_agent import BaseAgent

    class DemoSpecialist(BaseAgent):
        def __init__(self):
            super().__init__(name="DemoSpecialist")
            self.seen_queries: list[str] = []

        def run(self, state):
            self.seen_queries.append(str(state.get("query") or ""))
            return {"answer": f"handled:{state.get('query')}"}

    agent = DemoSpecialist()

    # Early completion + error isolation
    calls: list[str] = []

    def tool_search(q: str) -> list[str]:
        calls.append(f"search:{q}")
        return [f"hit({q})"]

    def tool_flaky() -> str:
        calls.append("flaky")
        raise RuntimeError("transient_tool_error")

    def planner(step_idx: int, history: list[dict[str, Any]]):
        if step_idx == 0:
            return ("search", {"q": "table 4"})
        if step_idx == 1:
            return ("flaky", {})
        return None

    loop_res = agent.run_bounded_tool_loop({"search": tool_search, "flaky": tool_flaky}, planner, max_steps=3)
    assert loop_res["stopped_reason"] == "completed"
    assert loop_res["step_count"] == 2
    assert loop_res["tools_used"] == ["search", "flaky"]
    assert loop_res["steps"][0]["ok"] is True
    assert loop_res["steps"][1]["ok"] is False
    assert "RuntimeError" in loop_res["steps"][1]["error"]

    # Hard cap enforcement (even if planner never returns None, capped at max_steps <= 4)
    capped = agent.run_bounded_tool_loop(
        {"search": tool_search},
        lambda idx, hist: ("search", {"q": f"step_{idx}"}),
        max_steps=99,
    )
    assert capped["step_count"] == 4
    assert capped["max_steps"] == 4
    assert capped["stopped_reason"] == "max_steps_reached"

    # Unknown tool stops safely
    unknown = agent.run_bounded_tool_loop(
        {"search": tool_search},
        lambda idx, hist: ("missing_tool", {}),
        max_steps=3,
    )
    assert unknown["stopped_reason"] == "unknown_tool"
    assert unknown["steps"][0]["ok"] is False

    # `run_task` scopes corrective handoff focus cleanly
    task = AgentTask(
        role="retrieval",
        objective="Fetch Table 4",
        sender="reader",
        reason="missing_table_reference",
        focus=("Table 4 throughput",),
        budget_category="read",
        round_index=1,
    )
    out = agent.run_task({"query": "Original question", "sub_queries": ["Original question"]}, task)
    assert out["answer"] == "handled:Original question"


# ---------------------------------------------------------------------------
# 2. Controlled Parallel Hub Multi-Task Dispatch via `TaskScheduler`
# ---------------------------------------------------------------------------


def test_hub_parallel_multi_task_dispatch_via_task_scheduler(tmp_path: Path):
    from core.graph import build_research_rag_graph

    pdf_a = _create_pdf(
        tmp_path / "paper_alpha.pdf",
        ["Paper Alpha introduces AlphaNet achieving 91.2% accuracy on Benchmark X with 12 ms latency."],
    )
    pdf_b = _create_pdf(
        tmp_path / "paper_beta.pdf",
        ["Paper Beta introduces BetaNet achieving 94.6% accuracy on Benchmark X with 18 ms latency."],
    )

    app = build_research_rag_graph(
        mode="cpu_min",
        auto_scout=False,
        parallel_retrieval=True,
        supervisor_mode="hub",
    )
    final_state = app.invoke({
        "query": "Compare AlphaNet and BetaNet accuracy and latency on Benchmark X.",
        "pdf_path": pdf_a,
        "pdf_paths": [pdf_a, pdf_b],
        "retry_count": 0,
        "max_reflection_loops": 1,
    })

    trace = final_state.get("reasoning_trace") or []
    messages = final_state.get("agent_messages") or []

    parallel_dispatches = [
        e for e in trace
        if isinstance(e, dict)
        and e.get("action") == "supervisor_hub_dispatched"
        and e.get("next_node") == "parallel_dispatch_node"
    ]
    batch_done = [
        e for e in trace
        if isinstance(e, dict) and e.get("action") == "supervisor_parallel_batch_completed"
    ]
    parallel_msgs = [
        m for m in messages
        if isinstance(m, dict) and m.get("kind") == "parallel_task_completed"
    ]

    assert parallel_dispatches, "SupervisorHub must dispatch parallel_dispatch_node when parallel_retrieval=True and >=2 tasks"
    assert batch_done and batch_done[0]["completed_tasks"] >= 2
    assert len(parallel_msgs) >= 2
    assert final_state.get("final_report"), "Hub parallel run must still complete full pipeline to final_report"


# ---------------------------------------------------------------------------
# 3. Async `RunnableLambda` Non-Blocking Graph Execution + Sync Parity
# ---------------------------------------------------------------------------


def test_async_graph_nodes_non_blocking_and_sync_parity(tmp_path: Path):
    from core.graph import build_research_rag_graph

    pdf_path = _create_pdf(
        tmp_path / "async_test.pdf",
        ["The Transformer base model has 6 encoder layers, 8 attention heads, and d_model = 512."],
    )
    app = build_research_rag_graph(mode="cpu_min", auto_scout=False, supervisor_mode="hub")
    init_state = {
        "query": "How many encoder layers and attention heads does the base Transformer have?",
        "pdf_path": pdf_path,
        "pdf_paths": [pdf_path],
        "retry_count": 0,
        "max_reflection_loops": 1,
    }

    sync_res = app.invoke(dict(init_state))

    async def _run_async_and_heartbeat():
        ticks = 0
        done_flag = False

        async def _heartbeat():
            nonlocal ticks
            while not done_flag:
                ticks += 1
                await asyncio.sleep(0.002)

        hb = asyncio.create_task(_heartbeat())
        seen_nodes: list[str] = []
        last_final_report = ""
        try:
            async for update in app.astream(dict(init_state), stream_mode="updates"):
                if isinstance(update, dict):
                    seen_nodes.extend(update.keys())
                    for val in update.values():
                        if isinstance(val, dict) and val.get("final_report"):
                            last_final_report = str(val["final_report"])
        finally:
            done_flag = True
            await hb
        return seen_nodes, last_final_report, ticks

    nodes, async_report, ticks = asyncio.run(_run_async_and_heartbeat())
    assert "orchestrator_router" in nodes
    assert "supervisor_node" in nodes
    assert "writer_node" in nodes
    assert ticks >= 1, "Event loop heartbeat must advance while graph nodes run off-thread"
    assert bool(async_report.strip()) == bool(str(sync_res.get("final_report") or "").strip())
    assert "6" in async_report and "512" in async_report


# ---------------------------------------------------------------------------
# 4. Role-Based `max_tokens` Caps
# ---------------------------------------------------------------------------


def test_role_max_tokens_caps_across_config_client_and_kaggle(monkeypatch):
    from core.config import ROLE_MAX_TOKENS, get_role_max_tokens
    from utils.llm_client import UnifiedLLMClient
    from kaggle import kaggle_finetune_server as kfs

    assert ROLE_MAX_TOKENS["router"] == 384
    assert ROLE_MAX_TOKENS["arbiter"] == 256
    assert ROLE_MAX_TOKENS["critic"] == 512
    assert ROLE_MAX_TOKENS["writer"] == 512
    assert ROLE_MAX_TOKENS["generator"] == 768
    assert get_role_max_tokens("CriticAgent") == 512
    assert get_role_max_tokens("unknown_role", default=600) == 600

    # `generate_json` accepts explicit `max_tokens` (used by SupervisorHub arbiter) and defaults by role
    client = UnifiedLLMClient()
    captured: list[int] = []

    def _fake_generate(prompt, system_instruction=None, temperature=0.2, max_tokens=2048, json_mode=False, agent=None):
        captured.append(max_tokens)
        return '{"next_role": "writer"}'

    monkeypatch.setattr(client, "generate", _fake_generate)
    parsed = client.generate_json("pick", agent="arbiter", max_tokens=256)
    assert parsed["next_role"] == "writer"
    assert captured[-1] == 256

    client.generate_json("pick", agent="critic")
    assert captured[-1] == 512

    # Kaggle Pydantic schema defaults reflect P0-hardened role caps & hybrid_sota_base canonical profile
    assert kfs.RouterRequest(query="q").max_new_tokens == 384
    assert kfs.CriticRequest(query="q", answer="a", context="c").max_tokens == 512
    assert kfs.WriterRequest(query="q").max_tokens == 512
    assert kfs.SummarizeRequest(query="q", context="c").max_tokens == 768
    assert kfs.GenerateRequest(prompt="p").max_tokens == 1024
    assert kfs._canonical_worker_profile("hybrid_sota_base") == "qa_dual_gpu"


# ---------------------------------------------------------------------------
# 5. Authenticated Auto-Tunnel Registration & `hybrid_sota_base` Preset
# ---------------------------------------------------------------------------


def test_authenticated_tunnel_register_endpoint_and_kaggle_helper(monkeypatch):
    from api import app as api_app
    from core.config import settings
    from utils.llm_client import llm_client
    from kaggle import run_kaggle

    # Direct async invocation avoids Windows asyncio.socketpair() TestClient hang
    orig_url = settings.REMOTE_INFERENCE_URL
    orig_tok = settings.REMOTE_INFERENCE_TOKEN

    class _FakeRequest:
        def __init__(self, headers: dict[str, str]):
            self.headers = headers

    try:
        monkeypatch.setattr(api_app, "_write_env_settings", lambda vals: None)

        # 503 when neither TUNNEL_REGISTER_SECRET nor REMOTE_INFERENCE_TOKEN is configured
        monkeypatch.delenv("TUNNEL_REGISTER_SECRET", raising=False)
        settings.REMOTE_INFERENCE_TOKEN = ""
        with pytest.raises(api_app.HTTPException) as exc_503:
            asyncio.run(api_app.register_kaggle_tunnel(
                api_app.RegisterTunnelRequest(remote_url="https://demo.trycloudflare.com"),
                _FakeRequest({}),
            ))
        assert exc_503.value.status_code == 503

        # 401 on wrong secret
        monkeypatch.setenv("TUNNEL_REGISTER_SECRET", "super-secret-xyz")
        with pytest.raises(api_app.HTTPException) as exc_401:
            asyncio.run(api_app.register_kaggle_tunnel(
                api_app.RegisterTunnelRequest(remote_url="https://demo.trycloudflare.com"),
                _FakeRequest({"x-tunnel-secret": "wrong"}),
            ))
        assert exc_401.value.status_code == 401

        # 400 on cloud metadata host or non-https external host
        with pytest.raises(api_app.HTTPException) as exc_ssrf:
            asyncio.run(api_app.register_kaggle_tunnel(
                api_app.RegisterTunnelRequest(remote_url="https://169.254.169.254"),
                _FakeRequest({"x-tunnel-secret": "super-secret-xyz"}),
            ))
        assert exc_ssrf.value.status_code == 400

        with pytest.raises(api_app.HTTPException) as exc_http:
            asyncio.run(api_app.register_kaggle_tunnel(
                api_app.RegisterTunnelRequest(remote_url="http://evil.example.com"),
                _FakeRequest({"x-tunnel-secret": "super-secret-xyz"}),
            ))
        assert exc_http.value.status_code == 400

        # 200 on valid secret + https URL; updates settings & clears circuit breaker
        llm_client._circuit_breaker_until["https://old.example::api/factcheck"] = time.time() + 60.0
        res = asyncio.run(api_app.register_kaggle_tunnel(
            api_app.RegisterTunnelRequest(
                remote_url="https://fresh-worker.trycloudflare.com/",
                remote_token="new-inference-tok",
                worker_profile="hybrid_sota_base",
                git_commit_sha="f3b146e",
            ),
            _FakeRequest({"x-tunnel-secret": "super-secret-xyz"}),
        ))
        assert res["status"] == "registered"
        assert res["remote_worker"] == "https://fresh-worker.trycloudflare.com"
        assert res["token_updated"] is True
        assert settings.REMOTE_INFERENCE_URL == "https://fresh-worker.trycloudflare.com"
        assert settings.REMOTE_INFERENCE_TOKEN == "new-inference-tok"
        assert llm_client._circuit_breaker_until == {}
    finally:
        settings.REMOTE_INFERENCE_URL = orig_url
        settings.REMOTE_INFERENCE_TOKEN = orig_tok

    # Test `hybrid_sota_base` env defaults (non-destructive)
    env_box = {"BASE_QWEN_MODEL_ID": "Qwen/Qwen3-14B"}  # explicit user override must be preserved
    applied = run_kaggle.apply_profile_env_defaults("hybrid_sota_base", env=env_box)
    assert env_box["BASE_QWEN_MODEL_ID"] == "Qwen/Qwen3-14B"
    assert env_box["QWEN_USE_BASE_INSTRUCT"] == "true"
    assert env_box["PREFER_HF_SOTA_NLI"] == "true"
    assert env_box["EMBEDDING_MODEL"] == "BAAI/bge-m3"
    assert env_box["RERANKER_MODEL"] == "BAAI/bge-reranker-v2-m3"
    assert "BASE_QWEN_MODEL_ID" not in applied

    # Test `register_tunnel_with_backend` no-op & success path
    noop = run_kaggle.register_tunnel_with_backend("https://t.trycloudflare.com", "tok", callback_url="")
    assert noop["registered"] is False
    assert noop["reason"] == "callback_url_not_configured"

    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.read.return_value = json.dumps({"status": "registered"}).encode("utf-8")
    fake_resp.__enter__.return_value = fake_resp
    fake_resp.__exit__.return_value = False
    captured_reqs: list[Any] = []

    def _fake_urlopen(req, timeout=10.0):
        captured_reqs.append(req)
        return fake_resp

    monkeypatch.setattr(run_kaggle.urllib.request, "urlopen", _fake_urlopen)
    ok = run_kaggle.register_tunnel_with_backend(
        "https://fresh.trycloudflare.com",
        "inf-tok-123",
        callback_url="http://127.0.0.1:8000",
        register_secret="reg-sec-999",
        worker_profile="hybrid_sota_base",
        git_commit_sha="f3b146e",
    )
    assert ok["registered"] is True
    assert captured_reqs[0].full_url == "http://127.0.0.1:8000/api/config/remote-url/register"
    assert captured_reqs[0].headers.get("X-tunnel-secret") == "reg-sec-999"


# ---------------------------------------------------------------------------
# 6. Per-Agent Autonomy Ablation & Held-Out 40-Pair NLI Harness
# ---------------------------------------------------------------------------


def test_disabled_agent_handoffs_and_per_agent_ablation(tmp_path: Path):
    from core.coordination import AgentDecision, supervisor_evaluate_decision
    from eval.ablation_study import run_per_agent_autonomy_ablation

    decision = AgentDecision(
        source_agent="reader",
        action="request_more_evidence",
        target_agent="retrieval",
        reason="missing_table_reference",
        focus=("Table 4",),
        round_index=0,
    )
    approved, audit = supervisor_evaluate_decision(
        decision,
        {
            "query": "What is in Table 4?",
            "retry_count": 0,
            "max_reflection_loops": 2,
            "disabled_agent_handoffs": ["reader"],
        },
        expected_sender="reader",
    )
    assert approved is None
    assert audit is not None
    assert audit["approved"] is False
    assert audit["rejection_reason"] == "agent_handoff_disabled:reader"

    pdf_path = _create_pdf(
        tmp_path / "ablation_doc.pdf",
        ["Method Z achieves 88.5% accuracy on Dataset Q with 10 ms latency."],
    )
    ablation = run_per_agent_autonomy_ablation(
        [
            {
                "id": "q1",
                "question": "What accuracy does Method Z achieve on Dataset Q?",
                "pdf_paths": [pdf_path],
                "gold_keywords": ["88.5%"],
                "is_unanswerable": False,
            }
        ],
        roles=["reader", "writer"],
        mode="cpu_min",
        supervisor_mode="hub",
        repeats=1,
    )
    assert ablation["experiment_type"] == "per_agent_autonomy_ablation"
    assert set(ablation["per_role"].keys()) == {"reader", "writer"}
    assert ablation["per_role"]["reader"]["total_questions"] == 1
    assert "reader_handoff_off" in ablation["per_role"]["reader"]
    assert "reader_handoff_on" in ablation["per_role"]["reader"]


def test_nli_holdout_40_dataset_and_metrics():
    from eval.nli_holdout_eval import (
        LABELS,
        compute_multiclass_metrics,
        evaluate_nli_predictor,
        load_nli_holdout_dataset,
        predict_with_local_deterministic_verifier,
    )

    dataset = load_nli_holdout_dataset()
    assert len(dataset) == 40
    label_counts = {lbl: sum(1 for x in dataset if x["gold_label"] == lbl) for lbl in LABELS}
    assert label_counts == {"SUPPORT": 14, "CONTRADICT": 14, "NOT_ENOUGH_INFO": 12}

    # Sanity check multiclass metric math
    m = compute_multiclass_metrics(
        ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO", "CONTRADICT"],
        ["SUPPORT", "CONTRADICT", "SUPPORT", "CONTRADICT"],
    )
    assert m["sample_count"] == 4
    assert m["correct"] == 3
    assert m["accuracy"] == 0.75

    # Run real local deterministic verifier on the full 40-pair holdout set
    report = evaluate_nli_predictor(
        dataset,
        predict_with_local_deterministic_verifier,
        engine_name="local_deterministic_verifier",
    )
    assert report["metrics"]["sample_count"] == 40
    assert 0.0 <= report["metrics"]["macro_f1"] <= 1.0
    assert set(report["by_language"].keys()) >= {"en", "vi", "bilingual"}


# ---------------------------------------------------------------------------
# 7. Default Hub Mode Promotion, Real Specialist Tool Loops & API Config
# ---------------------------------------------------------------------------


def test_default_hub_mode_real_specialist_tool_loops_and_api_controls(tmp_path: Path, monkeypatch):
    from api import app as api_app
    from core.config import settings
    from core.graph import build_research_rag_graph

    pdf_path = _create_pdf(
        tmp_path / "hub_default_tools.pdf",
        ["Our system achieves 93.4% accuracy on SciBench using hybrid retrieval and cross-family verification."],
    )

    # Default `build_research_rag_graph("cpu_min")` runs in Hub-and-Spoke mode without passing supervisor_mode
    app = build_research_rag_graph("cpu_min", auto_scout=False)
    out = app.invoke({
        "query": "What accuracy does the system achieve on SciBench?",
        "pdf_path": pdf_path,
        "pdf_paths": [pdf_path],
        "retry_count": 0,
        "max_reflection_loops": 1,
    })
    assert out["supervisor_mode"] == "hub"
    trace = out.get("reasoning_trace") or []
    assert any(isinstance(e, dict) and e.get("action") == "supervisor_hub_dispatched" for e in trace)

    # Real specialist tool loops record `tools_used` in trace details
    ret_events = [e for e in trace if isinstance(e, dict) and e.get("action") == "cpu_min_agentic_retrieval_completed"]
    read_events = [e for e in trace if isinstance(e, dict) and e.get("action") == "cpu_min_reading_completed"]
    assert ret_events and "fts5_bm25_search" in (ret_events[0].get("details") or {}).get("tools_used", [])
    assert read_events and "sanitize_source_units" in (read_events[0].get("details") or {}).get("tools_used", [])
    assert "table_completeness_audit" in (read_events[0].get("details") or {}).get("tools_used", [])

    # Explicit `supervisor_mode="pipeline"` remains available for A/B ablation
    pipe_app = build_research_rag_graph("cpu_min", auto_scout=False, supervisor_mode="pipeline")
    pipe_out = pipe_app.invoke({
        "query": "What accuracy does the system achieve on SciBench?",
        "pdf_path": pdf_path,
        "pdf_paths": [pdf_path],
        "retry_count": 0,
        "max_reflection_loops": 1,
    })
    assert pipe_out["supervisor_mode"] == "pipeline"

    # API config & supervisor-mode toggle + 7-entry `/api/agents` catalog
    orig_sup = settings.SUPERVISOR_MODE
    try:
        monkeypatch.setattr(api_app, "_write_env_settings", lambda vals: None)
        cfg = api_app.get_config()
        assert cfg["supervisor_mode"] in {"hub", "pipeline"}
        agents_payload = api_app.get_agents()
        roles = [a["role"] for a in agents_payload["agents"]]
        assert roles[:2] == ["SupervisorHub", "ScoutAgent"]
        assert len(roles) == 7

        toggled = api_app.set_supervisor_mode(api_app.SetSupervisorModeRequest(supervisor_mode="pipeline"))
        assert toggled["supervisor_mode"] == "pipeline"
        restored = api_app.set_supervisor_mode(api_app.SetSupervisorModeRequest(supervisor_mode="hub"))
        assert restored["supervisor_mode"] == "hub"
        with pytest.raises(api_app.HTTPException) as bad_exc:
            api_app.set_supervisor_mode(api_app.SetSupervisorModeRequest(supervisor_mode="invalid"))
        assert bad_exc.value.status_code == 400
    finally:
        settings.SUPERVISOR_MODE = orig_sup
