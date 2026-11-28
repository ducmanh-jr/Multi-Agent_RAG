"""Phase 0 regression tests verifying all 10 audited bug fixes."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import httpx
import pytest

from core.config import settings
from core.coordination import (
    AgentDecision,
    evaluate_agent_handoff,
    handoff_budget_category,
    next_handoff_budgets,
    supervisor_evaluate_decision,
)
from core.provider_router import (
    is_cross_family_verifier,
    is_gemini_permitted,
    load_hybrid_provider_policy,
    provider_router,
)
from eval.evaluate_30_deep_questions import format_scorecard_rating
from modes.registry import get_agent_class, normalize_mode
from utils.llm_client import UnifiedLLMClient


def test_0_1_deepseek_uses_dedicated_deepseek_model_never_gemini_fallback(monkeypatch):
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "ds-test-key")
    monkeypatch.setattr(settings, "DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setattr(settings, "FALLBACK_LLM_MODEL", "gemini-3.5-flash-lite")

    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "DeepSeek OK"}}]},
        )

    client = UnifiedLLMClient()
    monkeypatch.setattr(client, "http_client", httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        out = client._call_openai_compatible("Hello", "Be concise", 0.1, 256, False)
        assert out == "DeepSeek OK"
        assert captured[0]["model"] == "deepseek-chat"
        assert "gemini" not in captured[0]["model"].lower()
    finally:
        client.http_client.close()


def test_0_2_strict_local_roles_and_cross_family_fact_checker(monkeypatch):
    monkeypatch.setattr(settings, "HYBRID_PROVIDER_PROFILE", "gemini_first")
    caps = {
        "gemini": {"online": True},
        "gpu_finetune": {"online": True},
        "kaggle_max": {"online": True, "has_vllm_32b": True},
    }

    # STRICT_LOCAL_ROLES must never be permitted to call Gemini in hybrid mode
    for strict_role in ("fact_checker", "embedding", "reranker", "citation_audit"):
        assert is_gemini_permitted("hybrid", role=strict_role) is False

    # Hybrid policy override cannot promote strict roles to Gemini
    policy = load_hybrid_provider_policy(
        {"fact_checker": "gemini", "embedding": "gemini", "reranker": "gemini", "citation_audit": "gemini"}
    )
    for strict_role in ("fact_checker", "embedding", "reranker", "citation_audit"):
        assert policy[strict_role] != "gemini"

    # fact_checker chain must NEVER include gemini
    chain_gemini_gen = provider_router.resolve_chain(
        "fact_checker",
        state={"generator_model_family": "gemini_flash"},
        caps=caps,
    )
    assert "gemini" not in chain_gemini_gen
    assert chain_gemini_gen[0] == "gpu_finetune"

    # Cross-family guard blocks Qwen-32B verifier when generator is Qwen family
    chain_qwen_gen = provider_router.resolve_chain(
        "fact_checker",
        state={"generator_model_family": "qwen3_8b"},
        caps=caps,
    )
    assert "gemini" not in chain_qwen_gen
    assert "gpu_max" not in chain_qwen_gen
    assert is_cross_family_verifier("gemini_flash", "gemini") is False
    assert is_cross_family_verifier("qwen2.5_7b", "gpu_max") is False
    assert is_cross_family_verifier("qwen2.5_7b", "gpu_finetune") is True


def test_0_3_call_gemini_uses_native_system_instruction_thinking_config_and_max_tokens_retry(monkeypatch):
    monkeypatch.setattr(settings, "DEFAULT_LLM_MODEL", "gemini-3.5-flash")
    monkeypatch.setattr(settings, "FALLBACK_LLM_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-key")

    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        payloads.append(body)
        # First call simulates MAX_TOKENS truncation -> should retry with larger maxOutputTokens
        if len(payloads) == 1:
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "finishReason": "MAX_TOKENS",
                            "content": {"parts": [{"text": '{"answer": "trunc'}]},
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": '{"answer": "ok"}'}]},
                    }
                ]
            },
        )

    client = UnifiedLLMClient()
    monkeypatch.setattr(client, "http_client", httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        out = client._call_gemini(
            "Return JSON",
            system_instruction="Strict JSON only.",
            max_tokens=256,
            json_mode=True,
        )
        assert out == '{"answer": "ok"}'
        assert len(payloads) == 2
        # Native systemInstruction (not spoofed user/model turns)
        assert payloads[0]["systemInstruction"] == {"parts": [{"text": "Strict JSON only."}]}
        assert len(payloads[0]["contents"]) == 1
        assert payloads[0]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
        # Retry bumped maxOutputTokens
        assert payloads[1]["generationConfig"]["maxOutputTokens"] > payloads[0]["generationConfig"]["maxOutputTokens"]
    finally:
        client.http_client.close()

    # Generic 400 INVALID_ARGUMENT (as returned by gemini-3.5-flash-lite without the word "thinking")
    # must strip thinkingConfig, cache the model in _no_thinking_budget_models, and succeed on retry.
    lite_payloads: list[dict] = []

    def lite_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        lite_payloads.append(body)
        if "thinkingConfig" in body.get("generationConfig", {}):
            return httpx.Response(
                400,
                json={"error": {"code": 400, "message": "Request contains an invalid argument.", "status": "INVALID_ARGUMENT"}},
            )
        return httpx.Response(
            200,
            json={
                "modelVersion": "gemini-3.5-flash-lite",
                "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": '{"ok": true}'}]}}],
            },
        )

    client_lite = UnifiedLLMClient()
    monkeypatch.setattr(client_lite, "http_client", httpx.Client(transport=httpx.MockTransport(lite_handler)))
    try:
        assert client_lite._call_gemini("say ok", json_mode=True) == '{"ok": true}'
        assert len(lite_payloads) == 2
        assert "thinkingConfig" not in lite_payloads[1]["generationConfig"]
        # Subsequent call remembers the model doesn't accept thinkingBudget=0 (1 call only)
        assert client_lite._call_gemini("say ok 2", json_mode=True) == '{"ok": true}'
        assert len(lite_payloads) == 3
        assert "thinkingConfig" not in lite_payloads[2]["generationConfig"]
    finally:
        client_lite.http_client.close()


def test_0_4_kaggle_base_revision_reads_pinned_manifest_with_env_override(monkeypatch, tmp_path):
    server = importlib.import_module("kaggle.kaggle_finetune_server")
    manifest = tmp_path / "weights_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "base_models": {
                    "qwen_7b_base": {
                        "model_id": "Qwen/Qwen2.5-7B-Instruct",
                        "revision": "d4094e97c9b83b6b15802fb8ce335a9f6d4d14b4",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("BASE_QWEN_REVISION", raising=False)
    assert (
        server._resolve_pinned_base_revision("Qwen/Qwen2.5-7B-Instruct", manifest_path=manifest)
        == "d4094e97c9b83b6b15802fb8ce335a9f6d4d14b4"
    )
    assert server._resolve_pinned_base_revision("Qwen/Qwen3-8B", manifest_path=manifest) is None

    monkeypatch.setenv("BASE_QWEN_REVISION", "override-commit-123")
    assert (
        server._resolve_pinned_base_revision("Qwen/Qwen2.5-7B-Instruct", manifest_path=manifest)
        == "override-commit-123"
    )


def test_0_5_orchestrator_router_fresh_run_matches_dispatch_next_node(monkeypatch):
    graph_mod = importlib.import_module("core.graph")
    visited: list[str] = []

    def fake_router(state, **kwargs):
        visited.append("router")
        return {
            "reasoning_trace": [
                {
                    "action": "agent_handoff",
                    "sender": "supervisor",
                    "recipient": "retrieval",
                    "reason": "skip_scout_with_selected_pdfs",
                    "signature": "sig_skip_scout",
                }
            ]
        }

    def fake_scout(self, state):
        visited.append("scout")
        return {}

    runners = {
        "retrieval": lambda s: visited.append("retrieval") or {},
        "reader": lambda s: visited.append("reader") or {"source_units": [{"id": "u1", "text": "x"}]},
        "summarizer": lambda s: visited.append("summarizer") or {"answer": "x"},
        "fact_checker": lambda s: visited.append("fact_checker") or {"fact_check_label": "support"},
        "writer": lambda s: visited.append("writer") or {"final_report": "x"},
    }
    monkeypatch.setattr(graph_mod, "orchestrator_router_node", fake_router)
    monkeypatch.setattr(graph_mod, "get_agent_runners", lambda mode: runners)
    monkeypatch.setattr(graph_mod.ScoutAgent, "run", fake_scout)

    app = graph_mod.build_research_rag_graph("cpu_min")
    app.invoke({"query": "Test", "pdf_paths": ["a.pdf"]})
    # Fresh run obeys supervisor->retrieval handoff just like resume dispatch_next_node!
    assert visited[:2] == ["router", "retrieval"]
    assert "scout" not in visited


def test_0_6_per_category_budgets_protect_fact_checker_from_starvation():
    # Budget limit = 2: Reader consumes 1 evidence handoff -> Summarizer cannot consume the last slot
    # reserved for Fact-Checker; Fact-Checker still retains its contradiction repair slot!
    budgets_after_reader = next_handoff_budgets({}, "evidence", fallback_total=0)
    state_after_reader = {
        "query": "What is latency?",
        "retry_count": 1,
        "max_reflection_loops": 2,
        "handoff_budgets": budgets_after_reader,
        "source_units": [{"id": "u1", "text": "Hybrid-X p99 latency is 14.6 ms."}],
        "critic_action": "rewrite",
        "critic_feedback": {"issues_found": ["polish wording"]},
    }
    sum_handoff, sum_audit = evaluate_agent_handoff("summarizer", state_after_reader)
    assert sum_handoff is None
    assert sum_audit is not None and sum_audit["rejection_reason"] == "verifier_budget_reserved"

    # Now Fact-Checker detects a contradiction and still has budget!
    fc_state = {
        **state_after_reader,
        "fact_check_label": "contradict",
        "rejected_claims": [{"claim_text": "Latency is 99.9 ms", "status": "CONTRADICT"}],
    }
    fc_handoff, fc_audit = evaluate_agent_handoff("fact_checker", fc_state)
    assert fc_handoff is not None
    assert fc_handoff.recipient == "summarizer"
    assert fc_audit is not None and fc_audit["approved"] is True
    assert fc_audit["budget_category"] == "verify"

    # Explicit category budget exhaustion is enforced independently
    dec = AgentDecision(
        source_agent="reader",
        action="request_more_evidence",
        target_agent="retrieval",
        reason="missing_table",
        round_index=1,
    )
    assert handoff_budget_category(dec) == "evidence"
    h, a = supervisor_evaluate_decision(
        dec,
        {
            "retry_count": 1,
            "max_reflection_loops": 3,
            "handoff_budgets": {"evidence": 1, "total": 1},
            "handoff_budget_limits": {"evidence": 1, "total": 3},
        },
        expected_sender="reader",
    )
    assert h is None
    assert a is not None and a["rejection_reason"] == "category_budget_exhausted:evidence"


def test_0_7_registry_rejects_unknown_mode_but_preserves_empty_default():
    assert normalize_mode(None) == "cpu_min"
    assert normalize_mode("") == "cpu_min"
    assert normalize_mode("gpu_ft") == "gpu_finetune"
    assert normalize_mode("cpu_max") == "cpu_max"

    with pytest.raises(ValueError, match="Unknown execution mode"):
        normalize_mode("typo_mode_xyz")
    with pytest.raises(ValueError, match="Unknown execution mode"):
        get_agent_class("retrieval", "nonexistent_mode")


def test_0_8_scorecard_ratings_reflect_actual_scores():
    low_badge, _ = format_scorecard_rating(0.32)
    high_badge, _ = format_scorecard_rating(0.95)
    slow_badge, _ = format_scorecard_rating(22.5, metric_kind="latency_s")
    fast_badge, _ = format_scorecard_rating(0.4, metric_kind="latency_s")
    assert "Cần cải thiện" in low_badge
    assert "Xuất sắc" not in low_badge
    assert "Cao" in high_badge
    assert "Chậm" in slow_badge
    assert "Rất nhanh" in fast_badge


def test_kaggle_worker_bounded_queue_returns_503_with_retry_after_and_health_queue_metrics(monkeypatch):
    from fastapi import HTTPException

    server = importlib.import_module("kaggle.kaggle_finetune_server")
    monkeypatch.setenv("QWEN_MAX_QUEUE_DEPTH", "1")
    monkeypatch.setenv("QWEN_LOCK_WAIT_SECONDS", "0.05")
    monkeypatch.setattr(server, "_qwen_queue_depth", 1)
    try:
        with pytest.raises(HTTPException) as exc_info:
            server.generate_with_qwen("hello")
        assert exc_info.value.status_code == 503
        assert "Retry-After" in (exc_info.value.headers or {})
    finally:
        monkeypatch.setattr(server, "_qwen_queue_depth", 0)

    health = server.healthcheck()
    assert "queue" in health
    assert health["queue"]["max_queue_depth"] == 1
    assert "avg_tokens_per_sec" in health["queue"]


def test_kaggle_client_endpoint_timeouts_and_isolated_circuit_breaker(monkeypatch):
    monkeypatch.setattr(settings, "REMOTE_INFERENCE_URL", "https://kaggle.isolated.test")
    monkeypatch.setattr(settings, "REMOTE_TIMEOUT_SECONDS", 90)
    client = UnifiedLLMClient()
    assert client._endpoint_timeout("api/router", None) == 20.0
    assert client._endpoint_timeout("api/factcheck", None) == 30.0
    assert client._endpoint_timeout("api/summarize/generator", None) == 90.0

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/summarize/generator"):
            return httpx.Response(503, headers={"Retry-After": "6"}, json={"detail": "busy"})
        return httpx.Response(200, json={"results": [{"status": "SUPPORT"}]})

    monkeypatch.setattr(client, "http_client", httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        # Generator 503 trips ONLY the generator circuit breaker
        with pytest.raises(RuntimeError, match="HTTP 503"):
            client.call_kaggle_specialist("api/summarize/generator", {"query": "q", "context": "c"})

        # GPU 1 factcheck endpoint stays completely unaffected!
        fc_res = client.call_kaggle_specialist("api/factcheck", {"claims": ["a"], "context": "a"}, allow_generate_fallback=False)
        assert fc_res == {"results": [{"status": "SUPPORT"}]}

        # Immediate second generator call fails fast from per-endpoint circuit breaker
        with pytest.raises(RuntimeError, match="Circuit Breaker"):
            client.call_kaggle_specialist("api/summarize/generator", {"query": "q", "context": "c"})
    finally:
        client.http_client.close()


def test_cross_lingual_target_guard_and_cache_isolation(monkeypatch):
    from core import tools as tools_mod
    from core.verifier import review_abstention
    from utils.domain_guard import check_out_of_domain

    greenhouse_units = [
        {
            "id": "gh_p1",
            "doc_id": "greenhouse-trial",
            "page_no": 1,
            "text": "Greenhouse irrigation trial\nThe greenhouse trial ran for 35 days. The trial measured soil moisture at noon.",
        }
    ]
    orchard_units = [
        {
            "id": "or_p1",
            "doc_id": "orchard-trial",
            "page_no": 1,
            "text": "Orchard irrigation trial\nThe orchard trial ran for 21 days. The trial measured soil moisture at sunrise.",
        }
    ]

    # Pollute global _PDF_CHUNK_CACHE with an unrelated PDF containing "nitrate" to prove isolation.
    monkeypatch.setitem(
        tools_mod._PDF_CHUNK_CACHE,
        "/tmp/unrelated_other_paper.pdf",
        [{"doc_id": "unrelated_other_paper", "text": "Soil nitrate concentration reached 12.5 mg/L."}],
    )

    # 1. Cross-lingual unanswerable trap (VI query asking for nitrate concentration on EN duration-only doc)
    trap_q = "Thí nghiệm trong nhà kính đo nồng độ nitrat trong đất là bao nhiêu?"
    is_ood, reason = check_out_of_domain(trap_q, greenhouse_units)
    assert is_ood is True
    assert reason.startswith("cross_lingual_target_absent:")

    # 2. Fact-Checker review_abstention accepts justified cross-lingual missing-target abstention
    fc = review_abstention(
        {
            "query": trap_q,
            "answer": "Thông tin hoàn toàn không được đề cập trong tài liệu.",
            "answer_status": "NOT_FOUND_IN_DOCUMENT",
            "is_out_of_domain": True,
            "source_units": greenhouse_units,
        }
    )
    assert fc is not None
    assert fc["fact_check_label"] == "support"
    assert fc["rejected_claims"] == []

    # 3. Valid cross-lingual query (VI duration question on EN doc) must NOT false-positive
    valid_q = "Thí nghiệm tưới nước trong vườn cây kéo dài bao nhiêu ngày?"
    is_ood_valid, reason_valid = check_out_of_domain(valid_q, orchard_units)
    assert is_ood_valid is False
    assert reason_valid == "cross_lingual_in_domain"

    # 4. Valid cross-lingual nitrate query when English doc actually contains nitrate concentration
    nitrate_en_units = [
        {
            "id": "gh_nit",
            "doc_id": "greenhouse-nitrate",
            "page_no": 1,
            "text": "The greenhouse trial measured an average soil nitrate concentration of 12.5 mg/L.",
        }
    ]
    is_ood_present, _ = check_out_of_domain(trap_q, nitrate_en_units)
    assert is_ood_present is False


def test_offline_cpu_pilot_trap_and_table_regression(tmp_path):
    from eval.run_benchmark import load_pilot, run_mode_evaluation

    questions = load_pilot(tmp_path)
    for regime in ("cold", "warm_no_cache"):
        summary = run_mode_evaluation("cpu_min", questions, documents_dir=tmp_path, cache_regime=regime)
        assert summary["errors"] == []
        assert summary["scored_questions"] == 4
        assert summary["cache_hit_total"] == 0
        by_id = {r["id"]: r for r in summary["records"]}
        assert by_id["pilot-unanswerable-vi"]["abstention_accuracy"] == 1.0
        assert by_id["pilot-unanswerable-vi"]["hallucination_rate"] == 0.0
        assert by_id["pilot-single-vi"]["numeric_exactness"] == 1.0
        assert by_id["pilot-multi-en"]["numeric_exactness"] == 1.0
        assert by_id["pilot-table-en"]["numeric_exactness"] == 1.0


def test_kaggle_server_health_and_routes_work_without_torch(monkeypatch):
    server = importlib.import_module("kaggle.kaggle_finetune_server")
    monkeypatch.setattr(server, "torch", None)
    assert server._cuda_available() is False
    assert server._cuda_device_count() == 0
    assert server.get_bnb_config() is None
    health = server.healthcheck()
    assert health["gpus_available"] == 0
    assert "queue" in health


def test_find_page_tables_robust_recovers_vector_tables_when_pymupdf_layout_misses(tmp_path, monkeypatch):
    import fitz
    from core.pdf_ingest import _extract_page_table_entries_robust

    pdf_path = tmp_path / "compact_grid.pdf"
    with fitz.open() as doc:
        p = doc.new_page(width=595, height=842)
        x_cols = [48.0, 188.0, 308.0, 428.0, 543.0]
        rows = [["Method-Gamma", "87%", "83%", "18.2 ms"], ["Method-Delta", "89%", "85%", "16.4 ms"]]
        y0, rh = 68.0, 28.0
        for r_i in range(len(rows) + 1):
            p.draw_line((x_cols[0], y0 + r_i * rh), (x_cols[-1], y0 + r_i * rh), color=(0, 0, 0), width=0.8)
        for x in x_cols:
            p.draw_line((x, y0), (x, y0 + len(rows) * rh), color=(0, 0, 0), width=0.8)
        for r_i, row in enumerate(rows):
            for c_i, val in enumerate(row):
                p.insert_text((x_cols[c_i] + 5, y0 + r_i * rh + 18), val, fontsize=9)
        doc.save(pdf_path, no_new_id=True)

    with fitz.open(pdf_path) as doc:
        page = doc[0]
        globs = page.get_layout.__func__.__globals__
        # 1. Simulate legacy pymupdf_layout ONNX returning an empty layout_information list
        monkeypatch.setitem(globs, "_get_layout", lambda _pg: [])
        entries = _extract_page_table_entries_robust(page)
        assert len(entries) == 1
        _bbox, extracted = entries[0]
        assert extracted[0][0] == "Method-Gamma"
        assert extracted[1][0] == "Method-Delta"

    # 2. Simulate PyMuPDF 1.28+/2.0 `use_layout=True` returning empty while `use_layout=False` returns the grid
    class FakeTable:
        def __init__(self, bbox, rows):
            self.bbox = bbox
            self._rows = rows

        def extract(self):
            return self._rows

    class FakeTabs:
        def __init__(self, tables):
            self.tables = tables

    class FakePyMuPDF2Page:
        derotation_matrix = fitz.Identity

        def get_drawings(self):
            return [{"rect": (48, 68, 543, 124)}]

        def find_tables(self, *, use_layout=True, **kwargs):
            if use_layout:
                return FakeTabs([])
            return FakeTabs([FakeTable((48, 68, 543, 124), [["Method-Gamma", "87%", "83%", "18.2 ms"]])])

    modern_entries = _extract_page_table_entries_robust(FakePyMuPDF2Page())
    assert len(modern_entries) == 1
    assert modern_entries[0][1][0][0] == "Method-Gamma"

    # 3. Simulate PyMuPDF 2.0 ONNX returning a loose padded bbox around the same compact 2-row table
    class FakePyMuPDF2LooseBoxPage:
        derotation_matrix = fitz.Identity

        def get_drawings(self):
            return [{"rect": (48, 420, 543, 476)}]

        def find_tables(self, *, use_layout=True, **kwargs):
            if use_layout:
                return FakeTabs([FakeTable((40, 360, 555, 520), [["Hardware", "VRAM (GB)", "TDP (W)", "Nodes"], ["Tesla-T4", "16 GB", "70 W", "2"]])])
            return FakeTabs([FakeTable((48, 420, 543, 476), [["Hardware", "VRAM (GB)", "TDP (W)", "Nodes"], ["Tesla-T4", "16 GB", "70 W", "2"]])])

    deduped_entries = _extract_page_table_entries_robust(FakePyMuPDF2LooseBoxPage())
    assert len(deduped_entries) == 1
    assert deduped_entries[0][0] == [48.0, 420.0, 543.0, 476.0]




