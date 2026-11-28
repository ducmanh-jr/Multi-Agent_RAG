"""
Milestone M3 Regression Suite:
1. Bidirectional VI<->EN query expansion (including unaccented Vietnamese) & equation/table structural boost.
2. Conversational follow-up coreference resolution (`Còn trên tập test?` / `What about on the test set?`)
   inheriting entity pairs & metrics from `chat_history`.
3. SQLite checkpoint persistence & `ResearchRunService.resume()` preserving `chat_history`, `resolved_query`,
   `inherited_entities`, `answer_language`, and `document_scope`.
4. Unified `run_manifest` retrieval fallback & strategy observability across `gpu_finetune`, `hybrid`, `gemini`, `cpu_min`.
5. Live Gemini per-model permission probe (`200 verified`, `403 permission_denied`, `404 model_not_found`, `429 quota_exhausted`).
6. Full M3 dev `sparse -> hybrid -> hybrid_rerank` & 4-language matrix gate (`Recall@5 >= 0.90`, `Region Recall@5 >= 0.90`, `lang >= 0.98`).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import fitz
import httpx
import pytest

from agents.agent1_retrieval.tools.semantic_pipeline import (
    boost_entities_tables_and_equations,
    build_cross_lingual_subqueries,
)
from api.research_runs import CHECKPOINT_STATE_KEYS, ResearchRequest, ResearchRunService
from api.run_manifest import RUN_MANIFEST_ACTION, get_run_manifest, manifest_event
from core import db
from core.config import RunConfig, settings, use_run_config
from core.graph import orchestrator_router_node
from core.supervisor import supervisor_planner
from eval.m3_dev_evaluation import run_m3_dev_evaluation
from eval.retrieval_metrics import bbox_containment, bbox_iou, score_ranked_regions
from modes.cpu_min.retrieval import CPUMinRetrievalAgent
from modes.gpu_finetune.retrieval import GPUFinetuneRetrievalAgent
from utils.bilingual import expand_bilingual_tokens
from utils.llm_client import llm_client
from utils.local_vulkan_setup import _GEMINI_ACCESS_CACHE, probe_all_capabilities, probe_gemini_model_access


def test_bilingual_expansion_handles_four_directions_and_unaccented_vietnamese():
    # VI -> EN
    vi_en = expand_bilingual_tokens("Thử nghiệm tưới nhỏ giọt dùng bao nhiêu lít nước?", set())
    assert {"drip", "irrigation", "liters", "trial"} <= vi_en

    # Unaccented VI -> EN
    unaccented = expand_bilingual_tokens("Ket qua tuoi nho giot va do chinh xac cua cong thuc", set())
    assert {"drip", "accuracy", "equation"} <= unaccented

    # EN -> VI
    en_vi = expand_bilingual_tokens("How many liters did drip irrigation use in the greenhouse soil nitrate trial?", set())
    assert {"lít", "nhỏ giọt", "nhà kính", "nitrat", "đất"} <= en_vi

    # Structural equation/table subqueries
    en_corpus = [{"text": "The orchard trial ran for 21 consecutive days and used 42 liters."}]
    bridged = build_cross_lingual_subqueries("Trong phương trình (2), tốc độ học α lớn hơn 0?", en_corpus)
    assert any("(2)" in q for q in bridged)
    assert any("alpha" in q.lower() or "learning" in q.lower() for q in bridged)

    # Structural boost elevates matching equation number and table number ahead of generic text
    candidates = [
        {"id": "txt1", "unit_type": "text", "text": "General background discussion about optimization."},
        {"id": "eq1", "unit_type": "equation", "equation_number": "(1)", "text": "y = a/b (1)"},
        {"id": "eq2", "unit_type": "equation", "equation_number": "(2)", "text": "Q(s,a) <- Q(s,a) + alpha * delta (2)"},
    ]
    ranked = boost_entities_tables_and_equations("Explain equation (2) and variable α", candidates, [], top_k=3)
    assert ranked[0]["id"] == "eq2"


def test_conversational_coreference_inherits_method_pair_and_persists_in_checkpoint(tmp_path, research_db):
    chat_history = [
        {"role": "user", "content": "Trong Hình 3, phương pháp B hơn A bao nhiêu điểm phần trăm?"},
        {"role": "assistant", "content": "Phương pháp B đạt 92%, phương pháp A đạt 81%, cao hơn 11 điểm phần trăm."},
    ]
    coref = supervisor_planner.resolve_conversational_query("Còn trên tập test?", chat_history)
    assert coref["is_followup"] is True
    assert "Method B" in coref["inherited_entities"] and "Method A" in coref["inherited_entities"]
    assert "tập test" in coref["resolved_query"]
    assert "Method B" in coref["resolved_query"]

    # English follow-up works symmetrically
    en_coref = supervisor_planner.resolve_conversational_query(
        "What about on the test set?",
        [{"role": "user", "content": "Compare Method A and Method B validation accuracy."}],
    )
    assert en_coref["is_followup"] is True
    assert "Method A" in en_coref["resolved_query"] and "Method B" in en_coref["resolved_query"]

    # Router node writes resolved_query, inherited_entities, document_scope into state
    pdf_path = tmp_path / "study.pdf"
    with fitz.open() as doc:
        p1 = doc.new_page()
        p1.insert_text((60, 80), "Validation: Method A 81%, Method B 92%. Test: Method A 85%, Method B 94%.")
        doc.save(str(pdf_path))

    routed = orchestrator_router_node(
        {
            "query": "Còn trên tập test?",
            "chat_history": chat_history,
            "pdf_path": str(pdf_path),
            "pdf_paths": [str(pdf_path)],
            "answer_language": "vi",
        },
        execution_mode="cpu_min",
    )
    assert "Method B" in routed["resolved_query"]
    assert routed["document_scope"] == [{"path": str(pdf_path)}]
    assert "chat_history" in CHECKPOINT_STATE_KEYS
    assert "resolved_query" in CHECKPOINT_STATE_KEYS
    assert "document_scope" in CHECKPOINT_STATE_KEYS

    # Persist a mid-run checkpoint after orchestrator_router and verify ResearchRunService.resume() restores
    # chat_history, resolved_query, inherited_entities, answer_language, and document_scope intact!
    saved_doc = db.save_document(
        doc_id="doc_m3_study",
        name="study.pdf",
        path=str(pdf_path),
        size=pdf_path.stat().st_size,
        doc_type="pdf",
    )
    base_cfg = RunConfig.capture(remote_inference_url="", corpus_path=str(tmp_path / "empty.jsonl"))
    run_cfg = RunConfig({**base_cfg._values, "EXECUTION_MODE": "cpu_min"})
    admission = manifest_event(run_cfg, mode="cpu_min", auto_scout=False, scout_enabled=False, reflection_loops=1)
    src_run = db.save_run(
        run_id="run_m3_src",
        question="Còn trên tập test?",
        status="failed",
        reasoning_trace=[admission],
        documents=[{
            "id": saved_doc["id"],
            "name": saved_doc["name"],
            "path": saved_doc["path"],
            "sha256": "a" * 64,
        }],
    )
    db.save_run_checkpoint(
        src_run["id"],
        1,
        "orchestrator_router",
        {
            "query": "Còn trên tập test?",
            "chat_history": chat_history,
            "resolved_query": routed["resolved_query"],
            "query_with_context": routed["query_with_context"],
            "inherited_entities": routed["inherited_entities"],
            "answer_language": "vi",
            "query_language": "vi",
            "source_language": "en",
            "pdf_path": str(pdf_path),
            "pdf_paths": [str(pdf_path)],
            "document_scope": [{"id": saved_doc["id"], "name": "study.pdf", "path": str(pdf_path), "catalogued": True}],
            "sub_queries": routed["sub_queries"],
            "required_entities": routed["required_entities"],
            "retry_count": 0,
        },
    )

    service = ResearchRunService()

    async def _run_resume():
        resumed = await service.resume(src_run["id"], config=run_cfg, mode="cpu_min")
        return await resumed.result()

    res = asyncio.run(_run_resume())
    assert res.record["status"] == "completed"
    assert res.state["chat_history"] == chat_history
    assert res.state["resolved_query"] == routed["resolved_query"]
    assert res.state["answer_language"] == "vi"
    assert res.state["document_scope"][0]["id"] == saved_doc["id"]
    assert "94%" in res.state["answer"] or "85%" in res.state["answer"]


def test_run_manifest_captures_retrieval_strategy_and_fallback_across_modes(tmp_path, monkeypatch):
    pdf_path = tmp_path / "en_paper.pdf"
    with fitz.open() as doc:
        p1 = doc.new_page()
        p1.insert_text((60, 80), "Subsurface drip irrigation ran for 21 consecutive days and used 42 liters.")
        doc.save(str(pdf_path))

    # 1. GPU Finetune fallback from dense 503 -> sparse with full fallback_chain in manifest
    cfg = RunConfig.capture(remote_inference_url="https://worker.invalid", corpus_path=str(tmp_path / "none.jsonl"))
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503))) as client, use_run_config(cfg):
        monkeypatch.setattr(llm_client, "http_client", client)
        gpu_res = GPUFinetuneRetrievalAgent(enable_dense=True, enable_rerank=True, enable_remote_router=False).run({
            "query": "Tưới nhỏ giọt dùng bao nhiêu lít nước?",
            "pdf_paths": [str(pdf_path)],
            "top_k": 5,
        })
    # 2. CPU Min sparse retrieval trace also captured by run_manifest
    with use_run_config(cfg):
        cpu_res = CPUMinRetrievalAgent().run({
            "query": "Tưới nhỏ giọt dùng bao nhiêu lít nước?",
            "pdf_paths": [str(pdf_path)],
            "top_k": 5,
        })

    admission = manifest_event(cfg, mode="gpu_finetune", auto_scout=False, scout_enabled=False, reflection_loops=1)
    manifest = get_run_manifest({
        "id": "run_manifest_test",
        "status": "completed",
        "created_at": "2026-11-27T00:00:00Z",
        "documents": [],
        "reasoning_trace": [admission, *gpu_res["reasoning_trace"], *cpu_res["reasoning_trace"]],
    })
    attempts = manifest["observed"]["retrieval_attempts"]
    assert len(attempts) == 2
    assert attempts[0]["strategy_used"] == "sparse"
    assert attempts[0]["dense"]["status"] == "fallback"
    assert attempts[0]["rerank"]["status"] == "skipped"
    assert any(f["stage"] == "dense" for f in attempts[0]["fallback_chain"])
    assert attempts[0]["cross_lingual_queries"]
    assert attempts[1]["action"] == "cpu_min_agentic_retrieval_completed"
    assert attempts[1]["strategy_used"] == "sparse"


def test_gemini_live_model_permission_probe_verifies_http_status(monkeypatch):
    _GEMINI_ACCESS_CACHE.clear()
    key = "AIzaSyTestKey1234567890"

    # 200 OK with generateContent supported -> verified
    with httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]})
    )) as ok_client:
        res_ok = probe_gemini_model_access(key, "gemini-2.5-flash", http_client=ok_client)
    assert res_ok["ok"] is True and res_ok["status"] == "verified"

    # 403 Permission Denied -> ok=False
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(403))) as deny_client:
        res_403 = probe_gemini_model_access(key, "gemini-4-carbon", http_client=deny_client)
    assert res_403["ok"] is False and res_403["status"] == "permission_denied"

    # 404 Model Not Found -> ok=False
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(404))) as nf_client:
        res_404 = probe_gemini_model_access(key, "gemini-nonexistent", http_client=nf_client)
    assert res_404["ok"] is False and res_404["status"] == "model_not_found"

    # probe_all_capabilities reflects denied permission instead of falsely reporting online=True!
    base_cfg = RunConfig.capture()
    cfg = RunConfig({**base_cfg._values, "GEMINI_API_KEY": key, "DEFAULT_LLM_MODEL": "gemini-4-carbon"})
    with use_run_config(cfg):
        monkeypatch.setattr(
            "utils.local_vulkan_setup.probe_gemini_model_access",
            lambda *a, **kw: {"ok": False, "status": "permission_denied", "model": "gemini-4-carbon", "http_status": 403, "detail": "denied"},
        )
        caps = probe_all_capabilities(force_refresh=True, live_gemini_probe=True)
    assert caps["gemini"]["online"] is False
    assert caps["gemini"]["model_access_status"] == "permission_denied"


def test_region_level_retrieval_metrics_and_m3_dev_gate():
    assert bbox_iou([0, 0, 100, 100], [0, 0, 100, 100]) == pytest.approx(1.0)
    assert bbox_containment([0, 0, 200, 200], [50, 50, 150, 150]) == pytest.approx(1.0)

    r_scores = score_ranked_regions(
        [
            ("doc.pdf", {"page_no": 1, "unit_type": "text", "bbox": [0, 0, 100, 40], "text": "Intro text"}),
            ("doc.pdf", {"page_no": 1, "unit_type": "equation", "bbox": [180, 130, 460, 170], "text": "y = a/b (1)"}),
        ],
        [
            {"document": "doc.pdf", "page": 1, "unit_type": "equation", "bbox": [185, 135, 455, 165], "anchors": ["y = a/b"], "grade": 2},
        ],
    )
    assert r_scores["region_recall_at_5"] == 1.0
    assert r_scores["region_mrr_at_10"] == 0.5

    # Full M3 dev evaluation across sparse -> hybrid -> hybrid_rerank & 4 language directions
    summary = run_m3_dev_evaluation()
    assert summary["gate_m3_passed"] is True
    for strat in ("sparse", "hybrid", "hybrid_rerank"):
        ov = summary["strategies"][strat]["overall"]
        assert ov["recall_at_5"] >= 0.90, f"{strat} page Recall@5 too low: {ov}"
        assert ov["region_recall_at_5"] >= 0.90, f"{strat} region Recall@5 too low: {ov}"
    assert summary["qa_4lang_matrix"]["language_accuracy"] >= 0.98
    for direction in ("VI->VI", "VI->EN", "EN->VI", "EN->EN"):
        assert summary["strategies"]["hybrid_rerank"]["by_direction"][direction]["region_recall_at_5"] >= 0.90
