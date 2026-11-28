# -*- coding: utf-8 -*-
"""
Milestone M5 Verification Suite:
1. Kaggle notebook & boot.sh security hardening (zero hardcoded secrets, pinned 40-char SHA URL & checkout, LF endings).
2. Strict Admin vs Inference Bearer token separation on kaggle_supervisor_gateway.py.
3. Three canonical Kaggle worker profiles (`qa_dual_gpu`, `ingest`, `32b_tp2`) + peak per-GPU VRAM + reproducible runtime manifest.
4. Sealed 12-PDF / 36-question multi-document holdout disjointness, 12-PDF simultaneous retrieval & fan-out accuracy.
5. Full 3-cache-regime (`cold`, `warm_no_cache`, `warm_cached`) evaluation across `hybrid_new`, `kaggle_only`,
   `gemini_direct`, `cpu_min`, and `oracle_evidence` plus disconnect resume & Gemini outage resilience.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from eval.m5_holdout_evaluation import (
    build_m5_holdout_corpus,
    run_m5_holdout_evaluation,
    run_twelve_pdf_corpus_stress_test,
    verify_holdout_disjointness,
)
from kaggle import kaggle_supervisor_gateway as gw
from kaggle.run_kaggle import configure_runtime_tokens, normalize_worker_profile
from kaggle.worker_manager import GPUVRAMProbe, WorkerManager

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_m5_notebook_and_boot_sh_have_no_hardcoded_secrets_and_pin_sha():
    nb_path = REPO_ROOT / "kaggle" / "kaggle_master_worker.ipynb"
    nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source") or []) for cell in nb_data.get("cells") or [])

    # Zero leaked / hardcoded tokens
    assert "multiagent" not in source.lower()
    assert 'REMOTE_INFERENCE_TOKEN"] =' not in source
    # Kaggle Secrets + distinct Admin & Inference tokens
    assert "UserSecretsClient" in source
    assert "SUPERVISOR_ADMIN_TOKEN" in source
    assert "SUPERVISOR_API_TOKEN" in source
    assert 'SEPARATE_ADMIN_INFERENCE_TOKENS"] = "true"' in source
    # Pinned 40-char commit SHA for BOTH curl boot.sh URL AND repo checkout
    assert 'REQUIRE_PINNED_COMMIT"] = "true"' in source
    assert "/tatmanh/kaggle/boot.sh" not in source
    assert "/{GIT_COMMIT_SHA}/kaggle/boot.sh" in source
    assert re.search(r'"[0-9a-fA-F]{40}"', source), "Notebook must contain a 40-char pinned commit SHA"
    # Documents all 3 canonical worker profiles
    for profile in ("qa_dual_gpu", "ingest", "32b_tp2"):
        assert profile in source

    # boot.sh checks: LF-only line endings + strict SHA validation + detached checkout
    boot_bytes = (REPO_ROOT / "kaggle" / "boot.sh").read_bytes()
    assert b"\r\n" not in boot_bytes, "kaggle/boot.sh must use LF line endings for Linux/Kaggle bash"
    boot_text = boot_bytes.decode("utf-8")
    assert "GIT_COMMIT_SHA" in boot_text
    assert "REQUIRE_PINNED_COMMIT" in boot_text
    assert "validate_sha40" in boot_text
    assert "checkout -q --detach" in boot_text
    assert "ACTIVE_GIT_COMMIT_SHA" in boot_text


def test_m5_gateway_strict_admin_vs_inference_token_separation(monkeypatch):
    monkeypatch.setattr(gw, "ADMIN_TOKEN", None)
    monkeypatch.setattr(gw, "INFERENCE_TOKEN", None)
    monkeypatch.setattr(gw, "PUBLIC_TUNNEL", True)
    monkeypatch.setenv("SEPARATE_ADMIN_INFERENCE_TOKENS", "true")
    monkeypatch.setattr(gw, "RATE_LIMIT_REQUESTS", 0)
    gw.worker_manager.transition_state = "ready"
    gw.worker_manager.active_worker = "gpu_finetune"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{"ok": true}'
    mock_resp.headers = {"content-type": "application/json"}

    client = TestClient(gw.app, raise_server_exceptions=False)

    # 1. Identical admin & inference tokens -> fail closed with 503 on both admin and inference
    monkeypatch.setenv("SUPERVISOR_ADMIN_TOKEN", "identical-token-123")
    monkeypatch.setenv("SUPERVISOR_API_TOKEN", "identical-token-123")
    assert client.get("/api/admin/status", headers={"Authorization": "Bearer identical-token-123"}).status_code == 503
    assert client.post("/api/embed", json={"texts": ["x"]}, headers={"Authorization": "Bearer identical-token-123"}).status_code == 503

    # 2. Missing inference token -> inference endpoint NEVER falls back to admin token
    monkeypatch.setenv("SUPERVISOR_ADMIN_TOKEN", "admin-only-secret-AAA")
    monkeypatch.delenv("SUPERVISOR_API_TOKEN", raising=False)
    monkeypatch.delenv("REMOTE_INFERENCE_TOKEN", raising=False)
    assert client.post("/api/embed", json={"texts": ["x"]}, headers={"Authorization": "Bearer admin-only-secret-AAA"}).status_code == 503

    # 3. Distinct tokens -> strict least-privilege separation
    monkeypatch.setenv("SUPERVISOR_ADMIN_TOKEN", "admin-only-secret-AAA")
    monkeypatch.setenv("SUPERVISOR_API_TOKEN", "infer-only-secret-BBB")
    with patch.object(gw.worker_manager, "get_status_payload", return_value={"runtime_status": "running"}), \
         patch.object(gw._http_client, "request", new_callable=AsyncMock, return_value=mock_resp):
        # Admin endpoint accepts admin token, rejects inference token
        assert client.get("/api/admin/status", headers={"Authorization": "Bearer admin-only-secret-AAA"}).status_code == 200
        assert client.get("/api/admin/status", headers={"Authorization": "Bearer infer-only-secret-BBB"}).status_code == 403
        # Inference endpoint accepts inference token, rejects admin token
        assert client.post("/api/embed", json={"texts": ["x"]}, headers={"Authorization": "Bearer infer-only-secret-BBB"}).status_code == 200
        assert client.post("/api/embed", json={"texts": ["x"]}, headers={"Authorization": "Bearer admin-only-secret-AAA"}).status_code == 403


def test_m5_run_kaggle_configure_tokens_and_three_worker_profiles(monkeypatch):
    # Auto-generation creates TWO DISTINCT 32-hex tokens
    env_auto: dict[str, str] = {}
    meta_auto = configure_runtime_tokens(env=env_auto, secret_loader=lambda _name: False)
    assert meta_auto["tokens_distinct"] is True
    assert meta_auto["separation_enforced"] is True
    assert len(env_auto["SUPERVISOR_ADMIN_TOKEN"]) == 32
    assert len(env_auto["SUPERVISOR_API_TOKEN"]) == 32
    assert env_auto["SUPERVISOR_ADMIN_TOKEN"] != env_auto["SUPERVISOR_API_TOKEN"]

    # Identical tokens under strict separation fail fast
    with pytest.raises(RuntimeError, match="không được trùng nhau"):
        configure_runtime_tokens(
            env={
                "SUPERVISOR_ADMIN_TOKEN": "same-secret",
                "SUPERVISOR_API_TOKEN": "same-secret",
                "SEPARATE_ADMIN_INFERENCE_TOKENS": "true",
            },
            secret_loader=lambda _name: False,
        )

    # All 3 canonical M5 profiles + legacy aliases normalize cleanly
    assert normalize_worker_profile("qa_dual_gpu")["initial_mode"] == "gpu_finetune"
    assert normalize_worker_profile("ingest")["prefetch_qwen"] is False
    assert normalize_worker_profile("32b_tp2")["initial_mode"] == "gpu_max"
    assert normalize_worker_profile("gpu_finetune")["canonical_profile"] == "qa_dual_gpu"
    with pytest.raises(ValueError):
        normalize_worker_profile("unknown_profile")

    # Finetune server supports `ingest` profile (ready without Qwen 8B) and reports git SHA
    srv_path = REPO_ROOT / "kaggle" / "kaggle_finetune_server.py"
    spec = importlib.util.spec_from_file_location("m5_kaggle_finetune_server", srv_path)
    srv = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(srv)
    monkeypatch.setenv("KAGGLE_WORKER_PROFILE", "ingest")
    monkeypatch.setenv("ACTIVE_GIT_COMMIT_SHA", "5993b5aa760f508a93bc3143cea95f443d9c5f1d")
    monkeypatch.setattr(srv, "_models", {"embedding": object()})
    monkeypatch.setattr(srv, "_loaded_adapters", [])
    health = srv.healthcheck()
    assert health["ready"] is True
    assert health["canonical_profile"] == "ingest"
    assert health["git_commit_sha"] == "5993b5aa760f508a93bc3143cea95f443d9c5f1d"
    assert health["capabilities"]["layout_reader"] is True


def test_m5_dual_kaggle_session_manifest_reproducibility_and_vram_peaks(monkeypatch):
    pinned_sha = "5993b5aa760f508a93bc3143cea95f443d9c5f1d"
    monkeypatch.setenv("GIT_COMMIT_SHA", pinned_sha)
    monkeypatch.setenv("ACTIVE_GIT_COMMIT_SHA", pinned_sha)
    monkeypatch.setenv("KAGGLE_WORKER_PROFILE", "qa_dual_gpu")
    monkeypatch.setenv("SEPARATE_ADMIN_INFERENCE_TOKENS", "true")
    monkeypatch.setenv("SUPERVISOR_ADMIN_TOKEN", "top-secret-admin-xyz")
    monkeypatch.setenv("SUPERVISOR_API_TOKEN", "top-secret-infer-abc")

    vram_samples = [
        {0: {"used_mb": 2048.0, "total_mb": 16384.0, "free_mb": 14336.0}, 1: {"used_mb": 1024.0, "total_mb": 16384.0, "free_mb": 15360.0}},
        {0: {"used_mb": 9216.0, "total_mb": 16384.0, "free_mb": 7168.0}, 1: {"used_mb": 5120.0, "total_mb": 16384.0, "free_mb": 11264.0}},
        {0: {"used_mb": 6144.0, "total_mb": 16384.0, "free_mb": 10240.0}, 1: {"used_mb": 3072.0, "total_mb": 16384.0, "free_mb": 13312.0}},
    ]
    sample_iter = iter(vram_samples)
    with patch.object(GPUVRAMProbe, "get_vram_info", side_effect=lambda: next(sample_iter, vram_samples[-1])):
        wm_session_1 = WorkerManager(project_dir=str(REPO_ROOT))
        status_high = wm_session_1.get_status_payload()
        status_after_drop = wm_session_1.get_status_payload()

    wm_session_2 = WorkerManager(project_dir=str(REPO_ROOT))
    # Two independent Kaggle sessions produce byte-identical canonical runtime manifests
    assert wm_session_1.resolve_runtime_manifest() == wm_session_2.resolve_runtime_manifest()
    assert status_high["runtime_manifest"]["git_commit_sha"] == pinned_sha
    assert status_high["runtime_manifest"]["git_commit_pinned"] is True
    assert status_high["runtime_manifest"]["canonical_profile"] == "qa_dual_gpu"
    assert status_high["runtime_manifest"]["token_separation_enforced"] is True
    # Peak per-GPU VRAM persists after current usage drops
    assert status_after_drop["vram_peak_gb"]["gpu_0"] == 9.0
    assert status_after_drop["vram_peak_gb"]["gpu_1"] == 5.0
    assert status_after_drop["vram_allocation"]["gpu_0"]["used_gb"] == 6.0
    # Zero credential leakage in status payload
    serialized = json.dumps(status_after_drop)
    assert "top-secret-admin-xyz" not in serialized
    assert "top-secret-infer-abc" not in serialized


def test_m5_sealed_twelve_pdf_holdout_disjointness_and_global_retrieval_accuracy(tmp_path):
    pdf_paths, pdf_hashes, questions = build_m5_holdout_corpus(tmp_path / "holdout_12")
    assert len(pdf_paths) == 12
    assert len(questions) == 36

    disjoint = verify_holdout_disjointness(set(pdf_paths.keys()), pdf_hashes)
    assert disjoint["disjoint"] is True
    assert disjoint["overlap_filenames"] == []
    assert disjoint["overlap_arxiv_ids"] == []
    assert disjoint["all_holdout_hashes_unique"] is True

    stress = run_twelve_pdf_corpus_stress_test(pdf_paths)
    assert stress["total_pdfs"] == 12
    assert stress["top5_rate"] == 1.0, f"12-PDF global retrieval misses: {stress['misses']}"
    assert stress["top3_rate"] >= 11 / 12
    assert stress["fanout_status"] == "completed"
    assert stress["fanout_completed_documents"] == 12
    assert stress["fanout_all_12_sources_preserved"] is True


def test_m5_holdout_three_cache_regimes_multi_system_and_operational_gates(tmp_path):
    report_md = tmp_path / "M5_HOLDOUT_REPORT.md"
    report_json = tmp_path / "m5_holdout.json"
    result = run_m5_holdout_evaluation(
        regimes=("cold", "warm_no_cache", "warm_cached"),
        write_report_path=report_md,
        write_json_path=report_json,
    )

    assert result["gate_passed"] is True, f"M5 gate failed: {result['gate_checks']}"
    assert report_md.is_file() and report_json.is_file()

    # Check all 3 cache regimes ran independently
    for reg in ("cold", "warm_no_cache", "warm_cached"):
        assert reg in result["systems_by_regime"]
        for sys_name in ("hybrid_new", "kaggle_only", "gemini_direct", "cpu_min"):
            assert sys_name in result["systems_by_regime"][reg]
            assert result["systems_by_regime"][reg][sys_name]["fallback_cache_rows"] == 0

    # Cache behavior verified on gemini_direct (uses llm_client.generate cache):
    # cold & warm_no_cache have 0 cache hits; warm_cached has positive cache hits!
    assert result["systems_by_regime"]["cold"]["gemini_direct"]["overall"]["cache_hits_total"] == 0
    assert result["systems_by_regime"]["warm_no_cache"]["gemini_direct"]["overall"]["cache_hits_total"] == 0
    assert result["systems_by_regime"]["warm_cached"]["gemini_direct"]["overall"]["cache_hits_total"] > 0

    # Zero Gemini calls in kaggle_only mode
    assert result["systems_by_regime"]["warm_no_cache"]["kaggle_only"]["provider_call_counts"].get("gemini", 0) == 0

    # Cluster-bootstrap 95% CI populated across 12 document families
    ci = result["systems_by_regime"]["warm_no_cache"]["hybrid_new"]["overall"]["three_layer_cluster_ci95"]
    assert ci["clusters"] == 12
    assert ci["n"] == 36
    assert ci["ci95_low"] <= ci["mean"] <= ci["ci95_high"]

    # Operational M5 gates verified
    ops = result["operational_gates"]
    assert ops["dual_kaggle_session_reproducible"] is True
    assert ops["disconnect_resume_verified"] is True
    assert ops["gemini_outage_fallback_verified"] is True
    assert ops["gemini_outage_fallback_cache_rows"] == 0
