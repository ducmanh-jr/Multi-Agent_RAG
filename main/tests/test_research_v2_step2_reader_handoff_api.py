"""
Research V2 — Step 2 Real Autonomous Reader Handoff Acceptance Suite (NO fake_reader / NO fake_retrieval):
Tests the REAL FastAPI `/api/runs` pipeline (real Retrieval -> real Reader -> real Supervisor ->
real Summarizer -> real Fact-Checker -> real Writer) on REAL PyMuPDF multi-page PDFs across 4 cases:

  1. Initial evidence already contains Table 4 -> answers accurately with ZERO Reader handoff.
  2. Page 1 says "see Table 4" while Table 4 is buried on Page 7 -> real Reader emits AgentDecision,
     Supervisor approves, real Retrieval fetches Page 7, real Reader reads Table 4, and the final
     report states the exact row/column/value/page (Page 7, Hybrid-X, p99 = 14.6).
  3. Document references Table 4, but Table 4 is completely absent -> stops within reflection budget
     with `no_new_evidence` and clearly states the value is unverified / insufficient evidence.
  4. Two PDFs report conflicting values for the same metric -> attributes each value to its own
     document and surfaces the conflict/discrepancy without conflating sources.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import fitz
import pytest
from fastapi.testclient import TestClient

from api import app as api_module, auth
from core import db
from core.tools import clear_pdf_cache


def _make_html_pdf(path: Path, html_pages: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open() as doc:
        for html in html_pages:
            page = doc.new_page(width=595, height=842)
            page.insert_htmlbox(
                (48, 48, 547, 794),
                f'<div style="font-family:sans-serif; font-size:10.5pt; line-height:1.4;">{html}</div>',
            )
        doc.save(str(path))
    return path


def _poll_run(client: TestClient, run_id: str, timeout: float = 25.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/api/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] != "processing":
            return data
        time.sleep(0.03)
    pytest.fail(f"Run {run_id} did not complete within {timeout}s")


@pytest.fixture
def real_api_client(monkeypatch, research_db, tmp_path):
    clear_pdf_cache()
    monkeypatch.setattr(auth, "AUTH_SECRET_PASSWORD", "v2-step2-secret")
    monkeypatch.setattr(auth, "AUTH_DEV_MODE", False)
    monkeypatch.setattr(auth, "_sessions", {})
    monkeypatch.setattr(api_module.settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(api_module.settings, "CORPUS_PATH", str(tmp_path / "empty.jsonl"))
    monkeypatch.setattr(api_module, "_legacy_mode_override", "cpu_min")
    with TestClient(api_module.app, base_url="https://workspace.test") as client:
        assert client.post("/api/auth/login", json={"password": "v2-step2-secret"}).status_code == 200
        yield client
    clear_pdf_cache()


_TABLE_4_HTML = """
<p><b>Table 4: Empirical Latency Breakdown</b></p>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse; width:100%;">
  <tr><th>Engine</th><th>p50 (ms)</th><th>p99 (ms)</th></tr>
  <tr><td>Baseline-V1</td><td>22.1</td><td>48.5</td></tr>
  <tr><td>Hybrid-X</td><td>8.2</td><td>14.6</td></tr>
  <tr><td>Dense-Only</td><td>15.4</td><td>31.9</td></tr>
</table>
"""


def test_step2_case1_sufficient_initial_evidence_answers_with_zero_reader_handoff(
    real_api_client: TestClient, tmp_path: Path
):
    pdf = _make_html_pdf(
        tmp_path / "case1_inline_table4.pdf",
        [
            f"<h3>Latency Evaluation</h3>"
            f"<p>We evaluated Engine tail latency under production load as shown below.</p>"
            f"{_TABLE_4_HTML}",
            "<p>Section 5 discusses deployment tooling and logging configuration.</p>",
        ],
    )
    db.save_document("doc_case1", pdf.name, str(pdf), pdf.stat().st_size)

    resp = real_api_client.post(
        "/api/runs",
        json={
            "question": "What is the p99 latency of Hybrid-X in Table 4?",
            "document_ids": ["doc_case1"],
            "mode": "cpu_min",
            "max_iterations": 2,
            "auto_scout": False,
            "parallel_retrieval": False,
        },
    )
    assert resp.status_code == 200
    run = _poll_run(real_api_client, resp.json()["id"])
    assert run["status"] == "completed"

    trace = run["reasoning_trace"]
    reader_handoffs = [e for e in trace if e.get("action") == "agent_handoff" and e.get("sender") == "reader"]
    assert reader_handoffs == [], f"Expected zero Reader handoffs when Table 4 is already present, got: {reader_handoffs}"

    report = run["final_report"]
    assert "Hybrid-X" in report
    assert "14.6" in report
    assert any((ref.get("page") or ref.get("page_no")) == 1 for ref in run.get("citation_references") or [])


def test_step2_case2_real_reader_handoff_fetches_distant_table4_page_via_supervisor(
    real_api_client: TestClient, tmp_path: Path
):
    """
    Page 1 mentions Hybrid-X p99 latency under production load and says 'see Table 4'.
    Pages 2-6 and 8-10 discuss Hybrid-X production load p99 latency methodology (no numbers),
    filling initial top-k prose hits while compact Table 4 sits on Page 7.
    Real Reader MUST detect missing Table 4, propose AgentDecision -> Supervisor approves ->
    Real Retrieval fetches Page 7 -> Real Reader reads Table 4 -> final report cites Page 7 & 14.6!
    """
    prose_pages = [
        (
            "<h3>Section 1: Production Load Latency Overview</h3>"
            "<p>This paper evaluates the p99 latency of Hybrid-X under production load across multi-tenant clusters. "
            "For exact empirical p99 latency measurements of Hybrid-X under production load, please see Table 4.</p>"
        ),
        (
            "<h3>Section 2: Production Load Traffic Model for Hybrid-X</h3>"
            "<p>Our production load generator replays anonymized search queries to profile p99 latency of Hybrid-X "
            "without warm-cache bias. All production load traces keep arrival intervals steady while measuring Hybrid-X p99 latency.</p>"
        ),
        (
            "<h3>Section 3: Tail Latency Instrumentation for Hybrid-X</h3>"
            "<p>To capture p99 latency of Hybrid-X under production load accurately, each worker records monotonic timestamps "
            "at request admission and response flush. Hybrid-X production load experiments isolate queuing delay from parsing overhead.</p>"
        ),
        (
            "<h3>Section 4: Cluster Hardware for Hybrid-X Production Load</h3>"
            "<p>Every Hybrid-X production load run executes on isolated nodes so p99 latency reflects deterministic scheduling "
            "rather than noisy-neighbor contention. Production load monitors verify zero packet loss during Hybrid-X p99 latency tests.</p>"
        ),
        (
            "<h3>Section 5: Warmup Protocol for Hybrid-X</h3>"
            "<p>Before recording Hybrid-X p99 latency under production load, each instance completes a warmup pass so JIT caches stabilize. "
            "Production load benchmarks confirm reproducible Hybrid-X p99 latency distributions across repeated seeds.</p>"
        ),
        (
            "<h3>Section 6: Baseline Comparison Methodology</h3>"
            "<p>We compare Hybrid-X against baseline engines under identical production load to study p99 latency behavior. "
            "Qualitative latency trade-offs for Hybrid-X production load pipelines are discussed before reporting exact numbers.</p>"
        ),
        # PAGE 7: Actual Table 4 (compact structured table, does NOT repeat 'production load' prose!)
        f"<h3>Appendix A</h3>{_TABLE_4_HTML}",
        (
            "<h3>Section 7: Production Load Error Analysis for Hybrid-X</h3>"
            "<p>Under bursty production load, Hybrid-X p99 latency remains stable because bounded queues shed stale retries. "
            "Operators tuning Hybrid-X for production load prioritize tail p99 latency predictability.</p>"
        ),
        (
            "<h3>Section 8: Related Work on Hybrid-X Tail Latency</h3>"
            "<p>Prior retrieval studies examine production load p99 latency on single-stage indexes, whereas Hybrid-X coordinates "
            "structured verification under production load to guard p99 latency budgets.</p>"
        ),
        (
            "<h3>Section 9: Conclusions on Hybrid-X Production Load</h3>"
            "<p>Overall, Hybrid-X delivers dependable p99 latency under production load across heterogeneous workloads. "
            "Future work will test Hybrid-X p99 latency under geo-distributed production load.</p>"
        ),
    ]
    pdf = _make_html_pdf(tmp_path / "case2_distant_table4.pdf", prose_pages)
    db.save_document("doc_case2", pdf.name, str(pdf), pdf.stat().st_size)

    # First verify Handoff ON (max_iterations=2) triggers REAL Reader -> Supervisor -> Retrieval -> Reader
    resp = real_api_client.post(
        "/api/runs",
        json={
            "question": "What is the p99 latency of Hybrid-X under production load?",
            "document_ids": ["doc_case2"],
            "mode": "cpu_min",
            "max_iterations": 2,
            "auto_scout": False,
            "parallel_retrieval": False,
        },
    )
    assert resp.status_code == 200
    run = _poll_run(real_api_client, resp.json()["id"])
    assert run["status"] == "completed"

    trace = run["reasoning_trace"]
    reader_evals = [
        e for e in trace
        if e.get("action") == "supervisor_decision_evaluated" and e.get("source_agent") == "reader"
    ]
    reader_handoffs = [
        e for e in trace
        if e.get("action") == "agent_handoff" and e.get("sender") == "reader"
    ]
    retrieval_receipts = [
        e for e in trace
        if e.get("action") == "agent_handoff_received" and e.get("sender") == "reader" and e.get("recipient") == "retrieval"
    ]
    retrieval_outcomes = [
        e for e in trace
        if e.get("action") == "handoff_outcome_recorded" and e.get("sender") == "reader" and e.get("recipient") == "retrieval"
    ]

    assert len(reader_evals) == 1
    assert reader_evals[0]["approved"] is True
    assert reader_evals[0]["decision_state"] == "APPROVED"
    assert reader_evals[0]["target_agent"] == "retrieval"

    assert len(reader_handoffs) == 1
    assert reader_handoffs[0]["recipient"] == "retrieval"
    assert any("Table 4" in f for f in reader_handoffs[0]["focus"])

    assert len(retrieval_receipts) == 1
    assert retrieval_receipts[0]["signature"] == reader_handoffs[0]["signature"]
    assert len(retrieval_outcomes) == 1
    assert retrieval_outcomes[0]["outcome_status"] == "evidence_retrieved"

    report = run["final_report"]
    assert "Hybrid-X" in report
    assert "p99" in report.lower()
    assert "14.6" in report
    assert any((ref.get("page") or ref.get("page_no")) == 7 for ref in run.get("citation_references") or [])

    # Paired control on the exact same PDF with handoff OFF (max_iterations=0):
    # Supervisor rejects Reader's handoff (`reflection_budget_exhausted`), so Page 7's 14.6 is NOT reached!
    resp_off = real_api_client.post(
        "/api/runs",
        json={
            "question": "What is the p99 latency of Hybrid-X under production load?",
            "document_ids": ["doc_case2"],
            "mode": "cpu_min",
            "max_iterations": 0,
            "auto_scout": False,
            "parallel_retrieval": False,
        },
    )
    assert resp_off.status_code == 200
    run_off = _poll_run(real_api_client, resp_off.json()["id"])
    assert run_off["status"] == "completed"
    assert "14.6" not in run_off["final_report"]


def test_step2_case3_missing_referenced_table_stops_in_budget_and_reports_unverified(
    real_api_client: TestClient, tmp_path: Path
):
    pdf = _make_html_pdf(
        tmp_path / "case3_absent_table4.pdf",
        [
            "<h3>Latency Study</h3>"
            "<p>We benchmarked Hybrid-X under production load. For exact p99 latency measurements of Hybrid-X, please see Table 4.</p>",
            "<h3>Hardware Setup</h3>"
            "<p>All servers ran Linux with NVMe storage and standard network interfaces.</p>",
        ],
    )
    db.save_document("doc_case3", pdf.name, str(pdf), pdf.stat().st_size)

    resp = real_api_client.post(
        "/api/runs",
        json={
            "question": "What is the exact p99 latency of Hybrid-X in Table 4?",
            "document_ids": ["doc_case3"],
            "mode": "cpu_min",
            "max_iterations": 2,
            "auto_scout": False,
            "parallel_retrieval": False,
        },
    )
    assert resp.status_code == 200
    run = _poll_run(real_api_client, resp.json()["id"])
    assert run["status"] == "completed"

    trace = run["reasoning_trace"]
    reader_handoffs = [e for e in trace if e.get("action") == "agent_handoff" and e.get("sender") == "reader"]
    # Real Reader tried bounded lookup once, Retrieval returned no_new_evidence, and loop terminated cleanly!
    assert len(reader_handoffs) == 1
    retrieval_outcomes = [
        e for e in trace
        if e.get("action") == "handoff_outcome_recorded" and e.get("sender") == "reader" and e.get("recipient") == "retrieval"
    ]
    assert len(retrieval_outcomes) == 1
    assert retrieval_outcomes[0]["outcome_status"] == "no_new_evidence"

    report_lower = run["final_report"].lower()
    assert any(
        phrase in report_lower
        for phrase in ("unverified", "insufficient evidence", "missing", "chưa đủ bằng chứng", "không thể kiểm chứng", "không có")
    ), f"Expected honest unverified/missing-table message, got:\n{run['final_report']}"
    assert "table 4" in report_lower
    assert "14.6" not in report_lower


def test_step2_case4_two_pdfs_conflicting_values_attributes_both_sources_and_flags_conflict(
    real_api_client: TestClient, tmp_path: Path
):
    pdf_a = _make_html_pdf(
        tmp_path / "study_alpha.pdf",
        [
            "<h3>Study Alpha Evaluation</h3>"
            "<p>In our controlled evaluation on Dataset Omega, System Nova achieved 92.4% accuracy across all test splits.</p>"
        ],
    )
    pdf_b = _make_html_pdf(
        tmp_path / "study_beta.pdf",
        [
            "<h3>Study Beta Replication</h3>"
            "<p>In our independent replication on Dataset Omega, System Nova achieved 84.1% accuracy across all test splits.</p>"
        ],
    )
    db.save_document("doc_alpha", pdf_a.name, str(pdf_a), pdf_a.stat().st_size)
    db.save_document("doc_beta", pdf_b.name, str(pdf_b), pdf_b.stat().st_size)

    resp = real_api_client.post(
        "/api/runs",
        json={
            "question": "What accuracy did System Nova achieve on Dataset Omega?",
            "document_ids": ["doc_alpha", "doc_beta"],
            "mode": "cpu_min",
            "max_iterations": 2,
            "auto_scout": False,
            "parallel_retrieval": False,
        },
    )
    assert resp.status_code == 200
    run = _poll_run(real_api_client, resp.json()["id"])
    assert run["status"] == "completed"

    report = run["final_report"]
    report_lower = report.lower()
    assert "92.4%" in report
    assert "84.1%" in report
    assert "alpha" in report_lower
    assert "beta" in report_lower
    assert any(
        marker in report_lower
        for marker in ("conflicting", "discrepancy", "khác nhau", "mâu thuẫn", "sai khác")
    ), f"Expected explicit conflict/discrepancy notice, got:\n{report}"

    cited_docs = {
        str(ref.get("document_id") or ref.get("doc_id") or ref.get("title") or "").lower()
        for ref in (run.get("citation_references") or [])
    }
    assert any("alpha" in d for d in cited_docs)
    assert any("beta" in d for d in cited_docs)


def test_step2_case4b_different_metrics_across_two_pdfs_never_triggers_false_conflict(
    real_api_client: TestClient, tmp_path: Path
):
    """
    Negative test for Cross-Document Discrepancy Notice:
    Doc A reports System Nova accuracy = 92.4%, Doc B reports System Nova latency = 14.6 ms.
    Different metric families and different units MUST NOT trigger a conflict banner!
    """
    from core.claim_semantics import detect_cross_document_metric_conflicts

    # Direct unit checks covering positive + all 4 negative dimensions (metric, percentile, entity, split)
    assert len(
        detect_cross_document_metric_conflicts([
            ("doc_a", "On Dataset Omega test split, System Nova achieved 92.4% accuracy."),
            ("doc_b", "On Dataset Omega test split, System Nova achieved 84.1% accuracy."),
        ])
    ) == 1
    assert detect_cross_document_metric_conflicts([
        ("doc_a", "System Nova achieved 92.4% accuracy."),
        ("doc_b", "System Nova achieved 14.6 ms latency."),
    ]) == []
    assert detect_cross_document_metric_conflicts([
        ("doc_a", "Hybrid-X p50 latency is 8.2 ms."),
        ("doc_b", "Hybrid-X p99 latency is 14.6 ms."),
    ]) == []
    assert detect_cross_document_metric_conflicts([
        ("doc_a", "Model Orion achieved 92.4% accuracy."),
        ("doc_b", "Model Zephyr achieved 84.1% accuracy."),
    ]) == []
    assert detect_cross_document_metric_conflicts([
        ("doc_a", "System Nova achieved 92.4% accuracy on validation set."),
        ("doc_b", "System Nova achieved 84.1% accuracy on test set."),
    ]) == []

    # End-to-end API test on 2 real PDFs: accuracy (92.4%) vs latency (14.6 ms)
    pdf_acc = _make_html_pdf(
        tmp_path / "nova_accuracy.pdf",
        ["<p>In benchmark evaluation, System Nova achieved 92.4% accuracy on Dataset Omega.</p>"],
    )
    pdf_lat = _make_html_pdf(
        tmp_path / "nova_latency.pdf",
        ["<p>In benchmark evaluation, System Nova achieved 14.6 ms latency on Dataset Omega.</p>"],
    )
    db.save_document("doc_nova_acc", pdf_acc.name, str(pdf_acc), pdf_acc.stat().st_size)
    db.save_document("doc_nova_lat", pdf_lat.name, str(pdf_lat), pdf_lat.stat().st_size)

    resp = real_api_client.post(
        "/api/runs",
        json={
            "question": "Summarize the accuracy and latency of System Nova on Dataset Omega.",
            "document_ids": ["doc_nova_acc", "doc_nova_lat"],
            "mode": "cpu_min",
            "max_iterations": 2,
            "auto_scout": False,
            "parallel_retrieval": False,
        },
    )
    assert resp.status_code == 200
    run = _poll_run(real_api_client, resp.json()["id"])
    assert run["status"] == "completed"
    report = run["final_report"]
    assert "92.4%" in report
    assert "14.6 ms" in report
    assert "cross-document discrepancy notice" not in report.lower()
    assert "conflicting numeric values" not in report.lower()


def test_step2_case5_same_table_number_wrong_topic_or_missing_p99_column_triggers_reader_handoff_and_abstains(
    real_api_client: TestClient, tmp_path: Path
):
    """
    Verifies Reader checks target row/entity, column/metric, and unit — NOT just table number:
      5a. Table 4 exists, but is about participant age demographics (no Hybrid-X, no p99, no latency).
      5b. Table 4 exists for Hybrid-X, but only reports p50 (8.2 ms) and is missing p99.
    In both cases, real Reader MUST propose a Retrieval handoff and, when no matching table exists,
    record unresolved missing cells so the final report honestly reports unverified / insufficient evidence.
    """
    from agents.agent2_reader.tools.reader_pipeline import evaluate_reader_table_completeness

    # Direct Reader contract check on wrong-topic Table 4 (age demographics)
    age_units = [
        {
            "id": "u_age",
            "doc_id": "demo",
            "page_no": 1,
            "text": "Table 4: Participant Age Demographics\n| Cohort | Mean Age (years) | Median Age (years) |\n| Group A | 24.5 | 23.0 |\n| Group B | 31.2 | 30.5 |",
        }
    ]
    dec_age, unres_age = evaluate_reader_table_completeness(
        {"query": "What is the p99 latency of Hybrid-X in Table 4?", "retry_count": 0, "no_new_evidence": False},
        age_units,
    )
    assert dec_age is not None
    assert dec_age.target_agent == "retrieval"
    assert unres_age == []

    # Direct Reader contract check on partial Table 4 (has Hybrid-X + p50, MISSING p99)
    p50_only_units = [
        {
            "id": "u_p50",
            "doc_id": "partial",
            "page_no": 1,
            "text": "Table 4: Median Latency Breakdown\n| Engine | p50 (ms) |\n| Baseline-V1 | 22.1 |\n| Hybrid-X | 8.2 |",
        }
    ]
    dec_p50, _ = evaluate_reader_table_completeness(
        {"query": "What is the p99 latency of Hybrid-X in Table 4?", "retry_count": 0, "no_new_evidence": False},
        p50_only_units,
    )
    assert dec_p50 is not None
    assert "p99" in dec_p50.reason.lower()

    # End-to-end API test on real PDF whose Table 4 ONLY contains p50 (no p99)
    partial_html = """
    <p><b>Table 4: Median Latency Breakdown</b></p>
    <table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse; width:100%;">
      <tr><th>Engine</th><th>p50 (ms)</th></tr>
      <tr><td>Baseline-V1</td><td>22.1</td></tr>
      <tr><td>Hybrid-X</td><td>8.2</td></tr>
    </table>
    """
    pdf_partial = _make_html_pdf(
        tmp_path / "case5_partial_table4_no_p99.pdf",
        [
            f"<h3>Latency Summary</h3><p>Median latency results are shown in Table 4.</p>{partial_html}",
            "<p>Section 4 summarizes deployment logging.</p>",
        ],
    )
    db.save_document("doc_case5_partial", pdf_partial.name, str(pdf_partial), pdf_partial.stat().st_size)

    resp = real_api_client.post(
        "/api/runs",
        json={
            "question": "What is the p99 latency of Hybrid-X in Table 4?",
            "document_ids": ["doc_case5_partial"],
            "mode": "cpu_min",
            "max_iterations": 2,
            "auto_scout": False,
            "parallel_retrieval": False,
        },
    )
    assert resp.status_code == 200
    run = _poll_run(real_api_client, resp.json()["id"])
    assert run["status"] == "completed"

    trace = run["reasoning_trace"]
    reader_handoffs = [e for e in trace if e.get("action") == "agent_handoff" and e.get("sender") == "reader"]
    assert len(reader_handoffs) == 1
    report_lower = run["final_report"].lower()
    assert any(
        phrase in report_lower
        for phrase in ("unverified", "insufficient evidence", "missing", "chưa đủ bằng chứng", "không thể kiểm chứng")
    ), f"Expected unverified/missing-cell notice when p99 is absent from Table 4, got:\n{run['final_report']}"
    assert "p99" in report_lower
    assert "8.2" not in report_lower

