"""
Milestone M1 End-to-End Regression Suite:
Covers 100% of Section 1 P0/P1 findings and Gate M1 invariants across unit, graph,
FastAPI (`/api/research`), and SSE streaming (`/api/research/stream`):
  1. P0 · Writer lock (`GeminiWriterAgent`, `HybridWriterAgent`, `write_with_citation_audit`, SSE stream):
     blocks fabricated `999 layers` when source/claim states `12 layers`, blocks subject swaps and
     negation flips in post-Writer prose, and never exposes raw `source_units` to Writer.
  2. P0 · Negation, conditional framing, quantifiers, and directed subject-object relations
     (`CPUMinFactCheckerAgent`, `CPUMaxFactCheckerAgent`, `review_claims`, `FactCheckJudge`):
     blocks `"Treatment works. [1]"` from `"It is false that treatment works."` (EN & VI).
  3. P0 · Math inequalities, subscripts vs superscripts, signs, and domain constraints:
     preserves `"x < 3 and y > 2"` in CPU readers/cleaners, rejects `x_2 = 4` against `x^2 = 4`,
     and enforces `x != 0` on `x / x = 1`.
  4. P0 · Derived numerical claims (`derived_claim`):
     certifies `"B cao hơn A 11 điểm phần trăm"` from `"A = 81%, B = 92%"` with intact `[1]` citations,
     while rejecting `11%`, `13.58 điểm phần trăm`, swapped `"A cao hơn B"`, and Validation-vs-Test splits.
  5. P1 · Vietnamese–English detection, `query_language`/`answer_language`/`source_language` separation,
     and localized status/refusal messages (`"This is a test"` -> EN, `"Ket qua o trang nao?"` -> VI,
     English OOD refusal in English).
"""

from __future__ import annotations

import json

import fitz
import pytest
from fastapi.testclient import TestClient

from agents.agent1_retrieval.tools.semantic_pipeline import is_vietnamese_text
from agents.agent3_summarizer.components.generator import bind_evidence_spans
from agents.agent5_writer.components.report_builder import build_academic_report
from api import app as api_module, auth
from core import db
from core.citation_audit import audit_claim_citations, get_citation_audit, write_with_citation_audit
from core.config import settings
from core.state import EvidenceSpan, SourceUnit
from core.tools import clear_pdf_cache
from core.verifier import review_claims, verify_evidence_span_provenance
from eval.three_layer_evaluator import detect_response_language
from modes.cpu_max.fact_checker import CPUMaxFactCheckerAgent
from modes.cpu_max.reader import clean_source_text
from modes.cpu_min.fact_checker import CPUMinFactCheckerAgent
from modes.cpu_min.reader import sanitize_unit_text
from modes.cpu_min.summarizer import clean_vietnamese_text
from modes.cpu_min.writer import CPUMinWriterAgent
from modes.gemini.writer import GeminiWriterAgent
from modes.hybrid.writer import HybridWriterAgent
from utils.language import detect_language, resolve_language_bundle
from utils.llm_client import llm_client


@pytest.fixture
def isolated_api_client(monkeypatch, research_db, tmp_path):
    clear_pdf_cache()
    monkeypatch.setattr(auth, "AUTH_SECRET_PASSWORD", "m1-service-test")
    monkeypatch.setattr(auth, "AUTH_DEV_MODE", False)
    monkeypatch.setattr(auth, "_sessions", {})
    monkeypatch.setattr(api_module.settings, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(api_module.settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(api_module.settings, "CORPUS_PATH", str(tmp_path / "empty_corpus.jsonl"))
    monkeypatch.setattr(api_module, "_legacy_mode_override", "cpu_min")
    with TestClient(api_module.app, base_url="https://workspace.test") as client:
        assert client.post("/api/auth/login", json={"password": "m1-service-test"}).status_code == 200
        yield client


# =====================================================================
# 1. P0 · Writer Gate (GeminiWriter, HybridWriter, Post-Writer Audit & Stream)
# =====================================================================


def test_gemini_and_hybrid_writers_block_999_layers_hallucination(monkeypatch):
    quote = "The Transformer encoder consists of 12 layers."
    state = {
        "query": "How many layers does the Transformer encoder have?",
        "answer_language": "en",
        "source_units": [SourceUnit("u1", quote, "arch.pdf", 3).to_dict()],
        "evidence_spans": [EvidenceSpan("ev1", quote, "u1", "arch.pdf", 3).to_dict()],
        "claims": [{"id": "c1", "text": quote, "evidence_ids": ["ev1"]}],
        "verified_claims": [{"claim_id": "c1", "text": quote, "status": "SUPPORT", "evidence_ids": ["ev1"]}],
        "rejected_claims": [],
        "retrieved_docs": [{"doc_id": "arch.pdf", "page_no": 3, "title": "arch.pdf", "text": quote}],
    }

    monkeypatch.setattr(settings, "GEMINI_API_KEY", "AIzaSyTestFakeKeyForM1WriterGate")
    monkeypatch.setattr(
        llm_client,
        "generate",
        lambda *args, **kwargs: "The Transformer encoder consists of 999 layers [1].",
    )

    # 1. GeminiWriterAgent under write_with_citation_audit
    gem_out = write_with_citation_audit(state, GeminiWriterAgent().run)
    assert "999" not in gem_out["answer"]
    assert "999" not in gem_out["short_answer"]
    assert "999" not in gem_out["final_report"]
    assert "12 layers. [1]" in gem_out["answer"]
    assert gem_out["short_answer"] == gem_out["answer"]

    # 2. HybridWriterAgent delegating to GeminiWriterAgent
    monkeypatch.setattr(
        "modes.hybrid.writer.probe_all_capabilities",
        lambda *a, **kw: {"gemini": {"online": True}, "kaggle": {"online": False}, "ollama": {"online": False}},
    )
    monkeypatch.setattr(
        "modes.hybrid.writer.provider_router.resolve_chain",
        lambda *a, **kw: ["gemini", "cpu_max"],
    )
    hyb_out = write_with_citation_audit(state, HybridWriterAgent().run)
    assert "999" not in hyb_out["answer"]
    assert "999" not in hyb_out["short_answer"]
    assert "999" not in hyb_out["final_report"]
    assert "12 layers. [1]" in hyb_out["answer"]


def test_post_writer_audit_reverts_mutated_numbers_swapped_subjects_and_stripped_raw_units():
    quote = "Method B achieved 92% accuracy while Method A achieved 81% accuracy."
    approved_claim = "Method B outperforms Method A (92% vs 81%)."
    state = {
        "query": "Which method performed better?",
        "answer_language": "en",
        "source_units": [SourceUnit("u1", quote + " Unverified footnote 777.", "bench.pdf", 2).to_dict()],
        "evidence_spans": [EvidenceSpan("ev1", quote, "u1", "bench.pdf", 2).to_dict()],
        "verified_claims": [{"claim_id": "c1", "text": approved_claim, "status": "SUPPORT", "evidence_ids": ["ev1"]}],
        "rejected_claims": [],
        "retrieved_docs": [{"doc_id": "bench.pdf", "page_no": 2, "title": "bench.pdf", "text": quote}],
    }

    observed_writer_units = []

    def rogue_rewrite_writer(writer_state):
        observed_writer_units.extend(writer_state.get("source_units") or [])
        md, data = build_academic_report(
            query=writer_state["query"],
            verified_claims=writer_state["verified_claims"],
            rejected_claims=[],
            retrieved_docs=writer_state["retrieved_docs"],
            evidence_spans=writer_state["evidence_spans"],
            answer_language="en",
        )
        # Simulate a free-form rewrite reversing direction ("Method A outperforms Method B") while keeping same numbers!
        mutated = "Method A outperforms Method B (92% vs 81%) [1]."
        data["answer"] = mutated
        data["short_answer"] = mutated
        return {"final_report": md.replace(approved_claim + " [1]", mutated), "written_report_data": data}

    out = write_with_citation_audit(state, rogue_rewrite_writer)
    # Writer never receives raw source_units
    assert observed_writer_units == []
    # Directed relation check catches subject swap and reverts to approved claim!
    assert "Method A outperforms Method B" not in out["answer"]
    assert out["answer"] == f"{approved_claim} [1]"
    assert out["short_answer"] == f"{approved_claim} [1]"
    audit = get_citation_audit(out["reasoning_trace"])
    assert audit["post_writer_fallback"]["triggered"] is True


def test_sync_api_and_sse_stream_enforce_same_audit_gate_without_leaking_draft(
    isolated_api_client, monkeypatch, tmp_path
):
    pdf_path = tmp_path / "encoder_arch.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "Section 2. Architecture: The encoder stack contains 12 layers.")
        doc.save(pdf_path)
    db.save_document("doc_arch", "encoder_arch.pdf", str(pdf_path), pdf_path.stat().st_size)

    # Inject a Summarizer draft that hallucinates "999 layers" alongside a grounded claim "12 layers"
    from modes.cpu_min.summarizer import CPUMinSummarizerAgent

    orig_run = CPUMinSummarizerAgent.run

    def poisoned_summarizer(self, state):
        res = orig_run(self, state)
        res["answer"] = "The encoder stack contains 999 layers [1]."
        return res

    monkeypatch.setattr(CPUMinSummarizerAgent, "run", poisoned_summarizer)

    # 1. Synchronous /api/research
    sync_resp = isolated_api_client.post(
        "/api/research",
        json={"query": "How many layers are in the encoder stack?", "document_ids": ["doc_arch"], "auto_scout": False},
    )
    assert sync_resp.status_code == 200
    sync_body = sync_resp.json()
    assert "999" not in sync_body["answer"]
    assert "999" not in sync_body["short_answer"]
    assert "999" not in sync_body["final_report"]
    assert "12 layers" in sync_body["answer"]

    # 2. Streaming /api/research/stream
    stream_resp = isolated_api_client.post(
        "/api/research/stream",
        json={"query": "How many layers are in the encoder stack?", "document_ids": ["doc_arch"], "auto_scout": False},
    )
    assert stream_resp.status_code == 200
    events = [json.loads(line[6:]) for line in stream_resp.text.splitlines() if line.startswith("data: ")]
    for ev in events:
        if ev["event"] == "node_update":
            partial_ans = ev["data"].get("partial_answer") or ""
            assert "999" not in partial_ans, f"Unaudited draft leaked in SSE node {ev['node']}: {partial_ans}"
    finished = events[-1]
    assert finished["event"] == "finished"
    assert "999" not in finished["final_report"]
    assert "12 layers" in finished["final_report"]


# =====================================================================
# 2. P0 · Negation, Conditionals, Quantifiers & Subject Swaps (EN + VI)
# =====================================================================


def test_cpu_min_and_cpu_max_reject_affirmative_substring_from_negated_source():
    source_text = "It is false that treatment works."
    state = {
        "query": "Does treatment work?",
        "answer_language": "en",
        "answer": "Treatment works.",
        "claims": [{"id": "c1", "text": "Treatment works.", "evidence_ids": ["u1"]}],
        "source_units": [SourceUnit("u1", source_text, "clinical.pdf", 1).to_dict()],
        "evidence_spans": [EvidenceSpan("ev1", source_text, "u1", "clinical.pdf", 1).to_dict()],
        "retrieved_docs": [{"id": "u1", "doc_id": "clinical.pdf", "page_no": 1, "text": source_text}],
    }

    min_res = CPUMinFactCheckerAgent().run(state)
    max_res = CPUMaxFactCheckerAgent().run(state)

    assert min_res["verified_claims"] == []
    assert min_res["fact_check_label"] == "contradict"
    assert max_res["verified_claims"] == []
    assert max_res["fact_check_label"] == "contradict"

    # Even if forced through Writer, "Treatment works. [1]" must NEVER be published
    published_min = write_with_citation_audit({**state, **min_res}, CPUMinWriterAgent().run)
    published_max = write_with_citation_audit({**state, **max_res}, CPUMinWriterAgent().run)
    assert "Treatment works. [1]" not in published_min["answer"]
    assert "Treatment works. [1]" not in published_max["answer"]
    assert published_min["written_report_data"]["status"] == "INSUFFICIENT_EVIDENCE"


@pytest.mark.parametrize(
    ("source_text", "bad_claim"),
    [
        ("Sai khi nói rằng phương pháp điều trị có hiệu quả.", "Phương pháp điều trị có hiệu quả."),
        ("If treatment works, recovery time drops.", "Treatment works."),
        ("Some cohorts converged within 5 epochs.", "All cohorts converged within 5 epochs."),
        ("Model B was faster than Model A on the benchmark.", "Model A was faster than Model B on the benchmark."),
        ("Phương pháp B vượt trội hơn Phương pháp A.", "Phương pháp A vượt trội hơn Phương pháp B."),
    ],
)
def test_verifier_rejects_vietnamese_negation_conditionals_quantifiers_and_swaps(source_text, bad_claim):
    state = {
        "query": "Verify finding",
        "answer": bad_claim,
        "claims": [{"id": "c1", "text": bad_claim, "evidence_ids": ["u1"]}],
        "source_units": [SourceUnit("u1", source_text, "paper.pdf", 1).to_dict()],
        "evidence_spans": [EvidenceSpan("ev1", source_text, "u1", "paper.pdf", 1).to_dict()],
    }
    res = review_claims(state)
    assert res["verified_claims"] == [], f"Should have rejected {bad_claim!r} against {source_text!r}"


# =====================================================================
# 3. P0 · Math Inequalities, Subscripts vs Superscripts & Domain Constraints
# =====================================================================


def test_cpu_readers_and_cleaners_preserve_inequalities_while_stripping_real_html():
    raw = "<p>For stability, x < 3 and y > 2 must hold.<br/>Also z<sub>1</sub> = 0.</p>"
    for cleaner in (clean_source_text, sanitize_unit_text, clean_vietnamese_text):
        cleaned = cleaner(raw)
        assert "x < 3 and y > 2" in cleaned
        assert "<p>" not in cleaned and "<br" not in cleaned
        assert "_{1}" in cleaned


def test_quote_matcher_verifier_and_citation_audit_reject_subscript_for_superscript():
    source_units = [SourceUnit("u1", "The quadratic equation is x^2 = 4.", "math.pdf", 2).to_dict()]
    bad_span = {"id": "ev1", "source_unit_id": "u1", "doc_id": "math.pdf", "page_no": 2, "quote": "x_2 = 4"}

    ok, _ = verify_evidence_span_provenance(bad_span, source_units)
    assert ok is False
    assert bind_evidence_spans([bad_span], source_units) == []

    # Even with a valid quote "x^2 = 4", a claim stating "x_2 = 4" must fail BOTH verifier and citation_audit
    valid_span = EvidenceSpan("ev1", "The quadratic equation is x^2 = 4.", "u1", "math.pdf", 2).to_dict()
    state = {
        "query": "What is the equation?",
        "answer": "The quadratic equation is x_2 = 4.",
        "claims": [{"id": "c1", "text": "The quadratic equation is x_2 = 4.", "evidence_ids": ["ev1"]}],
        "source_units": source_units,
        "evidence_spans": [valid_span],
    }
    ver = review_claims(state)
    assert ver["verified_claims"] == []
    assert ver["fact_check_label"] == "contradict"

    forced_audit = audit_claim_citations({
        **state,
        "verified_claims": [{"claim_id": "c1", "text": "The quadratic equation is x_2 = 4.", "status": "SUPPORT", "evidence_ids": ["ev1"]}],
    })
    assert forced_audit["claims"]["c1"]["status"] != "verified"
    assert forced_audit["claims"]["c1"]["evidence_ids"] == []


def test_verifier_preserves_sign_inequality_direction_and_nonzero_domain_constraint():
    # 1. Domain constraint x / x = 1 for x != 0
    domain_unit = SourceUnit("u_dom", "The simplified ratio x / x = 1 holds for x != 0.", "calc.pdf", 1).to_dict()
    bad_domain_res = review_claims({
        "query": "When does x/x = 1 hold?",
        "answer": "The simplified ratio x / x = 1 holds for all x.",
        "claims": [{"id": "c1", "text": "The simplified ratio x / x = 1 holds for all x.", "evidence_ids": ["u_dom"]}],
        "source_units": [domain_unit],
    })
    assert bad_domain_res["verified_claims"] == []

    good_domain_res = review_claims({
        "query": "When does x/x = 1 hold?",
        "answer": "The simplified ratio x / x = 1 holds for x != 0.",
        "claims": [{"id": "c1", "text": "The simplified ratio x / x = 1 holds for x != 0.", "evidence_ids": ["u_dom"]}],
        "source_units": [domain_unit],
    })
    assert len(good_domain_res["verified_claims"]) == 1

    # 2. Inequality flip & sign flip
    ineq_unit = SourceUnit("u_in", "Convergence requires x < 3 and w = -4.", "calc.pdf", 2).to_dict()
    for wrong in ("Convergence requires x > 3.", "Convergence requires w = 4."):
        res = review_claims({
            "query": "What are the bounds?",
            "answer": wrong,
            "claims": [{"id": "c1", "text": wrong, "evidence_ids": ["u_in"]}],
            "source_units": [ineq_unit],
        })
        assert res["verified_claims"] == []


# =====================================================================
# 4. P0 · Derived Numerical Claims (`derived_claim`: 11 pp vs 13.58% vs Split)
# =====================================================================


def test_derived_claim_11_percentage_points_passes_verifier_audit_and_writer():
    source_text = "In Figure 3 (Validation), Method A = 81%, Method B = 92%."
    unit = SourceUnit("u_fig3", source_text, "vision.pdf", 4).to_dict()
    span = EvidenceSpan("ev_fig3", source_text, "u_fig3", "vision.pdf", 4).to_dict()
    good_claim = "Method B cao hơn Method A 11 điểm phần trăm."

    state = {
        "query": "Trong Hình 3, Method B cao hơn Method A bao nhiêu điểm phần trăm?",
        "answer_language": "vi",
        "answer": good_claim,
        "claims": [{"id": "c_pp", "text": good_claim, "evidence_ids": ["ev_fig3"]}],
        "source_units": [unit],
        "evidence_spans": [span],
        "retrieved_docs": [{"doc_id": "vision.pdf", "page_no": 4, "title": "vision.pdf", "text": source_text}],
    }

    # 1. Verifier certifies 92% - 81% = 11 pp as SUPPORT (derived_claim)
    ver = review_claims(state)
    assert ver["fact_check_label"] == "support"
    assert len(ver["verified_claims"]) == 1
    assert ver["verified_claims"][0]["claim_type"] == "derived_claim"

    # 2. Citation audit preserves evidence_ids and does NOT flag '11' as missing_numeric_evidence
    merged = {**state, **ver}
    audit = audit_claim_citations(merged)
    assert audit["claims"]["c_pp"]["status"] == "verified"
    assert audit["claims"]["c_pp"]["evidence_ids"] == ["ev_fig3"]
    assert audit["claims"]["c_pp"]["missing_values"] == []

    # 3. Writer publishes "Method B cao hơn Method A 11 điểm phần trăm. [1]"
    published = write_with_citation_audit(merged, CPUMinWriterAgent().run)
    assert "Method B cao hơn Method A 11 điểm phần trăm. [1]" in published["answer"]


@pytest.mark.parametrize(
    ("source_text", "bad_derived_claim"),
    [
        # Wrong unit: 11% instead of 11 percentage points
        ("Method A = 81%, Method B = 92%.", "Method B cao hơn Method A 11%."),
        # Confusing relative increase (13.58%) with percentage points
        ("Method A = 81%, Method B = 92%.", "Method B cao hơn Method A 13.58 điểm phần trăm."),
        # Swapped comparative direction
        ("Method A = 81%, Method B = 92%.", "Method A cao hơn Method B 11 điểm phần trăm."),
        # Cross-split subtraction (Validation vs Test)
        ("Method A = 81% (Validation), Method B = 92% (Test).", "Method B cao hơn Method A 11 điểm phần trăm."),
    ],
)
def test_derived_claim_rejects_wrong_unit_relative_pct_confusion_swaps_and_cross_split(
    source_text, bad_derived_claim
):
    unit = SourceUnit("u1", source_text, "vision.pdf", 4).to_dict()
    span = EvidenceSpan("ev1", source_text, "u1", "vision.pdf", 4).to_dict()
    state = {
        "query": "Compare Method A and Method B",
        "answer": bad_derived_claim,
        "claims": [{"id": "c1", "text": bad_derived_claim, "evidence_ids": ["ev1"]}],
        "source_units": [unit],
        "evidence_spans": [span],
    }
    ver = review_claims(state)
    assert ver["verified_claims"] == [], f"Verifier must reject {bad_derived_claim!r}"

    # Even if manually marked SUPPORT, citation_audit must reject it!
    audit = audit_claim_citations({
        **state,
        "verified_claims": [{"claim_id": "c1", "text": bad_derived_claim, "status": "SUPPORT", "evidence_ids": ["ev1"]}],
    })
    assert audit["claims"]["c1"]["status"] != "verified"
    assert audit["claims"]["c1"]["evidence_ids"] == []


def test_api_end_to_end_answers_derived_percentage_point_question(isolated_api_client, tmp_path):
    pdf_path = tmp_path / "ab_results.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            "Table 1 (Validation): Method A = 81%, Method B = 92% on the benchmark dataset.",
        )
        doc.save(pdf_path)
    db.save_document("doc_ab", "ab_results.pdf", str(pdf_path), pdf_path.stat().st_size)

    resp = isolated_api_client.post(
        "/api/research",
        json={
            "query": "Trong tài liệu, Method B cao hơn Method A bao nhiêu điểm phần trăm?",
            "document_ids": ["doc_ab"],
            "auto_scout": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "11 điểm phần trăm" in body["answer"]
    assert "[1]" in body["answer"]
    assert body["citation_references"][0]["document_id"] == "doc_ab"


# =====================================================================
# 5. P1 · Vietnamese–English Language Detection & Localized Refusals
# =====================================================================


def test_language_detection_handles_english_s_unaccented_vietnamese_and_greek_math():
    # Exact P1 cases from Section 1
    assert is_vietnamese_text("This is a test") is False
    assert detect_language("This is a test") == "en"

    assert is_vietnamese_text("Ket qua o trang nao?") is True
    assert detect_language("Ket qua o trang nao?") == "vi"

    # Mixed Greek symbols & technical terms
    assert detect_language("What are α, β, and θ in Equation 3?") == "en"
    assert detect_language("Gia tri α va β trong phuong trinh 3 la bao nhieu?") == "vi"

    # Explicit answer language override (in query and via explicit parameter)
    bundle_in_query = resolve_language_bundle("Hãy tóm tắt bài báo này bằng tiếng Anh")
    assert bundle_in_query["query_language"] == "vi"
    assert bundle_in_query["answer_language"] == "en"

    bundle_param = resolve_language_bundle("What accuracy was reported?", explicit_answer_language="vi")
    assert bundle_param["query_language"] == "en"
    assert bundle_param["answer_language"] == "vi"


def test_english_out_of_domain_query_returns_english_refusal_in_report_and_api(
    isolated_api_client, tmp_path
):
    pdf_path = tmp_path / "agronomy.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "The orchard drip irrigation trial lasted 21 days and consumed 42 liters.")
        doc.save(pdf_path)
    db.save_document("doc_agro", "agronomy.pdf", str(pdf_path), pdf_path.stat().st_size)

    resp = isolated_api_client.post(
        "/api/research",
        json={
            "query": "How does Apache ZooKeeper elect a leader in Kubernetes?",
            "document_ids": ["doc_agro"],
            "auto_scout": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer_status"] == "NOT_FOUND_IN_DOCUMENT"
    assert detect_response_language(body["answer"]) == "en"
    assert detect_response_language(body["final_report"]) == "en"
    assert "hoàn toàn không được đề cập" not in body["final_report"].lower()
    assert "not mentioned" in body["answer"].lower()
