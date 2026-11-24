"""Regression & capability tests for Phase 1 upgrades:
1. Layout-aware 2-column PDF reading order & intact Markdown table chunk extraction.
2. Cross-lingual Vietnamese <-> English retrieval bridging in Hybrid & Gemini modes.
3. Hybrid mode Kaggle BGE-M3 Dense + Guarded MiniLM Reranker integration (with fail-open fallback).
4. Gemini mode table-priority boost & optional metered Gemini Embedding dense retrieval.
5. Verifier quantified single-digit guard & entity-value binding anti-hallucination guard.
"""

from __future__ import annotations

import json
import httpx
import fitz
import pytest

from agents.agent1_retrieval.tools.semantic_pipeline import build_cross_lingual_subqueries
from core.config import settings
from core.tools import clear_pdf_cache, load_or_parse_selected_pdf
from core.verifier import (
    check_entity_value_binding,
    check_numerical_and_date_consistency,
    deterministic_verify_claims,
    extract_numbers_and_years,
    review_claims,
)
from modes.gemini.retrieval import GeminiRetrievalAgent
from modes.hybrid.retrieval import HybridRetrievalAgent
from utils.llm_client import llm_client


def _make_two_column_pdf(path):
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    # Spanning title
    page.insert_text((72, 60), "1. Introduction to Dual Column Academic Parsing", fontsize=14)
    # Left column blocks
    page.insert_textbox(
        fitz.Rect(50, 110, 270, 220),
        "LEFT_FIRST: The encoder processes input tokens sequentially across six attention blocks.",
        fontsize=10,
    )
    page.insert_textbox(
        fitz.Rect(50, 240, 270, 350),
        "LEFT_SECOND: Positional encodings preserve relative order within the left column.",
        fontsize=10,
    )
    # Right column blocks (placed at same y-coordinates as left column!)
    page.insert_textbox(
        fitz.Rect(330, 110, 550, 220),
        "RIGHT_FIRST: Experimental evaluation measures recall and precision on holdout benchmarks.",
        fontsize=10,
    )
    page.insert_textbox(
        fitz.Rect(330, 240, 550, 350),
        "RIGHT_SECOND: Guarded reranking preserves the top three hybrid passages during scoring.",
        fontsize=10,
    )
    doc.save(str(path))
    doc.close()


def _make_table_pdf(path):
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((60, 60), "3. Experimental Results", fontsize=13)
    page.insert_text((60, 95), "Table 1: Model accuracy comparison on holdout benchmark.", fontsize=10)

    # Draw a crisp 3x3 bordered table so PyMuPDF find_tables() extracts it cleanly
    x_cols = [60, 220, 380, 520]
    y_rows = [110, 140, 170, 200]
    for y in y_rows:
        page.draw_line((x_cols[0], y), (x_cols[-1], y))
    for x in x_cols:
        page.draw_line((x, y_rows[0]), (x, y_rows[-1]))

    rows = [
        ["Model", "Accuracy", "Latency"],
        ["Model A", "91.2%", "120 ms"],
        ["Model B", "85.4%", "80 ms"],
    ]
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            page.insert_text((x_cols[c_idx] + 8, y_rows[r_idx] + 20), val, fontsize=10)

    doc.save(str(path))
    doc.close()


def _make_english_trial_pdf(path):
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((60, 80), "Orchard irrigation trial ran for 21 days with drip sensors.", fontsize=11)
    page.insert_text((60, 120), "Greenhouse water study lasted 35 days under controlled temperature.", fontsize=11)
    doc.save(str(path))
    doc.close()


def test_two_column_pdf_preserves_column_reading_order(tmp_path):
    clear_pdf_cache()
    pdf = tmp_path / "two_col.pdf"
    _make_two_column_pdf(pdf)
    chunks = load_or_parse_selected_pdf(str(pdf))
    text_chunks = [c for c in chunks if c.get("unit_type") == "text"]
    assert text_chunks
    combined = " ".join(c["text"] for c in text_chunks)
    assert combined.index("LEFT_FIRST") < combined.index("LEFT_SECOND") < combined.index("RIGHT_FIRST") < combined.index("RIGHT_SECOND")
    assert "Introduction" in (text_chunks[0].get("section_title") or "")


def test_pdf_table_extracted_as_intact_markdown_chunk(tmp_path):
    clear_pdf_cache()
    pdf = tmp_path / "with_table.pdf"
    _make_table_pdf(pdf)
    chunks = load_or_parse_selected_pdf(str(pdf))
    table_chunks = [c for c in chunks if c.get("unit_type") == "table"]
    assert len(table_chunks) == 1
    table_text = table_chunks[0]["text"]
    assert "Table 1" in table_text
    assert "Model A" in table_text and "91.2%" in table_text
    assert "Model B" in table_text and "85.4%" in table_text
    assert "|" in table_text
    assert table_chunks[0]["id"].endswith("_p1_t1")


def test_cross_lingual_vietnamese_query_retrieves_english_pdf(tmp_path, monkeypatch):
    clear_pdf_cache()
    pdf = tmp_path / "orchard.pdf"
    _make_english_trial_pdf(pdf)
    monkeypatch.setattr(settings, "GEMINI_DENSE_ENABLED", False)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")

    chunks = load_or_parse_selected_pdf(str(pdf))
    bridged = build_cross_lingual_subqueries("Thử nghiệm vườn cây tưới tiêu kéo dài bao nhiêu ngày?", chunks)
    assert bridged
    assert any("orchard" in q.lower() and "days" in q.lower() for q in bridged)

    agent = GeminiRetrievalAgent()
    result = agent.run({
        "query": "Thử nghiệm vườn cây tưới tiêu kéo dài bao nhiêu ngày?",
        "pdf_path": str(pdf),
        "pdf_paths": [str(pdf)],
        "top_k": 5,
    })
    assert result["retrieved_docs"]
    assert any("21 days" in d["text"] for d in result["retrieved_docs"])


def test_hybrid_retrieval_uses_kaggle_dense_and_guarded_rerank_with_fallback(tmp_path, monkeypatch):
    clear_pdf_cache()
    pdf = tmp_path / "multi_page.pdf"
    doc = fitz.open()
    for p in range(1, 6):
        page = doc.new_page()
        page.insert_text((60, 80), f"Page {p} discusses quantum lattice routing metric {p * 10}%.", fontsize=11)
    doc.save(str(pdf))
    doc.close()

    monkeypatch.setattr(settings, "REMOTE_INFERENCE_URL", "https://kaggle.example.invalid")
    monkeypatch.setattr(settings, "HYBRID_DENSE_ENABLED", True)
    monkeypatch.setattr(settings, "HYBRID_RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "HYBRID_RERANK_KEEP_PREFIX", 3)
    monkeypatch.setattr(settings, "GEMINI_DENSE_ENABLED", False)
    monkeypatch.setattr("modes.hybrid.retrieval.probe_all_capabilities", lambda: {
        "gemini": {"online": False},
        "gpu_finetune": {"online": True},
    })

    fake_rev_embed = "a" * 40
    fake_rev_rerank = "b" * 40

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        if request.url.path == "/api/embed":
            texts = body["texts"]
            vectors = []
            for t in texts:
                vec = [0.0] * 1024
                if "Page 5" in t or "metric" in t.lower():
                    vec[0] = 1.0
                else:
                    vec[1] = 1.0
                vectors.append(vec)
            return httpx.Response(200, json={
                "embeddings": vectors,
                "model": settings.EMBEDDING_MODEL,
                "revision": fake_rev_embed,
            })
        if request.url.path == "/api/rerank":
            candidates = body["candidates"]
            scores = [float(i) for i in range(len(candidates))]
            ranked_indices = sorted(range(len(candidates)), key=scores.__getitem__, reverse=True)
            return httpx.Response(200, json={
                "scores": scores,
                "ranked_indices": ranked_indices,
                "model": settings.RERANKER_MODEL,
                "revision": fake_rev_rerank,
            })
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(llm_client, "http_client", httpx.Client(transport=transport))

    agent = HybridRetrievalAgent()
    result = agent.run({
        "query": "What is the routing metric on Page 5?",
        "pdf_path": str(pdf),
        "pdf_paths": [str(pdf)],
        "top_k": 5,
    })
    trace_details = result["reasoning_trace"][-1]["details"]
    assert trace_details["dense"]["status"] == "used"
    assert trace_details["dense"]["backend"] == "kaggle_bge_m3"
    assert trace_details["rerank"]["status"] == "used"
    assert trace_details["rerank"]["kept_prefix"] == 3

    # Now simulate Kaggle worker 503 outage -> must fall back cleanly to sparse without raising
    def failing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "worker switching"})

    monkeypatch.setattr(llm_client, "http_client", httpx.Client(transport=httpx.MockTransport(failing_handler)))
    fallback_res = agent.run({
        "query": "What is the routing metric on Page 5?",
        "pdf_path": str(pdf),
        "pdf_paths": [str(pdf)],
        "top_k": 5,
    })
    fb_details = fallback_res["reasoning_trace"][-1]["details"]
    assert fb_details["dense"]["status"] == "fallback"
    assert fb_details["rerank"]["status"] == "skipped"
    assert fallback_res["retrieved_docs"]


def test_gemini_retrieval_prioritizes_table_chunks_for_table_queries(tmp_path, monkeypatch):
    clear_pdf_cache()
    pdf = tmp_path / "table_priority.pdf"
    _make_table_pdf(pdf)
    monkeypatch.setattr(settings, "GEMINI_DENSE_ENABLED", False)

    agent = GeminiRetrievalAgent()
    result = agent.run({
        "query": "So sánh accuracy trong bảng Table 1 của Model A và Model B",
        "pdf_path": str(pdf),
        "pdf_paths": [str(pdf)],
        "top_k": 3,
    })
    assert result["retrieved_docs"]
    assert any(d.get("unit_type") == "table" for d in result["retrieved_docs"])


def test_verifier_catches_quantified_single_digit_hallucinations():
    # Quantified single digits must be extracted
    nums = extract_numbers_and_years("The architecture uses 7 layers across 2 GPUs for 5 epochs.")
    assert {"7", "2", "5"}.issubset(set(nums))

    # Leading list ordinals and figure/table numbers must NOT trigger false positives
    list_nums = extract_numbers_and_years("1. According to Table 2 and Figure 3 on Page 4, accuracy improved.")
    assert list_nums == []

    # Fabricated single-digit layer count must fail numerical consistency
    num_ok, date_ok, missing = check_numerical_and_date_consistency(
        "The encoder uses 7 layers.",
        "The encoder uses 3 layers with hidden size 768.",
    )
    assert not num_ok
    assert any("7" in m for m in missing)


def test_verifier_catches_entity_value_swap_in_local_and_nli_review():
    units = [
        {"id": "u1", "doc_id": "paper", "page_no": 1, "text": "Model A achieved 91.2% accuracy on the test split."},
        {"id": "u2", "doc_id": "paper", "page_no": 1, "text": "Model B achieved 85.4% accuracy on the test split."},
    ]
    swapped_claim = "Model A achieved 85.4% accuracy on the test split."

    # 1. Direct entity-value binding check catches swapped model name
    ok, missing = check_entity_value_binding(swapped_claim, units[1]["text"])
    assert not ok
    assert "Model A" in missing

    # 2. Deterministic offline verifier rejects swapped claim as CONTRADICT (even though 85.4% exists in u2!)
    verified, rejected, _ = deterministic_verify_claims(
        [{"id": "c1", "text": swapped_claim}],
        units,
    )
    assert not verified
    assert len(rejected) == 1
    assert rejected[0]["status"] == "CONTRADICT"

    # 3. Even if an NLI model mistakenly votes SUPPORT quoting Model B's sentence, review_claims downgrades it
    state = {
        "query": "What accuracy did Model A achieve?",
        "answer": swapped_claim,
        "claims": [{"id": "c1", "text": swapped_claim, "evidence_ids": ["u2"]}],
        "source_units": units,
    }
    def mistaken_reviewer(pairs):
        return [{"claim_id": "c1", "status": "SUPPORT", "quote": units[1]["text"], "rationale": "matched 85.4%"}]

    result = review_claims(state, reviewer=mistaken_reviewer)
    assert not result["verified_claims"]
    assert result["fact_check_label"] == "not_enough_info"
    assert result["rejected_claims"][0]["claim_id"] == "c1"
