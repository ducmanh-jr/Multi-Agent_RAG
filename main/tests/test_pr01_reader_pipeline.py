"""PR-01 Shared Reader Pipeline verification:
1. Exact-text deduplication across multiple documents while preserving multi-source provenance.
2. Strict preservation of primary SourceUnit context_hash for CitationAudit & ComparisonPlan.
3. Resilient partial-failure handling: one corrupted/missing PDF never discards units from valid PDFs
   across cpu_min, cpu_max, gemini, and gpu_finetune Readers.
"""

from __future__ import annotations

import fitz
import pytest

from core.state import compute_context_hash
from core.tools import clear_pdf_cache
from modes.cpu_max.reader import CPUMaxReaderAgent
from modes.cpu_min.reader import CPUMinReaderAgent
from modes.gemini.reader import GeminiReaderAgent
from modes.gpu_finetune.reader import GPUFinetuneReaderAgent


def _make_pdf(path, text: str) -> str:
    with fitz.open() as doc:
        page = doc.new_page(width=500, height=400)
        page.insert_text((40, 80), text, fontsize=11)
        doc.save(str(path))
    return str(path)


@pytest.mark.parametrize(
    "reader_cls",
    [CPUMinReaderAgent, CPUMaxReaderAgent, GeminiReaderAgent, GPUFinetuneReaderAgent],
)
def test_readers_deduplicate_identical_text_and_preserve_multi_source_provenance(reader_cls):
    shared_text = "Self-attention relates different positions of a single sequence to compute a representation."
    docs = [
        {"id": "paperA_p5", "doc_id": "PaperA", "page_no": 5, "text": shared_text},
        {"id": "paperB_p12", "doc_id": "PaperB", "page_no": 12, "text": shared_text},
        {"id": "paperB_p13", "doc_id": "PaperB", "page_no": 13, "text": "Multi-head attention projects queries in parallel."},
    ]

    result = reader_cls().run({"query": "Explain self-attention", "retrieved_docs": docs})
    units = result["source_units"]
    assert len(units) == 2

    merged = units[0]
    assert merged["doc_id"] == "PaperA"
    assert merged["page_no"] == 5
    # Primary context_hash must remain strictly valid for CitationAudit
    assert merged["context_hash"] == compute_context_hash(merged["text"], "PaperA", 5)
    # Multi-source provenance retains BOTH PaperA (p5) and PaperB (p12)
    prov_pairs = {(p["doc_id"], p["page_no"]) for p in merged["provenance"]}
    assert prov_pairs == {("PaperA", 5), ("PaperB", 12)}


@pytest.mark.parametrize(
    "reader_cls",
    [CPUMinReaderAgent, CPUMaxReaderAgent, GeminiReaderAgent, GPUFinetuneReaderAgent],
)
def test_readers_handle_partial_pdf_failure_without_losing_valid_pdf(tmp_path, reader_cls):
    clear_pdf_cache()
    corrupted = tmp_path / "corrupted.pdf"
    corrupted.write_bytes(b"NOT A VALID PDF FILE CONTENT")
    missing = tmp_path / "does_not_exist.pdf"
    valid = _make_pdf(tmp_path / "valid_study.pdf", "Valid clinical trial achieved 94.5% sensitivity.")

    result = reader_cls().run({
        "query": "What sensitivity was achieved?",
        "pdf_path": str(corrupted),
        "pdf_paths": [str(corrupted), str(missing), valid],
        "retrieved_docs": [],
    })
    units = result["source_units"]
    assert units, f"{reader_cls.__name__} failed to extract units from valid PDF after partial failure"
    assert any("94.5%" in u["text"] for u in units)

    trace_details = result["reasoning_trace"][-1]["details"]
    assert trace_details["failed_pdf_count"] == 2
    failed_paths = {item["path"] for item in trace_details["failed_pdfs"]}
    assert str(corrupted) in failed_paths and str(missing) in failed_paths


def test_multi_pdf_fallback_deduplicates_across_two_pdfs_and_tracks_provenance(tmp_path):
    clear_pdf_cache()
    shared = "Scaled dot-product attention divides by the square root of dimension dk."
    pdf_a = _make_pdf(tmp_path / "paper_a.pdf", shared)
    pdf_b = _make_pdf(tmp_path / "paper_b.pdf", shared)

    result = CPUMinReaderAgent().run({
        "query": "Explain scaled dot-product attention",
        "pdf_paths": [pdf_a, pdf_b],
        "retrieved_docs": [],
    })
    assert len(result["source_units"]) == 1
    unit = result["source_units"][0]
    assert {p["doc_id"] for p in unit["provenance"]} == {"paper_a", "paper_b"}
    assert result["document_schema"]["dedup_merged_count"] == 1
