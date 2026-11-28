"""
Milestone M2a End-to-End Regression Suite:
1. Bilingual scan OCR at ingest + figure selection without explicit 'figure'/'hình' keyword.
2. 2D stacked-fraction (`y = a/b`) & multi-span equation reconstruction + real `1911.10635v2.pdf` page 5 check.
3. Region-level citation numbering (`[1]`, `[2]` for two distinct `bbox` regions on the same page) + `/crop` PNG endpoint.
4. Rotated PDF (`90°`, `270°`) PDF.js viewport coordinate transformation & upright region crop rendering.
5. Header/unit-repeating `<= 512`-token table sharding + deterministic row/column cell-swap rejection (`CONTRADICT`).
6. Chart entity–value tuple swap rejection (`entity_value_conflict`) & `printed_values` vs `axis_estimated_values`.
7. Per-page fault isolation, upload SHA-256 + `pages_status`, zero parser-error `SourceUnit`s, and live `/api/reader/layout` PNG probe.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import fitz
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from agents.agent1_retrieval.tools.semantic_pipeline import extract_searchable_figure_chunks
from agents.agent5_writer.components.report_builder import build_academic_report
from api import app as api_module, auth
from api.citations import citation_references, format_run_evidence, pin_citation_provenance
from core.citation_audit import write_with_citation_audit
from core.config import settings
from core.figure_qa import _select, cross_check_figure_reading
from core.figure_reader import _parse_structured_vision
from core.pdf_figures import extract_pdf_figures
from core.pdf_ingest import (
    MAX_TABLE_SHARD_TOKENS,
    build_pdfjs_region_descriptor,
    clear_ingest_caches,
    compute_file_sha256,
    extract_page_equations_2d,
    extract_page_tables_sharded,
    get_ingest_manifest,
    is_parser_error_text,
    render_pdf_region_crop,
)
from core.tools import clear_pdf_cache as clear_pdf_chunk_cache, load_or_parse_selected_pdf
from core.verifier import deterministic_verify_claims, review_claims
from modes.cpu_min.writer import CPUMinWriterAgent
from modes.gpu_finetune.reader import GPUFinetuneReaderAgent


def _make_scanned_pdf(path: Path, text_lines: list[str], *, rotation: int = 0) -> str:
    """Renders text into a high-DPI raster image and embeds ONLY the image (zero PDF text layer)."""
    with fitz.open() as src:
        p = src.new_page(width=520, height=260)
        y = 70
        for ln in text_lines:
            p.insert_text((36, y), ln, fontsize=22, color=(0, 0, 0))
            y += 42
        png_bytes = p.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False).tobytes("png")

    with fitz.open() as doc:
        page = doc.new_page(width=520, height=260)
        page.insert_image(page.rect, stream=png_bytes)
        if rotation:
            page.set_rotation(rotation)
        doc.save(str(path))
    return str(path)


def _make_two_equations_same_page_pdf(path: Path) -> str:
    """
    Creates a 1-page PDF containing:
      - Stacked fraction equation 1: `y = a / b (1)` drawn with horizontal fraction bar.
      - Display equation 2: `z = c^2 + 1 (2)` followed by `where c > 0 is the gain factor.`
    """
    with fitz.open() as doc:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 70), "Section 2. Mathematical Formulation", fontsize=13)
        page.insert_text((72, 105), "The primary ratio is defined by the stacked fraction below:", fontsize=11)

        # Equation 1: y = \frac{a}{b} (1) around y=145..175
        page.insert_text((190, 158), "y =", fontsize=12)
        page.insert_text((232, 149), "a", fontsize=12)
        page.draw_line((224, 154), (252, 154), color=(0, 0, 0), width=1.0)
        page.insert_text((232, 168), "b", fontsize=12)
        page.insert_text((460, 158), "(1)", fontsize=11)
        page.insert_text((72, 195), "where b != 0 denotes the normalization denominator.", fontsize=11)

        # Equation 2: z = c^2 + 1 (2) around y=260
        page.insert_text((190, 260), "z = c^2 + 1 (2)", fontsize=12)
        page.insert_text((72, 290), "where c > 0 is the gain factor.", fontsize=11)
        doc.save(str(path))
    return str(path)


def _make_wide_long_table_pdf(path: Path, *, num_rows: int = 34) -> str:
    """Creates a bordered multi-column benchmark table exceeding 512 subword tokens."""
    with fitz.open() as doc:
        page = doc.new_page(width=680, height=1200)
        page.insert_text((50, 42), "Table 1: Multi-split evaluation across benchmark variants.", fontsize=11)

        x_cols = [50, 200, 340, 480, 620]
        row_h = 24
        y0 = 58
        headers = ["Method", "Validation Acc (%)", "Test Acc (%)", "Latency (ms)"]
        rows = [headers]
        rows.append(["MethodR01_BaselineConfig", "84.2%", "91.4%", "12.5 ms"])
        rows.append(["MethodR02_AblationConfig", "88.0%", "95.8%", "18.3 ms"])
        for i in range(3, num_rows + 1):
            rows.append([
                f"MethodR{i:02d}_ExtendedVariantArchitecture",
                f"{70.0 + (i % 15):.1f}%",
                f"{75.0 + (i % 17):.1f}%",
                f"{20.0 + i:.1f} ms",
            ])

        total_h = row_h * len(rows)
        for r_i in range(len(rows) + 1):
            y = y0 + r_i * row_h
            page.draw_line((x_cols[0], y), (x_cols[-1], y), color=(0, 0, 0), width=0.7)
        for x in x_cols:
            page.draw_line((x, y0), (x, y0 + total_h), color=(0, 0, 0), width=0.7)

        for r_i, row in enumerate(rows):
            cy = y0 + r_i * row_h + 16
            for c_i, cell in enumerate(row):
                page.insert_text((x_cols[c_i] + 4, cy), cell, fontsize=8.5)

        doc.save(str(path))
    return str(path)


# =====================================================================
# 1. Scanned PDF OCR at Ingest & Non-'Figure' Query Retrieval
# =====================================================================


def test_scanned_pdf_ocr_produces_chunks_and_answers_plain_query(tmp_path, monkeypatch):
    clear_pdf_chunk_cache()
    clear_ingest_caches()
    monkeypatch.setattr(settings, "CORPUS_PATH", str(tmp_path / "empty_corpus"))

    scan_pdf = _make_scanned_pdf(tmp_path / "scan_acc.pdf", ["Accuracy was 83 percent."])

    # Verify zero native text layer in the PDF
    with fitz.open(scan_pdf) as doc:
        assert doc[0].get_text("text").strip() == ""

    chunks = load_or_parse_selected_pdf(scan_pdf)
    assert len(chunks) >= 1
    ocr_chunks = [c for c in chunks if c.get("unit_type") == "ocr_scan"]
    assert ocr_chunks, f"Expected ocr_scan chunk, got: {chunks}"
    combined_ocr = " ".join(c["text"] for c in ocr_chunks).lower()
    assert "83" in combined_ocr and "accuracy" in combined_ocr
    assert len(ocr_chunks[0]["bbox"]) == 4
    assert len(ocr_chunks[0]["document_sha256"]) == 64

    # Semantic pipeline figure indexer also surfaces scan OCR chunk
    fig_chunks = extract_searchable_figure_chunks([scan_pdf])
    assert any("83" in fc["text"] for fc in fig_chunks)

    # Figure selector selects the scan page even when query does NOT contain 'figure' / 'hình'
    catalog = extract_pdf_figures(scan_pdf)
    assert len(catalog) == 1 and catalog[0]["kind"] == "scan"
    sel = _select("What accuracy was reported?", catalog)
    assert sel["status"] == "matched"
    assert len(sel["figures"]) == 1

    # End-to-end graph answers plain query with 83 and citation [1]
    from core.graph import build_research_rag_graph

    graph = build_research_rag_graph("cpu_min")
    result = graph.invoke({"query": "What accuracy was reported?", "pdf_path": scan_pdf})
    assert "83" in result["answer"]
    assert "[1]" in result["answer"]


# =====================================================================
# 2. Stacked Fractions, Real 1911.10635v2 Page 5, & Same-Page [1]/[2] Region Citations
# =====================================================================


def test_stacked_fraction_and_two_same_page_equations_get_distinct_region_citations(tmp_path, monkeypatch):
    clear_pdf_chunk_cache()
    clear_ingest_caches()
    eq_pdf = _make_two_equations_same_page_pdf(tmp_path / "equations_page1.pdf")
    chunks = load_or_parse_selected_pdf(eq_pdf)

    eq_chunks = [c for c in chunks if c.get("unit_type") == "equation"]
    assert len(eq_chunks) >= 2

    frac_chunk = next((c for c in eq_chunks if "\\frac{a}{b}" in c["text"]), None)
    assert frac_chunk is not None, f"Stacked fraction not reconstructed! Equations: {[c['text'] for c in eq_chunks]}"
    assert frac_chunk["equation_metadata"]["numerator"] == "a"
    assert frac_chunk["equation_metadata"]["denominator"] == "b"
    assert frac_chunk["equation_metadata"]["equation_number"] == "(1)"
    assert any("b != 0" in d for d in frac_chunk["equation_metadata"]["domain_constraints"])

    second_eq = next((c for c in eq_chunks if "c^2 + 1" in c["text"]), None)
    assert second_eq is not None
    assert second_eq["equation_metadata"]["equation_number"] == "(2)"
    assert any("c > 0" in d for d in second_eq["equation_metadata"]["domain_constraints"])
    assert frac_chunk["bbox"] != second_eq["bbox"]

    # Verify two distinct equations on the SAME page 1 receive [1] and [2] (NOT collapsed to [1]!)
    q1 = frac_chunk["text"].splitlines()[0]
    q2 = second_eq["text"].splitlines()[0]
    state = {
        "query": "State both equations (1) and (2) on page 1.",
        "answer_language": "en",
        "source_units": [
            {
                "id": frac_chunk["id"],
                "doc_id": "equations_page1",
                "page_no": 1,
                "text": frac_chunk["text"],
                "bbox": frac_chunk["bbox"],
                "source_path": eq_pdf,
            },
            {
                "id": second_eq["id"],
                "doc_id": "equations_page1",
                "page_no": 1,
                "text": second_eq["text"],
                "bbox": second_eq["bbox"],
                "source_path": eq_pdf,
            },
        ],
        "evidence_spans": [
            {
                "id": "ev_eq1",
                "source_unit_id": frac_chunk["id"],
                "doc_id": "equations_page1",
                "page_no": 1,
                "quote": q1,
                "bbox": frac_chunk["bbox"],
            },
            {
                "id": "ev_eq2",
                "source_unit_id": second_eq["id"],
                "doc_id": "equations_page1",
                "page_no": 1,
                "quote": q2,
                "bbox": second_eq["bbox"],
            },
        ],
        "claims": [
            {"id": "c1", "text": f"Equation 1 defines {q1}.", "evidence_ids": ["ev_eq1"]},
            {"id": "c2", "text": f"Equation 2 defines {q2}.", "evidence_ids": ["ev_eq2"]},
        ],
        "verified_claims": [
            {"claim_id": "c1", "claim_text": f"Equation 1 defines {q1}.", "status": "SUPPORT", "evidence_ids": ["ev_eq1"]},
            {"claim_id": "c2", "claim_text": f"Equation 2 defines {q2}.", "status": "SUPPORT", "evidence_ids": ["ev_eq2"]},
        ],
    }

    out = write_with_citation_audit(state, CPUMinWriterAgent().run)
    assert "[1]" in out["answer"] and "[2]" in out["answer"]

    doc_entry = {
        "id": "doc_eq1",
        "name": "equations_page1.pdf",
        "path": eq_pdf,
        "catalogued": True,
        "sha256": compute_file_sha256(eq_pdf),
    }
    pin_citation_provenance(out["reasoning_trace"], state, [doc_entry])
    resolved_refs = citation_references(out["reasoning_trace"], state["evidence_spans"], [doc_entry])
    assert len(resolved_refs) == 2
    assert [r["ref_id"] for r in resolved_refs] == [1, 2]
    assert resolved_refs[0]["page"] == 1 and resolved_refs[1]["page"] == 1
    assert resolved_refs[0]["bbox"] != resolved_refs[1]["bbox"]
    assert "crop_url" in resolved_refs[0] and "pdfjs_region" in resolved_refs[0]

    formatted_ev = format_run_evidence(state["evidence_spans"], resolved_refs)
    assert len(formatted_ev[0]["regions"]) == 1
    assert formatted_ev[0]["regions"][0]["bbox"] == resolved_refs[0]["bbox"]

    # Test GET /api/documents/{doc_id}/crop endpoint via FastAPI TestClient
    monkeypatch.setattr(auth, "AUTH_SECRET_PASSWORD", "m2a-test")
    monkeypatch.setattr(auth, "AUTH_DEV_MODE", False)
    monkeypatch.setattr(auth, "_sessions", {})
    monkeypatch.setattr(api_module.db, "get_document", lambda did: doc_entry if did == "doc_eq1" else None)
    with TestClient(api_module.app, base_url="https://workspace.test") as client:
        assert client.post("/api/auth/login", json={"password": "m2a-test"}).status_code == 200
        for ref in resolved_refs:
            resp = client.get(ref["crop_url"])
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("image/png")
            assert resp.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_real_workspace_pdf_1911_10635v2_page5_equations_intact():
    real_pdf = Path(__file__).resolve().parents[1] / "uploads" / "doc_0051427d_1911.10635v2.pdf"
    if not real_pdf.is_file():
        # Create a faithful geometric replica of 1911.10635v2 page 5 when running in clean checkout without uploads/
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp())
        replica = tmp_dir / "replica_1911_10635v2.pdf"
        with fitz.open() as doc:
            for _ in range(4):
                doc.new_page(width=612, height=792)
            p5 = doc.new_page(width=612, height=792)
            p5.insert_htmlbox((158.3, 122.0, 225.0, 145.0), '<span style="font-size:11pt">Qπ(s,a) = E</span>')
            p5.insert_htmlbox((226.6, 138.0, 260.0, 156.0), '<span style="font-size:8pt">t≥0</span>')
            p5.insert_htmlbox((318.9, 122.0, 520.0, 145.0), '<span style="font-size:10pt">γ^t r_t | a_t ∼ π(·|s_t), a_0 = a, s_0 = s</span>')
            p5.insert_htmlbox((170.1, 162.0, 225.0, 185.0), '<span style="font-size:11pt">Vπ(s) = E</span>')
            p5.insert_htmlbox((226.6, 178.0, 260.0, 196.0), '<span style="font-size:8pt">t≥0</span>')
            p5.insert_htmlbox((318.9, 162.0, 520.0, 185.0), '<span style="font-size:10pt">γ^t r_t | a_t ∼ π(·|s_t), s_0 = s</span>')
            p5.insert_htmlbox((140.0, 522.0, 540.0, 548.0), '<span style="font-size:11pt">Q(s,a) ← (1 − α) Q(s,a) + α [r + γ max Q(s\',a\')]</span>')
            p5.insert_htmlbox((90.0, 558.0, 480.0, 580.0), '<span style="font-size:10pt">where α &gt; 0 is the stepsize/learning rate.</span>')
            doc.save(str(replica))
        real_pdf = replica

    sha = compute_file_sha256(real_pdf)
    with fitz.open(str(real_pdf)) as doc:
        eqs = extract_page_equations_2d(
            doc[4],
            doc_title="1911.10635v2",
            doc_sha256=sha,
            page_no=5,
            resolved=str(real_pdf),
            section_title="Reinforcement Learning",
        )
    assert len(eqs) >= 3
    texts = [e["text"] for e in eqs]
    assert any("Q" in t and "t≥0" in t for t in texts), f"Expected intact Q_pi with t>=0 limit, got: {texts}"
    assert any("V" in t and "t≥0" in t for t in texts), f"Expected intact V_pi with t>=0 limit, got: {texts}"
    assert any("α > 0" in t for t in texts), f"Expected Q-learning equation with α > 0 domain, got: {texts}"
    for eq in eqs:
        bx = eq["bbox"]
        assert len(bx) == 4 and bx[2] > bx[0] and bx[3] > bx[1]


# =====================================================================
# 3. Rotated PDF (`90°` & `270°`) PDF.js Viewport Mapping & Crop
# =====================================================================


@pytest.mark.parametrize("rotation", [90, 270])
def test_rotated_pdf_region_mapping_and_crop(tmp_path, rotation):
    pdf_path = tmp_path / f"rotated_{rotation}.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=400, height=200)
        page.draw_rect((20, 20, 120, 70), color=None, fill=(0.9, 0.1, 0.1))
        page.insert_text((25, 50), "x = 42", fontsize=14)
        page.set_rotation(rotation)
        doc.save(str(pdf_path))

    desc = build_pdfjs_region_descriptor(pdf_path, 1, [20.0, 20.0, 120.0, 70.0])
    assert desc is not None
    assert desc["page_rotation"] == rotation
    # Unrotated width=400, height=200 -> rotated 90/270 visual viewport has width=200, height=400!
    assert desc["unrotated_page_size"] == [400.0, 200.0]
    assert desc["page_size"] == [200.0, 400.0]
    nb = desc["normalized_bbox"]
    assert all(0.0 <= v <= 1.0 for v in nb) and nb[2] > nb[0] and nb[3] > nb[1]

    crop_png = render_pdf_region_crop(pdf_path, 1, [20.0, 20.0, 120.0, 70.0])
    pix = fitz.Pixmap(crop_png)
    assert pix.width > 10 and pix.height > 10


# =====================================================================
# 4. Wide/Long Table `<= 512`-Token Sharding & Row/Col Cell-Swap Rejection
# =====================================================================


def test_table_sharding_repeats_headers_and_rejects_cell_swaps(tmp_path):
    table_pdf = _make_wide_long_table_pdf(tmp_path / "bench_table.pdf", num_rows=34)
    sha = compute_file_sha256(table_pdf)
    with fitz.open(table_pdf) as doc:
        shards = extract_page_tables_sharded(
            doc[0],
            doc_title="bench_table",
            doc_sha256=sha,
            page_no=1,
            resolved=table_pdf,
            section_title="Experiments",
        )

    assert len(shards) >= 2, "Expected 34-row table to be sharded below 512 tokens"
    for idx, sh in enumerate(shards, start=1):
        meta = sh["table_metadata"]
        assert meta["estimated_tokens"] <= MAX_TABLE_SHARD_TOKENS < 512
        assert meta["shard_index"] == idx
        assert meta["shard_count"] == len(shards)
        # Every shard MUST repeat caption, column headers, and units!
        assert "Table 1:" in sh["text"]
        assert "Validation Acc (%)" in sh["text"]
        assert "Test Acc (%)" in sh["text"]
        assert "Latency (ms)" in sh["text"]
        assert "%" in meta["units"] and "ms" in meta["units"]

    first_shard = shards[0]
    # True cell: MethodR01_BaselineConfig has Test Acc = 91.4% -> SUPPORT
    verified, rejected, _ = deterministic_verify_claims(
        [
            {"id": "ok", "text": "MethodR01_BaselineConfig achieves 91.4% on Test Acc.", "evidence_ids": [first_shard["id"]]},
            # Split/column swap: 84.2% is Validation Acc, NOT Test Acc!
            {"id": "col_swap", "text": "MethodR01_BaselineConfig achieves 84.2% on Test Acc.", "evidence_ids": [first_shard["id"]]},
            # Row swap: 95.8% belongs to MethodR02_AblationConfig, NOT MethodR01_BaselineConfig!
            {"id": "row_swap", "text": "MethodR01_BaselineConfig achieves 95.8% on Test Acc.", "evidence_ids": [first_shard["id"]]},
        ],
        [first_shard],
    )
    assert [v["claim_id"] for v in verified] == ["ok"]
    rejected_by_id = {r["claim_id"]: r["status"] for r in rejected}
    assert rejected_by_id["col_swap"] == "CONTRADICT"
    assert rejected_by_id["row_swap"] == "CONTRADICT"

    # Also verify external NLI reviewer returning SUPPORT on swapped cell is overridden to CONTRADICT
    nli_state = {
        "query": "What is MethodR01_BaselineConfig Test Acc?",
        "answer": "MethodR01_BaselineConfig achieves 95.8% on Test Acc.",
        "claims": [{"id": "c_swap", "text": "MethodR01_BaselineConfig achieves 95.8% on Test Acc.", "evidence_ids": [first_shard["id"]]}],
        "source_units": [first_shard],
    }
    res_nli = review_claims(
        nli_state,
        reviewer=lambda pairs: [{"claim_id": "c_swap", "status": "SUPPORT", "quote": pairs[0]["evidence"][:120], "rationale": "ok"}],
    )
    assert res_nli["fact_check_label"] == "contradict"
    assert res_nli["rejected_claims"][0]["status"] == "CONTRADICT"


# =====================================================================
# 5. Chart Entity-Value Swap Conflict & Printed vs Axis-Estimated Split
# =====================================================================


def test_chart_entity_value_swap_is_flagged_conflict_never_corroborated():
    # Source PDF says Method A = 81%, Method B = 92%. Vision swaps A and B!
    swapped_fig = {
        "caption": "Figure 2: Accuracy comparison across methods.",
        "reference": {"text": "In Figure 2, Method A achieved 81% accuracy while Method B achieved 92% accuracy."},
        "reading": {
            "method": "vision",
            "engine": "gemini/test",
            "text": "Method A achieved 92% accuracy and Method B achieved 81% accuracy.",
        },
    }
    ver = cross_check_figure_reading(swapped_fig)
    assert ver["status"] == "entity_value_conflict"
    assert ver["conflicts"]

    # Matching entity-value bindings -> corroborated
    aligned_fig = {
        **swapped_fig,
        "reading": {
            "method": "vision",
            "engine": "gemini/test",
            "text": "Method A achieved 81% accuracy and Method B achieved 92% accuracy.",
            "structured": {"printed_values": ["81%", "92%"], "axis_estimated_values": []},
        },
    }
    ver_ok = cross_check_figure_reading(aligned_fig)
    assert ver_ok["status"] == "corroborated"
    assert ver_ok["value_source"] == "printed_direct"

    # Axis-estimated value separated from printed value -> flagged as axis_estimated_values
    raw_json = (
        '{"chart_type": "bar", "axes": "Accuracy (%)", "legend": "A, B", '
        '"printed_values": ["92%"], "axis_estimated_values": ["76%"], '
        '"trend_summary": "B higher than C", "uncertainty": "Bar C has no printed label", '
        '"observation": "Method B prints 92% while Method C is estimated near 76% from the y-axis."}'
    )
    txt, structured = _parse_structured_vision(raw_json)
    assert structured["printed_values"] == ["92%"]
    assert structured["axis_estimated_values"] == ["76%"]
    est_check = cross_check_figure_reading({
        "caption": "Figure 3: Method B reaches 92%.",
        "reading": {"method": "vision", "engine": "gemini/test", "text": txt, "structured": structured},
    })
    assert est_check["status"] == "axis_estimated_values"
    assert est_check["value_source"] == "axis_estimated"


# =====================================================================
# 6. Fault Isolation, Upload SHA-256 + `pages_status`, & Zero Parser-Error SourceUnits
# =====================================================================


def test_per_page_fault_isolation_and_zero_parser_error_source_units(tmp_path, monkeypatch):
    clear_pdf_chunk_cache()
    clear_ingest_caches()

    pdf_path = tmp_path / "three_pages.pdf"
    with fitz.open() as doc:
        p1 = doc.new_page()
        p1.insert_text((72, 100), "Page one reports baseline throughput of 145 req/s in production.")
        p2 = doc.new_page()
        p2.insert_text((72, 100), "Page two content that will simulate a corrupted page exception.")
        p3 = doc.new_page()
        p3.insert_text((72, 100), "Page three reports tail latency p95 of 19 ms under load.")
        doc.save(str(pdf_path))

    real_eq_fn = __import__("core.tools", fromlist=["extract_page_equations_2d"]).extract_page_equations_2d

    def _faulty_page2(page, **kwargs):
        if kwargs.get("page_no") == 2:
            raise RuntimeError("Simulated MuPDF stream corruption on page 2")
        return real_eq_fn(page, **kwargs)

    monkeypatch.setattr("core.tools.extract_page_equations_2d", _faulty_page2)
    chunks = load_or_parse_selected_pdf(str(pdf_path))

    # Pages 1 and 3 MUST succeed even though page 2 raised an exception!
    pages_returned = {c["page_no"] for c in chunks}
    assert 1 in pages_returned and 3 in pages_returned
    manifest = get_ingest_manifest(str(pdf_path.resolve()))
    assert manifest is not None
    assert manifest["status"] == "partial_completed"
    status_by_page = {p["page_no"]: p["status"] for p in manifest["pages"]}
    assert status_by_page[1] == "text_ready"
    assert status_by_page[2] == "failed"
    assert status_by_page[3] == "text_ready"

    # Corrupted file + stale retrieved_docs containing parser error text -> ZERO parser-error SourceUnits!
    bad_pdf = tmp_path / "corrupt.pdf"
    bad_pdf.write_bytes(b"NOT_A_VALID_PDF_BINARY")
    reader = GPUFinetuneReaderAgent()
    monkeypatch.setattr(reader.settings, "REMOTE_INFERENCE_URL", "")
    reader_out = reader.run({
        "query": "What is inside corrupt.pdf?",
        "pdf_path": str(bad_pdf),
        "retrieved_docs": [
            {"id": "err_1", "doc_id": "corrupt", "page_no": 1, "text": "Lỗi trích xuất bảng biểu/công thức (PDF Parser): broken xref"},
            {"id": "valid_1", "doc_id": "three_pages", "page_no": 1, "text": "Baseline throughput was 145 req/s."},
        ],
    })
    units = reader_out["source_units"]
    assert all(not is_parser_error_text(u.get("text")) for u in units)
    assert [u["id"] for u in units] == ["valid_1"]


def test_upload_endpoint_returns_sha256_and_pages_status(tmp_path, monkeypatch, research_db):
    pdf_path = tmp_path / "upload_sample.pdf"
    with fitz.open() as doc:
        p1 = doc.new_page()
        p1.insert_text((72, 100), "Uploaded document page 1 content for SHA verification.")
        doc.save(str(pdf_path))

    expected_sha = compute_file_sha256(pdf_path)
    upload_dir = tmp_path / "uploads_store"
    upload_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(auth, "AUTH_SECRET_PASSWORD", "m2a-test")
    monkeypatch.setattr(auth, "AUTH_DEV_MODE", False)
    monkeypatch.setattr(auth, "_sessions", {})
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(upload_dir))

    with TestClient(api_module.app, base_url="https://workspace.test") as client:
        assert client.post("/api/auth/login", json={"password": "m2a-test"}).status_code == 200
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("upload_sample.pdf", pdf_path.read_bytes(), "application/pdf")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["sha256"] == expected_sha
    assert body["ingest_status"] == "completed"
    assert len(body["pages"]) == 1
    assert body["pages"][0]["page_no"] == 1
    assert body["pages"][0]["status"] == "text_ready"


# =====================================================================
# 7. `/api/reader/layout` Live PNG Handler & `WorkerManager` Capability Probe
# =====================================================================


def test_layout_reader_endpoint_and_worker_manager_live_png_probe(tmp_path, monkeypatch):
    from kaggle import kaggle_finetune_server as srv
    from kaggle.worker_manager import WorkerManager

    # Render a small valid PNG
    with fitz.open() as d:
        p = d.new_page(width=120, height=80)
        p.insert_text((15, 45), "E = mc^2", fontsize=14)
        png_bytes = p.get_pixmap().tobytes("png")
    b64_png = base64.b64encode(png_bytes).decode("ascii")

    # 1. Disabled layout_reader -> HTTP 501
    with patch.dict(srv._models, {}, clear=True), monkeypatch.context() as m:
        m.delenv("ENABLE_LAYOUT_READER", raising=False)
        with pytest.raises(HTTPException) as exc_info:
            srv.reader_layout_endpoint(srv.ReaderLayoutRequest(pdf_path="doc.pdf", pages=[1]))
        assert exc_info.value.status_code == 501

    # 2. Enabled layout_reader with crop PNG -> HTTP 200 + valid 4-coord bbox + provenance headers
    from fastapi import Response

    with monkeypatch.context() as m:
        m.setenv("ENABLE_LAYOUT_READER", "true")
        req = srv.ReaderLayoutRequest(
            pdf_path="doc.pdf",
            pages=[1],
            images=[
                srv.LayoutImagePayload(
                    page_no=1,
                    kind="crop",
                    image_b64=b64_png,
                    bbox_hint=[50.0, 100.0, 220.0, 160.0],
                )
            ],
        )
        http_resp = Response()
        payload = srv.reader_layout_endpoint(req, response=http_resp)
        assert http_resp.headers.get("X-Research-Model")
        assert payload["status"] == "ok"
        assert len(payload["regions"]) == 1
        bbox = payload["regions"][0]["bbox"]
        assert len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]

    # 3. WorkerManager only advertises layout_reader=True AFTER live PNG probe succeeds
    with patch.object(WorkerManager, "_load_manifest", return_value={"checkpoints": [], "base_models": {}}), \
         patch.object(WorkerManager, "validate_manifest_checkpoints", return_value={"router": True, "generator": True, "critic": True, "writer": True, "fact_checker_nli": True, "layout_reader": True}):
        wm = WorkerManager(str(tmp_path))
        wm.active_worker = "gpu_finetune"
        # Before live probe: MUST be False even if disk checkpoint validated!
        assert wm.get_status_payload()["active_worker_capabilities"]["layout_reader"] is False

        # Probe fails (501) -> stays False
        fail_client = MagicMock()
        fail_client.post = AsyncMock(return_value=MagicMock(status_code=501))
        assert asyncio.run(wm._probe_layout_reader_capability(fail_client, "http://127.0.0.1:8001")) is False
        assert wm.get_status_payload()["active_worker_capabilities"]["layout_reader"] is False

        # Probe succeeds (200 with valid 4-coord bbox) -> becomes True!
        ok_resp = MagicMock(status_code=200)
        ok_resp.json.return_value = {"status": "ok", "regions": [{"page_no": 1, "bbox": [10.0, 10.0, 90.0, 60.0]}]}
        ok_client = MagicMock()
        ok_client.post = AsyncMock(return_value=ok_resp)
        assert asyncio.run(wm._probe_layout_reader_capability(ok_client, "http://127.0.0.1:8001")) is True
        assert wm.get_status_payload()["active_worker_capabilities"]["layout_reader"] is True
