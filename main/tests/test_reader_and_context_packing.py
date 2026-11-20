"""
Automated Multi-Source Reader & Context Packing Coverage Suite (Đợt 4 — Criteria E & F).
"""

from __future__ import annotations

from unittest.mock import patch

from agents.agent2_reader.node import ReaderAgent
from agents.agent3_summarizer.components.generator import pack_context_with_coverage
from core.state import SourceUnit


def test_reader_single_pdf_and_multi_pdf_and_retrieved_docs():
    u_pdf1 = SourceUnit(id="p1_u1", text="Transformer uses multi-head attention.", doc_id="doc_1", page_no=2)
    u_pdf2_dup = SourceUnit(id="p2_u1", text="Transformer uses multi-head attention.", doc_id="doc_2", page_no=7)
    u_pdf2_unique = SourceUnit(id="p2_u2", text="Positional encodings inject sequence order.", doc_id="doc_2", page_no=8)

    def _fake_parse(path: str):
        if path == "doc_1.pdf":
            return {"source": "doc_1.pdf", "pages": [2], "sections": []}, [u_pdf1]
        if path == "doc_2.pdf":
            return {"source": "doc_2.pdf", "pages": [7, 8], "sections": []}, [u_pdf2_dup, u_pdf2_unique]
        raise RuntimeError(f"Corrupted PDF: {path}")

    reader = ReaderAgent()

    with patch("agents.agent2_reader.node.parse_pdf_document", side_effect=_fake_parse):
        # 1. Single PDF
        out_single = reader.run({"pdf_path": "doc_1.pdf", "retrieved_docs": []})
        assert out_single["processing_status"] == "SUCCESS"
        assert len(out_single["source_units"]) == 1

        # 2. Multi-PDF + retrieved_docs + duplicate content across doc_1 & doc_2
        out_multi = reader.run(
            {
                "pdf_path": "doc_1.pdf",
                "pdf_paths": ["doc_1.pdf", "doc_2.pdf"],
                "retrieved_docs": [
                    {"id": "ret_1", "doc_id": "ext_web", "page_no": 1, "text": "External benchmark note."}
                ],
            }
        )
        assert out_multi["processing_status"] == "SUCCESS"
        # 3 unique texts: shared passage (merged), doc_2 unique, external retrieved doc
        assert len(out_multi["source_units"]) == 3

        shared_unit = next(
            u for u in out_multi["source_units"] if "multi-head attention" in u["text"]
        )
        prov_pairs = {(p["doc_id"], p["page_no"]) for p in shared_unit["provenance"]}
        assert ("doc_1", 2) in prov_pairs
        assert ("doc_2", 7) in prov_pairs

        # 3. Partial PDF failure (corrupted.pdf fails, doc_1.pdf succeeds -> valid units preserved!)
        out_partial_fail = reader.run(
            {"pdf_paths": ["corrupted.pdf", "doc_1.pdf"], "retrieved_docs": []}
        )
        assert out_partial_fail["processing_status"] == "SUCCESS"
        assert out_partial_fail["error_code"] == "PARTIAL_EXTRACTION_FAILURE"
        assert len(out_partial_fail["source_units"]) == 1

        # 4. Total PDF failure with no retrieved_docs -> EXTRACTION_FAILED
        out_total_fail = reader.run({"pdf_path": "corrupted.pdf", "retrieved_docs": []})
        assert out_total_fail["processing_status"] == "EXTRACTION_FAILED"
        assert out_total_fail["error_stage"] == "reader"
        assert out_total_fail["source_units"] == []


def test_context_packing_tracks_accurate_coverage():
    units = [
        {"id": f"u_{i}", "doc_id": "doc_1", "page_no": i + 1, "text": "word " * 80}
        for i in range(10)
    ]

    # Tight budget -> truncated=True with accurate counts
    _, included, cov = pack_context_with_coverage(units, max_tokens=500)
    assert cov["total_units"] == 10
    assert cov["included_units"] == len(included)
    assert cov["omitted_units"] == 10 - len(included)
    assert cov["omitted_units"] > 0
    assert cov["truncated"] is True
    assert len(cov["omitted_unit_ids"]) == cov["omitted_units"]

    # Generous budget -> truncated=False, omitted_units=0
    _, included_all, cov_all = pack_context_with_coverage(units, max_tokens=20000)
    assert cov_all["total_units"] == 10
    assert cov_all["included_units"] == 10
    assert cov_all["omitted_units"] == 0
    assert cov_all["truncated"] is False
    assert cov_all["omitted_unit_ids"] == []
