# -*- coding: utf-8 -*-
"""
Milestone M2b Comprehensive Verification Suite:
1. Cross-page table stitching across 2–3 consecutive pages (vector grid tables & pipe tables,
   headerless continuations & explicit `(continued)` / `(tiếp theo)` captions) while keeping
   distinct tables (`Table 1` vs `Table 2`) strictly separate, enforcing `<= 512` token shards,
   and preserving exact physical `(page_no, bbox)` per row group for PDF.js navigation.
2. Cross-page `derived_claim` verification (binding operands from Page 1 & Page 3 of the same
   logical table, catching cross-split traps and swapped continuation cells).
3. 2D Vector Box-and-Arrow Flowchart / Architecture Diagram extraction (`nodes`, `edges`,
   triangle arrowheads, `[Yes]`/`[No]`/`[Đúng]`/`[Sai]` branch labels) + inline arrow specs.
4. Deterministic diagram graph path & branch condition verification in `verifier` (blocking
   reversed arrows, skipped-hop direct edge claims, and swapped branch conditions).
5. Multi-document accuracy test across an 8-PDF bilingual corpus combining multi-page tables
   and branching architecture diagrams simultaneously.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz
import pytest

from agents.agent2_reader.tools.reader_pipeline import build_source_units_from_docs
from core.config import RunConfig, use_run_config
from core.graph import build_research_rag_graph
from core.pdf_ingest import (
    MAX_TABLE_SHARD_TOKENS,
    build_pdfjs_region_descriptor,
    clear_ingest_caches,
    get_ingest_manifest,
)
from core.tools import clear_pdf_cache, load_or_parse_selected_pdf
from core.verifier import deterministic_verify_claims, review_claims
from modes.cpu_min.retrieval import CPUMinRetrievalAgent


def _draw_vector_grid_table(
    page: fitz.Page,
    rows: list[list[str]],
    *,
    x0: float = 48.0,
    y0: float = 90.0,
    col_widths: tuple[float, ...] = (140.0, 120.0, 120.0, 115.0),
    row_h: float = 28.0,
    caption: str = "",
) -> list[float]:
    """Draws a genuine vector-bordered table on `page` so `page.find_tables()` extracts exact cells & bbox."""
    if caption:
        page.insert_htmlbox(
            fitz.Rect(x0, max(20.0, y0 - 34.0), x0 + sum(col_widths), y0 - 4.0),
            f'<div style="font-family:monospace; font-size:10pt;">{caption}</div>',
        )
    total_w = sum(col_widths)
    total_h = row_h * len(rows)
    # Horizontal lines
    for r_i in range(len(rows) + 1):
        y = y0 + r_i * row_h
        page.draw_line((x0, y), (x0 + total_w, y), color=(0, 0, 0), width=0.8)
    # Vertical lines
    cx = x0
    for cw in col_widths:
        page.draw_line((cx, y0), (cx, y0 + total_h), color=(0, 0, 0), width=0.8)
        cx += cw
    page.draw_line((x0 + total_w, y0), (x0 + total_w, y0 + total_h), color=(0, 0, 0), width=0.8)

    # Cell text
    for r_i, row in enumerate(rows):
        cy = y0 + r_i * row_h
        cx = x0
        for c_i, val in enumerate(row):
            cw = col_widths[c_i]
            page.insert_htmlbox(
                fitz.Rect(cx + 4.0, cy + 4.0, cx + cw - 4.0, cy + row_h - 2.0),
                f'<div style="font-family:monospace; font-size:9pt;">{val}</div>',
            )
            cx += cw
    return [x0, y0, x0 + total_w, y0 + total_h]


def _draw_arrow_with_triangle_head(
    page: fitz.Page,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str = "",
) -> None:
    """Draws a vector connector segment from `start` to `end` with a crisp triangle arrowhead at `end`."""
    import math

    sx, sy = start
    ex, ey = end
    page.draw_line((sx, sy), (ex, ey), color=(0.1, 0.1, 0.1), width=1.2)

    dx, dy = ex - sx, ey - sy
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    px, py = -uy, ux  # perpendicular
    head_len = 8.0
    half_w = 4.0
    b1 = fitz.Point(ex - ux * head_len + px * half_w, ey - uy * head_len + py * half_w)
    b2 = fitz.Point(ex - ux * head_len - px * half_w, ey - uy * head_len - py * half_w)
    tip = fitz.Point(ex, ey)
    shape = page.new_shape()
    shape.draw_polyline([b1, tip, b2, b1])
    shape.finish(color=(0.1, 0.1, 0.1), fill=(0.1, 0.1, 0.1), width=0.8)
    shape.commit()

    if label:
        mx, my = 0.5 * (sx + ex), 0.5 * (sy + ey)
        page.insert_htmlbox(
            fitz.Rect(mx + 4.0, my - 12.0, mx + 58.0, my + 12.0),
            f'<div style="font-family:monospace; font-size:8.5pt;">{label}</div>',
        )


def _draw_node_box(page: fitz.Page, rect: tuple[float, float, float, float], label: str) -> None:
    r = fitz.Rect(*rect)
    page.draw_rect(r, color=(0.1, 0.2, 0.5), fill=(0.95, 0.97, 1.0), width=1.1)
    page.insert_htmlbox(
        fitz.Rect(r.x0 + 5.0, r.y0 + 6.0, r.x1 - 5.0, r.y1 - 4.0),
        f'<div style="font-family:monospace; font-size:9.5pt;">{label}</div>',
    )


def test_m2b_three_page_headerless_continuation_table_stitching_and_pdfjs_bboxes(tmp_path):
    clear_pdf_cache()
    clear_ingest_caches()
    pdf_path = tmp_path / "three_page_benchmark.pdf"

    with fitz.open() as doc:
        # Page 1: Table 1 starts near middle/bottom with explicit headers & units
        p1 = doc.new_page(width=595, height=842)
        p1.insert_htmlbox(
            fitz.Rect(48, 42, 545, 95),
            '<div style="font-family:monospace; font-size:11pt;">Section 4. Multi-Page Benchmark Evaluation</div>',
        )
        b_p1 = _draw_vector_grid_table(
            p1,
            [
                ["Model", "Validation (%)", "Test (%)", "Latency (ms)"],
                ["Method-Alpha", "81%", "78%", "24.5 ms"],
                ["Method-Beta", "84%", "80%", "21.0 ms"],
            ],
            y0=520.0,
            caption="Table 1: Comprehensive Accuracy and Latency Across Models",
        )

        # Page 2: Headerless continuation at top of page (NO header row repeated!)
        p2 = doc.new_page(width=595, height=842)
        b_p2 = _draw_vector_grid_table(
            p2,
            [
                ["Method-Gamma", "87%", "83%", "18.2 ms"],
                ["Method-Delta", "89%", "85%", "16.4 ms"],
            ],
            y0=68.0,
        )

        # Page 3: Explicit `(continued)` continuation at top + a SEPARATE `Table 2` below!
        p3 = doc.new_page(width=595, height=842)
        b_p3 = _draw_vector_grid_table(
            p3,
            [
                ["Method-Epsilon", "91%", "87%", "15.1 ms"],
                ["Method-Omega", "94%", "90%", "12.8 ms"],
            ],
            y0=88.0,
            caption="Table 1 (continued)",
        )
        _draw_vector_grid_table(
            p3,
            [
                ["Hardware", "VRAM (GB)", "TDP (W)", "Nodes"],
                ["Tesla-T4", "16 GB", "70 W", "2"],
            ],
            y0=420.0,
            caption="Table 2: Cluster Hardware Specification",
        )
        doc.save(pdf_path, no_new_id=True)

    chunks = load_or_parse_selected_pdf(str(pdf_path))
    manifest = get_ingest_manifest(str(pdf_path))
    assert manifest is not None
    assert len(manifest["stitched_logical_tables"]) == 1
    stitched_info = manifest["stitched_logical_tables"][0]
    assert stitched_info["spanned_pages"] == [1, 2, 3]
    assert stitched_info["total_rows"] == 6
    assert stitched_info["headers"] == ["Model", "Validation (%)", "Test (%)", "Latency (ms)"]

    table_chunks = [c for c in chunks if c.get("unit_type") == "table"]
    ltab1_chunks = [c for c in table_chunks if (c.get("table_metadata") or {}).get("spanned_pages") == [1, 2, 3]]
    ltab2_chunks = [c for c in table_chunks if (c.get("table_metadata") or {}).get("spanned_pages") == [3]]
    assert len(ltab1_chunks) == 3
    assert len(ltab2_chunks) == 1
    assert ltab1_chunks[0]["table_metadata"]["logical_table_id"] != ltab2_chunks[0]["table_metadata"]["logical_table_id"]

    # Every continuation shard repeats canonical headers, stays <= token limit, and keeps its own physical page & bbox!
    for ch, expected_page, expected_box in zip(ltab1_chunks, (1, 2, 3), (b_p1, b_p2, b_p3)):
        tmeta = ch["table_metadata"]
        assert ch["page_no"] == expected_page
        assert tmeta["headers"] == ["Model", "Validation (%)", "Test (%)", "Latency (ms)"]
        assert tmeta["units"] == ["", "%", "%", "ms"]
        assert tmeta["estimated_tokens"] <= MAX_TABLE_SHARD_TOKENS
        assert "| Model | Validation (%) | Test (%) | Latency (ms) |" in ch["text"]
        assert abs(ch["bbox"][1] - expected_box[1]) <= 6.0
        desc = build_pdfjs_region_descriptor(pdf_path, ch["page_no"], ch["bbox"])
        assert desc is not None and desc["bbox"][1] == pytest.approx(ch["bbox"][1], abs=6.0)

    # Verify cross-page `derived_claim` (`Method-Omega` 94% on Page 3 vs `Method-Alpha` 81% on Page 1 = 13 pp on Validation)
    units, *_ = build_source_units_from_docs(ltab1_chunks, limit=10)
    unit_dicts = [u.to_dict() for u in units]

    verified_good, rejected_good, _ = deterministic_verify_claims(
        [{"id": "c_good", "text": "On Validation, Method-Omega exceeds Method-Alpha by 13 percentage points."}],
        unit_dicts,
    )
    assert len(verified_good) == 1 and not rejected_good
    assert verified_good[0]["claim_type"] == "derived_claim"
    # Cites BOTH Page 1 and Page 3 table units!
    cited_pages = {next(u["page_no"] for u in unit_dicts if u["id"] == eid) for eid in verified_good[0]["evidence_ids"]}
    assert cited_pages == {1, 3}

    # Cross-page cross-split trap (94% Validation on Page 3 minus 78% Test on Page 1 = 16 pp) MUST be blocked!
    verified_trap, rejected_trap, _ = deterministic_verify_claims(
        [{
            "id": "c_trap",
            "text": "Method-Omega exceeds Method-Alpha by 16 percentage points.",
            "claim_type": "derived_claim",
            "derivation": {
                "op": "diff_pp",
                "operands": [
                    {"entity": "Method-Omega", "split": "validation", "value": 94.0, "unit": "%"},
                    {"entity": "Method-Alpha", "split": "test", "value": 78.0, "unit": "%"},
                ],
            },
        }],
        unit_dicts,
    )
    assert not verified_trap and rejected_trap[0]["status"] == "CONTRADICT"

    # Swapped column value on Page 2 headerless continuation (`Method-Gamma` Test is 83%, NOT 87%!) MUST be blocked!
    verified_swap, rejected_swap, _ = deterministic_verify_claims(
        [{"id": "c_swap", "text": "On Test, Method-Gamma achieved 87%."}],
        unit_dicts,
    )
    assert not verified_swap and rejected_swap[0]["status"] == "CONTRADICT"


def test_m2b_vietnamese_pipe_table_continuation_across_pages(tmp_path):
    clear_pdf_cache()
    clear_ingest_caches()
    pdf_vi = tmp_path / "bao_cao_bang_lien_trang_vi.pdf"

    with fitz.open() as doc:
        p1 = doc.new_page(width=595, height=842)
        p1.insert_htmlbox(
            fitz.Rect(42, 460, 553, 760),
            '<div style="font-family:monospace; font-size:10pt; line-height:1.4;">'
            "Bảng 1: Kết quả độ chính xác nhận dạng tiếng Việt theo mô hình<br/>"
            "| Mô hình | Tập kiểm định (%) | Độ trễ (ms) |<br/>"
            "| VietRAG-Base | 82.5% | 34.0 ms |<br/>"
            "| VietRAG-Large | 88.0% | 29.5 ms |"
            "</div>",
        )
        p2 = doc.new_page(width=595, height=842)
        p2.insert_htmlbox(
            fitz.Rect(42, 60, 553, 260),
            '<div style="font-family:monospace; font-size:10pt; line-height:1.4;">'
            "Bảng 1 (tiếp theo)<br/>"
            "| VietRAG-Pro | 93.5% | 19.2 ms |<br/>"
            "| VietRAG-Ultra | 95.5% | 16.8 ms |"
            "</div>",
        )
        doc.save(pdf_vi, no_new_id=True)

    chunks = load_or_parse_selected_pdf(str(pdf_vi))
    t_chunks = [c for c in chunks if c.get("unit_type") == "table"]
    assert len(t_chunks) == 2
    assert t_chunks[0]["table_metadata"]["logical_table_id"] == t_chunks[1]["table_metadata"]["logical_table_id"]
    assert t_chunks[1]["table_metadata"]["spanned_pages"] == [1, 2]
    assert "| Mô hình | Tập kiểm định (%) | Độ trễ (ms) |" in t_chunks[1]["text"]

    units, *_ = build_source_units_from_docs(t_chunks, limit=6)
    u_dicts = [u.to_dict() for u in units]
    ver, rej, _ = deterministic_verify_claims(
        [{"id": "vi_deriv", "text": "VietRAG-Ultra cao hơn VietRAG-Base 13 điểm phần trăm."}],
        u_dicts,
    )
    assert len(ver) == 1 and not rej
    assert ver[0]["claim_type"] == "derived_claim"


def test_m2b_vector_2d_flowchart_with_branching_and_arrowheads(tmp_path):
    clear_pdf_cache()
    clear_ingest_caches()
    pdf_diag = tmp_path / "rag_architecture_flowchart.pdf"

    with fitz.open() as doc:
        page = doc.new_page(width=595, height=842)
        page.insert_htmlbox(
            fitz.Rect(48, 42, 545, 76),
            '<div style="font-family:monospace; font-size:11pt;">Figure 1: Multi-Agent Verification Pipeline</div>',
        )
        # Boxes: Input Parser -> Hybrid Retriever -> Confidence Gate -> [Yes] Guarded Writer / [No] Scout Fallback
        _draw_node_box(page, (60, 100, 190, 140), "Input Parser")
        _draw_node_box(page, (235, 100, 385, 140), "Hybrid Retriever")
        _draw_node_box(page, (235, 200, 385, 245), "Confidence Gate")
        _draw_node_box(page, (80, 320, 230, 365), "Guarded Writer")
        _draw_node_box(page, (365, 320, 515, 365), "Scout Fallback")

        _draw_arrow_with_triangle_head(page, (190, 120), (235, 120))
        _draw_arrow_with_triangle_head(page, (310, 140), (310, 200))
        _draw_arrow_with_triangle_head(page, (275, 245), (155, 320), label="Yes")
        _draw_arrow_with_triangle_head(page, (345, 245), (440, 320), label="No")
        doc.save(pdf_diag, no_new_id=True)

    chunks = load_or_parse_selected_pdf(str(pdf_diag))
    diag_chunks = [c for c in chunks if c.get("unit_type") == "diagram"]
    assert len(diag_chunks) == 1
    dmeta = diag_chunks[0]["diagram_metadata"]
    assert dmeta["extraction_mode"] == "vector_2d"
    node_labels = {n["label"] for n in dmeta["nodes"]}
    assert node_labels == {"Input Parser", "Hybrid Retriever", "Confidence Gate", "Guarded Writer", "Scout Fallback"}

    edge_tuples = {(e["source_label"], e["target_label"], e["condition"]) for e in dmeta["edges"]}
    assert ("Input Parser", "Hybrid Retriever", "") in edge_tuples
    assert ("Hybrid Retriever", "Confidence Gate", "") in edge_tuples
    assert ("Confidence Gate", "Guarded Writer", "Yes") in edge_tuples
    assert ("Confidence Gate", "Scout Fallback", "No") in edge_tuples

    units, *_ = build_source_units_from_docs(diag_chunks, limit=4)
    u_dicts = [u.to_dict() for u in units]

    # 1. Valid direct connections & conditional branches -> SUPPORT
    for good_claim in (
        "Input Parser connects directly to Hybrid Retriever.",
        "If Yes, Confidence Gate routes to Guarded Writer.",
        "Khi Sai, Confidence Gate chuyển sang Scout Fallback.",
        "Từ Input Parser dẫn tới Guarded Writer.",
    ):
        v_ok, r_ok, _ = deterministic_verify_claims([{"id": "g", "text": good_claim}], u_dicts)
        assert len(v_ok) == 1 and not r_ok, f"Expected SUPPORT for {good_claim!r}, got {r_ok}"

    # 2. Reversed flow, skipped-hop direct edge, and swapped branch conditions -> CONTRADICT (even with permissive NLI!)
    def _fake_permissive_nli(pairs):
        return [{"claim_id": p["claim_id"], "status": "SUPPORT", "quote": p["evidence"][:80], "rationale": "ok"} for p in pairs]

    for bad_claim, expected_tag in (
        ("Hybrid Retriever connects directly to Input Parser.", "diagram_flow_reversed"),
        ("Guarded Writer đi thẳng vào Confidence Gate.", "diagram_flow_reversed"),
        ("Input Parser connects directly to Guarded Writer.", "diagram_direct_edge_absent"),
        ("If Yes, Confidence Gate routes to Scout Fallback.", "diagram_branch_condition_mismatch"),
        ("Khi Đúng, Confidence Gate chuyển sang Scout Fallback.", "diagram_branch_condition_mismatch"),
    ):
        state_check = review_claims(
            {
                "query": "Explain Figure 1 pipeline",
                "answer": bad_claim,
                "claims": [{"id": "b1", "text": bad_claim, "unit_id": u_dicts[0]["id"]}],
                "source_units": u_dicts,
            },
            reviewer=_fake_permissive_nli,
        )
        assert state_check["verified_claims"] == [], f"Trap leaked: {bad_claim}"
        assert state_check["rejected_claims"][0]["status"] == "CONTRADICT"
        assert expected_tag in state_check["rejected_claims"][0]["reason"]


def test_m2b_eight_pdf_bilingual_corpus_multi_document_accuracy(tmp_path):
    """
    Tests M2b simultaneously across an 8-PDF bilingual multi-page corpus (4 multi-page stitched
    table PDFs + 4 branching diagram PDFs in EN & VI) to ensure high accuracy at scale.
    """
    clear_pdf_cache()
    clear_ingest_caches()
    corpus_dir = tmp_path / "m2b_eight_pdfs"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths: list[str] = []

    # 1. Doc 1 (EN 2-page stitched table: Solar perovskite cells)
    p_solar = corpus_dir / "m2b_01_solar_cells_en.pdf"
    with fitz.open() as d:
        pg1 = d.new_page(width=595, height=842)
        _draw_vector_grid_table(
            pg1,
            [["Cell", "PCE (%)", "Voc (V)", "T80_Hours (h)"], ["Perov-Std", "21.4%", "1.12 V", "620 h"]],
            y0=560, caption="Table 1: Tandem Perovskite Solar Stability",
        )
        pg2 = d.new_page(width=595, height=842)
        _draw_vector_grid_table(pg2, [["Perov-Passivated", "25.8%", "1.19 V", "1450 h"]], y0=70)
        d.save(p_solar, no_new_id=True)
    pdf_paths.append(str(p_solar))

    # 2. Doc 2 (VI 2-page stitched table: Năng suất lúa chịu mặn)
    p_rice = corpus_dir / "m2b_02_lua_chiu_man_vi.pdf"
    with fitz.open() as d:
        pg1 = d.new_page(width=595, height=842)
        pg1.insert_htmlbox(
            fitz.Rect(42, 520, 553, 760),
            '<div style="font-family:monospace; font-size:10pt;">'
            "Bảng 2: Năng suất giống lúa chịu mặn vùng ven biển<br/>"
            "| Giống lúa | Năng suất (tấn/ha) | Độ mặn chịu đựng (g/L) |<br/>"
            "| ST25-ĐốiChứng | 6.2 | 3.0 g/L |"
            "</div>",
        )
        pg2 = d.new_page(width=595, height=842)
        pg2.insert_htmlbox(
            fitz.Rect(42, 65, 553, 240),
            '<div style="font-family:monospace; font-size:10pt;">'
            "Bảng 2 (tiếp theo)<br/>"
            "| SaltTol-09 | 7.9 | 6.5 g/L |"
            "</div>",
        )
        d.save(p_rice, no_new_id=True)
    pdf_paths.append(str(p_rice))

    # 3. Doc 3 (EN 2-page stitched table: Database Raft consensus latency)
    p_raft = corpus_dir / "m2b_03_raft_consensus_en.pdf"
    with fitz.open() as d:
        pg1 = d.new_page(width=595, height=842)
        _draw_vector_grid_table(
            pg1,
            [["Protocol", "WAN_p99 (ms)", "Throughput (ops/s)", "Quorum"], ["Raft-Classic", "142 ms", "18500", "3"]],
            y0=560, caption="Table 3: Geo-Distributed Consensus Tail Latency",
        )
        pg2 = d.new_page(width=595, height=842)
        _draw_vector_grid_table(pg2, [["EPaxos-Fast", "68 ms", "31200", "3"]], y0=70)
        d.save(p_raft, no_new_id=True)
    pdf_paths.append(str(p_raft))

    # 4. Doc 4 (VI 2-page stitched table: Lọc màng RO nước lợ)
    p_ro = corpus_dir / "m2b_04_mang_loc_ro_vi.pdf"
    with fitz.open() as d:
        pg1 = d.new_page(width=595, height=842)
        pg1.insert_htmlbox(
            fitz.Rect(42, 520, 553, 760),
            '<div style="font-family:monospace; font-size:10pt;">'
            "Bảng 4: Hiệu suất khử muối của màng lọc nano-composite<br/>"
            "| Màng lọc | Tỷ lệ khử muối (%) | Áp suất vận hành (bar) |<br/>"
            "| RO-TiêuChuẩn | 96.2% | 15.5 bar |"
            "</div>",
        )
        pg2 = d.new_page(width=595, height=842)
        pg2.insert_htmlbox(
            fitz.Rect(42, 65, 553, 240),
            '<div style="font-family:monospace; font-size:10pt;">'
            "Bảng 4 (tiếp theo)<br/>"
            "| GrapheneRO-X | 99.4% | 11.2 bar |"
            "</div>",
        )
        d.save(p_ro, no_new_id=True)
    pdf_paths.append(str(p_ro))

    # 5. Doc 5 (EN Vector Diagram: Compiler Optimization Pipeline)
    p_comp = corpus_dir / "m2b_05_compiler_flow_en.pdf"
    with fitz.open() as d:
        pg = d.new_page(width=595, height=842)
        pg.insert_htmlbox(fitz.Rect(48, 40, 540, 75), '<div style="font-family:monospace; font-size:11pt;">Figure 2: MLIR Dialect Lowering Flow</div>')
        _draw_node_box(pg, (60, 110, 190, 150), "Tensor AST")
        _draw_node_box(pg, (230, 110, 370, 150), "Affine Fusion")
        _draw_node_box(pg, (410, 110, 545, 150), "PTX Codegen")
        _draw_arrow_with_triangle_head(pg, (190, 130), (230, 130))
        _draw_arrow_with_triangle_head(pg, (370, 130), (410, 130))
        d.save(p_comp, no_new_id=True)
    pdf_paths.append(str(p_comp))

    # 6. Doc 6 (VI Vector Diagram: Lưu đồ cấp cứu đột quỵ)
    p_stroke = corpus_dir / "m2b_06_luu_do_dot_quy_vi.pdf"
    with fitz.open() as d:
        pg = d.new_page(width=595, height=842)
        pg.insert_htmlbox(fitz.Rect(48, 40, 540, 75), '<div style="font-family:monospace; font-size:11pt;">Hình 3: Lưu đồ phân loại đột quỵ cấp</div>')
        _draw_node_box(pg, (190, 95, 390, 135), "Chụp CT Sọ Não")
        _draw_node_box(pg, (65, 220, 245, 265), "Tiêu Sợi Huyết tPA")
        _draw_node_box(pg, (345, 220, 525, 265), "Hội Chẩn Ngoại Thần Kinh")
        _draw_arrow_with_triangle_head(pg, (260, 135), (155, 220), label="Yes")
        _draw_arrow_with_triangle_head(pg, (320, 135), (435, 220), label="No")
        d.save(p_stroke, no_new_id=True)
    pdf_paths.append(str(p_stroke))

    # 7. Doc 7 (EN Inline Arrow Diagram: Telemetry Ingestion Pipeline)
    p_tel = corpus_dir / "m2b_07_telemetry_dag_en.pdf"
    with fitz.open() as d:
        pg = d.new_page(width=595, height=842)
        pg.insert_htmlbox(
            fitz.Rect(45, 60, 550, 200),
            '<div style="font-family:monospace; font-size:10pt; line-height:1.4;">'
            "Figure 4: Edge Telemetry Streaming Architecture<br/>"
            "Flow: SensorCollector -&gt; SchemaValidator -&gt; [Valid] ClickHouseSink<br/>"
            "Flow: SchemaValidator -&gt; [Invalid] DeadLetterQueue"
            "</div>",
        )
        d.save(p_tel, no_new_id=True)
    pdf_paths.append(str(p_tel))

    # 8. Doc 8 (VI Inline Arrow Diagram: Quy trình kiểm định chất lượng dược phẩm)
    p_pharma = corpus_dir / "m2b_08_kiem_dinh_duoc_vi.pdf"
    with fitz.open() as d:
        pg = d.new_page(width=595, height=842)
        pg.insert_htmlbox(
            fitz.Rect(45, 60, 550, 200),
            '<div style="font-family:monospace; font-size:10pt; line-height:1.4;">'
            "Sơ đồ 5: Quy trình kiểm nghiệm lô thuốc tiêm vô trùng<br/>"
            "Luồng: Lấy Mẫu Lô -&gt; Sắc Ký HPLC -&gt; [Đúng] Xuất Xưởng Kho Lạnh<br/>"
            "Luồng: Sắc Ký HPLC -&gt; [Sai] Biệt Trữ Hủy Lô"
            "</div>",
        )
        d.save(p_pharma, no_new_id=True)
    pdf_paths.append(str(p_pharma))

    # Simultaneous retrieval accuracy across ALL 8 multi-page / diagram PDFs
    retriever = CPUMinRetrievalAgent()
    probes = [
        ("Perov-Passivated PCE 25.8% T80_Hours Tandem Perovskite", "m2b_01_solar_cells_en", 2),
        ("giống lúa SaltTol-09 độ mặn chịu đựng 6.5 g/L năng suất", "m2b_02_lua_chiu_man_vi", 2),
        ("EPaxos-Fast WAN_p99 tail latency 68 ms throughput", "m2b_03_raft_consensus_en", 2),
        ("màng lọc GrapheneRO-X tỷ lệ khử muối 99.4% áp suất", "m2b_04_mang_loc_ro_vi", 2),
        ("Tensor AST Affine Fusion PTX Codegen MLIR lowering", "m2b_05_compiler_flow_en", 1),
        ("Chụp CT Sọ Não Tiêu Sợi Huyết tPA Hội Chẩn Ngoại Thần Kinh", "m2b_06_luu_do_dot_quy_vi", 1),
        ("SensorCollector SchemaValidator ClickHouseSink DeadLetterQueue", "m2b_07_telemetry_dag_en", 1),
        ("Lấy Mẫu Lô Sắc Ký HPLC Xuất Xưởng Kho Lạnh Biệt Trữ", "m2b_08_kiem_dinh_duoc_vi", 1),
    ]

    hits = 0
    for q_text, expected_doc, expected_page in probes:
        res = retriever.run({"query": q_text, "pdf_paths": pdf_paths, "top_k": 5})
        top3 = (res.get("retrieved_docs") or [])[:3]
        if any(expected_doc in str(c.get("doc_id") or "") and int(c.get("page_no") or 0) == expected_page for c in top3):
            hits += 1
    assert hits == 8, f"Expected 8/8 exact (doc, page) Top-3 accuracy across 8 M2b PDFs, got {hits}/8"

    # End-to-end LangGraph QA on a Page-2 continuation row (requiring Page-1 stitched headers!)
    cfg = RunConfig.capture(llm_cache_enabled=False, remote_inference_url="").with_overrides(
        GEMINI_API_KEY="", LOCAL_CPU_URL="", HYBRID_PROVIDER_PROFILE="local_only"
    )
    with use_run_config(cfg):
        graph = build_research_rag_graph("cpu_min", run_config=cfg, auto_scout=False)
        ans_state = graph.invoke({
            "query": "In Table 1, what PCE (%) and T80_Hours did Perov-Passivated achieve?",
            "pdf_path": str(p_solar),
            "pdf_paths": pdf_paths,
            "retry_count": 0,
        })
    ans_txt = str((ans_state.get("written_report_data") or {}).get("answer") or ans_state.get("answer") or "")
    assert "25.8%" in ans_txt and "1450" in ans_txt
