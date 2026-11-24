"""Regression & capability tests for Phase 2 Multimodal Figure QA upgrades:
1. Structured VLM JSON parser with json_repair resilience.
2. Explicit opt-in FIGURE_VISION_PROVIDER=auto resolution (preserving empty = local-only privacy).
3. Cross-modal grounding (vision numbers corroborated vs visual-only caution note).
4. Figure indexing into Hybrid & Gemini retrieval for implicit data queries + implicit body reference attachment.
"""

from __future__ import annotations

import fitz

from agents.agent1_retrieval.tools.semantic_pipeline import extract_searchable_figure_chunks
from core import figure_reader
from core.config import settings
from core.figure_qa import add_figure_evidence, cross_check_figure_reading, draft_visual_answer
from core.pdf_figures import extract_pdf_figures, render_pdf_figure
from core.tools import clear_pdf_cache
from modes.gemini.retrieval import GeminiRetrievalAgent
from modes.hybrid.retrieval import HybridRetrievalAgent


def _make_figure_paper(path):
    with fitz.open() as doc:
        p1 = doc.new_page(width=420, height=380)
        p1.insert_text((24, 30), "Section 4. Efficiency Analysis on Dual T4 Hardware.")
        p1.draw_line((60, 60), (60, 190), color=(0, 0, 0), width=2)
        p1.draw_line((60, 190), (340, 190), color=(0, 0, 0), width=2)
        for idx, height in enumerate((35, 85, 60)):
            left = 85 + idx * 75
            p1.draw_rect((left, 190 - height, left + 36, 190), color=None, fill=(0.15, 0.45, 0.75))
        p1.insert_text((60, 215), "Figure 1: Throughput (tokens/s) vs Batch Size for Qwen2.5-7B.")

        p2 = doc.new_page(width=420, height=380)
        p2.insert_text(
            (24, 50),
            "As illustrated in Figure 1, Qwen2.5-7B peaks at 142 tokens/s at batch size 16.",
        )
        doc.save(str(path))
    return str(path)


def test_structured_vision_parser_and_json_repair():
    # Valid structured JSON
    raw = (
        '{"chart_type": "bar", "axes": "X: Batch Size, Y: tokens/s", '
        '"legend": "Qwen2.5-7B", "printed_values": ["142 tokens/s"], '
        '"trend_summary": "Throughput peaks at batch size 16.", '
        '"uncertainty": "Exact value for batch size 8 is not printed.", '
        '"observation": "Throughput peaks at batch size 16 reaching 142 tokens/s."}'
    )
    text, structured = figure_reader._parse_structured_vision(raw)
    assert "Throughput peaks at batch size 16" in text
    assert "Trục/đơn vị: X: Batch Size, Y: tokens/s" in text
    assert "Giới hạn đọc ảnh: Exact value for batch size 8 is not printed." in text
    assert structured["chart_type"] == "bar"

    # Slightly malformed markdown-wrapped JSON with trailing comma -> repaired cleanly
    noisy = '```json\n{"observation": "Bar 2 is tallest.", "axes": "Y: ms",}\n```'
    text2, structured2 = figure_reader._parse_structured_vision(noisy)
    assert "Bar 2 is tallest." in text2
    assert structured2["axes"] == "Y: ms"


def test_auto_vision_provider_routing_preserves_opt_in(monkeypatch, tmp_path):
    path = _make_figure_paper(tmp_path / "auto_vision.pdf")
    fig, = extract_pdf_figures(path)
    png = render_pdf_figure(path, fig["figure_id"])

    calls = []
    monkeypatch.setattr(figure_reader, "_vision", lambda data, q, prov: (calls.append(prov) or f"{prov}/m", "Trend"))
    monkeypatch.setattr(figure_reader, "_recognize", lambda data: ("", ""))

    # Empty provider stays strictly local (no remote upload)
    monkeypatch.delenv("FIGURE_VISION_PROVIDER", raising=False)
    assert figure_reader.read_figure(png, "Describe", "hybrid")["method"] == "unavailable"
    assert calls == []

    # Explicit provider=auto routes hybrid -> gemini when API key present, cpu_max -> ollama when model set
    monkeypatch.setenv("FIGURE_VISION_PROVIDER", "auto")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("FIGURE_VISION_MODEL", "qwen2.5vl:7b")
    assert figure_reader.read_figure(png, "Describe", "hybrid")["method"] == "vision"
    assert figure_reader.read_figure(png, "Describe", "cpu_max")["method"] == "vision"
    assert figure_reader.read_figure(png, "Describe", "cpu_min")["method"] == "unavailable"
    assert calls == ["gemini", "ollama"]


def test_cross_modal_grounding_flags_visual_only_numbers():
    base_fig = {
        "figure_id": "fig-1",
        "doc_id": "paper",
        "page_no": 1,
        "label": "Figure 1",
        "caption": "Figure 1: Throughput for Qwen2.5-7B.",
        "embedded_text": "Batch Size 16",
        "reference": {"text": "As illustrated in Figure 1, Qwen2.5-7B peaks at 142 tokens/s.", "page_no": 2, "bbox": [0, 0, 10, 10]},
    }

    # Corroborated number (142 matches body reference)
    fig_ok = {**base_fig, "reading": {"method": "vision", "text": "Peak reaches 142 tokens/s at batch 16."}}
    cc_ok = cross_check_figure_reading(fig_ok)
    assert cc_ok["status"] == "corroborated"
    assert cc_ok["unverified_numbers"] == []

    # Unverified visual-only number (98.7% not in caption, labels, or author explanation)
    fig_warn = {**base_fig, "reading": {"method": "vision", "text": "Bar height appears to be 98.7%."}}
    cc_warn = cross_check_figure_reading(fig_warn)
    assert cc_warn["status"] == "visual_only_numbers"
    assert "98.7%" in cc_warn["unverified_numbers"]

    draft = draft_visual_answer({
        "query": "What is shown in Figure 1?",
        "document_schema": {"figure_evidence": {"explicit": True, "status": "matched", "figures": [{**fig_warn, "cross_check": cc_warn}]}},
        "source_units": [],
    })
    assert "Lưu ý đối chiếu ảnh–chữ" in draft["answer"]
    assert "98.7" in draft["answer"]


def test_figure_indexing_and_implicit_reference_in_hybrid_and_gemini(tmp_path, monkeypatch):
    clear_pdf_cache()
    pdf = _make_figure_paper(tmp_path / "efficiency.pdf")
    monkeypatch.setattr(settings, "GEMINI_DENSE_ENABLED", False)
    monkeypatch.setattr(settings, "HYBRID_DENSE_ENABLED", False)
    monkeypatch.setattr(settings, "HYBRID_RERANK_ENABLED", False)
    monkeypatch.setattr("modes.hybrid.retrieval.probe_all_capabilities", lambda: {
        "gemini": {"online": False}, "gpu_finetune": {"online": False},
    })

    fig_chunks = extract_searchable_figure_chunks([pdf])
    assert len(fig_chunks) == 1
    assert fig_chunks[0]["unit_type"] == "figure"
    assert "Qwen2.5-7B" in fig_chunks[0]["text"]

    # Query does NOT mention "figure" or "hình", yet retrieval indexes & returns the figure chunk!
    query = "What is the throughput tokens/s vs batch size for Qwen2.5-7B?"
    for agent in (GeminiRetrievalAgent(), HybridRetrievalAgent()):
        res = agent.run({"query": query, "pdf_path": pdf, "pdf_paths": [pdf], "top_k": 5})
        assert any(d.get("unit_type") == "figure" for d in res["retrieved_docs"])

    # Implicit add_figure_evidence also attaches the author's body explanation on page 2
    ev = add_figure_evidence({"query": query, "pdf_path": pdf}, {}, "hybrid")
    assert not ev["document_schema"]["figure_evidence"]["explicit"]
    assert any(u["id"].endswith(":reference") and u["page_no"] == 2 and "142 tokens/s" in u["text"]
               for u in ev["source_units"])
