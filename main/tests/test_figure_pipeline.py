"""Verify PDF figure extraction, pixel measurements and figure-specific graph routing."""

import fitz
import pytest

from core import figure_reader
from core.config import settings
from core.figure_qa import _select
from core.pdf_figures import extract_pdf_figures, render_pdf_figure


def make_chart(path, *, heights=(28, 74, 42)):
    with fitz.open() as doc:
        page = doc.new_page(width=360, height=320)
        page.insert_text((20, 30), "The experiment compares three runs.")
        page.draw_line((55, 55), (55, 175), color=(0, 0, 0), width=2)
        page.draw_line((55, 175), (295, 175), color=(0, 0, 0), width=2)
        for idx, height in enumerate(heights):
            left = 77 + idx * 68
            page.draw_rect((left, 175 - height, left + 34, 175), color=None, fill=(0.2, 0.5, 0.8))
        page.insert_text((60, 202), "Figure 1: Median latency by run.")
        page = doc.new_page(width=360, height=320)
        page.insert_text((20, 50), "As shown in Figure 1, the second run has the tallest bar.")
        doc.save(path)
    return str(path)


@pytest.mark.integration
def test_graph_reads_pixels_and_keeps_reference_page(tmp_path, monkeypatch):
    from core.graph import build_research_rag_graph

    monkeypatch.setattr(settings, "CORPUS_PATH", str(tmp_path / "absent"))
    monkeypatch.delenv("FIGURE_VISION_PROVIDER", raising=False)
    path = make_chart(tmp_path / "latency.pdf")
    graph = build_research_rag_graph("cpu_min")
    state = graph.invoke({"query": "Trong Hình 1, cột nào cao nhất?", "pdf_path": path})
    assert "cột thứ 2 từ trái sang cao nhất" in state["final_report"]
    assert "Tác giả giải thích ở trang 2" in state["final_report"]
    reference = next(s for s in state["evidence_spans"] if s["source_unit_id"].endswith(":reference"))
    assert reference["page_no"] == 2 and "tallest" in reference["quote"]
    caption = next(s for s in state["evidence_spans"] if s["source_unit_id"].endswith(":caption"))
    assert caption["bbox"][1] > 175
    attachment, = next(e["figures"] for e in state["reasoning_trace"] if e["action"] == "answer_figures")
    assert attachment["page_no"] == 1
    audit = next(e["citation_audit"] for e in state["reasoning_trace"] if e["action"] == "citation_audit")
    assert [r["page"] for r in audit["references"]] == [1, 2]
    assert fitz.Pixmap(render_pdf_figure(path, attachment["figure_id"])).height < 300

    missing = graph.invoke({"query": "Cho xem Hình 99", "pdf_path": path})
    assert "Chưa định vị" in missing["final_report"]
    assert not any(e["action"] in {"answer_figures", "agent_handoff"} for e in missing["reasoning_trace"])


def test_ambiguous_figures_in_distinct_papers_require_document_choice(tmp_path):
    one = make_chart(tmp_path / "latency.pdf")
    two = make_chart(tmp_path / "throughput.pdf", heights=(14, 28, 56))
    catalog = extract_pdf_figures(one) + extract_pdf_figures(two)
    assert _select("Cho xem Hình 1", catalog)["status"] == "ambiguous"
    assert _select("Cho xem Hình 1 trong throughput.pdf", catalog)["figures"][0]["source_path"] == two
    assert not _select("Mô hình phân loại có ảnh hưởng gì?", catalog)["explicit"]
    assert not _select("Explain graph neural networks", catalog)["explicit"]
    assert _select("Cho xem ảnh trang 1", catalog)["explicit"]
    with pytest.raises(KeyError):
        render_pdf_figure(two, catalog[0]["figure_id"])


def test_scan_and_rotation_preserve_original_pixels(tmp_path):
    with fitz.open() as source:
        page = source.new_page(width=200, height=110)
        page.draw_rect((0, 0, 100, 110), color=None, fill=(1, 0, 0))
        page.insert_text((10, 70), "Median latency 42 ms", fontsize=15)
        raster = page.get_pixmap().tobytes("png")
    path = tmp_path / "scan.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=200, height=110)
        page.insert_image(page.rect, stream=raster)
        page.set_rotation(90)
        doc.save(path)
    figure, = extract_pdf_figures(path)
    assert figure["kind"] == "scan" and not figure["embedded_text"]
    pix = fitz.Pixmap(render_pdf_figure(path, figure["figure_id"]))
    assert (pix.width, pix.height) == (220, 400)
    assert pix.pixel(100, 10) == (255, 0, 0) and pix.pixel(100, 390) == (255, 255, 255)


def test_remote_vision_requires_explicit_provider_and_reader_mode(monkeypatch, tmp_path):
    path = make_chart(tmp_path / "plot.pdf")
    fig, = extract_pdf_figures(path)
    png = render_pdf_figure(path, fig["figure_id"])
    calls = []
    def observe(data, question, provider):
        calls.append((data, provider))
        return "test", "Observed trend"
    monkeypatch.setattr(figure_reader, "_vision", observe)
    monkeypatch.setattr(figure_reader, "_recognize", lambda data: ("", ""))
    monkeypatch.delenv("FIGURE_VISION_PROVIDER", raising=False)
    assert figure_reader.read_figure(png, "Explain", "gemini")["method"] == "unavailable"
    monkeypatch.setenv("FIGURE_VISION_PROVIDER", "gemini")
    assert figure_reader.read_figure(png, "Explain", "cpu_min")["method"] == "unavailable" and not calls
    assert figure_reader.read_figure(png, "Explain", "hybrid")["method"] == "vision"
    assert calls == [(png, "gemini")]


def test_related_text_answer_attaches_only_an_illustration_supported_by_its_source(tmp_path):
    from core.figure_qa import add_figure_evidence, attach_answer_figures

    pdf = make_chart(tmp_path / "experiment.pdf")
    state = {"query": "What was the median latency in the experiment?", "pdf_path": pdf}
    state.update(add_figure_evidence(state, {}, "cpu_min"))
    assert not state["document_schema"]["figure_evidence"]["explicit"]
    span = {"doc_id": "experiment", "page_no": 2, "quote": "Median latency is reported by run."}
    state["evidence_spans"] = [span]
    trace = attach_answer_figures(state, {})["reasoning_trace"]
    assert trace[0]["figures"][0]["page_no"] == 1
    state["evidence_spans"] = [{**span, "doc_id": "other"}]
    assert attach_answer_figures(state, {}) == {}
