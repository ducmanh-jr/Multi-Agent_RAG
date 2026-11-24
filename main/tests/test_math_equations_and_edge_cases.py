"""
End-to-End Mathematical Formula Extraction, Verification, Citation Audit & Edge-Case Tests:
  1. PDF extraction of short formulas (`E = mc^2 (1)`), display equations (`unit_type='equation'`),
     and `extract_table_or_equations_tool` output.
  2. FTS5 tokenization preserving single-letter math variables (`Q`, `K`, `V`, `W`, `x`, `α`, `λ`).
  3. Verifier & Citation Audit handling of Unicode math operators (`−` vs `-`, `×` vs `*`),
     NFKC ligatures (`ﬁ` vs `fi`), superscripts/subscripts (`x²` vs `x^2`), equation locators (`Eq. (1)`,
     `Công thức (4)`), and math variable coefficients (`2m`, `8h`) vs physical units.
  4. Multi-line display math (`$$...$$`) and academic abbreviation (`Eq.`, `Fig.`) protection during
     atomic claim splitting.
  5. Multi-round cumulative `EvidenceStore` source preservation across replanning loops.
"""

from __future__ import annotations

import fitz

from agents.agent1_retrieval.tools.fts5_search import SQLiteFTS5Searcher
from agents.agent3_summarizer.components.generator import bind_evidence_spans
from core.citation_audit import audit_claim_citations
from core.evidence_store import EvidenceStore
from core.tools import (
    clear_pdf_cache,
    extract_table_or_equations_tool,
    load_or_parse_selected_pdf,
)
from core.verifier import (
    check_numerical_and_date_consistency,
    split_answer_into_atomic_claims,
    verify_evidence_span_provenance,
)


def test_pdf_equation_chunk_extraction_and_tool_output(tmp_path):
    clear_pdf_cache()
    pdf_path = tmp_path / "math_paper.pdf"
    page_content = (
        "1. Methodology\n\n"
        "E = mc^2 (1)\n\n"
        "The Scaled Dot-Product Attention is computed as:\n"
        "Attention(Q, K, V) = softmax(QK^T / √d_k) V (2)\n"
        "where Q, K, V are query, key, and value matrices with d_k = 64.\n\n"
        "The regularized objective function is defined as:\n"
        "L(θ) = −∑ y_i log(p_i) + λ||θ||² (3)\n"
    )
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), page_content)
        doc.save(pdf_path)

    chunks = load_or_parse_selected_pdf(str(pdf_path))
    assert chunks

    all_text = "\n".join(c["text"] for c in chunks)
    # Short formula E = mc^2 (1) must NOT be dropped
    assert "E = mc^2 (1)" in all_text

    eq_chunks = [c for c in chunks if c.get("unit_type") == "equation"]
    assert len(eq_chunks) >= 2
    assert any("Attention(Q, K, V)" in c["text"] and "d_k = 64" in c["text"] for c in eq_chunks)

    tool_out = extract_table_or_equations_tool(str(pdf_path), page_number=1)
    assert "Công thức / Biểu thức Toán học" in tool_out
    assert "Attention(Q, K, V)" in tool_out


def test_fts5_retains_single_letter_math_variables_and_retrieves_equations():
    searcher = SQLiteFTS5Searcher()
    docs = [
        {
            "id": "eq_chunk",
            "doc_id": "transformer.pdf",
            "title": "Attention Is All You Need",
            "text": "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V where Q, K, V are matrices.",
            "page_no": 4,
        },
        {
            "id": "other_chunk",
            "doc_id": "transformer.pdf",
            "title": "Introduction",
            "text": "Sequence transduction models are widely used in machine translation.",
            "page_no": 1,
        },
    ]
    searcher.fit(docs)
    results = searcher.search("Công thức tính Q K V là gì?", top_k=3)
    assert results
    assert results[0]["id"] == "eq_chunk"
    searcher.close()


def test_verifier_and_citation_audit_handle_unicode_math_exponents_equation_numbers_and_variables():
    # Source unit extracted from PDF contains Unicode minus '−', superscript '²', ligature 'ﬁ', and '2 * m'
    source_units = [
        {
            "id": "u_math_1",
            "doc_id": "algebra.pdf",
            "page_no": 3,
            "text": "We deﬁne the polynomial loss f(x) = x² − 2x + 1 with margin 2 * m + 1 and head count h = 8.",
        }
    ]

    # LLM outputs ASCII '-' and LaTeX '^2' and 'fi' in the quote
    raw_spans = [
        {
            "id": "sp_math_1",
            "source_unit_id": "u_math_1",
            "doc_id": "algebra.pdf",
            "page_no": 3,
            "quote": "We define the polynomial loss f(x) = x^2 - 2x + 1 with margin 2 * m + 1 and head count h = 8.",
        }
    ]
    bound = bind_evidence_spans(raw_spans, source_units)
    assert len(bound) == 1

    ok, reason = verify_evidence_span_provenance(raw_spans[0], source_units)
    assert ok is True
    assert reason == "verified"

    # Claim references 'Công thức (4)' / 'Eq. (4)', writes LaTeX '$f(x) = x^2 - 2x + 1$', and uses '2m + 1' and '8h'
    claim_str = "Theo Công thức (4) (Eq. 4), hàm mất mát là $f(x) = x^2 - 2x + 1$ với biên $2m + 1$ và cấu hình $8h$."
    num_ok, date_ok, missing = check_numerical_and_date_consistency(
        claim_str,
        source_units[0]["text"],
    )
    assert num_ok is True, f"Unexpected missing numbers: {missing}"
    assert date_ok is True

    # Citation audit must also pass without flagging '4' (from Công thức 4) or '2' (from x^2 vs x²)
    state = {
        "source_units": source_units,
        "evidence_spans": [
            {
                "id": "sp_math_1",
                "source_unit_id": "u_math_1",
                "doc_id": "algebra.pdf",
                "page_no": 3,
                "quote": source_units[0]["text"],
            }
        ],
        "verified_claims": [
            {
                "id": "c_math_1",
                "text": claim_str,
                "status": "SUPPORT",
                "evidence_ids": ["sp_math_1"],
            }
        ],
    }
    audit = audit_claim_citations(state)
    assert audit["claims"]["c_math_1"]["status"] == "verified"
    assert audit["claims"]["c_math_1"]["missing_values"] == []


def test_split_atomic_claims_preserves_multiline_display_math_and_eq_abbreviations():
    source_units = [
        {
            "id": "u_attn",
            "doc_id": "paper.pdf",
            "page_no": 4,
            "text": "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V with d_k = 64.",
        }
    ]
    answer = (
        "Theo Eq. (1) và Fig. 2 trong tài liệu, cơ chế chú ý được tính bằng công thức:\n"
        "$$\n"
        "Attention(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V\n"
        "$$\n"
        "Trong công thức trên, chiều khóa được đặt là $d_k = 64$ để ổn định gradient."
    )
    claims = split_answer_into_atomic_claims(answer, source_units)
    assert len(claims) >= 2
    # Display math block must remain intact inside a claim, never split across newlines
    assert any("$$" in c["text"] and "Attention(Q, K, V)" in c["text"] for c in claims)
    # 'Eq.' and 'Fig.' must not split the first sentence into a fragment ending at 'Eq.'
    assert not any(c["text"].endswith("Eq.") or c["text"].endswith("Fig.") for c in claims)


def test_multi_round_evidence_store_accumulates_prior_round_sources():
    round0_state = {
        "source_units": [
            {"id": "u_r0", "doc_id": "paper_a.pdf", "page_no": 2, "text": "Round 0 baseline finding."}
        ],
        "evidence_spans": [],
        "verified_claims": [],
        "rejected_claims": [],
        "evidence_gaps": [],
    }
    store_r0 = EvidenceStore.from_state(round0_state)

    round1_state = {
        "document_schema": {"evidence_store": store_r0.to_dict()},
        "source_units": [
            {"id": "u_r1", "doc_id": "paper_b.pdf", "page_no": 5, "text": "Round 1 gap investigation finding."}
        ],
        "evidence_spans": [
            {"id": "sp_r0", "source_unit_id": "u_r0", "doc_id": "paper_a.pdf", "page_no": 2, "quote": "Round 0 baseline finding."},
            {"id": "sp_r1", "source_unit_id": "u_r1", "doc_id": "paper_b.pdf", "page_no": 5, "quote": "Round 1 gap investigation finding."},
        ],
        "verified_claims": [
            {"id": "c_combined", "text": "Combined finding from both rounds.", "status": "SUPPORT", "evidence_ids": ["sp_r0", "sp_r1"]}
        ],
        "rejected_claims": [],
        "evidence_gaps": [],
    }
    store_r1 = EvidenceStore.from_state(round1_state)
    summary = store_r1.summary()
    assert summary["source_count"] == 2
    assert summary["verified_span_count"] == 2
    lineage = store_r1.trace_claim_lineage("c_combined")
    assert lineage is not None and lineage["lineage_complete"] is True
    assert lineage["supporting_documents"] == ["paper_a.pdf", "paper_b.pdf"]


def test_end_to_end_langgraph_pipeline_answers_math_equation_query_from_pdf(monkeypatch, tmp_path):
    """
    Full end-to-end LangGraph run on a real PDF containing a mathematical equation:
    Verifies PDF equation extraction -> FTS5/BM25 Retrieval -> Reader -> Summarizer ->
    Fact-Checker -> Citation Audit -> Writer produces a verified report containing the equation.
    """
    from core.config import settings
    from core.graph import build_research_rag_graph

    clear_pdf_cache()
    monkeypatch.setattr(settings, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "CORPUS_PATH", str(tmp_path / "empty.jsonl"))

    pdf_path = tmp_path / "attention_equation.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            "3.2 Scaled Dot-Product Attention\n\n"
            "In our architecture, the Scaled Dot-Product Attention formula is computed as:\n"
            "Attention(Q, K, V) = softmax(QK^T / √d_k) V (1)\n"
            "where Q, K, and V denote the query, key, and value matrices with key dimension d_k = 64.",
        )
        doc.save(pdf_path)

    app = build_research_rag_graph("cpu_min", auto_scout=False)
    result = app.invoke({
        "query": "Công thức tính Scaled Dot-Product Attention(Q, K, V) trong bài báo là gì?",
        "pdf_path": str(pdf_path),
        "pdf_paths": [str(pdf_path)],
        "max_reflection_loops": 0,
    })

    assert "Attention(Q, K, V)" in (result.get("answer") or "")
    assert "Attention(Q, K, V)" in (result.get("final_report") or "")
    assert result.get("fact_check_label") == "support"
    assert len(result.get("verified_claims") or []) >= 1
    ev_store = result["document_schema"]["evidence_store"]
    assert ev_store["summary"]["verified_claim_count"] >= 1
    assert ev_store["source_quality"]["verified_lineage_rate"] == 1.0


def test_multiline_broken_pdf_equation_reconstruction_and_scientific_math_eval(tmp_path):
    from core.tools import _reconstruct_broken_pdf_equations, safe_math_eval
    from modes.cpu_min.summarizer import clean_vietnamese_text

    # 1. Broken 4-line equation from PDF text layer is reconstructed into a single intact equation line
    broken_pdf_text = (
        "The attention weights are computed by:\n"
        "Attention(Q, K, V) = softmax(\n"
        "QK^T /\n"
        "√d_k\n"
        ") V (1)\n"
        "where d_k = 64."
    )
    reconstructed = _reconstruct_broken_pdf_equations(broken_pdf_text)
    assert "Attention(Q, K, V) = softmax( QK^T / √d_k ) V (1)" in reconstructed

    # 2. Scientific safe_math_eval supports exponents (^), Unicode operators (−, ×), log2, exp, sqrt, trig, pi, e
    assert safe_math_eval("2^10 − 24") == "1000"
    assert safe_math_eval("log2(64) × sqrt(16)") == "24"
    assert safe_math_eval("round(ln(e^3) + cos(0), 4)") == "4"

    # 3. CPU Min clean_vietnamese_text preserves LaTeX subscript braces like L_{total}
    cleaned = clean_vietnamese_text("Hàm mục tiêu L_{total} = α L_{ce} + β L_{kl}.")
    assert "L_{total}" in cleaned
    assert "L_{ce}" in cleaned
    assert "L_{kl}" in cleaned


