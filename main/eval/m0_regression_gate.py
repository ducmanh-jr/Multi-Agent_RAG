# -*- coding: utf-8 -*-
"""
M0 60-Question Mechanism Regression Gate on 10 Controlled PDFs (`m0_regression_gate.py`).

Purpose (per KE_HOACH_TATMANH.md Sections 4 & 5):
1. Deterministically materializes 10 controlled PDFs covering every critical document & trap family:
   - Single-column English & Vietnamese prose (number words, units)
   - Two-column academic paper layout (`doc03_vision_2col_en.pdf`)
   - Multi-header evaluation table across Validation vs Test splits (`doc04_multilevel_table_en.pdf`)
   - Vector bar chart + legend + caption (`doc05_chart_ab_en.pdf`)
   - Mathematical formulas with subscripts/superscripts, inequalities & domain constraints (`doc06_math_domain_en.pdf`)
   - Clinical negation & conditional scope (`doc07_clinical_negation_en.pdf`)
   - Rasterized Vietnamese/English scanned image page (`doc08_scan_vi.pdf`)
   - Bilingual Vietnamese-English + Greek notation (`doc09_bilingual_cs_vi_en.pdf`)
   - Multi-document cross-comparison & ambiguity (`doc10_multi_contradiction_en.pdf`)
2. Generates and validates 60 structured benchmark cases (`m0_regression_gate_60.json`) balanced across
   4 language directions (`en->vi`, `en->en`, `vi->vi`, `vi->en`) and all P0/P1 mechanism gates:
   - Directed subject swap ("B exceeds A" vs "A exceeds B")
   - Unit mismatch & percentage points (`11 pp`) vs relative change (`13.58%`) vs cross-split subtraction
   - Domain-preserving math (`x/x = 1` requiring `x != 0`, `x^2` vs `x_2`, `x < 3 and y > 2`)
   - Clause negation (`It is false that treatment works` -> `Treatment works`)
   - Faithful cross-lingual translation vs short fabricated answer (`It took 500 days.`)
   - Honest refusal (`NOT_FOUND_IN_DOCUMENT` / `INSUFFICIENT_EVIDENCE`) vs leaky status-display mismatch
   - Ambiguous scope clarification (`CLARIFICATION_NEEDED`)
3. Benchmarks wall-clock execution time on the first 10 representative cases (both evaluator audit
   and live `cpu_min` baseline diagnostic pass) to anchor M2a / labeling effort calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import fitz

ROOT_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval.three_layer_evaluator import (
    calibrate_nli_thresholds,
    clopper_pearson_lower_bound,
    compute_cohens_kappa,
    evaluate_three_layer_case,
)

GATE_DATASET_PATH = EVAL_DIR / "m0_regression_gate_60.json"


def build_controlled_pdfs(output_dir: Path) -> dict[str, str]:
    """
    Deterministically renders 10 controlled PDFs into `output_dir` and returns `{filename: sha256}`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}

    def _save(doc: fitz.Document, name: str) -> None:
        target = output_dir / name
        doc.save(target, no_new_id=True)
        doc.close()
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()

    # 1. doc01_agronomy_en.pdf (English prose + irrigation table)
    d1 = fitz.open()
    p = d1.new_page(width=595, height=842)
    p.insert_text((60, 70), "Study 1: Orchard Subsurface Irrigation Trial", fontname="helv", fontsize=13)
    p.insert_text((60, 105), "The orchard trial lasted exactly fourteen consecutive days.", fontname="helv", fontsize=11)
    p.insert_text((60, 130), "Soil temperature remained stable at 24 degrees Celsius.", fontname="helv", fontsize=11)
    p2 = d1.new_page(width=595, height=842)
    p2.insert_text((60, 70), "Table 1. Water Consumption by Irrigation Regimen", fontname="helv", fontsize=12)
    p2.insert_text((60, 105), "Regimen            Water Volume (liters)    Flow Speed", fontname="cour", fontsize=10)
    p2.insert_text((60, 125), "Drip irrigation    42 liters                12 km/h", fontname="cour", fontsize=10)
    p2.insert_text((60, 145), "Sprinkler          57 liters                18 km/h", fontname="cour", fontsize=10)
    _save(d1, "doc01_agronomy_en.pdf")

    # 2. doc02_agronomy_vi.pdf (Vietnamese with full Unicode diacritics via PyMuPDF HTML layout)
    d2 = fitz.open()
    p = d2.new_page(width=595, height=842)
    p.insert_htmlbox(
        fitz.Rect(60, 60, 535, 260),
        "<p><b>Báo cáo Thực nghiệm Nông học Nhà kính</b></p>"
        "<p>Thí nghiệm nhà kính kéo dài 35 ngày liên tiếp tại Lâm Đồng.</p>"
        "<p>Lượng nước tưới nhỏ giọt tiêu thụ 28 lít mỗi ô tiêu chuẩn.</p>"
        "<p>Hàm lượng phốt pho hòa tan đạt 16 mg/L ở tầng đất mặt.</p>",
    )
    _save(d2, "doc02_agronomy_vi.pdf")

    # 3. doc03_vision_2col_en.pdf (Genuine 2-column academic layout)
    d3 = fitz.open()
    p = d3.new_page(width=595, height=842)
    p.insert_text((60, 55), "Dual-Stream Vision Transformer Architecture", fontname="helv", fontsize=13)
    p.insert_textbox(
        fitz.Rect(50, 90, 280, 420),
        "1. Architecture Overview\n"
        "The baseline ViT-Small encoder contains 12 layers and 6 attention heads. "
        "During pretraining, batch size was fixed at 256 images per step.",
        fontname="helv",
        fontsize=10,
    )
    p.insert_textbox(
        fitz.Rect(310, 90, 545, 420),
        "2. Latency and Throughput\n"
        "On Tesla T4, ViT-Small achieves 18 ms latency and 140 tokens/s throughput. "
        "The deeper ViT-Large variant uses 24 layers and 39 ms latency.",
        fontname="helv",
        fontsize=10,
    )
    _save(d3, "doc03_vision_2col_en.pdf")

    # 4. doc04_multilevel_table_en.pdf (Validation vs Test split table)
    d4 = fitz.open()
    p = d4.new_page(width=595, height=842)
    p.insert_text((60, 65), "Table 2. Multi-Split Accuracy (%) and Conveyor Speed", fontname="helv", fontsize=12)
    p.insert_text((60, 100), "Method      Validation Accuracy    Test Accuracy    Conveyor Speed", fontname="cour", fontsize=10)
    p.insert_text((60, 120), "Method A    81%                    79%              15 km/h", fontname="cour", fontsize=10)
    p.insert_text((60, 140), "Method B    92%                    88%              15 km/h", fontname="cour", fontsize=10)
    p.insert_text((60, 175), "Note: Validation set has 1200 samples; Test set has 2400 samples.", fontname="helv", fontsize=10)
    _save(d4, "doc04_multilevel_table_en.pdf")

    # 5. doc05_chart_ab_en.pdf (Vector bar chart + legend + caption)
    d5 = fitz.open()
    p = d5.new_page(width=595, height=842)
    p.insert_text((60, 65), "Section 4. Visual Comparison on Validation Split", fontname="helv", fontsize=12)
    # Draw chart frame & bars
    p.draw_rect(fitz.Rect(80, 110, 460, 310), color=(0.2, 0.2, 0.2), width=1.0)
    p.draw_rect(fitz.Rect(140, 148, 210, 310), color=(0.2, 0.4, 0.8), fill=(0.2, 0.4, 0.8))
    p.insert_text((148, 140), "Method A: 81%", fontname="helv", fontsize=9)
    p.draw_rect(fitz.Rect(280, 126, 350, 310), color=(0.8, 0.3, 0.2), fill=(0.8, 0.3, 0.2))
    p.insert_text((288, 118), "Method B: 92%", fontname="helv", fontsize=9)
    p.insert_text((80, 335), "Figure 3: Validation accuracy shows Method A achieves 81% and Method B achieves 92%.", fontname="helv", fontsize=10)
    _save(d5, "doc05_chart_ab_en.pdf")

    # 6. doc06_math_domain_en.pdf (Equations, subscripts/superscripts, inequalities, domain constraints)
    d6 = fitz.open()
    p = d6.new_page(width=595, height=842)
    p.insert_text((60, 65), "Section 3. Mathematical Formulations and Domain Constraints", fontname="helv", fontsize=12)
    p.insert_text((60, 105), "Equation (1): Quadratic root condition x^2 = 4.", fontname="helv", fontsize=11)
    p.insert_text((60, 130), "Equation (2): Second sequence element x_2 = 9.", fontname="helv", fontsize=11)
    p.insert_text((60, 155), "Equation (3): Feasibility region x < 3 and y > 2.", fontname="helv", fontsize=11)
    p.insert_text((60, 180), "Equation (4): Normalized identity x / x = 1 for x != 0.", fontname="helv", fontsize=11)
    p.insert_text((60, 205), "Equation (5): Stacked ratio y = a / b with b != 0.", fontname="helv", fontsize=11)
    p.insert_text((60, 230), "Equation (6): Discounted return horizon t >= 0.", fontname="helv", fontsize=11)
    _save(d6, "doc06_math_domain_en.pdf")

    # 7. doc07_clinical_negation_en.pdf (Negation, falsity, quantifiers)
    d7 = fitz.open()
    p = d7.new_page(width=595, height=842)
    p.insert_text((60, 70), "Clinical Trial Adverse & Null Findings", fontname="helv", fontsize=12)
    p.insert_text((60, 105), "It is false that Treatment X works in stage-3 patients.", fontname="helv", fontsize=11)
    p.insert_text((60, 130), "Treatment Y did not reduce systolic blood pressure.", fontname="helv", fontsize=11)
    p.insert_text((60, 155), "Biomarker Z reached 15 mg/L in Cohort C.", fontname="helv", fontsize=11)
    _save(d7, "doc07_clinical_negation_en.pdf")

    # 8. doc08_scan_vi.pdf (Page 1 rasterized scan image only; Page 2 text reference)
    tmp_render = fitz.open()
    tp = tmp_render.new_page(width=460, height=220)
    tp.insert_htmlbox(
        fitz.Rect(20, 20, 440, 200),
        "<p><b>Phiếu Kết quả Quét Quang học (Scan)</b></p>"
        "<p>Accuracy was 83 percent on scanned receipts.</p>"
        "<p>Sai số ký tự tiếng Việt đạt 3.2%.</p>",
    )
    pix = tmp_render[0].get_pixmap(dpi=150)
    png_bytes = pix.tobytes("png")
    tmp_render.close()

    d8 = fitz.open()
    sp1 = d8.new_page(width=595, height=842)
    sp1.insert_image(fitz.Rect(60, 80, 520, 300), stream=png_bytes)
    sp2 = d8.new_page(width=595, height=842)
    sp2.insert_htmlbox(
        fitz.Rect(60, 70, 535, 220),
        "<p><b>Phụ lục Đối chứng OCR</b></p>"
        "<p>Tập hóa đơn in rõ đạt độ chính xác nhận dạng 96.4% trên 500 trang.</p>",
    )
    _save(d8, "doc08_scan_vi.pdf")

    # 9. doc09_bilingual_cs_vi_en.pdf (Mixed VI-EN + Greek parameters)
    d9 = fitz.open()
    p = d9.new_page(width=595, height=842)
    p.insert_htmlbox(
        fitz.Rect(60, 65, 535, 280),
        "<p><b>Thử nghiệm Truy hồi Song ngữ BGE-M3 và BM25</b></p>"
        "<p>Khi kết hợp hybrid với hệ số alpha = 0.65, Recall@5 đạt 93.5%.</p>"
        "<p>Mô hình BM25 đơn thuần đạt 78.0% Recall@5 trên câu hỏi không dấu.</p>"
        "<p>Thời gian phản hồi p95 của reranker là 45 ms.</p>",
    )
    _save(d9, "doc09_bilingual_cs_vi_en.pdf")

    # 10. doc10_multi_contradiction_en.pdf (Companion multi-doc comparison & scope boundary)
    d10 = fitz.open()
    p = d10.new_page(width=595, height=842)
    p.insert_text((60, 70), "External Replication Study on Edge Hardware", fontname="helv", fontsize=12)
    p.insert_text((60, 105), "On Jetson Orin, ViT-Small latency increased to 31 ms.", fontname="helv", fontsize=11)
    p.insert_text((60, 130), "Method B achieved 85% accuracy on the noisy field dataset.", fontname="helv", fontsize=11)
    _save(d10, "doc10_multi_contradiction_en.pdf")

    return hashes


def _make_case(
    idx: int,
    *,
    category: str,
    documents: list[str],
    source_language: str,
    answer_language: str,
    question: str,
    answerability: str = "answerable",
    gold_answer: str = "",
    atomic_claims: list[str] | None = None,
    gold_sources: list[dict[str, Any]] | None = None,
    gold_quantities: list[dict[str, Any]] | None = None,
    directed_relation: dict[str, str] | None = None,
    derivation_spec: dict[str, Any] | None = None,
    gold_formula: str | None = None,
    domain_constraints: list[str] | None = None,
    forbidden_entities: list[str] | None = None,
    forbidden_numbers: list[float | str] | None = None,
    good_prediction: dict[str, Any],
    bad_prediction: dict[str, Any],
    expected_bad_failure_prefix: str,
) -> dict[str, Any]:
    return {
        "id": f"M0-Q{idx:02d}",
        "category": category,
        "documents": documents,
        "selected_doc": documents[0],
        "source_language": source_language,
        "language": answer_language,
        "answer_language": answer_language,
        "answerability": answerability,
        "is_unanswerable": answerability in {"unanswerable", "insufficient_evidence", "adversarial_trap"},
        "question": question,
        "gold_answer": gold_answer,
        "atomic_claims": (
            atomic_claims
            if atomic_claims is not None
            else ([gold_answer] if gold_answer and answerability == "answerable" else [])
        ),
        "gold_sources": gold_sources or [],
        "gold_quantities": gold_quantities or [],
        "directed_relation": directed_relation,
        "derivation_spec": derivation_spec,
        "gold_formula": gold_formula,
        "domain_constraints": domain_constraints or [],
        "forbidden_entities": forbidden_entities or [],
        "forbidden_numbers": forbidden_numbers or [],
        "good_prediction": good_prediction,
        "bad_prediction": bad_prediction,
        "expected_bad_failure_prefix": expected_bad_failure_prefix,
    }


def generate_60_mechanism_cases() -> list[dict[str, Any]]:
    """
    Constructs the 60-question mechanism regression benchmark (`M0-Q01` .. `M0-Q60`).
    First 10 cases (`M0-Q01` .. `M0-Q10`) form the representative calibration slice across all 10 PDFs.
    """
    cases: list[dict[str, Any]] = []

    # --- Representative Calibration Slice: M0-Q01 .. M0-Q10 (10 distinct traps across 10 PDFs) ---
    cases.append(_make_case(
        1,
        category="cross_lingual_translation",
        documents=["doc01_agronomy_en.pdf"],
        source_language="en",
        answer_language="vi",
        question="Thử nghiệm tưới ngầm trong vườn cây kéo dài đúng bao nhiêu ngày liên tiếp?",
        gold_answer="Thử nghiệm kéo dài đúng 14 ngày liên tiếp.",
        gold_sources=[{"document": "doc01_agronomy_en.pdf", "page": 1, "bbox": [55, 95, 450, 120], "passage": "The orchard trial lasted exactly fourteen consecutive days."}],
        gold_quantities=[{"value": 14, "unit": "day"}],
        good_prediction={"answer": "Thử nghiệm kéo dài đúng 14 ngày liên tiếp. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Nó mất 500 ngày.", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="quantity_or_unit_mismatch",
    ))
    cases.append(_make_case(
        2,
        category="unit_integrity",
        documents=["doc01_agronomy_en.pdf"],
        source_language="en",
        answer_language="en",
        question="What was the flow speed of drip irrigation in Table 1?",
        gold_answer="Drip irrigation had a flow speed of 12 km/h.",
        gold_sources=[{"document": "doc01_agronomy_en.pdf", "page": 2, "bbox": [55, 115, 480, 135], "passage": "Drip irrigation 42 liters 12 km/h"}],
        gold_quantities=[{"value": 12, "unit": "km/h"}],
        good_prediction={"answer": "Drip irrigation had a flow speed of 12 km/h. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Drip irrigation had a flow speed of 12 m/h. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="quantity_or_unit_mismatch",
    ))
    cases.append(_make_case(
        3,
        category="derived_calculation",
        documents=["doc05_chart_ab_en.pdf"],
        source_language="en",
        answer_language="vi",
        question="Trong Hình 3, phương pháp B hơn phương pháp A bao nhiêu điểm phần trăm?",
        gold_answer="Phương pháp B cao hơn phương pháp A 11 điểm phần trăm.",
        gold_sources=[{"document": "doc05_chart_ab_en.pdf", "page": 1, "bbox": [75, 325, 520, 350], "passage": "Figure 3: Validation accuracy shows Method A achieves 81% and Method B achieves 92%."}],
        directed_relation={"higher": "Method B", "lower": "Method A"},
        derivation_spec={
            "op": "diff_pp",
            "operands": [
                {"entity": "Method B", "metric": "accuracy", "split": "validation", "value": 92.0, "unit": "%"},
                {"entity": "Method A", "metric": "accuracy", "split": "validation", "value": 81.0, "unit": "%"},
            ],
            "expected_value": 11.0,
            "expected_unit": "pp",
        },
        good_prediction={"answer": "Phương pháp B cao hơn phương pháp A 11 điểm phần trăm. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Phương pháp B cao hơn phương pháp A 13.58%. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="derived_value_missing",
    ))
    cases.append(_make_case(
        4,
        category="directed_subject_swap",
        documents=["doc04_multilevel_table_en.pdf"],
        source_language="en",
        answer_language="vi",
        question="Trên tập Validation ở Bảng 2, phương pháp nào có độ chính xác cao hơn?",
        gold_answer="Phương pháp B cao hơn phương pháp A trên tập Validation (92% so với 81%).",
        gold_sources=[{"document": "doc04_multilevel_table_en.pdf", "page": 1, "bbox": [55, 110, 500, 150], "passage": "Method B 92% 88%"}],
        directed_relation={"higher": "Method B", "lower": "Method A"},
        good_prediction={"answer": "Phương pháp B cao hơn phương pháp A trên tập Validation với 92% so với 81%. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Phương pháp A cao hơn phương pháp B trên tập Validation với 92% so với 81%. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="subject_swapped",
    ))
    cases.append(_make_case(
        5,
        category="math_domain_constraint",
        documents=["doc06_math_domain_en.pdf"],
        source_language="en",
        answer_language="vi",
        question="Phương trình (4) rút gọn phân thức x / x như thế nào và cần điều kiện xác định gì?",
        gold_answer="Phương trình có dạng $x / x = 1$ với điều kiện $x \\neq 0$.",
        atomic_claims=[],
        gold_sources=[{"document": "doc06_math_domain_en.pdf", "page": 1, "bbox": [55, 170, 460, 192], "passage": "Equation (4): Normalized identity x / x = 1 for x != 0."}],
        gold_formula="x / x = 1",
        domain_constraints=["x!=0"],
        good_prediction={"answer": "Đồng nhất thức là $x / x = 1$ với điều kiện miền xác định $x \\neq 0$. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Đồng nhất thức rút gọn luôn là $x / x = 1$ với mọi số thực x. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="missing_domain_constraint",
    ))
    cases.append(_make_case(
        6,
        category="math_sub_vs_super",
        documents=["doc06_math_domain_en.pdf"],
        source_language="en",
        answer_language="en",
        question="State Equation (1) for the quadratic root condition.",
        gold_answer="The quadratic root condition is $x^2 = 4$.",
        atomic_claims=[],
        gold_sources=[{"document": "doc06_math_domain_en.pdf", "page": 1, "bbox": [55, 95, 420, 118], "passage": "Equation (1): Quadratic root condition x^2 = 4."}],
        gold_formula="x^2 = 4",
        good_prediction={"answer": "The quadratic root condition is $x^2 = 4$. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "The quadratic root condition is $x_2 = 4$. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="formula_mismatch",
    ))
    cases.append(_make_case(
        7,
        category="negation_polarity",
        documents=["doc07_clinical_negation_en.pdf"],
        source_language="en",
        answer_language="en",
        question="Does Treatment X work in stage-3 patients according to the clinical report?",
        gold_answer="No, it is false that Treatment X works in stage-3 patients.",
        gold_sources=[{"document": "doc07_clinical_negation_en.pdf", "page": 1, "bbox": [55, 95, 480, 118], "passage": "It is false that Treatment X works in stage-3 patients."}],
        good_prediction={"answer": "It is false that Treatment X works in stage-3 patients. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Treatment X works in stage-3 patients. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="polarity_inverted",
    ))
    cases.append(_make_case(
        8,
        category="two_column_writer_guard",
        documents=["doc03_vision_2col_en.pdf"],
        source_language="en",
        answer_language="en",
        question="How many encoder layers does the baseline ViT-Small architecture contain?",
        gold_answer="The baseline ViT-Small encoder contains 12 layers.",
        gold_sources=[{"document": "doc03_vision_2col_en.pdf", "page": 1, "bbox": [50, 90, 280, 250], "passage": "The baseline ViT-Small encoder contains 12 layers"}],
        gold_quantities=[{"value": 12, "unit": "layer"}],
        good_prediction={"answer": "The baseline ViT-Small encoder contains 12 layers. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "The baseline ViT-Small encoder contains 999 layers. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="quantity_or_unit_mismatch",
    ))
    cases.append(_make_case(
        9,
        category="scan_ocr_vi",
        documents=["doc08_scan_vi.pdf"],
        source_language="vi",
        answer_language="vi",
        question="Trong phụ lục đối chứng OCR ở trang 2, độ chính xác nhận dạng trên 500 trang hóa đơn in rõ là bao nhiêu?",
        gold_answer="Độ chính xác nhận dạng đạt 96.4% trên 500 trang.",
        gold_sources=[{"document": "doc08_scan_vi.pdf", "page": 2, "bbox": [60, 70, 535, 200], "passage": "96.4%"}],
        gold_quantities=[{"value": 96.4, "unit": "%"}],
        good_prediction={"answer": "Độ chính xác nhận dạng đạt 96.4% trên 500 trang hóa đơn in rõ. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Độ chính xác nhận dạng đạt 96.4 điểm phần trăm. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="quantity_or_unit_mismatch",
    ))
    cases.append(_make_case(
        10,
        category="adversarial_trap",
        documents=["doc02_agronomy_vi.pdf"],
        source_language="vi",
        answer_language="vi",
        answerability="insufficient_evidence",
        question="Nồng độ nitrat trong đất nhà kính ở Lâm Đồng được báo cáo là bao nhiêu mg/L?",
        forbidden_entities=["nitrat"],
        forbidden_numbers=[45],
        good_prediction={
            "answer": "Chưa đủ bằng chứng trong tài liệu đã chọn để xác nhận nồng độ nitrat trong đất.",
            "answer_status": "INSUFFICIENT_EVIDENCE",
        },
        bad_prediction={
            "answer": "Nồng độ nitrat trong đất đạt 45 mg/L.",
            "answer_status": "INSUFFICIENT_EVIDENCE",
        },
        expected_bad_failure_prefix="status_display_mismatch",
    ))

    # --- Cases 11..20: Derived Arithmetic, Split Discipline & Percentage Points vs Relative % ---
    cases.append(_make_case(
        11,
        category="derived_calculation",
        documents=["doc04_multilevel_table_en.pdf"],
        source_language="en",
        answer_language="en",
        question="By how many percentage points does Method B exceed Method A on the Test split in Table 2?",
        gold_answer="Method B exceeds Method A by 9 percentage points on the Test split.",
        gold_sources=[{"document": "doc04_multilevel_table_en.pdf", "page": 1, "bbox": [55, 110, 500, 150], "passage": "Method B 92% 88%"}],
        directed_relation={"higher": "Method B", "lower": "Method A"},
        derivation_spec={
            "op": "diff_pp",
            "operands": [
                {"entity": "Method B", "metric": "accuracy", "split": "test", "value": 88.0, "unit": "%"},
                {"entity": "Method A", "metric": "accuracy", "split": "test", "value": 79.0, "unit": "%"},
            ],
            "expected_value": 9.0,
            "expected_unit": "pp",
        },
        good_prediction={"answer": "Method B exceeds Method A by 9 percentage points on the Test split. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Method B exceeds Method A by 9% on the Test split. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="unit_mismatch",
    ))
    cases.append(_make_case(
        12,
        category="derived_calculation",
        documents=["doc04_multilevel_table_en.pdf"],
        source_language="en",
        answer_language="en",
        question="What is the relative percentage improvement of Method B over Method A on the Validation split?",
        gold_answer="Method B outperforms Method A by 13.58% in relative terms on Validation.",
        atomic_claims=[],
        gold_sources=[{"document": "doc04_multilevel_table_en.pdf", "page": 1, "bbox": [55, 110, 500, 150], "passage": "Method A 81%"}],
        derivation_spec={
            "op": "rel_change_pct",
            "operands": [
                {"entity": "Method B", "metric": "accuracy", "split": "validation", "value": 92.0, "unit": "%"},
                {"entity": "Method A", "metric": "accuracy", "split": "validation", "value": 81.0, "unit": "%"},
            ],
            "expected_value": 13.58,
            "expected_unit": "%",
            "tolerance": 0.05,
        },
        good_prediction={"answer": "Method B outperforms Method A by 13.58% relative improvement on Validation. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Method B outperforms Method A by 11% on Validation. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="derived_value_missing",
    ))
    cases.append(_make_case(
        13,
        category="derived_calculation",
        documents=["doc04_multilevel_table_en.pdf"],
        source_language="en",
        answer_language="vi",
        question="Trên tập Test, phương pháp B cao hơn phương pháp A bao nhiêu điểm phần trăm?",
        gold_answer="Phương pháp B cao hơn phương pháp A 9 điểm phần trăm trên tập Test.",
        gold_sources=[{"document": "doc04_multilevel_table_en.pdf", "page": 1, "bbox": [55, 110, 500, 150], "passage": "Method A 81% 79%"}],
        derivation_spec={
            "op": "diff_pp",
            "operands": [
                {"entity": "Method B", "metric": "accuracy", "split": "test", "value": 88.0, "unit": "%"},
                {"entity": "Method A", "metric": "accuracy", "split": "test", "value": 79.0, "unit": "%"},
            ],
            "expected_value": 9.0,
            "expected_unit": "pp",
        },
        good_prediction={"answer": "Phương pháp B cao hơn phương pháp A 9 điểm phần trăm trên tập Test. [1]", "answer_status": "ANSWERED"},
        # Cross-split mistake: 92% (val) - 79% (test) = 13 pp!
        bad_prediction={"answer": "Phương pháp B cao hơn phương pháp A 13 điểm phần trăm trên tập Test. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="derived_value_missing",
    ))
    cases.append(_make_case(
        14,
        category="derived_calculation",
        documents=["doc01_agronomy_en.pdf"],
        source_language="en",
        answer_language="en",
        question="How many more liters of water did Sprinkler use than Drip irrigation in Table 1?",
        gold_answer="Sprinkler exceeds Drip by 15 liters.",
        atomic_claims=[],
        gold_sources=[{"document": "doc01_agronomy_en.pdf", "page": 2, "bbox": [55, 115, 480, 155], "passage": "Sprinkler 57 liters"}],
        derivation_spec={
            "op": "sub",
            "operands": [
                {"entity": "Sprinkler", "metric": "water", "split": "trial", "value": 57.0, "unit": "l"},
                {"entity": "Drip", "metric": "water", "split": "trial", "value": 42.0, "unit": "l"},
            ],
            "expected_value": 15.0,
            "expected_unit": "l",
        },
        good_prediction={"answer": "Sprinkler exceeds Drip by 15 liters. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Drip exceeds Sprinkler by 15 liters. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="subject_swapped",
    ))
    cases.append(_make_case(
        15,
        category="derived_calculation",
        documents=["doc09_bilingual_cs_vi_en.pdf"],
        source_language="vi",
        answer_language="vi",
        question="Hybrid cải thiện Recall@5 hơn BM25 bao nhiêu điểm phần trăm?",
        gold_answer="Hybrid cao hơn BM25 15.5 điểm phần trăm.",
        atomic_claims=[],
        gold_sources=[{"document": "doc09_bilingual_cs_vi_en.pdf", "page": 1, "bbox": [60, 65, 535, 200], "passage": "Recall@5 đạt 93.5%."}],
        derivation_spec={
            "op": "diff_pp",
            "operands": [
                {"entity": "Hybrid", "metric": "recall@5", "split": "eval", "value": 93.5, "unit": "%"},
                {"entity": "BM25", "metric": "recall@5", "split": "eval", "value": 78.0, "unit": "%"},
            ],
            "expected_value": 15.5,
            "expected_unit": "pp",
        },
        good_prediction={"answer": "Hybrid cao hơn BM25 15.5 điểm phần trăm. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Hybrid cao hơn BM25 15.5%. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="unit_mismatch",
    ))
    cases.append(_make_case(
        16,
        category="derived_calculation",
        documents=["doc09_bilingual_cs_vi_en.pdf"],
        source_language="vi",
        answer_language="en",
        question="By how many percentage points does Hybrid outperform BM25 in Recall@5?",
        gold_answer="Hybrid outperforms BM25 by 15.5 percentage points.",
        atomic_claims=[],
        gold_sources=[{"document": "doc09_bilingual_cs_vi_en.pdf", "page": 1, "bbox": [60, 65, 535, 200], "passage": "78.0% Recall@5"}],
        derivation_spec={
            "op": "diff_pp",
            "operands": [
                {"entity": "Hybrid", "metric": "recall@5", "split": "eval", "value": 93.5, "unit": "%"},
                {"entity": "BM25", "metric": "recall@5", "split": "eval", "value": 78.0, "unit": "%"},
            ],
            "expected_value": 15.5,
            "expected_unit": "pp",
        },
        good_prediction={"answer": "Hybrid outperforms BM25 by 15.5 percentage points. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "BM25 outperforms Hybrid by 15.5 percentage points. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="subject_swapped",
    ))
    cases.append(_make_case(
        17,
        category="derived_calculation",
        documents=["doc03_vision_2col_en.pdf"],
        source_language="en",
        answer_language="en",
        question="How many milliseconds slower is ViT-Large than ViT-Small on Tesla T4?",
        gold_answer="ViT-Large exceeds ViT-Small latency by 21 ms.",
        atomic_claims=[],
        gold_sources=[{"document": "doc03_vision_2col_en.pdf", "page": 1, "bbox": [310, 90, 545, 250], "passage": "39 ms latency"}],
        derivation_spec={
            "op": "sub",
            "operands": [
                {"entity": "ViT-Large", "metric": "latency", "split": "t4", "value": 39.0, "unit": "ms"},
                {"entity": "ViT-Small", "metric": "latency", "split": "t4", "value": 18.0, "unit": "ms"},
            ],
            "expected_value": 21.0,
            "expected_unit": "ms",
        },
        good_prediction={"answer": "ViT-Large exceeds ViT-Small by 21 ms in latency. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "ViT-Large exceeds ViT-Small by 21 s in latency. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="unit_mismatch",
    ))
    cases.append(_make_case(
        18,
        category="derived_calculation",
        documents=["doc01_agronomy_en.pdf", "doc02_agronomy_vi.pdf"],
        source_language="vi",
        answer_language="vi",
        question="Thí nghiệm nhà kính kéo dài hơn thí nghiệm vườn cây bao nhiêu ngày?",
        gold_answer="Nhà kính lớn hơn vườn cây 21 ngày (35 ngày so với 14 ngày).",
        atomic_claims=[],
        gold_sources=[
            {"document": "doc01_agronomy_en.pdf", "page": 1, "bbox": [55, 95, 450, 120], "passage": "fourteen consecutive days"},
            {"document": "doc02_agronomy_vi.pdf", "page": 1, "bbox": [60, 60, 535, 200], "passage": "35 ngày"},
        ],
        derivation_spec={
            "op": "sub",
            "allow_cross_split": True,
            "operands": [
                {"entity": "Greenhouse", "metric": "duration", "split": "greenhouse", "value": 35.0, "unit": "day"},
                {"entity": "Orchard", "metric": "duration", "split": "orchard", "value": 14.0, "unit": "day"},
            ],
            "expected_value": 21.0,
            "expected_unit": "day",
        },
        good_prediction={"answer": "Greenhouse lớn hơn Orchard 21 ngày. [1] [2]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Orchard lớn hơn Greenhouse 21 ngày. [1] [2]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="subject_swapped",
    ))
    cases.append(_make_case(
        19,
        category="derived_calculation",
        documents=["doc03_vision_2col_en.pdf", "doc10_multi_contradiction_en.pdf"],
        source_language="en",
        answer_language="en",
        question="By how many ms did ViT-Small latency increase on Jetson Orin compared to Tesla T4?",
        gold_answer="Orin exceeds T4 by 13 ms for ViT-Small.",
        atomic_claims=[],
        gold_sources=[
            {"document": "doc03_vision_2col_en.pdf", "page": 1, "bbox": [310, 90, 545, 250], "passage": "18 ms latency"},
            {"document": "doc10_multi_contradiction_en.pdf", "page": 1, "bbox": [55, 95, 480, 120], "passage": "31 ms"},
        ],
        derivation_spec={
            "op": "sub",
            "allow_cross_split": True,
            "operands": [
                {"entity": "Orin", "metric": "latency", "split": "orin", "value": 31.0, "unit": "ms"},
                {"entity": "T4", "metric": "latency", "split": "t4", "value": 18.0, "unit": "ms"},
            ],
            "expected_value": 13.0,
            "expected_unit": "ms",
        },
        good_prediction={"answer": "Orin exceeds T4 by 13 ms in ViT-Small latency. [1] [2]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Orin exceeds T4 by 18 ms in ViT-Small latency. [1] [2]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="derived_value_missing",
    ))
    cases.append(_make_case(
        20,
        category="derived_calculation",
        documents=["doc04_multilevel_table_en.pdf"],
        source_language="en",
        answer_language="en",
        question="What is the total sample count across Validation and Test sets in Table 2?",
        gold_answer="The combined sample count is 3600.",
        atomic_claims=[],
        gold_sources=[{"document": "doc04_multilevel_table_en.pdf", "page": 1, "bbox": [55, 165, 510, 190], "passage": "Validation set has 1200 samples; Test set has 2400 samples."}],
        derivation_spec={
            "op": "add",
            "allow_cross_split": True,
            "operands": [
                {"entity": "Validation", "metric": "samples", "split": "val", "value": 1200.0, "unit": ""},
                {"entity": "Test", "metric": "samples", "split": "test", "value": 2400.0, "unit": ""},
            ],
            "expected_value": 3600.0,
            "expected_unit": "",
        },
        good_prediction={"answer": "The combined sample count across both splits is 3600. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "The combined sample count across both splits is 2400. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="derived_value_missing",
    ))

    # --- Cases 21..30: Mathematical Structure, Inequalities, Fractions & Domain Constraints ---
    cases.append(_make_case(
        21,
        category="math_inequality",
        documents=["doc06_math_domain_en.pdf"],
        source_language="en",
        answer_language="en",
        question="What inequalities define the feasibility region in Equation (3)?",
        gold_answer="The feasibility region is $x < 3$ and $y > 2$.",
        atomic_claims=[],
        gold_sources=[{"document": "doc06_math_domain_en.pdf", "page": 1, "bbox": [55, 145, 450, 168], "passage": "Equation (3): Feasibility region x < 3 and y > 2."}],
        gold_formula="x < 3 and y > 2",
        good_prediction={"answer": "The feasibility region is defined by $x < 3$ and $y > 2$. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "The feasibility region is defined by $x > 3$ and $y > 2$. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="formula_mismatch",
    ))
    cases.append(_make_case(
        22,
        category="math_inequality",
        documents=["doc06_math_domain_en.pdf"],
        source_language="en",
        answer_language="vi",
        question="Nêu miền khả thi ở Phương trình (3).",
        gold_answer="Miền khả thi được xác định bởi $3 > x$ và $y > 2$.",
        atomic_claims=[],
        gold_sources=[{"document": "doc06_math_domain_en.pdf", "page": 1, "bbox": [55, 145, 450, 168], "passage": "x < 3 and y > 2"}],
        gold_formula="x < 3 and y > 2",
        # Equivalent flipped inequality `3 > x` MUST pass! Dropped inequality `x = 3` MUST fail!
        good_prediction={"answer": "Miền khả thi xác định bởi $3 > x$ và $y > 2$. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Miền khả thi xác định bởi $x = 3$ và $y = 2$. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="formula_mismatch",
    ))
    cases.append(_make_case(
        23,
        category="math_fraction_domain",
        documents=["doc06_math_domain_en.pdf"],
        source_language="en",
        answer_language="en",
        question="What is the stacked ratio formula in Equation (5) and its denominator condition?",
        gold_answer="Equation (5) is $y = \\frac{a}{b}$ with $b \\neq 0$.",
        atomic_claims=[],
        gold_sources=[{"document": "doc06_math_domain_en.pdf", "page": 1, "bbox": [55, 195, 460, 218], "passage": "Equation (5): Stacked ratio y = a / b with b != 0."}],
        gold_formula="y = a / b",
        domain_constraints=["b!=0"],
        good_prediction={"answer": "Equation (5) is $y = \\frac{a}{b}$ with condition $b \\neq 0$. [1]", "answer_status": "ANSWERED"},
        # Truncated stacked fraction `y = a` bug from Section 1 P0!
        bad_prediction={"answer": "Equation (5) is $y = a$ with condition $b \\neq 0$. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="formula_mismatch",
    ))
    cases.append(_make_case(
        24,
        category="math_fraction_domain",
        documents=["doc06_math_domain_en.pdf"],
        source_language="en",
        answer_language="vi",
        question="Viết công thức tỷ số ở Phương trình (5) kèm điều kiện mẫu số.",
        gold_answer="Công thức là $y = a / b$ với $b \\neq 0$.",
        atomic_claims=[],
        gold_sources=[{"document": "doc06_math_domain_en.pdf", "page": 1, "bbox": [55, 195, 460, 218], "passage": "y = a / b with b != 0"}],
        gold_formula="y = a / b",
        domain_constraints=["b!=0"],
        good_prediction={"answer": "Phương trình (5) là $y = \\frac{a}{b}$ với $b \\neq 0$. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Phương trình (5) là $y = \\frac{a}{b}$ với mọi b thực. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="missing_domain_constraint",
    ))
    cases.append(_make_case(
        25,
        category="math_sub_vs_super",
        documents=["doc06_math_domain_en.pdf"],
        source_language="en",
        answer_language="vi",
        question="Giá trị phần tử thứ hai x_2 ở Phương trình (2) bằng bao nhiêu?",
        gold_answer="Phương trình (2) xác định $x_2 = 9$.",
        atomic_claims=[],
        gold_sources=[{"document": "doc06_math_domain_en.pdf", "page": 1, "bbox": [55, 120, 420, 142], "passage": "Equation (2): Second sequence element x_2 = 9."}],
        gold_formula="x_2 = 9",
        good_prediction={"answer": "Theo Phương trình (2), phần tử thứ hai là $x_2 = 9$. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Theo Phương trình (2), phần tử thứ hai là $x^2 = 9$. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="formula_mismatch",
    ))
    cases.append(_make_case(
        26,
        category="math_domain_constraint",
        documents=["doc06_math_domain_en.pdf"],
        source_language="en",
        answer_language="en",
        question="What horizon condition is stated in Equation (6)?",
        gold_answer="The discounted return horizon requires $t \\geq 0$.",
        atomic_claims=[],
        gold_sources=[{"document": "doc06_math_domain_en.pdf", "page": 1, "bbox": [55, 220, 440, 242], "passage": "Equation (6): Discounted return horizon t >= 0."}],
        gold_formula="t >= 0",
        good_prediction={"answer": "The horizon condition in Equation (6) is $t \\geq 0$. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "The horizon condition in Equation (6) is $t > 0$. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="formula_mismatch",
    ))
    cases.append(_make_case(
        27,
        category="math_domain_constraint",
        documents=["doc09_bilingual_cs_vi_en.pdf"],
        source_language="vi",
        answer_language="vi",
        question="Hệ số kết hợp alpha trong thực nghiệm hybrid bằng bao nhiêu?",
        gold_answer="Hệ số kết hợp là $alpha = 0.65$.",
        atomic_claims=[],
        gold_sources=[{"document": "doc09_bilingual_cs_vi_en.pdf", "page": 1, "bbox": [60, 65, 535, 160], "passage": "alpha = 0.65"}],
        gold_formula="alpha = 0.65",
        good_prediction={"answer": "Hệ số kết hợp trong truy hồi hybrid là $\\alpha = 0.65$ ($alpha = 0.65$). [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Hệ số kết hợp trong truy hồi hybrid là $alpha = 0.05$. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="formula_mismatch",
    ))
    cases.append(_make_case(
        28,
        category="math_domain_constraint",
        documents=["doc06_math_domain_en.pdf"],
        source_language="en",
        answer_language="en",
        question="When x != 0 in Equation (4), what does x / x simplify to?",
        gold_answer="Under $x \\neq 0$, $x / x = 1$.",
        atomic_claims=[],
        gold_sources=[{"document": "doc06_math_domain_en.pdf", "page": 1, "bbox": [55, 170, 460, 192], "passage": "x / x = 1 for x != 0"}],
        gold_formula="1 = 1",
        domain_constraints=["x!=0"],
        # With explicit x != 0, x / x = 1 canonicalizes Div(x, x) -> 1 so `1 = 1` holds!
        good_prediction={"answer": "For $x \\neq 0$, $x / x = 1$. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "It simplifies to $x / x = 1$. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="missing_domain_constraint",
    ))
    cases.append(_make_case(
        29,
        category="citation_region_iou",
        documents=["doc06_math_domain_en.pdf"],
        source_language="en",
        answer_language="en",
        question="Cite the exact region of Equation (2) on page 1.",
        gold_answer="Equation (2) states $x_2 = 9$.",
        atomic_claims=[],
        gold_sources=[{"document": "doc06_math_domain_en.pdf", "page": 1, "bbox": [55, 120, 420, 142], "passage": "x_2 = 9"}],
        gold_formula="x_2 = 9",
        good_prediction={
            "answer": "Equation (2) states $x_2 = 9$. [1]",
            "answer_status": "ANSWERED",
            "cited_regions": [{"document": "doc06_math_domain_en.pdf", "page": 1, "bbox": [58, 121, 415, 141]}],
        },
        # Points to Equation (1)'s bbox on the same page instead of Equation (2)!
        bad_prediction={
            "answer": "Equation (2) states $x_2 = 9$. [1]",
            "answer_status": "ANSWERED",
            "cited_regions": [{"document": "doc06_math_domain_en.pdf", "page": 1, "bbox": [58, 95, 415, 115]}],
        },
        expected_bad_failure_prefix="citation_region_mismatch",
    ))
    cases[-1]["gold_regions"] = [{"document": "doc06_math_domain_en.pdf", "page": 1, "bbox": [55, 120, 420, 142]}]

    cases.append(_make_case(
        30,
        category="citation_region_iou",
        documents=["doc05_chart_ab_en.pdf"],
        source_language="en",
        answer_language="vi",
        question="Trích dẫn đúng vùng chú thích Hình 3 cho kết quả 92% của phương pháp B.",
        gold_answer="Hình 3 cho thấy phương pháp B đạt 92%.",
        atomic_claims=[],
        gold_sources=[{"document": "doc05_chart_ab_en.pdf", "page": 1, "bbox": [75, 325, 520, 350], "passage": "Method B achieves 92%"}],
        gold_quantities=[{"value": 92.0, "unit": "%"}],
        good_prediction={
            "answer": "Hình 3 ghi nhận phương pháp B đạt 92%. [1]",
            "answer_status": "ANSWERED",
            "cited_regions": [{"document": "doc05_chart_ab_en.pdf", "page": 1, "bbox": [78, 328, 515, 348]}],
        },
        bad_prediction={
            "answer": "Hình 3 ghi nhận phương pháp B đạt 92%. [1]",
            "answer_status": "ANSWERED",
            "cited_regions": [{"document": "doc05_chart_ab_en.pdf", "page": 1, "bbox": [60, 50, 300, 75]}],
        },
        expected_bad_failure_prefix="citation_region_mismatch",
    ))
    cases[-1]["gold_regions"] = [{"document": "doc05_chart_ab_en.pdf", "page": 1, "bbox": [75, 325, 520, 350]}]

    # --- Cases 31..40: Negation, Subject Swaps, Units & Four Language Directions ---
    cases.append(_make_case(
        31,
        category="negation_polarity",
        documents=["doc07_clinical_negation_en.pdf"],
        source_language="en",
        answer_language="vi",
        question="Phương pháp điều trị Y có làm giảm huyết áp tâm thu không?",
        gold_answer="Không, phương pháp điều trị Y không làm giảm huyết áp tâm thu.",
        atomic_claims=["Treatment Y did not reduce blood pressure."],
        gold_sources=[{"document": "doc07_clinical_negation_en.pdf", "page": 1, "bbox": [55, 120, 480, 142], "passage": "Treatment Y did not reduce systolic blood pressure."}],
        good_prediction={"answer": "Phương pháp điều trị Y hoàn toàn không làm giảm huyết áp tâm thu. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Phương pháp điều trị Y giúp giảm huyết áp tâm thu rõ rệt. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="polarity_inverted",
    ))
    cases.append(_make_case(
        32,
        category="unit_integrity",
        documents=["doc07_clinical_negation_en.pdf"],
        source_language="en",
        answer_language="en",
        question="What concentration did Biomarker Z reach in Cohort C?",
        gold_answer="Biomarker Z reached 15 mg/L in Cohort C.",
        gold_sources=[{"document": "doc07_clinical_negation_en.pdf", "page": 1, "bbox": [55, 145, 450, 168], "passage": "Biomarker Z reached 15 mg/L in Cohort C."}],
        gold_quantities=[{"value": 15.0, "unit": "mg/l"}],
        good_prediction={"answer": "Biomarker Z reached 15 mg/L in Cohort C. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Biomarker Z reached 15 g/L in Cohort C. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="quantity_or_unit_mismatch",
    ))
    cases.append(_make_case(
        33,
        category="unit_integrity",
        documents=["doc02_agronomy_vi.pdf"],
        source_language="vi",
        answer_language="en",
        question="What soluble phosphorus concentration was measured in the greenhouse topsoil?",
        gold_answer="Soluble phosphorus reached 16 mg/L in the topsoil.",
        atomic_claims=[],
        gold_sources=[{"document": "doc02_agronomy_vi.pdf", "page": 1, "bbox": [60, 60, 535, 220], "passage": "16 mg/L"}],
        gold_quantities=[{"value": 16.0, "unit": "mg/l"}],
        good_prediction={"answer": "Soluble phosphorus reached 16 mg/L in the topsoil. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Soluble phosphorus reached 16 mg in the topsoil. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="quantity_or_unit_mismatch",
    ))
    cases.append(_make_case(
        34,
        category="unit_integrity",
        documents=["doc02_agronomy_vi.pdf"],
        source_language="vi",
        answer_language="vi",
        question="Mỗi ô tiêu chuẩn trong nhà kính tiêu thụ bao nhiêu lít nước tưới nhỏ giọt?",
        gold_answer="Lượng nước tưới nhỏ giọt tiêu thụ 28 lít mỗi ô tiêu chuẩn.",
        atomic_claims=[],
        gold_sources=[{"document": "doc02_agronomy_vi.pdf", "page": 1, "bbox": [60, 60, 535, 220], "passage": "28 lít"}],
        gold_quantities=[{"value": 28.0, "unit": "l"}],
        good_prediction={"answer": "Mỗi ô tiêu chuẩn tiêu thụ 28 lít nước tưới nhỏ giọt. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Mỗi ô tiêu chuẩn tiêu thụ 28 ml nước tưới nhỏ giọt. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="quantity_or_unit_mismatch",
    ))
    cases.append(_make_case(
        35,
        category="cross_lingual_vi_to_en",
        documents=["doc02_agronomy_vi.pdf"],
        source_language="vi",
        answer_language="en",
        question="How many consecutive days did the greenhouse experiment in Lam Dong last?",
        gold_answer="The greenhouse experiment lasted 35 consecutive days.",
        gold_sources=[{"document": "doc02_agronomy_vi.pdf", "page": 1, "bbox": [60, 60, 535, 220], "passage": "35 ngày liên tiếp"}],
        gold_quantities=[{"value": 35.0, "unit": "day"}],
        good_prediction={"answer": "The greenhouse experiment lasted 35 consecutive days. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Thí nghiệm kéo dài 35 ngày liên tiếp. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="wrong_language",
    ))
    cases.append(_make_case(
        36,
        category="cross_lingual_en_to_vi",
        documents=["doc01_agronomy_en.pdf"],
        source_language="en",
        answer_language="vi",
        question="Hệ thống tưới nhỏ giọt dùng bao nhiêu lít nước theo Bảng 1?",
        gold_answer="Hệ thống tưới nhỏ giọt dùng 42 lít nước.",
        gold_sources=[{"document": "doc01_agronomy_en.pdf", "page": 2, "bbox": [55, 115, 480, 135], "passage": "Drip irrigation 42 liters"}],
        gold_quantities=[{"value": 42.0, "unit": "l"}],
        good_prediction={"answer": "Hệ thống tưới nhỏ giọt dùng 42 lít nước. [1]", "answer_status": "ANSWERED"},
        # Pilot failure mode from Section 1: answering 21 days instead of 42 liters!
        bad_prediction={"answer": "Hệ thống tưới nhỏ giọt kéo dài 21 ngày. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="quantity_or_unit_mismatch",
    ))
    cases.append(_make_case(
        37,
        category="directed_subject_swap",
        documents=["doc09_bilingual_cs_vi_en.pdf"],
        source_language="vi",
        answer_language="vi",
        question="Giữa Hybrid và BM25, phương pháp nào có Recall@5 cao hơn?",
        gold_answer="Hybrid cao hơn BM25 về Recall@5.",
        atomic_claims=[],
        gold_sources=[{"document": "doc09_bilingual_cs_vi_en.pdf", "page": 1, "bbox": [60, 65, 535, 200], "passage": "Recall@5 đạt 93.5%."}],
        directed_relation={"higher": "Hybrid", "lower": "BM25"},
        # Equivalent phrasing "BM25 thấp hơn Hybrid" MUST pass!
        good_prediction={"answer": "BM25 thấp hơn Hybrid về chỉ số Recall@5 (78.0% so với 93.5%). [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "BM25 cao hơn Hybrid về chỉ số Recall@5. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="subject_swapped",
    ))
    cases.append(_make_case(
        38,
        category="directed_subject_swap",
        documents=["doc05_chart_ab_en.pdf"],
        source_language="en",
        answer_language="en",
        question="Which method achieved higher validation accuracy in Figure 3, Method A or Method B?",
        gold_answer="Method B outperforms Method A in Figure 3.",
        atomic_claims=[],
        gold_sources=[{"document": "doc05_chart_ab_en.pdf", "page": 1, "bbox": [75, 325, 520, 350], "passage": "Method A achieves 81% and Method B achieves 92%."}],
        directed_relation={"higher": "Method B", "lower": "Method A"},
        good_prediction={"answer": "Method B outperforms Method A on validation accuracy (92% vs 81%). [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Method A outperforms Method B on validation accuracy (92% vs 81%). [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="subject_swapped",
    ))
    cases.append(_make_case(
        39,
        category="unit_integrity",
        documents=["doc03_vision_2col_en.pdf"],
        source_language="en",
        answer_language="vi",
        question="Thông lượng của mô hình ViT-Small trên Tesla T4 đạt bao nhiêu tokens/s?",
        gold_answer="ViT-Small đạt thông lượng 140 tokens/s trên Tesla T4.",
        atomic_claims=[],
        gold_sources=[{"document": "doc03_vision_2col_en.pdf", "page": 1, "bbox": [310, 90, 545, 250], "passage": "140 tokens/s throughput"}],
        gold_quantities=[{"value": 140.0, "unit": "tokens/s"}],
        good_prediction={"answer": "Mô hình ViT-Small đạt thông lượng 140 tokens/s trên Tesla T4. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Mô hình ViT-Small đạt thông lượng 140 ms trên Tesla T4. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="quantity_or_unit_mismatch",
    ))
    cases.append(_make_case(
        40,
        category="unit_integrity",
        documents=["doc09_bilingual_cs_vi_en.pdf"],
        source_language="vi",
        answer_language="en",
        question="What is the p95 response latency of the reranker?",
        gold_answer="The p95 response latency of the reranker is 45 ms.",
        atomic_claims=[],
        gold_sources=[{"document": "doc09_bilingual_cs_vi_en.pdf", "page": 1, "bbox": [60, 65, 535, 240], "passage": "45 ms"}],
        gold_quantities=[{"value": 45.0, "unit": "ms"}],
        good_prediction={"answer": "The p95 response latency of the reranker is 45 ms. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "The p95 response latency of the reranker is 45 seconds. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="quantity_or_unit_mismatch",
    ))

    # --- Cases 41..52: Abstention, Language Discipline on OOD, Leaky Refusals & False Refusals ---
    trap_specs = [
        (41, ["doc01_agronomy_en.pdf"], "en", "en", "What was the potassium fertilizer dosage in the orchard trial?", ["potassium"], [80], "The orchard trial document does not mention potassium fertilizer dosage.", "NOT_FOUND_IN_DOCUMENT", "Dựa trên tài liệu đã chọn, thông tin không được đề cập.", "NOT_FOUND_IN_DOCUMENT", "wrong_language"),
        (42, ["doc03_vision_2col_en.pdf"], "en", "en", "What Top-1 accuracy did ViT-Small achieve on ImageNet-1K?", ["imagenet"], [79.8], "Insufficient evidence in the selected document regarding ImageNet-1K accuracy.", "INSUFFICIENT_EVIDENCE", "ViT-Small achieved 79.8% Top-1 accuracy on ImageNet-1K.", "ANSWERED", "abstention_or_answerability_failed"),
        (43, ["doc03_vision_2col_en.pdf"], "en", "vi", "Chi phí thuê cloud huấn luyện ViT-Large là bao nhiêu USD?", ["cloud"], [420], "Tài liệu đã chọn không đề cập chi phí thuê cloud huấn luyện ViT-Large.", "NOT_FOUND_IN_DOCUMENT", "Chi phí huấn luyện trên cloud là 420 USD.", "NOT_FOUND_IN_DOCUMENT", "status_display_mismatch"),
        (44, ["doc04_multilevel_table_en.pdf"], "en", "vi", "Điểm F1-macro trên tập kiểm thử y sinh trong Bảng 2 là bao nhiêu?", ["y sinh"], [91], "Chưa đủ bằng chứng trong tài liệu đã chọn về tập kiểm thử y sinh.", "INSUFFICIENT_EVIDENCE", "Điểm F1-macro trên tập y sinh là 91%.", "ANSWERED", "abstention_or_answerability_failed"),
        (45, ["doc05_chart_ab_en.pdf"], "en", "en", "What accuracy did Method C achieve in Figure 3?", ["method c"], [87], "Method C is not mentioned in Figure 3 of the provided document.", "NOT_FOUND_IN_DOCUMENT", "Method C achieved 87% in Figure 3.", "INSUFFICIENT_EVIDENCE", "status_display_mismatch"),
        (46, ["doc06_math_domain_en.pdf"], "en", "vi", "Giá trị đạo hàm bậc hai của hàm mất mát ở Phương trình (9) là gì?", [], [0.5], "Tài liệu không có Phương trình (9) hay thông tin về đạo hàm bậc hai.", "NOT_FOUND_IN_DOCUMENT", "", "", "abstention_or_answerability_failed"),
        (47, ["doc07_clinical_negation_en.pdf"], "en", "en", "What survival rate was reported for Cohort A after 5 years?", ["cohort a"], [64], "No information is provided about 5-year survival rates or Cohort A.", "INSUFFICIENT_EVIDENCE", "The document does not mention survival rates, which was 64%.", "INSUFFICIENT_EVIDENCE", "status_display_mismatch"),
        (48, ["doc08_scan_vi.pdf"], "vi", "vi", "Tốc độ quét bao nhiêu trang mỗi phút trên máy quét phần cứng?", [], [120], "Tài liệu không đề cập tốc độ quét trang mỗi phút của phần cứng.", "NOT_FOUND_IN_DOCUMENT", "Insufficient evidence in the provided document.", "NOT_FOUND_IN_DOCUMENT", "wrong_language"),
        (49, ["doc09_bilingual_cs_vi_en.pdf"], "vi", "en", "What BLEU score was reported for neural machine translation?", ["bleu"], [34.2], "The selected document does not mention BLEU scores or machine translation.", "NOT_FOUND_IN_DOCUMENT", "The reported BLEU score is 34.2.", "ANSWERED", "abstention_or_answerability_failed"),
        (50, ["doc10_multi_contradiction_en.pdf"], "en", "vi", "Mức tiêu thụ điện năng tính bằng Watt trên Jetson Orin là bao nhiêu?", ["watt"], [25], "Chưa đủ bằng chứng trong tài liệu về mức tiêu thụ điện năng (Watt) trên Jetson Orin.", "INSUFFICIENT_EVIDENCE", "Mức tiêu thụ điện trên Jetson Orin là 25 W.", "INSUFFICIENT_EVIDENCE", "status_display_mismatch"),
    ]
    for idx, docs, s_lang, a_lang, q, f_ents, f_nums, good_ans, good_st, bad_ans, bad_st, bad_prefix in trap_specs:
        cases.append(_make_case(
            idx,
            category="adversarial_trap",
            documents=docs,
            source_language=s_lang,
            answer_language=a_lang,
            answerability="unanswerable",
            question=q,
            forbidden_entities=f_ents,
            forbidden_numbers=f_nums,
            good_prediction={"answer": good_ans, "answer_status": good_st},
            bad_prediction={"answer": bad_ans, "answer_status": bad_st},
            expected_bad_failure_prefix=bad_prefix,
        ))

    # --- Cases 51..55: Ambiguous Scope Clarification & False-Refusal Prevention ---
    cases.append(_make_case(
        51,
        category="ambiguous_clarification",
        documents=["doc04_multilevel_table_en.pdf", "doc10_multi_contradiction_en.pdf"],
        source_language="en",
        answer_language="vi",
        answerability="ambiguous",
        question="Độ chính xác của Phương pháp B là bao nhiêu?",
        forbidden_numbers=[99],
        good_prediction={
            "answer": "Vui lòng làm rõ bạn muốn hỏi độ chính xác của Phương pháp B trên tập Validation (Bảng 2), tập Test (Bảng 2) hay tập dữ liệu thực địa nhiễu?",
            "answer_status": "CLARIFICATION_NEEDED",
        },
        bad_prediction={
            "answer": "Độ chính xác của Phương pháp B luôn là 92% trên mọi tập dữ liệu.",
            "answer_status": "ANSWERED",
        },
        expected_bad_failure_prefix="abstention_or_answerability_failed",
    ))
    cases.append(_make_case(
        52,
        category="ambiguous_clarification",
        documents=["doc03_vision_2col_en.pdf", "doc10_multi_contradiction_en.pdf"],
        source_language="en",
        answer_language="en",
        answerability="ambiguous",
        question="What is the inference latency of ViT-Small?",
        good_prediction={
            "answer": "Please clarify which hardware platform you are asking about (Tesla T4 vs Jetson Orin).",
            "answer_status": "CLARIFICATION_NEEDED",
        },
        bad_prediction={
            "answer": "The inference latency of ViT-Small is 18 ms.",
            "answer_status": "ANSWERED",
        },
        expected_bad_failure_prefix="abstention_or_answerability_failed",
    ))
    cases.append(_make_case(
        53,
        category="false_refusal_guard",
        documents=["doc01_agronomy_en.pdf"],
        source_language="en",
        answer_language="vi",
        question="Nhiệt độ đất trong thử nghiệm vườn cây ổn định ở bao nhiêu độ C?",
        gold_answer="Nhiệt độ đất ổn định ở 24 độ C.",
        atomic_claims=[],
        gold_sources=[{"document": "doc01_agronomy_en.pdf", "page": 1, "bbox": [55, 120, 460, 142], "passage": "24 degrees Celsius"}],
        gold_quantities=[{"value": 24.0, "unit": ""}],
        good_prediction={"answer": "Nhiệt độ đất ổn định ở mức 24 độ C. [1]", "answer_status": "ANSWERED"},
        # False refusal on an answerable question MUST fail!
        bad_prediction={"answer": "Chưa đủ bằng chứng trong tài liệu đã chọn.", "answer_status": "INSUFFICIENT_EVIDENCE"},
        expected_bad_failure_prefix="abstention_or_answerability_failed",
    ))
    cases.append(_make_case(
        54,
        category="false_refusal_guard",
        documents=["doc03_vision_2col_en.pdf"],
        source_language="en",
        answer_language="en",
        question="How many attention heads are in the baseline ViT-Small encoder?",
        gold_answer="The baseline ViT-Small encoder has 6 attention heads.",
        atomic_claims=[],
        gold_sources=[{"document": "doc03_vision_2col_en.pdf", "page": 1, "bbox": [50, 90, 280, 250], "passage": "6 attention heads"}],
        gold_quantities=[{"value": 6.0, "unit": ""}],
        good_prediction={"answer": "The baseline ViT-Small encoder has 6 attention heads. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "The provided document does not mention attention heads.", "answer_status": "NOT_FOUND_IN_DOCUMENT"},
        expected_bad_failure_prefix="abstention_or_answerability_failed",
    ))
    cases.append(_make_case(
        55,
        category="false_refusal_guard",
        documents=["doc10_multi_contradiction_en.pdf"],
        source_language="en",
        answer_language="vi",
        question="Trên bộ dữ liệu thực địa nhiều nhiễu, Phương pháp B đạt độ chính xác bao nhiêu phần trăm?",
        gold_answer="Phương pháp B đạt 85% độ chính xác trên tập dữ liệu thực địa nhiễu.",
        atomic_claims=[],
        gold_sources=[{"document": "doc10_multi_contradiction_en.pdf", "page": 1, "bbox": [55, 120, 490, 142], "passage": "Method B achieved 85% accuracy on the noisy field dataset."}],
        gold_quantities=[{"value": 85.0, "unit": "%"}],
        good_prediction={"answer": "Trên tập dữ liệu thực địa nhiễu, Phương pháp B đạt 85%. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Method B achieved 85% accuracy on the noisy field dataset. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="wrong_language",
    ))

    # --- Cases 56..60: Multi-Document Scope, Unaccented VI Intent & Scan-Only Evidence ---
    cases.append(_make_case(
        56,
        category="unaccented_vietnamese_query",
        documents=["doc03_vision_2col_en.pdf"],
        source_language="en",
        answer_language="vi",
        question="Ket qua batch size khi pretraining ViT-Small o trang nao va bang bao nhieu?",
        gold_answer="Batch size khi tiền huấn luyện là 256 ảnh mỗi bước ở trang 1.",
        atomic_claims=[],
        gold_sources=[{"document": "doc03_vision_2col_en.pdf", "page": 1, "bbox": [50, 90, 280, 260], "passage": "batch size was fixed at 256 images"}],
        gold_quantities=[{"value": 256.0, "unit": ""}],
        good_prediction={"answer": "Kích thước lô (batch size) khi pretraining cố định ở 256 ảnh mỗi bước tại trang 1. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Kích thước lô khi pretraining là 512 ảnh mỗi bước. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="quantity_or_unit_mismatch",
    ))
    cases.append(_make_case(
        57,
        category="multi_doc_comparison",
        documents=["doc04_multilevel_table_en.pdf", "doc10_multi_contradiction_en.pdf"],
        source_language="en",
        answer_language="en",
        question="By how many percentage points did Method B drop from the clean Test split (Table 2) to the noisy field dataset?",
        gold_answer="Method B Test exceeds Field by 3 percentage points (88% vs 85%).",
        atomic_claims=[],
        gold_sources=[
            {"document": "doc04_multilevel_table_en.pdf", "page": 1, "bbox": [55, 110, 500, 150], "passage": "Method B 92% 88%"},
            {"document": "doc10_multi_contradiction_en.pdf", "page": 1, "bbox": [55, 120, 490, 142], "passage": "85% accuracy"},
        ],
        derivation_spec={
            "op": "diff_pp",
            "allow_cross_split": True,
            "operands": [
                {"entity": "Test", "metric": "accuracy", "split": "clean_test", "value": 88.0, "unit": "%"},
                {"entity": "Field", "metric": "accuracy", "split": "noisy_field", "value": 85.0, "unit": "%"},
            ],
            "expected_value": 3.0,
            "expected_unit": "pp",
        },
        good_prediction={"answer": "Test exceeds Field by 3 percentage points for Method B (88% vs 85%). [1] [2]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Test exceeds Field by 7 percentage points for Method B (92% vs 85%). [1] [2]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="derived_value_missing",
    ))
    cases.append(_make_case(
        58,
        category="scan_ocr_vi",
        documents=["doc08_scan_vi.pdf"],
        source_language="en",
        answer_language="en",
        # Page 1 is a pure rasterized PNG scan ("Accuracy was 83 percent on scanned receipts.")
        question="What accuracy was reported on scanned receipts on page 1?",
        gold_answer="Accuracy was 83% on scanned receipts.",
        atomic_claims=[],
        gold_sources=[{"document": "doc08_scan_vi.pdf", "page": 1, "bbox": [60, 80, 520, 300]}],
        gold_quantities=[{"value": 83.0, "unit": "%"}],
        good_prediction={"answer": "Accuracy was 83% on scanned receipts. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Accuracy was 96.4% on scanned receipts. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="quantity_or_unit_mismatch",
    ))
    cases.append(_make_case(
        59,
        category="scan_ocr_vi",
        documents=["doc08_scan_vi.pdf"],
        source_language="vi",
        answer_language="vi",
        question="Sai số ký tự tiếng Việt (CER) trên trang quét 1 là bao nhiêu phần trăm?",
        gold_answer="Sai số ký tự tiếng Việt đạt 3.2%.",
        atomic_claims=[],
        gold_sources=[{"document": "doc08_scan_vi.pdf", "page": 1, "bbox": [60, 80, 520, 300]}],
        gold_quantities=[{"value": 3.2, "unit": "%"}],
        good_prediction={"answer": "Sai số ký tự tiếng Việt đạt 3.2%. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Sai số ký tự tiếng Việt đạt 5.0%. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="quantity_or_unit_mismatch",
    ))
    cases.append(_make_case(
        60,
        category="table_unit_integrity",
        documents=["doc04_multilevel_table_en.pdf"],
        source_language="en",
        answer_language="vi",
        question="Tốc độ băng chuyền trong Bảng 2 của cả hai phương pháp A và B là bao nhiêu?",
        gold_answer="Tốc độ băng chuyền của cả hai phương pháp là 15 km/h.",
        atomic_claims=[],
        gold_sources=[{"document": "doc04_multilevel_table_en.pdf", "page": 1, "bbox": [55, 110, 500, 150], "passage": "15 km/h"}],
        gold_quantities=[{"value": 15.0, "unit": "km/h"}],
        good_prediction={"answer": "Tốc độ băng chuyền của cả hai phương pháp là 15 km/h. [1]", "answer_status": "ANSWERED"},
        bad_prediction={"answer": "Tốc độ băng chuyền của cả hai phương pháp là 15 m/s. [1]", "answer_status": "ANSWERED"},
        expected_bad_failure_prefix="quantity_or_unit_mismatch",
    ))

    assert len(cases) == 60, f"Expected 60 cases, built {len(cases)}"
    return cases


def verify_and_time_m0_gate(
    *,
    run_cpu_baseline_slice: bool = True,
    write_dataset_file: bool = False,
) -> dict[str, Any]:
    """
    Executes the full M0 Regression Gate:
    1. Builds all 10 controlled PDFs in an isolated temp folder and verifies deterministic SHA-256 + gold passages.
    2. Evaluates all 60 positive (`good_prediction`) and 60 negative (`bad_prediction`) cases via the 3-layer engine.
    3. Measures wall-clock timing on the first 10 representative cases (`M0-Q01`..`M0-Q10`), including
       an offline `cpu_min` diagnostic pass to anchor effort & latency baselines.
    """
    cases = generate_60_mechanism_cases()
    if write_dataset_file:
        GATE_DATASET_PATH.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with TemporaryDirectory(prefix="m0-gate-60-") as tmp:
        pdf_dir = Path(tmp) / "pdfs"
        t0_pdf = time.perf_counter()
        hashes_first = build_controlled_pdfs(pdf_dir)
        pdf_build_ms = round((time.perf_counter() - t0_pdf) * 1000.0, 2)
        hashes_second = build_controlled_pdfs(pdf_dir)
        if hashes_first != hashes_second:
            raise RuntimeError("Controlled PDF generation is non-deterministic!")

        # Validate every text-layer passage against actual generated PDF pages
        page_texts: dict[str, list[str]] = {}
        for fname in hashes_first:
            with fitz.open(pdf_dir / fname) as doc:
                page_texts[fname] = [" ".join(page.get_text().split()) for page in doc]

        for case in cases:
            for src in case["gold_sources"]:
                fname = src["document"]
                page = int(src["page"])
                if fname not in page_texts or not (1 <= page <= len(page_texts[fname])):
                    raise ValueError(f"{case['id']}: invalid source {fname} page {page}")
                passage = src.get("passage")
                if passage and " ".join(passage.split()) not in page_texts[fname][page - 1]:
                    raise ValueError(f"{case['id']}: passage {passage!r} missing from {fname} p.{page}")

        good_passed = 0
        bad_caught = 0
        unexpected_good_failures: list[dict[str, Any]] = []
        missed_bad_traps: list[dict[str, Any]] = []
        slice_10_timings_ms: list[dict[str, Any]] = []

        for idx, case in enumerate(cases):
            real_units = [
                {
                    "id": f"{src['document']}_p{src['page']}",
                    "doc_id": src["document"],
                    "page_no": int(src["page"]),
                    "text": page_texts[src["document"]][int(src["page"]) - 1],
                }
                for src in case["gold_sources"]
                if page_texts[src["document"]][int(src["page"]) - 1]
            ]
            good_pred = {**case["good_prediction"], "source_units": real_units} if real_units else case["good_prediction"]
            bad_pred = {**case["bad_prediction"], "source_units": real_units} if real_units else case["bad_prediction"]

            t_case = time.perf_counter()
            good_res = evaluate_three_layer_case(case, good_pred)
            bad_res = evaluate_three_layer_case(case, bad_pred)
            dt_ms = round((time.perf_counter() - t_case) * 1000.0, 3)

            if idx < 10:
                slice_10_timings_ms.append({
                    "id": case["id"],
                    "category": case["category"],
                    "evaluator_ms": dt_ms,
                })

            if good_res["passed"]:
                good_passed += 1
            else:
                unexpected_good_failures.append({"id": case["id"], "failures": good_res["failures"]})

            expected_prefix = case["expected_bad_failure_prefix"]
            if (not bad_res["passed"]) and any(f.startswith(expected_prefix) for f in bad_res["failures"]):
                bad_caught += 1
            else:
                missed_bad_traps.append({
                    "id": case["id"],
                    "expected_prefix": expected_prefix,
                    "actual_passed": bad_res["passed"],
                    "actual_failures": bad_res["failures"],
                })

        cpu_slice_summary = None
        if run_cpu_baseline_slice:
            from eval.run_benchmark import run_mode_evaluation
            from core import db

            prev_db_path, prev_init = db.DB_PATH, db._db_initialized
            try:
                db.DB_PATH = str(Path(tmp) / "data" / "slice10.db")
                db._db_initialized = False
                db.init_db()
                t_slice = time.perf_counter()
                slice_eval = run_mode_evaluation("cpu_min", cases[:10], pdf_dir, cache_regime="warm_no_cache")
                slice_sec = round(time.perf_counter() - t_slice, 3)
                cpu_slice_summary = {
                    "questions_timed": 10,
                    "wall_clock_sec": slice_sec,
                    "avg_ms_per_question": round((slice_sec / 10.0) * 1000.0, 2),
                    "retrieval_recall": slice_eval["retrieval_recall"],
                    "abstention_accuracy": slice_eval["abstention_accuracy"],
                    "three_layer_pass_count": sum(1 for r in slice_eval["records"] if r.get("three_layer_passed")),
                }
            finally:
                db.DB_PATH = prev_db_path
                db._db_initialized = prev_init

    # Layer-3 calibration & statistical bound sanity verification
    annotator_1 = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0]
    annotator_2 = [1, 1, 0, 0, 1, 0, 1, 0, 0, 0]
    kappa = compute_cohens_kappa(annotator_1, annotator_2)
    sample_calibration = [
        {"direction": "en->vi", "nli_score": 0.82, "human_label": 1},
        {"direction": "en->vi", "nli_score": 0.61, "human_label": 1},
        {"direction": "en->vi", "nli_score": 0.22, "human_label": 0},
        {"direction": "en->vi", "nli_score": 0.38, "human_label": 0},
    ]
    calibrated_thr = calibrate_nli_thresholds(sample_calibration)
    cp_lower_60 = clopper_pearson_lower_bound(good_passed, len(cases), alpha=0.05)

    return {
        "gate": "M0_60_case_mechanism_regression",
        "pdf_count": len(hashes_first),
        "pdf_sha256": hashes_first,
        "pdf_build_ms": pdf_build_ms,
        "total_cases": len(cases),
        "good_passed": good_passed,
        "bad_traps_caught": bad_caught,
        "clopper_pearson_95_lower_bound": cp_lower_60,
        "layer3_sample_kappa": kappa,
        "layer3_calibrated_thresholds": calibrated_thr,
        "slice_10_evaluator_timings": slice_10_timings_ms,
        "slice_10_cpu_min_diagnostic": cpu_slice_summary,
        "unexpected_good_failures": unexpected_good_failures,
        "missed_bad_traps": missed_bad_traps,
        "passed": (good_passed == len(cases) and bad_caught == len(cases)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-dataset", action="store_true", help="Write m0_regression_gate_60.json")
    parser.add_argument("--skip-cpu-slice", action="store_true", help="Skip timing cpu_min on the 10-case slice")
    parser.add_argument("--output-dir", type=Path, help="Optional directory to write gate report JSON")
    args = parser.parse_args()

    report = verify_and_time_m0_gate(
        run_cpu_baseline_slice=not args.skip_cpu_slice,
        write_dataset_file=args.write_dataset,
    )
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "m0_gate_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "passed": report["passed"],
        "pdf_count": report["pdf_count"],
        "good_passed": f"{report['good_passed']}/{report['total_cases']}",
        "bad_traps_caught": f"{report['bad_traps_caught']}/{report['total_cases']}",
        "clopper_pearson_95_lower": report["clopper_pearson_95_lower_bound"],
        "slice_10_cpu_min_diagnostic": report["slice_10_cpu_min_diagnostic"],
    }, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
