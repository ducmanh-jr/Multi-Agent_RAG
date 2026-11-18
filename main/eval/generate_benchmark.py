# -*- coding: utf-8 -*-
"""
Generates the 120-Question Bilingual Evaluation Benchmark (`main/eval/benchmark_120.jsonl`).
- 4 Categories (30 questions each = 120 total):
  1. single_hop (30): Factoid, Numeric & Formula Extraction
  2. multi_hop (30): Multi-Section / Multi-Page Reasoning
  3. table_comparison (30): Structured Benchmark & Architecture Comparison
  4. adversarial_trap (30): Unanswerable / Fake Entity / Out-of-Scope Trap Questions
- Language Distribution: 70% Vietnamese (84 questions) / 30% English (36 questions)
- Grounded in real documents in `main/uploads/` and `main/data/corpus.jsonl`.
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.tools import load_or_parse_selected_pdf


def _extract_doc_facts() -> List[Dict[str, Any]]:
    uploads_dir = ROOT_DIR / "uploads"
    facts: List[Dict[str, Any]] = []

    if uploads_dir.exists():
        for pdf_file in sorted(uploads_dir.glob("*.pdf")):
            chunks = load_or_parse_selected_pdf(pdf_file.name, str(ROOT_DIR))
            for ch in chunks:
                txt = (ch.get("text") or "").strip()
                if len(txt) < 120:
                    continue
                nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", txt)
                words = [w for w in re.findall(r"[A-Za-zÀ-ỹ]{5,}", txt) if len(w) >= 5][:6]
                facts.append({
                    "selected_doc": pdf_file.name,
                    "title": ch.get("title", pdf_file.name),
                    "page": ch.get("page", 1),
                    "text": txt,
                    "numbers": nums[:4],
                    "keywords": words[:5],
                })
    return facts


def build_benchmark_120() -> List[Dict[str, Any]]:
    facts = _extract_doc_facts()
    if not facts:
        raise RuntimeError("No PDF facts found in main/uploads/")

    benchmark: List[Dict[str, Any]] = []

    # Helper to pick language: 70% VI (21 per 30) / 30% EN (9 per 30) -> 84 VI + 36 EN = 120
    def _lang_for_idx(i: int) -> str:
        return "vi" if (i % 10) < 7 else "en"

    # 1. Category 1: single_hop (30 questions)
    num_facts = [f for f in facts if f["numbers"]] or facts
    for i in range(30):
        f = num_facts[(i * 3) % len(num_facts)]
        lang = _lang_for_idx(i)
        kw_str = ", ".join(f["keywords"][:3]) if f["keywords"] else f["selected_doc"]
        if lang == "vi":
            q = (
                f"Trong tài liệu {f['selected_doc']} (trang {f['page']}), các thông số hoặc chi tiết cụ thể "
                f"liên quan đến '{kw_str}' được trình bày như thế nào? Hãy trích dẫn chính xác [n]."
            )
        else:
            q = (
                f"According to {f['selected_doc']} (page {f['page']}), what specific metrics, values, or findings "
                f"are reported regarding '{kw_str}'? Cite exact evidence [n]."
            )
        benchmark.append({
            "id": f"SH-{i+1:03d}",
            "category": "single_hop",
            "language": lang,
            "selected_doc": f["selected_doc"],
            "question": q,
            "gold_numbers": f["numbers"][:2],
            "gold_keywords": f["keywords"][:4],
            "gold_page": f["page"],
            "is_unanswerable": False,
        })

    # 2. Category 2: multi_hop (30 questions)
    for i in range(30):
        f1 = facts[(i * 5) % len(facts)]
        same_doc = [x for x in facts if x["selected_doc"] == f1["selected_doc"] and x["page"] != f1["page"]]
        f2 = same_doc[i % len(same_doc)] if same_doc else f1
        lang = _lang_for_idx(i)
        kw1 = ", ".join(f1["keywords"][:2])
        kw2 = ", ".join(f2["keywords"][:2])
        if lang == "vi":
            q = (
                f"Kết hợp nội dung tại trang {f1['page']} ({kw1}) và trang {f2['page']} ({kw2}) của tài liệu "
                f"{f1['selected_doc']}, phương pháp được đề xuất giải quyết bài toán và đánh giá kết quả ra sao?"
            )
        else:
            q = (
                f"Synthesizing the findings from page {f1['page']} ({kw1}) and page {f2['page']} ({kw2}) in "
                f"{f1['selected_doc']}, how does the proposed methodology connect to its experimental outcomes?"
            )
        benchmark.append({
            "id": f"MH-{i+1:03d}",
            "category": "multi_hop",
            "language": lang,
            "selected_doc": f1["selected_doc"],
            "question": q,
            "gold_numbers": (f1["numbers"][:1] + f2["numbers"][:1]),
            "gold_keywords": list(dict.fromkeys(f1["keywords"][:2] + f2["keywords"][:2])),
            "gold_page": f1["page"],
            "is_unanswerable": False,
        })

    # 3. Category 3: table_comparison (30 questions)
    table_facts = [
        f for f in facts
        if any(tok in f["text"].lower() for tok in ("table", "bảng", "compare", "accuracy", "f1", "score", "model", "%"))
    ] or facts
    for i in range(30):
        f = table_facts[(i * 2) % len(table_facts)]
        lang = _lang_for_idx(i)
        kw = ", ".join(f["keywords"][:3])
        if lang == "vi":
            q = (
                f"Hãy so sánh và lập bảng tổng hợp các thông số/kết quả thực nghiệm liên quan đến '{kw}' "
                f"tại trang {f['page']} trong tài liệu {f['selected_doc']}."
            )
        else:
            q = (
                f"Compare the benchmark results, architectural components, or quantitative metrics related to '{kw}' "
                f"on page {f['page']} of {f['selected_doc']}."
            )
        benchmark.append({
            "id": f"TC-{i+1:03d}",
            "category": "table_comparison",
            "language": lang,
            "selected_doc": f["selected_doc"],
            "question": q,
            "gold_numbers": f["numbers"][:2],
            "gold_keywords": f["keywords"][:4],
            "gold_page": f["page"],
            "is_unanswerable": False,
        })

    # 4. Category 4: adversarial_trap (30 questions - Unanswerable / Fake Metric Traps)
    fake_traps = [
        ("Quantum-GPT-99B", "99.98%"),
        ("Hyperion-V12", "2029"),
        ("MegaLoRA-512B", "0.0001 ms"),
        ("BioSynth-X7", "42,500 GPU hours"),
        ("Chronos-Net-88", "148.7 BLEU"),
    ]
    for i in range(30):
        f = facts[(i * 4) % len(facts)]
        trap_entity, trap_num = fake_traps[i % len(fake_traps)]
        lang = _lang_for_idx(i)
        if lang == "vi":
            q = (
                f"Tài liệu {f['selected_doc']} có chứng minh rằng mô hình giả định '{trap_entity}' đạt độ chính xác "
                f"'{trap_num}' không, hay thông tin này không tồn tại trong tài liệu? Chỉ trả lời dựa trên bằng chứng thực tế."
            )
        else:
            q = (
                f"Does {f['selected_doc']} report that the hypothetical model '{trap_entity}' achieved '{trap_num}', "
                f"or is this absent from the document? Ground your response strictly in the actual text."
            )
        benchmark.append({
            "id": f"AT-{i+1:03d}",
            "category": "adversarial_trap",
            "language": lang,
            "selected_doc": f["selected_doc"],
            "question": q,
            "gold_numbers": [],
            "gold_keywords": f["keywords"][:3],
            "trap_entities": [trap_entity, trap_num],
            "gold_page": f["page"],
            "is_unanswerable": True,
        })

    return benchmark


def main():
    out_path = Path(__file__).resolve().parent / "benchmark_120.jsonl"
    items = build_benchmark_120()
    with open(out_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    vi_count = sum(1 for x in items if x["language"] == "vi")
    en_count = sum(1 for x in items if x["language"] == "en")
    print(f"[OK] Generated {len(items)} benchmark questions -> {out_path}")
    print(f"     Language split: VI={vi_count} ({vi_count/len(items)*100:.1f}%), EN={en_count} ({en_count/len(items)*100:.1f}%)")


if __name__ == "__main__":
    main()
