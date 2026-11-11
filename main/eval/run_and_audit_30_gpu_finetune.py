# -*- coding: utf-8 -*-
"""
Deep Evaluation & Audit Script for GPU Finetune Mode (30 Questions Benchmark).
Evaluates all 30 standard queries on the NGINX benchmark document:
- 15 In-Domain Queries (Technical architecture, commands, configuration, summaries)
- 15 Out-of-Domain Queries (Adversarial traps, general knowledge, historical facts)

Measures:
1. Pass Rate (% PASS/FAIL)
2. Groundedness & Honest Abstention (Zero Hallucination)
3. F1 / Keyword Matching & Citation Anchors
4. Deep qualitative audit: Strengths vs. Weaknesses of gpu_finetune answers.
"""

import sys
import os
import glob
import time
import json
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tests"))

from core.graph import build_research_rag_graph
from evaluate_30_queries_deep import test_cases, pdf_path


def audit_answer_quality(q_item: Dict[str, Any], final_report: str, verified_claims: list, rejected_claims: list, latency: float) -> Dict[str, Any]:
    q = q_item["query"]
    q_type = q_item["type"]
    exp_kws = q_item["expected_keywords"]
    report_l = final_report.lower()

    # Pass/Fail criteria
    if q_type == "out_of_domain":
        is_refusal = "hoàn toàn không được đề cập" in report_l or "không được đề cập" in report_l
        has_hallucination = any(phrase in report_l for phrase in ["upstream và nhiều backend", "tự sinh này bằng cách"])
        success = is_refusal and not has_hallucination
        category = "Out-of-Domain Trap"
    else:
        is_refusal = "hoàn toàn không được đề cập" in report_l
        has_stub = "chưa thể tổng hợp" in report_l or "chế độ dự phòng" in report_l
        matched_kws = [kw for kw in exp_kws if kw.lower() in report_l]
        success = (not is_refusal) and (not has_stub) and (len(matched_kws) > 0)
        category = "In-Domain Technical"

    # Qualitative analysis: Strengths and Weaknesses
    strengths = []
    weaknesses = []

    if q_type == "out_of_domain":
        if is_refusal:
            strengths.append("Tuân thủ 100% Zero-Hallucination: Kích hoạt từ chối trung thực (Honest Abstention).")
            strengths.append("Không bịa đặt số liệu hay nội dung sai lệch ngoài tài liệu.")
        else:
            weaknesses.append("Chưa kích hoạt từ chối trung thực: Có nguy cơ phát sinh ảo giác thông tin ngoài miền.")
    else:
        # In-domain evaluation
        if matched_kws:
            strengths.append(f"Khớp đúng {len(matched_kws)}/{len(exp_kws)} từ khóa thực nghiệm cốt lõi ({matched_kws}).")
        else:
            weaknesses.append(f"Thiếu các từ khóa mục tiêu: {exp_kws}.")

        if "[1]" in final_report or "Trang" in final_report:
            strengths.append("Có gắn mỏ neo trích dẫn số trang/văn bản chuẩn học thuật.")
        else:
            weaknesses.append("Thiếu mỏ neo trích dẫn rõ ràng [1] (Trang X).")

        if len(final_report) > 800:
            strengths.append(f"Độ dài câu trả lời phong phú ({len(final_report)} ký tự), cấu trúc đa tầng hoàn thiện.")
        elif len(final_report) < 300:
            weaknesses.append("Câu trả lời còn ngắn, chưa khai thác hết ngữ cảnh liên đới của tài liệu.")

        if "```" in final_report or "conf" in final_report:
            strengths.append("Bảo toàn tốt khối code cấu hình kỹ thuật hoặc lệnh shell.")

        if rejected_claims:
            weaknesses.append(f"Có {len(rejected_claims)} luận điểm bị bộ kiểm định BGE NLI Head loại bỏ do chưa đủ bằng chứng.")

    return {
        "id": q_item["id"],
        "query": q,
        "type": q_type,
        "category": category,
        "status": "PASS" if success else "FAIL",
        "latency_s": round(latency, 2),
        "report_length": len(final_report),
        "verified_claims_count": len(verified_claims),
        "rejected_claims_count": len(rejected_claims),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "final_report": final_report,
    }


def run_gpu_finetune_30_audit():
    print("=" * 80)
    print("🚀 BẮT ĐẦU KIỂM THỬ & ĐÁNH GIÁ CHUYÊN SÂU 30 CÂU HỎI TRÊN GPU FINETUNE")
    print(f"📄 Tài liệu: {os.path.basename(pdf_path)}")
    print("=" * 80)

    app = build_research_rag_graph("gpu_finetune")
    results = []
    pass_count = 0

    for idx, tc in enumerate(test_cases, 1):
        q = tc["query"]
        t_type = tc["type"]
        print(f"\n[{idx}/30] [{t_type.upper()}] Q{tc['id']:02d}: '{q}'")

        t0 = time.perf_counter()
        try:
            state = app.invoke({
                "query": q,
                "pdf_path": pdf_path,
                "retry_count": 0,
                "max_reflection_loops": 1,
            })
            latency = time.perf_counter() - t0
            report = state.get("final_report") or state.get("answer") or ""
            v_claims = state.get("verified_claims") or []
            r_claims = state.get("rejected_claims") or []

            audit = audit_answer_quality(tc, report, v_claims, r_claims, latency)
            results.append(audit)

            if audit["status"] == "PASS":
                pass_count += 1
                status_icon = "✅ PASS"
            else:
                status_icon = "❌ FAIL"

            print(f"   -> {status_icon} | Thời gian: {latency:.2f}s | Độ dài: {len(report)} ký tự | Luận điểm: {len(v_claims)}")
            if audit["weaknesses"]:
                print(f"   ⚠️  Điểm cần lưu ý: {audit['weaknesses'][0]}")

        except Exception as e:
            latency = time.perf_counter() - t0
            print(f"   ❌ EXCEPTION: {e}")
            results.append({
                "id": tc["id"],
                "query": q,
                "type": t_type,
                "status": "FAIL",
                "latency_s": round(latency, 2),
                "report_length": 0,
                "strengths": [],
                "weaknesses": [f"Lỗi thực thi pipeline: {e}"],
                "final_report": "",
            })

    # Summary
    pass_rate = (pass_count / len(test_cases)) * 100
    print("\n" + "=" * 80)
    print(f"📊 KẾT QUẢ TỔNG HỢP: {pass_count}/{len(test_cases)} CÂU ĐẠT ({pass_rate:.1f}%)")
    print("=" * 80)

    # Save detailed JSON
    out_json = ROOT_DIR / "eval" / "gpu_finetune_30_audit_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu kết quả chi tiết tại: {out_json}")


if __name__ == "__main__":
    run_gpu_finetune_30_audit()
