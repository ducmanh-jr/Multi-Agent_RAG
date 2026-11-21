#!/usr/bin/env python3
"""
Multi-Agent RAG - Gate 3 End-to-End Acceptance Test Runner
Tests the full LangGraph 5-Agent pipeline executing in 'gpu_finetune' mode
backed by the live Dual Tesla T4 remote worker cluster on Kaggle.

Stages Verified:
  C01: Router Intent & Sub-query Decomposition (/api/router)
  C02: Document Chunk Retrieval & Structural Reader (FTS5/BM25 + SourceUnits)
  C03: Grounded Synthesis with Citation Tags (/api/summarize/generator) + Critic (/api/summarize/critic)
  C04: Multi-Claim NLI Fact-Checking Audit on GPU 1 (/api/factcheck)
  C05: Publication-grade Academic Report Synthesis (/api/writer)
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

# Ensure project root and main/ are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAIN_DIR = os.path.join(PROJECT_ROOT, "main")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MAIN_DIR not in sys.path:
    sys.path.insert(0, MAIN_DIR)

from core.config import settings
from core.graph import build_research_rag_graph


class Gate3E2EVerifier:
    def __init__(self, base_url: str, admin_token: str = "", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.admin_token = admin_token
        self.timeout = timeout
        self.results: List[Dict[str, Any]] = []

        # Configure environment settings for gpu_finetune mode
        settings.REMOTE_INFERENCE_URL = self.base_url
        settings.REMOTE_TIMEOUT_SECONDS = int(timeout)
        settings.EXECUTION_MODE = "gpu_finetune"

    def record(self, test_id: str, stage: str, component: str, latency: float, details: str, passed: bool):
        self.results.append({
            "id": test_id,
            "stage": stage,
            "component": component,
            "latency": f"{latency:.2f}s" if latency > 0 else "N/A",
            "passed": passed,
            "details": details,
        })
        status_symbol = "✔ PASS" if passed else "❌ FAIL"
        print(f"  [{status_symbol}] {test_id} - {stage}: {component} ({latency:.2f}s) -> {details[:80]}")

    def run_e2e_pipeline(self, pdf_rel_path: str, query: str) -> Dict[str, Any]:
        pdf_full_path = os.path.join(PROJECT_ROOT, pdf_rel_path)
        if not os.path.exists(pdf_full_path):
            raise FileNotFoundError(f"PDF not found at {pdf_full_path}")

        print("\n" + "=" * 78)
        print(" 🚀 [GATE 3] KHỞI CHẠY KIỂM THỬ TOÀN TRÌNH LANGGRAPH 5-AGENT PIPELINE (E2E)")
        print(f" 📄 Target Document: {os.path.basename(pdf_full_path)}")
        print(f" ❓ Research Query: {query}")
        print("=" * 78)

        # Build compiled LangGraph workflow in gpu_finetune mode
        t_build_0 = time.time()
        graph = build_research_rag_graph(mode="gpu_finetune")
        t_build = time.time() - t_build_0
        print(f"  • Đã biên dịch LangGraph cho chế độ gpu_finetune trong {t_build:.2f}s\n")

        initial_state = {
            "query": query,
            "pdf_path": pdf_full_path,
            "max_reflection_loops": 2,
            "auto_scout": False,
        }

        print("  • Đang thực thi chuỗi LangGraph workflow...")
        t_start = time.time()
        final_state = graph.invoke(initial_state)
        total_time = time.time() - t_start
        print(f"\n  • Hoàn thành chuỗi workflow trong {total_time:.2f}s!\n")

        return final_state

    def audit_final_state(self, state: Dict[str, Any]):
        traces = state.get("reasoning_trace") or []
        
        # C01: Router stage
        sub_queries = state.get("sub_queries") or []
        primary_intent = state.get("primary_intent", "")
        intent = state.get("intent", "")
        alpha = state.get("alpha", 0.0)
        top_k = state.get("top_k", 0)
        c01_passed = bool(sub_queries and primary_intent == "DOCUMENT_QA")
        c01_detail = f"intent={intent}, sub_queries={len(sub_queries)}, alpha={alpha:.2f}, top_k={top_k}"
        self.record("C01", "Agent 1 Router", "Query Decomposition", 0.0, c01_detail, c01_passed)

        # C02: Retrieval & Reader stage
        retrieved_docs = state.get("retrieved_docs") or []
        source_units = state.get("source_units") or []
        c02_passed = bool(retrieved_docs and source_units)
        c02_detail = f"retrieved_chunks={len(retrieved_docs)}, source_units={len(source_units)}"
        self.record("C02", "Agent 1 & 2", "Retrieval & Structural Reader", 0.0, c02_detail, c02_passed)

        # C03: Summarizer Generator & Critic stage
        answer = state.get("answer", "")
        claims = state.get("claims") or []
        evidence_spans = state.get("evidence_spans") or []
        critic_action = state.get("critic_action", "")
        c03_passed = bool(answer and len(answer) > 50)
        c03_detail = f"answer_len={len(answer)}, claims={len(claims)}, critic_action='{critic_action}'"
        self.record("C03", "Agent 3 Summarizer", "Generator & Critic Loop", 0.0, c03_detail, c03_passed)

        # C04: Fact-Checker NLI stage (GPU 1)
        fact_check_label = state.get("fact_check_label", "")
        verified_claims = state.get("verified_claims") or []
        rejected_claims = state.get("rejected_claims") or []
        verifier_model = state.get("verifier_model_family", "")
        c04_passed = bool(fact_check_label in ("support", "contradict", "not_enough_info"))
        c04_detail = f"label='{fact_check_label}', verified={len(verified_claims)}, rejected={len(rejected_claims)}, verifier={verifier_model}"
        self.record("C04", "Agent 4 Fact-Checker", "BGE NLI Head (GPU 1)", 0.0, c04_detail, c04_passed)

        # C05: Writer stage
        final_report = state.get("final_report", "")
        confidence = state.get("confidence", "")
        c05_passed = bool(final_report and len(final_report) > 200)
        has_sections = ("#" in final_report)
        c05_detail = f"report_len={len(final_report)} chars, confidence='{confidence}', has_markdown={has_sections}"
        self.record("C05", "Agent 5 Writer", "Publication Academic Report", 0.0, c05_detail, c05_passed and has_sections)

    def print_summary_table(self) -> bool:
        print("\n" + "─" * 96)
        print(" 📊 BẢNG TỔNG HỢP NGHIỆM THU GATE 3: FULL END-TO-END PIPELINE (DUAL TESLA T4)")
        print("─" * 96)
        print(f"| {'ID':<4} | {'Giai đoạn / Agent':<22} | {'Hạng mục kiểm thử':<28} | {'Kết quả':<7} | {'Chi tiết / Preview':<25} |")
        print(f"|{'-'*6}|{'-'*24}|{'-'*30}|{'-'*9}|{'-'*27}|")
        for r in self.results:
            status = "✔ PASS" if r["passed"] else "❌ FAIL"
            print(f"| {r['id']:<4} | {r['stage']:<22} | {r['component']:<28} | {status:<7} | {r['details'][:25]:<25} |")
        print("─" * 96)

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        pct = (passed / total) * 100 if total > 0 else 0.0
        print(f"🎯 KẾT LUẬN GATE 3: {passed}/{total} giai đoạn đã hoàn thành thành công ({pct:.1f}%)!")
        if passed == total:
            print("🟢 TOÀN BỘ CHUỖI 5-AGENT MULTI-AGENT RAG TRÊN KAGGLE DUAL T4 ĐẠT CHUẨN NGHIỆM THU GATE 3!")
        else:
            print("🔴 CÓ GIAI ĐOẠN CHƯA HOÀN TẤT. Xem chi tiết ở bảng trên.")
        print("─" * 96 + "\n")
        return passed == total and total > 0


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent RAG Gate 3 E2E Verifier")
    parser.add_argument(
        "--url",
        default=os.environ.get("REMOTE_INFERENCE_URL", "http://127.0.0.1:8000"),
        help="Kaggle Gateway URL (defaults to REMOTE_INFERENCE_URL or http://127.0.0.1:8000)"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("SUPERVISOR_ADMIN_TOKEN", ""),
        help="Admin Token (defaults to SUPERVISOR_ADMIN_TOKEN)"
    )
    parser.add_argument("--pdf", default="main/uploads/Hu_2021_LoRA.pdf", help="Target PDF path relative to repo root")
    parser.add_argument("--query", default="Tóm tắt ý tưởng cốt lõi của phương pháp LoRA và phân tích ưu điểm so với full fine-tuning.", help="Query")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout in seconds")
    args = parser.parse_args()

    verifier = Gate3E2EVerifier(base_url=args.url, admin_token=args.token, timeout=args.timeout)
    try:
        final_state = verifier.run_e2e_pipeline(pdf_rel_path=args.pdf, query=args.query)
        verifier.audit_final_state(final_state)
        all_ok = verifier.print_summary_table()

        # Print preview of final outputs
        print("\n" + "=" * 78)
        print(" 📝 ĐẦU RA BÁO CÁO KHOA HỌC CUỐI CÙNG (FINAL ACADEMIC REPORT PREVIEW)")
        print("=" * 78)
        report = final_state.get("final_report", "")
        print(report[:1200] + ("\n... [TRUNCATED] ..." if len(report) > 1200 else ""))
        print("=" * 78 + "\n")

        if not all_ok:
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ LỖI TRONG QUÁ TRÌNH KIỂM THỬ GATE 3: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
