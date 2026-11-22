#!/usr/bin/env python3
"""
Real-World Academic & Technical Evaluation Runner (Step 1 & Step 2).
Executes the 16 real-world test cases from evaluation/dataset/real_world_academic_qa_v1.json
through the complete LangGraph Multi-Agent RAG pipeline against the live GPU worker.

Strict Compliance Enforced:
  1. No Mock Policy (AGENT.md Mục 15): Real queries, real PDF, real remote GPU inference.
  2. Worker Status Verification: Asserts active_worker and transition_state == 'ready'.
  3. PDF Integrity Check: Verifies SHA256 of target PDF before executing document cases.
  4. Node-level Evidence Collection: Captures Intent, Retrieval, Reader, Generator, Critic,
     Fact-Checker, Writer, Citations, and Latency.
  5. Accurate Categorization: GQA-04 classified as TASK_EXECUTION (no RAG forcing).
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
import httpx
import re
import traceback

# Ensure project roots are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MAIN_DIR = PROJECT_ROOT / "main"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from core.config import settings
from core.graph import build_research_rag_graph

UPLOADS_DIR = MAIN_DIR / "uploads"
DATASET_PATH = PROJECT_ROOT / "evaluation" / "dataset" / "real_world_academic_qa_v1.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def get_git_commit_sha() -> str:
    """Retrieves current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5.0
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest().upper()


def resolve_pdf_path(filename: str) -> Optional[str]:
    """Finds absolute path of PDF in main/uploads/."""
    if not filename:
        return None
    direct = UPLOADS_DIR / filename
    if direct.exists():
        return str(direct.resolve())
    for f in UPLOADS_DIR.iterdir():
        if f.name.endswith(filename) or filename in f.name:
            if f.suffix.lower() == ".pdf":
                return str(f.resolve())
    return None


def verify_worker_status(gateway_url: str, admin_token: str, expected_worker: str) -> Dict[str, Any]:
    """
    Verifies that the Kaggle Supervisor Gateway has the expected worker active and ready.
    Fails closed if the worker is offline, transitioning, or in an incorrect mode.
    """
    if not gateway_url:
        raise RuntimeError("REMOTE_INFERENCE_URL chưa được cấu hình. Không thể kết nối GPU Worker.")

    url = f"{gateway_url.rstrip('/')}/api/admin/status"
    headers = {"Authorization": f"Bearer {admin_token}"} if admin_token else {}
    
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(url, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"Gateway trả về HTTP {resp.status_code}: {resp.text}")
        status = resp.json()

    active_w = status.get("active_worker")
    state = status.get("transition_state")
    if active_w != expected_worker:
        raise RuntimeError(
            f"Worker hiện tại là '{active_w}', không khớp với worker kỳ vọng '{expected_worker}'."
        )
    if state != "ready":
        raise RuntimeError(
            f"Trạng thái worker là '{state}', chưa sẵn sàng (yêu cầu 'ready')."
        )
    return status


def evaluate_case_response(case: Dict[str, Any], result: Dict[str, Any], latency: float) -> Dict[str, Any]:
    """
    Evaluates response quality, factuality, citations, and abstention against golden reference.
    """
    category = case.get("category", "")
    golden = case.get("golden_reference", {})
    expected_intent = case.get("expected_intent", "")

    # Extract node outputs from LangGraph state
    final_output = str(result.get("final_report") or result.get("template_response") or result.get("answer") or "")
    detected_intent = str(result.get("primary_intent") or "")
    processing_status = str(result.get("processing_status") or "")
    evidence_status = str(result.get("evidence_status") or "")
    input_status = str(result.get("input_status") or "")
    fact_check_label = str(result.get("fact_check_label") or "")
    critic_action = str(result.get("critic_action") or "")
    critic_feedback = str(result.get("critic_feedback") or "")
    verified_claims = result.get("verified_claims") or []

    # Read genuine provenance and citations from AgentState contracts
    evidence_spans = result.get("evidence_spans") or []
    source_units = result.get("source_units") or []
    written_report_data = result.get("written_report_data") or {}
    report_refs = written_report_data.get("references") or []

    citations_count = len(report_refs) if report_refs else len(evidence_spans)
    if citations_count == 0:
        cited_markers = re.findall(r"\[(\d+)\]", final_output)
        page_markers = re.findall(r"(?:Page|Trang)\s*(\d+)", final_output, re.I)
        citations_count = len(set(cited_markers + page_markers))

    retrieved_chunks = source_units or result.get("retrieved_docs") or []

    # 1. Intent check
    intent_matched = (detected_intent == expected_intent)
    # Special tolerance: TASK_EXECUTION vs TASK_ASSISTED_RESPONSE, and GENERAL_QA vs TASK_ASSISTED_RESPONSE
    if expected_intent in ("TASK_EXECUTION", "TASK_ASSISTED_RESPONSE") and detected_intent in ("TASK_EXECUTION", "TASK_ASSISTED_RESPONSE"):
        intent_matched = True
    elif expected_intent in ("GENERAL_QA", "TASK_ASSISTED_RESPONSE") and detected_intent in ("GENERAL_QA", "TASK_ASSISTED_RESPONSE"):
        intent_matched = True

    eval_record: Dict[str, Any] = {
        "id": case["id"],
        "category": category,
        "query": case["query"],
        "expected_intent": expected_intent,
        "detected_intent": detected_intent,
        "intent_matched": intent_matched,
        "latency_sec": round(latency, 2),
        "processing_status": processing_status,
        "evidence_status": evidence_status,
        "fact_check_label": fact_check_label,
        "critic_action": critic_action,
        "critic_feedback": critic_feedback[:150] if critic_feedback else "",
        "verified_claims_count": len(verified_claims),
        "citations_count": citations_count,
        "retrieved_chunks_count": len(retrieved_chunks),
        "output_length": len(final_output),
        "output_sample": final_output[:300] + ("..." if len(final_output) > 300 else ""),
        "status": "PASS",
        "notes": [],
    }

    if not intent_matched:
        eval_record["notes"].append(f"Intent mismatch: detected '{detected_intent}', expected '{expected_intent}'")

    # 2. Adversarial Trap / Abstention Check
    if golden.get("is_adversarial_trap"):
        abstention_keywords = [
            "không đề cập", "không có", "không tìm thấy", "chưa xuất hiện",
            "không chứa", "không nhắc", "sau năm 2017", "ra đời sau",
            "not mentioned", "not found"
        ]
        has_abstention = any(kw in final_output.lower() for kw in abstention_keywords)
        truthful_refusal = (
            has_abstention
            or evidence_status in ("NOT_FOUND", "PARTIAL")
            or fact_check_label in ("refute", "not_enough_info")
        )
        eval_record["abstention_verified"] = truthful_refusal
        if truthful_refusal:
            eval_record["notes"].append("Passed adversarial trap: honest refusal/abstention observed.")
        else:
            eval_record["status"] = "FAIL"
            eval_record["notes"].append("FAILED adversarial trap: system hallucinated nonexistent entities.")

    # 3. Must-contain / Key concepts check
    must_contain = golden.get("must_contain") or []
    matched_must = [kw for kw in must_contain if kw.lower() in final_output.lower()]
    must_score = len(matched_must) / max(1, len(must_contain)) if must_contain else 1.0
    eval_record["must_contain_score"] = round(must_score, 2)
    eval_record["matched_keywords"] = matched_must

    # 4. Citations check for Document QA
    if category == "document_qa" and not golden.get("is_adversarial_trap"):
        has_citations = citations_count > 0 or ("Trang" in final_output or "page" in final_output.lower() or "[" in final_output)
        eval_record["has_citations"] = has_citations
        if not has_citations:
            eval_record["status"] = "FAIL"
            eval_record["notes"].append("FAILED: Missing required citations in document QA response.")

    # 5. Edge cases check
    if category == "edge_cases":
        if case["id"] == "EDGE-01":
            is_no_doc = (input_status == "NO_DOCUMENT_SELECTED") or any(
                kw in final_output.lower() for kw in ["tài liệu", "chưa", "chọn", "tải lên"]
            )
            eval_record["edge_valid"] = is_no_doc
            if not is_no_doc:
                eval_record["status"] = "FAIL"
                eval_record["notes"].append("EDGE-01 failed: NO_DOCUMENT_SELECTED template not triggered.")
        elif case["id"] == "EDGE-02":
            is_clarify = (detected_intent == "CLARIFICATION_REQUIRED") or any(
                kw in final_output.lower() for kw in ["làm rõ", "cụ thể", "câu hỏi"]
            )
            eval_record["edge_valid"] = is_clarify
            if not is_clarify:
                eval_record["status"] = "FAIL"
                eval_record["notes"].append("EDGE-02 failed: Clarification required not triggered.")
        elif case["id"] == "EDGE-03":
            is_unsupported = (detected_intent == "UNSUPPORTED_CAPABILITY") or any(
                kw in final_output.lower() for kw in ["chưa có công cụ", "không thể", "hướng dẫn", "từ chối"]
            )
            eval_record["edge_valid"] = is_unsupported
            if not is_unsupported:
                eval_record["status"] = "FAIL"
                eval_record["notes"].append("EDGE-03 failed: Security refusal not triggered.")
        elif case["id"] == "EDGE-04":
            is_p12_valid = ("12" in final_output) and ("trang" in final_output.lower() or "page" in final_output.lower())
            eval_record["edge_valid"] = is_p12_valid
            if not is_p12_valid:
                eval_record["status"] = "FAIL"
                eval_record["notes"].append("EDGE-04 failed: Page 12 content or attribution missing.")
        elif case["id"] == "EDGE-05":
            is_oob_valid = (result.get("answer_status") == "NOT_FOUND_IN_DOCUMENT" or "OUT_OF_BOUNDS" in final_output) and any(
                kw in final_output.lower() for kw in ["15 trang", "không tồn tại", "ngoài phạm vi"]
            )
            eval_record["edge_valid"] = is_oob_valid
            if not is_oob_valid:
                eval_record["status"] = "FAIL"
                eval_record["notes"].append("EDGE-05 failed: Out-of-bounds page detection not triggered.")

    # Determine final case status
    if eval_record["status"] == "PASS":
        if processing_status != "SUCCESS" and category != "edge_cases":
            eval_record["status"] = "FAIL"
            eval_record["notes"].append(f"Non-success processing_status: '{processing_status}' (error: {result.get('error_message')})")

        error_indicators = [
            "không thể xử lý yêu cầu trực tiếp",
            "lỗi mô hình",
            "empty response from direct assistant",
            "circuit breaker active",
            "không thể kết nối",
        ]
        if any(err in final_output.lower() for err in error_indicators) and category != "edge_cases":
            eval_record["status"] = "FAIL"
            eval_record["notes"].append(f"Model error message returned instead of substantive answer: '{final_output[:60]}...'")
        elif len(final_output) < 20:
            eval_record["status"] = "FAIL"
            eval_record["notes"].append("Output too short (<20 chars).")
        elif not intent_matched and category != "task_execution":
            eval_record["status"] = "FAIL"

    return eval_record


def run_real_qa_evaluation(
    mode: str = "gpu_finetune",
    dataset_path: str = str(DATASET_PATH),
    output_path: Optional[str] = None,
    gateway_url: Optional[str] = None,
    admin_token: Optional[str] = None,
    case_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs real-world academic evaluation suite."""
    settings.EXECUTION_MODE = mode
    os.environ["EXECUTION_MODE"] = mode

    gateway_url = (gateway_url or getattr(settings, "REMOTE_INFERENCE_URL", "") or os.environ.get("REMOTE_INFERENCE_URL", "")).rstrip("/")
    admin_token = admin_token or getattr(settings, "SUPERVISOR_ADMIN_TOKEN", "") or os.environ.get("SUPERVISOR_ADMIN_TOKEN", "")
    
    if gateway_url:
        settings.REMOTE_INFERENCE_URL = gateway_url
        os.environ["REMOTE_INFERENCE_URL"] = gateway_url
    if admin_token:
        settings.SUPERVISOR_ADMIN_TOKEN = admin_token
        settings.INFERENCE_API_KEY = admin_token
        os.environ["SUPERVISOR_ADMIN_TOKEN"] = admin_token
        os.environ["INFERENCE_API_KEY"] = admin_token
    
    if not output_path:
        output_path = str(RESULTS_DIR / f"real_qa_{mode}.json")

    print("\n" + "=" * 90)
    print(f" 🚀 MULTI-AGENT RAG: REAL-WORLD ACADEMIC EVALUATION (Mode: {mode.upper()})")
    print(f" 🌐 Target Gateway: {gateway_url or 'N/A'}")
    print(f" 📂 Dataset: {dataset_path}")
    print("=" * 90)

    # 1. Step 4 Verification: Assert worker is active and ready
    print("\n[BƯỚC 1/3] XÁC MINH TRẠNG THÁI CLUSTER THEO ĐIỀU KIỆN NGHIỆM THU:")
    worker_status = {}
    try:
        worker_status = verify_worker_status(gateway_url, admin_token, expected_worker=mode)
        vram = worker_status.get("vram_allocation") or {}
        g0 = vram.get("gpu_0", {}).get("used_gb", 0.0)
        g1 = vram.get("gpu_1", {}).get("used_gb", 0.0)
        caps = worker_status.get("active_worker_capabilities") or {}
        print(f"  ✔ Worker đã sẵn sàng: active_worker='{worker_status.get('active_worker')}', state='{worker_status.get('transition_state')}'")
        print(f"  ✔ VRAM Probe: GPU 0 = {g0:.1f}GB, GPU 1 = {g1:.1f}GB")
        print(f"  ✔ Capabilities: {caps}")
    except Exception as exc:
        print(f"  ❌ XÁC MINH WORKER THẤT BẠI: {exc}")
        print("  🚨 Tuân thủ No Mock Policy: Dừng đánh giá ngay lập tức, không giả lập kết quả.")
        sys.exit(1)

    # 2. Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        cases: List[Dict[str, Any]] = json.load(f)

    if case_filter:
        cases = [c for c in cases if c["id"].lower() == case_filter.lower()]
        print(f"  ℹ Lọc theo Case ID '{case_filter}': tìm thấy {len(cases)} ca kiểm thử.")

    # 3. Build LangGraph for target mode
    print(f"\n[BƯỚC 2/3] KHỞI TẠO ĐỒ THỊ LANGGRAPH (Mode: {mode})...")
    settings.EXECUTION_MODE = mode
    graph = build_research_rag_graph(mode=mode)
    print("  ✔ Đồ thị LangGraph 5-Agent đã khởi tạo thành công.")

    # 4. Execute test cases
    print(f"\n[BƯỚC 3/3] THỰC THI {len(cases)} CA KIỂM THỬ THỰC TẾ:")
    executed_results: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for idx, case in enumerate(cases, start=1):
        test_id = case["id"]
        cat = case["category"]
        query = case["query"]
        target_pdf = case.get("target_pdf")
        expected_sha = case.get("target_pdf_sha256")

        print(f"\n  • [{idx}/{len(cases)}] {test_id} ({cat}): {query[:65]}...")

        # Integrity Check on Target PDF
        pdf_path = None
        if target_pdf and not case.get("simulate_no_pdf"):
            pdf_path = resolve_pdf_path(target_pdf)
            if not pdf_path or not os.path.exists(pdf_path):
                print(f"    ❌ LỖI TÀI LIỆU: Không tìm thấy PDF '{target_pdf}' tại main/uploads/.")
                executed_results.append({
                    "id": test_id,
                    "category": cat,
                    "query": query,
                    "status": "BLOCKED",
                    "notes": [f"Missing target PDF {target_pdf}"]
                })
                continue
            
            # Verify SHA256
            if expected_sha:
                real_sha = compute_file_sha256(pdf_path)
                if real_sha != expected_sha.upper():
                    print(f"    ❌ LỖI TOÀN VẸN: SHA256 không khớp! Thực tế={real_sha}, kỳ vọng={expected_sha}")
                    executed_results.append({
                        "id": test_id,
                        "category": cat,
                        "query": query,
                        "status": "BLOCKED",
                        "notes": [f"PDF SHA256 mismatch: {real_sha} != {expected_sha}"]
                    })
                    continue
                print(f"    ✔ Xác minh SHA256 PDF khớp 100% ({real_sha[:16]}...)")

        # Initial state for LangGraph
        initial_state: Dict[str, Any] = {
            "query": query,
            "user_id": "eval_user",
            "session_id": f"real_eval_{mode}_{test_id}",
            "enable_scout": True,
        }
        if pdf_path:
            initial_state["pdf_path"] = pdf_path
            initial_state["pdf_paths"] = [pdf_path]
            initial_state["selected_docs"] = [target_pdf]

        # Invoke Graph with timing
        t0 = time.perf_counter()
        try:
            res = graph.invoke(initial_state)
            lat = time.perf_counter() - t0
            latencies.append(lat)

            eval_res = evaluate_case_response(case, res, lat)
            executed_results.append(eval_res)

            status_icon = "✔ PASS" if eval_res["status"] == "PASS" else "❌ FAIL"
            print(f"    [{status_icon}] ({lat:.2f}s) -> Intent: {eval_res.get('detected_intent')} | "
                  f"Evidence: {eval_res.get('evidence_status')} | Citations: {eval_res.get('citations_count')}")
            if eval_res.get("notes"):
                print(f"    ℹ Ghi chú: {'; '.join(eval_res['notes'])}")

        except Exception as exc:
            lat = time.perf_counter() - t0
            latencies.append(lat)
            print(f"    ❌ EXCEPTION ({lat:.2f}s): {exc}")
            traceback.print_exc()
            executed_results.append({
                "id": test_id,
                "category": cat,
                "query": query,
                "status": "FAIL",
                "latency_sec": round(lat, 2),
                "error": str(exc),
                "notes": [f"Runtime exception in graph execution: {exc}"]
            })

        # Incremental checkpoint save after each test case
        try:
            interim_summary = {
                "metadata": {
                    "evaluation_name": "Real-World Academic Evaluation (Step 2)",
                    "execution_mode": mode,
                    "target_gateway": gateway_url,
                    "git_commit": get_git_commit_sha(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "target_pdf": "Vaswani_2017_Attention.pdf",
                    "target_pdf_sha256": "BDFAA68D8984F0DC02BEACA527B76F207D99B666D31D1DA728EE0728182DF697",
                },
                "worker_verification": worker_status,
                "metrics": {
                    "total_cases_run": len(executed_results),
                    "passed_cases": sum(1 for r in executed_results if r.get("status") == "PASS"),
                    "failed_cases": sum(1 for r in executed_results if r.get("status") == "FAIL"),
                    "blocked_cases": sum(1 for r in executed_results if r.get("status") == "BLOCKED"),
                },
                "case_results": executed_results,
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(interim_summary, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # 5. Print Summary Table
    total = len(executed_results)
    passed = sum(1 for r in executed_results if r.get("status") == "PASS")
    blocked = sum(1 for r in executed_results if r.get("status") == "BLOCKED")
    failed = total - passed - blocked
    avg_lat = round(sum(latencies) / max(1, len(latencies)), 2)
    pass_pct = round((passed / max(1, total)) * 100, 1)

    print("\n" + "─" * 105)
    print(f" 📊 BẢNG TỔNG HỢP NGHIỆM THU CÂU HỎI THẬT TRÊN {mode.upper()} (DUAL TESLA T4)")
    print("─" * 105)
    print(f"| {'ID':<8} | {'Phân loại':<18} | {'Intent':<18} | {'Status':<8} | {'Độ trễ':<8} | {'Chi tiết / Ghi chú':<34} |")
    print(f"|{'-'*10}|{'-'*20}|{'-'*20}|{'-'*10}|{'-'*10}|{'-'*36}|")
    for r in executed_results:
        s = r.get("status", "NOT_RUN")
        status_str = "✔ PASS" if s == "PASS" else ("⛔ BLOCKED" if s == "BLOCKED" else "❌ FAIL")
        intent_str = str(r.get("detected_intent") or "N/A")[:18]
        lat_str = f"{r.get('latency_sec', 0.0):.1f}s"
        notes_str = "; ".join(r.get("notes") or [])[:34]
        if not notes_str:
            notes_str = str(r.get("output_sample", ""))[:34]
        print(f"| {r['id']:<8} | {r.get('category',''):<18} | {intent_str:<18} | {status_str:<8} | {lat_str:<8} | {notes_str:<34} |")
    print("─" * 105)
    print(f"🎯 KẾT LUẬN: {passed}/{total} ca ĐẠT ({pass_pct}%), {failed} THẤT BẠI, {blocked} BỊ CHẶN | Độ trễ trung bình: {avg_lat}s")
    print("─" * 105)

    # 6. Save Reproducible JSON Artifact
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    summary_data = {
        "metadata": {
            "evaluation_name": "Real-World Academic Evaluation (Step 2)",
            "execution_mode": mode,
            "target_gateway": gateway_url,
            "git_commit": get_git_commit_sha(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target_pdf": "Vaswani_2017_Attention.pdf",
            "target_pdf_sha256": "BDFAA68D8984F0DC02BEACA527B76F207D99B666D31D1DA728EE0728182DF697",
        },
        "worker_verification": worker_status,
        "metrics": {
            "total_cases": total,
            "passed_cases": passed,
            "failed_cases": failed,
            "blocked_cases": blocked,
            "pass_percentage": pass_pct,
            "avg_latency_sec": avg_lat,
        },
        "case_results": executed_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\n📁 File minh chứng nghiệm thu JSON đã được lưu tại:\n   {os.path.abspath(output_path)}\n")

    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Real-World Academic & Technical Evaluation Runner")
    parser.add_argument("--mode", default="gpu_finetune", help="Execution mode (gpu_finetune | gpu_max)")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="Path to test dataset JSON")
    parser.add_argument("--output", default=None, help="Path to save evaluation results JSON")
    parser.add_argument("--gateway-url", default=None, help="Supervisor Gateway URL")
    parser.add_argument("--admin-token", default=None, help="Supervisor Admin Token")
    parser.add_argument("--case-id", default=None, help="Filter single case ID (e.g. DOC-01)")
    args = parser.parse_args()

    run_real_qa_evaluation(
        mode=args.mode,
        dataset_path=args.dataset,
        output_path=args.output,
        gateway_url=args.gateway_url,
        admin_token=args.admin_token,
        case_filter=args.case_id,
    )


if __name__ == "__main__":
    main()
