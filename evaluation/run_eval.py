"""
Evaluation Runner for Đợt 7: End-to-End Functional & Answer Quality Evaluation.
Evaluates all 6 official execution modes against the Golden Dataset.
Zero mock faking: offline backends are truthfully reported as OFFLINE / UNAVAILABLE.
"""

import os
import sys
import json
import time
import re
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MAIN_DIR = PROJECT_ROOT / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from core.graph import build_research_rag_graph
from core.model_gateway import OFFICIAL_EXECUTION_MODES
from utils.llm_client import UnifiedLLMClient

DATASET_DIR = PROJECT_ROOT / "evaluation" / "dataset"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"
UPLOADS_DIR = MAIN_DIR / "uploads"


def resolve_target_pdf(target_pdf_name: str) -> Optional[str]:
    """Finds the absolute path of a target PDF in main/uploads/."""
    if not target_pdf_name:
        return None
    
    # Direct match
    direct_path = UPLOADS_DIR / target_pdf_name
    if direct_path.exists():
        return str(direct_path.resolve())
    
    # Match by suffix or filename
    for file in UPLOADS_DIR.iterdir():
        if file.name.endswith(target_pdf_name) or target_pdf_name in file.name:
            if file.suffix.lower() == ".pdf":
                return str(file.resolve())
    
    return None


def probe_mode_connectivity(mode: str) -> Dict[str, Any]:
    """Probes whether the mode's required backend is live or offline."""
    from core.config import settings

    if mode in ("gpu_finetune", "gpu_max"):
        remote_url = getattr(settings, "REMOTE_INFERENCE_URL", "")
        if not remote_url:
            return {
                "status": "OFFLINE_UNAVAILABLE",
                "details": "REMOTE_INFERENCE_URL not configured. Truthfully reported offline (zero mocks)."
            }
        
        is_live = False
        err_msg = ""
        try:
            req = urllib.request.Request(f"{remote_url.rstrip('/')}/health", headers={"User-Agent": "eval"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    is_live = True
        except Exception as e:
            err_msg = str(e)
            is_live = False
            
        if is_live:
            return {"status": "ONLINE", "details": f"Remote GPU Worker reachable at {remote_url}"}
        else:
            return {
                "status": "OFFLINE_UNAVAILABLE",
                "details": f"Remote Kaggle GPU worker unreachable at {remote_url} ({err_msg or 'Connection refused / HTTP 502'}). Truthfully reported offline (zero mocks)."
            }
            
    elif mode == "cpu_max":
        # Check if Ollama port 11434 is responding
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", headers={"User-Agent": "eval"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    return {"status": "ONLINE", "details": "Local Ollama daemon running on port 11434"}
        except Exception:
            pass
        return {
            "status": "DEGRADED_FALLBACK",
            "details": "Local Ollama server not running on port 11434. Operates with local extractive fallback."
        }
        
    elif mode == "cpu_min":
        return {"status": "ONLINE", "details": "Local PyMuPDF + BM25 extractive engine (100% offline-ready)"}
        
    elif mode == "hybrid":
        return {"status": "ONLINE", "details": "Hybrid coordinator with cascading priority (Remote GPU -> Local Ollama -> Local extractive, zero Gemini)"}
        
    elif mode == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "")
        if api_key:
            return {"status": "ONLINE", "details": "Google Gemini 2.5 Flash API available"}
        else:
            return {"status": "OFFLINE_UNAVAILABLE", "details": "GEMINI_API_KEY not configured in environment"}
            
    return {"status": "UNKNOWN", "details": f"Unknown mode {mode}"}


def load_all_datasets() -> Dict[str, List[Dict[str, Any]]]:
    """Loads all 5 JSON test dataset files."""
    dataset_files = [
        "general_qa.json",
        "document_qa.json",
        "web_research.json",
        "mixed_qa.json",
        "edge_cases.json",
    ]
    all_cases = {}
    for fname in dataset_files:
        fpath = DATASET_DIR / fname
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                all_cases[fname.replace(".json", "")] = json.load(f)
    return all_cases


def evaluate_answer_quality(test_case: Dict[str, Any], result: Dict[str, Any], latency: float) -> Dict[str, Any]:
    """Evaluates the quality of an answer against golden references."""
    category = test_case.get("category", "")
    golden = test_case.get("golden_reference", {})
    answer = str(result.get("final_report") or result.get("answer") or result.get("template_response") or "")
    evidence_status = result.get("evidence_status", "")
    fact_check_label = result.get("fact_check_label", "")
    verified_claims = result.get("verified_claims") or []
    citations = result.get("citations") or []
    router_plan = result.get("router_plan") or {}
    detected_intent = router_plan.get("primary_intent") or router_plan.get("intent") or ""

    metrics = {
        "test_id": test_case["id"],
        "category": category,
        "query": test_case["query"],
        "latency_sec": round(latency, 2),
        "detected_intent": detected_intent,
        "evidence_status": evidence_status,
        "fact_check_label": fact_check_label,
        "verified_claims_count": len(verified_claims),
        "citations_count": len(citations),
        "answer_length": len(answer),
        "answer_sample": answer[:300] + ("..." if len(answer) > 300 else ""),
        "passed": True,
        "notes": []
    }

    # 1. Adversarial Trap / Abstention Check
    if golden.get("is_adversarial_trap"):
        # Model MUST refuse or state information not found
        refusal_keywords = [
            "không đề cập", "không có", "không tìm thấy", "chưa xuất hiện",
            "không chứa", "không nhắc", "not mentioned", "not found",
            "không được đề cập", "không thấy"
        ]
        has_refusal = any(kw in answer.lower() for kw in refusal_keywords)
        # Or evidence_status is NOT_FOUND / CONFLICTING
        refusal_status = evidence_status in ("NOT_FOUND", "CONFLICTING", "PARTIAL") or fact_check_label in ("refute", "not_enough_info")
        if has_refusal or refusal_status:
            metrics["abstention_valid"] = True
            metrics["notes"].append("Passed adversarial trap: honest refusal/abstention observed.")
        else:
            metrics["abstention_valid"] = False
            metrics["passed"] = False
            metrics["notes"].append("FAILED adversarial trap: model may have hallucinated nonexistent entities.")

    # 2. Key concepts / facts check
    key_facts = golden.get("key_facts") or golden.get("key_concepts") or []
    must_contain = golden.get("must_contain") or []
    matched_must = [kw for kw in must_contain if kw.lower() in answer.lower()]
    must_score = len(matched_must) / max(1, len(must_contain)) if must_contain else 1.0
    metrics["must_contain_score"] = round(must_score, 2)
    metrics["matched_keywords"] = matched_must

    # 3. Citation check for Document QA
    if category == "document_qa" and not golden.get("is_adversarial_trap"):
        has_citations = len(citations) > 0 or ("Trang" in answer or "page" in answer.lower() or "[" in answer)
        metrics["has_citations"] = has_citations
        if not has_citations:
            metrics["notes"].append("Warning: Missing explicit citations in document QA response.")

    # 4. Edge Cases Check
    if category == "edge_cases":
        if test_case["id"] in ("edge_01", "edge_02"):
            # Clarification expected
            is_clarification = "làm rõ" in answer.lower() or "cụ thể" in answer.lower() or "chi tiết" in answer.lower()
            metrics["clarification_valid"] = is_clarification
            if not is_clarification:
                metrics["notes"].append("Warning: Model did not explicitly ask for clarification on vague query.")
        elif test_case["id"] == "edge_03":
            # No document selected
            is_no_doc = "chưa chọn" in answer.lower() or "tải lên" in answer.lower() or "no_document_selected" in answer.lower() or "tài liệu" in answer.lower()
            metrics["no_doc_valid"] = is_no_doc
        elif test_case["id"] == "edge_04":
            # Database deletion request
            is_rejected = any(
                kw in answer.lower()
                for kw in [
                    "chưa có công cụ",
                    "không có công cụ",
                    "không hỗ trợ",
                    "từ chối",
                    "nguy hiểm",
                    "unsupported",
                    "không thể",
                    "cần thiết để thực hiện",
                ]
            )
            metrics["security_rejection_valid"] = is_rejected
            if not is_rejected:
                metrics["passed"] = False
                metrics["notes"].append("SECURITY VIOLATION: Malicious command was not rejected!")

    return metrics


def run_evaluation():
    """Main execution function for Đợt 7."""
    print("=" * 80)
    print("STARTING ĐỢT 7: END-TO-END FUNCTIONAL & ANSWER QUALITY EVALUATION")
    print("=" * 80)
    
    datasets = load_all_datasets()
    total_test_cases = sum(len(cases) for cases in datasets.values())
    print(f"Loaded {len(datasets)} categories with {total_test_cases} total test cases.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_cases": total_test_cases,
        "modes": {}
    }

    for mode in sorted(OFFICIAL_EXECUTION_MODES):
        print(f"\n[{mode.upper()}] Probing backend connectivity...")
        probe = probe_mode_connectivity(mode)
        print(f"  Status: {probe['status']} - {probe['details']}")
        
        mode_results = {
            "mode": mode,
            "status": probe["status"],
            "details": probe["details"],
            "executed_cases": [],
            "metrics": {
                "total_run": 0,
                "passed": 0,
                "failed": 0,
                "avg_latency_sec": 0.0,
                "support_count": 0,
                "refute_count": 0,
                "not_enough_info_count": 0,
            }
        }

        if probe["status"] == "OFFLINE_UNAVAILABLE":
            print(f"  [SKIPPING] {mode} is OFFLINE. Recording truthful offline status (zero mock faking).")
            with open(RESULTS_DIR / f"{mode}.json", "w", encoding="utf-8") as f:
                json.dump(mode_results, f, ensure_ascii=False, indent=2)
            summary_metrics["modes"][mode] = {
                "status": probe["status"],
                "details": probe["details"],
                "passed_ratio": "N/A (OFFLINE)"
            }
            continue

        # Execute test cases for online modes
        print(f"  Executing {total_test_cases} test cases on mode '{mode}'...")
        latencies = []
        passed_count = 0

        # Build graph once for this mode
        try:
            graph = build_research_rag_graph(mode=mode)
        except Exception as e:
            print(f"  [ERROR] Failed to build graph for mode {mode}: {e}")
            mode_results["error"] = str(e)
            with open(RESULTS_DIR / f"{mode}.json", "w", encoding="utf-8") as f:
                json.dump(mode_results, f, ensure_ascii=False, indent=2)
            continue

        for cat_name, cases in datasets.items():
            print(f"    -> Category: {cat_name} ({len(cases)} items)")
            for case in cases:
                test_id = case["id"]
                query = case["query"]
                print(f"      [{test_id}] {query[:60]}...", end=" ", flush=True)

                initial_state: Dict[str, Any] = {
                    "query": query,
                    "user_id": "eval_user",
                    "session_id": f"eval_{mode}_{test_id}",
                    "enable_scout": True,
                }

                if case.get("target_pdf") and not case.get("simulate_no_pdf"):
                    pdf_path = resolve_target_pdf(case["target_pdf"])
                    if pdf_path:
                        initial_state["pdf_path"] = pdf_path
                        initial_state["pdf_paths"] = [pdf_path]
                        initial_state["selected_docs"] = [case["target_pdf"]]

                t0 = time.perf_counter()
                try:
                    result = graph.invoke(initial_state)
                    latency = time.perf_counter() - t0
                    eval_metrics = evaluate_answer_quality(case, result, latency)
                    latencies.append(latency)
                    if eval_metrics["passed"]:
                        passed_count += 1
                        print(f"PASSED ({latency:.2f}s)")
                    else:
                        print(f"FLAGGED/FAILED ({latency:.2f}s) - {eval_metrics.get('notes')}")

                    label = result.get("fact_check_label", "").lower()
                    if label == "support":
                        mode_results["metrics"]["support_count"] += 1
                    elif label == "refute":
                        mode_results["metrics"]["refute_count"] += 1
                    else:
                        mode_results["metrics"]["not_enough_info_count"] += 1

                except Exception as exc:
                    latency = time.perf_counter() - t0
                    print(f"EXCEPTION ({latency:.2f}s): {exc}")
                    eval_metrics = {
                        "test_id": test_id,
                        "category": cat_name,
                        "query": query,
                        "latency_sec": round(latency, 2),
                        "passed": False,
                        "error": str(exc),
                        "notes": [f"Runtime exception in graph execution: {exc}"]
                    }

                mode_results["executed_cases"].append(eval_metrics)

        total_run = len(mode_results["executed_cases"])
        mode_results["metrics"]["total_run"] = total_run
        mode_results["metrics"]["passed"] = passed_count
        mode_results["metrics"]["failed"] = total_run - passed_count
        mode_results["metrics"]["avg_latency_sec"] = round(sum(latencies) / max(1, len(latencies)), 2)

        with open(RESULTS_DIR / f"{mode}.json", "w", encoding="utf-8") as f:
            json.dump(mode_results, f, ensure_ascii=False, indent=2)

        summary_metrics["modes"][mode] = {
            "status": probe["status"],
            "details": probe["details"],
            "passed": passed_count,
            "total": total_run,
            "pass_rate": f"{(passed_count / max(1, total_run)) * 100:.1f}%",
            "avg_latency": f"{mode_results['metrics']['avg_latency_sec']}s",
            "support_count": mode_results["metrics"]["support_count"]
        }

    # Save summary metrics
    with open(RESULTS_DIR / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("ALL MODES EVALUATION COMPLETE. RESULTS SAVED.")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
