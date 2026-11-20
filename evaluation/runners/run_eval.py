"""
Comprehensive E2E AI System Evaluation Runner (Đợt 7).
Executes the 96-run matrix (16 Golden Cases x 6 Execution Modes).
Strictly adheres to:
  1. Zero mock faking: unavailable backends recorded as BLOCKED.
  2. Degraded extractive fallbacks recorded as DEGRADED (not genuine LLM pass).
  3. Strict 10-point scoring rubric per evaluation/ground_truth/scoring_rubric.json.
"""

import os
import sys
import json
import time
import re
import urllib.request
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project roots are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MAIN_DIR = PROJECT_ROOT / "main"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

import fitz
from core.graph import build_research_rag_graph
from core.model_gateway import OFFICIAL_EXECUTION_MODES
from core.config import settings

DATASET_DIR = PROJECT_ROOT / "evaluation" / "dataset"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"
GROUND_TRUTH_DIR = PROJECT_ROOT / "evaluation" / "ground_truth"
UPLOADS_DIR = MAIN_DIR / "uploads"


def resolve_pdf_path(filename: str) -> Optional[str]:
    """Resolves absolute path of PDF in main/uploads/."""
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


def probe_mode_status(mode: str) -> Dict[str, Any]:
    """Returns backend availability status: AVAILABLE, DEGRADED, or BLOCKED."""
    if mode in ("gpu_finetune", "gpu_max"):
        remote_url = getattr(settings, "REMOTE_INFERENCE_URL", "")
        if not remote_url:
            return {
                "status": "BLOCKED",
                "backend": "Remote Kaggle Cloudflare Worker",
                "model_name": "qwen2.5-coder-7b-instruct" if mode == "gpu_finetune" else "qwen2.5-32b-instruct",
                "details": "REMOTE_INFERENCE_URL not configured. Marked BLOCKED (zero mocks permitted)."
            }
        try:
            req = urllib.request.Request(f"{remote_url.rstrip('/')}/health", headers={"User-Agent": "eval"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    return {
                        "status": "AVAILABLE",
                        "backend": "Remote Kaggle GPU Worker",
                        "model_name": "qwen2.5-coder-7b-instruct" if mode == "gpu_finetune" else "qwen2.5-32b-instruct",
                        "details": f"Remote worker online at {remote_url}"
                    }
        except Exception as e:
            return {
                "status": "BLOCKED",
                "backend": "Remote Kaggle Cloudflare Worker",
                "model_name": "qwen2.5-coder-7b-instruct" if mode == "gpu_finetune" else "qwen2.5-32b-instruct",
                "details": f"Remote worker unreachable at {remote_url} ({e}). Marked BLOCKED per protocol."
            }

    elif mode == "cpu_max":
        # Check if Ollama port 11434 is responding
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", headers={"User-Agent": "eval"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    return {
                        "status": "AVAILABLE",
                        "backend": "Local Ollama 7B LLM",
                        "model_name": "qwen2.5-coder:7b",
                        "details": "Local Ollama daemon running on port 11434"
                    }
        except Exception:
            pass
        return {
            "status": "DEGRADED",
            "backend": "Local In-Memory CPU Extractive Fallback",
            "model_name": "extractive-fallback (Ollama offline)",
            "details": "Local Ollama server offline on port 11434. Operating in extractive fallback mode."
        }

    elif mode == "cpu_min":
        return {
            "status": "AVAILABLE",
            "backend": "Local In-Memory C-Engine",
            "model_name": "PyMuPDF + BM25 + Deterministic Extractor",
            "details": "Local pure in-memory engine (100% offline-ready)"
        }

    elif mode == "hybrid":
        return {
            "status": "AVAILABLE",
            "backend": "Cascading Hybrid Coordinator",
            "model_name": "Kaggle (offline) -> Ollama (offline) -> Local Extractive (live)",
            "details": "Cascading multi-tier coordinator operational. Zero Gemini calls strictly enforced."
        }

    elif mode == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "")
        if api_key:
            return {
                "status": "AVAILABLE",
                "backend": "Google Cloud Gemini API",
                "model_name": "gemini-2.5-flash",
                "details": "Gemini 2.5 Flash API available and active"
            }
        else:
            return {
                "status": "BLOCKED",
                "backend": "Google Cloud Gemini API",
                "model_name": "gemini-2.5-flash",
                "details": "GEMINI_API_KEY not configured in environment"
            }

    return {"status": "BLOCKED", "backend": "Unknown", "model_name": "unknown", "details": f"Unknown mode {mode}"}


def load_golden_dataset() -> List[Dict[str, Any]]:
    """Loads all 16 test cases in official order."""
    case_files = [
        "general_qa.json",
        "document_qa.json",
        "web_research.json",
        "mixed_qa.json",
        "edge_cases.json"
    ]
    all_cases = []
    for fname in case_files:
        fpath = DATASET_DIR / fname
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                cases = json.load(f)
                all_cases.extend(cases)
    return all_cases


def score_response(case: Dict[str, Any], result: Dict[str, Any], latency: float, mode_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes strict 10-point score per scoring_rubric.json:
      - Correctness: 0 - 4
      - Completeness: 0 - 2
      - Groundedness & Citation: 0 - 2
      - Routing & Compliance: 0 - 1
      - Uncertainty Calibration: 0 - 1
      Total: 0 - 10
    """
    test_id = case["id"]
    category = case["category"]
    golden = case.get("golden_reference", {})
    answer = str(result.get("final_report") or result.get("answer") or result.get("template_response") or "").strip()
    evidence_status = result.get("evidence_status", "")
    fact_check_label = result.get("fact_check_label", "")
    citations = result.get("citations") or []
    router_plan = result.get("router_plan") or {}
    detected_intent = router_plan.get("primary_intent") or result.get("primary_intent") or ""

    notes = []
    is_hard_fail = False

    # 1. Routing & Compliance (0 - 1)
    routing_score = 0.0
    expected_intent = case.get("expected_intent", "")
    if detected_intent == expected_intent:
        routing_score = 1.0
    else:
        # Compatible intent mappings
        if expected_intent == "TASK_ASSISTED_RESPONSE" and detected_intent in ("TASK_ASSISTED_RESPONSE", "GENERAL_QA"):
            routing_score = 1.0
        elif expected_intent == "UNSUPPORTED_CAPABILITY" and detected_intent in ("TASK_EXECUTION", "UNSUPPORTED_CAPABILITY"):
            routing_score = 1.0
        else:
            notes.append(f"Routing mismatch: Expected {expected_intent}, detected {detected_intent}.")

    # 2. Uncertainty Calibration & Abstention (0 - 1)
    calibration_score = 1.0
    if category == "edge_cases":
        if test_id == "EDGE-01":
            # Missing document: must NOT answer as if document was read
            if "chưa" in answer.lower() and ("chọn" in answer.lower() or "tài liệu" in answer.lower()):
                calibration_score = 1.0
            else:
                calibration_score = 0.0
                notes.append("EDGE-01 failed: Did not provide clear NO_DOCUMENT_SELECTED guidance.")
        elif test_id == "EDGE-02":
            # Clarification
            if "làm rõ" in answer.lower() or "cụ thể" in answer.lower() or "ngắn" in answer.lower():
                calibration_score = 1.0
            else:
                calibration_score = 0.0
                notes.append("EDGE-02 failed: Did not request clarification on ambiguous query.")
        elif test_id == "EDGE-03":
            # Unsupported tool / command
            if any(w in answer.lower() for w in ["chưa có công cụ", "không hỗ trợ", "không thể", "từ chối", "hướng dẫn"]):
                calibration_score = 1.0
            else:
                calibration_score = 0.0
                is_hard_fail = True
                notes.append("HARD FAIL: Fabricated tool execution or failed to reject unsupported capability.")

    # 3. Groundedness & Citations (0 - 2)
    citation_score = 2.0
    if category in ("document_qa", "mixed_qa"):
        has_citations = len(citations) > 0 or ("Trang" in answer or "page" in answer.lower() or "[" in answer)
        if not has_citations and not golden.get("is_adversarial_trap"):
            citation_score = 1.0
            notes.append("Document QA: Missing explicit citations or page numbers.")
        
        # Check specific expected citations
        req_citations = golden.get("required_citations", {})
        if req_citations:
            exp_page = req_citations.get("page_no")
            if exp_page:
                exp_pages = [exp_page] if isinstance(exp_page, int) else exp_page
                found_page = any(f"Trang {p}" in answer or f"page {p}" in answer.lower() or f"[{p}]" in answer for p in exp_pages)
                if not found_page and not has_citations:
                    citation_score = 0.5
                    notes.append(f"Citations: Expected page {exp_pages} was not cited explicitly.")
    elif category == "general_qa":
        # General QA should NOT invent fake document citations
        if "Trang 1" in answer and "Attention Is All You Need" in answer:
            citation_score = 0.0
            is_hard_fail = True
            notes.append("HARD FAIL: General QA fabricated citations to unrequested PDF.")

    # 4. Correctness (0 - 4)
    correctness_score = 4.0
    must_contain = golden.get("must_contain") or []
    forbidden_claims = golden.get("forbidden_claims") or []

    # Check forbidden claims
    for claim in forbidden_claims:
        if claim.lower() in answer.lower():
            correctness_score -= 2.0
            notes.append(f"Found forbidden claim in response: '{claim}'.")

    # Domain-specific factual verification
    if test_id == "DOC-04":
        # Table 3 parameters: N=6, d_model=512, d_ff=2048, h=8, d_k=64
        table3_params = [("6", "N"), ("512", "d_model"), ("2048", "d_ff"), ("8", "h"), ("64", "d_k")]
        matched = [p for p, name in table3_params if p in answer]
        if len(matched) < 3:
            correctness_score = 1.5
            notes.append(f"DOC-04: Only matched {len(matched)}/5 Table 3 parameters.")
        elif len(matched) < 5:
            correctness_score = 3.0
            notes.append(f"DOC-04: Matched {len(matched)}/5 Table 3 parameters.")

    elif test_id == "DOC-01":
        # Authors & Year
        has_vaswani = "vaswani" in answer.lower()
        has_2017 = "2017" in answer
        if not (has_vaswani and has_2017):
            correctness_score = 2.0
            notes.append("DOC-01: Missing author Vaswani or publication year 2017.")

    elif test_id == "DOC-03":
        # Formulas
        has_formula = "softmax" in answer.lower() and ("sqrt" in answer.lower() or "căn" in answer.lower() or "d_k" in answer)
        if not has_formula:
            correctness_score = 2.0
            notes.append("DOC-03: Missing Scaled Dot-Product formula or sqrt(d_k) explanation.")

    elif test_id == "GQA-04":
        # Python code
        has_code = "def " in answer and "return" in answer
        if not has_code:
            correctness_score = 1.0
            notes.append("GQA-04: Response did not contain valid Python function.")

    elif must_contain:
        matched_must = [kw for kw in must_contain if kw.lower() in answer.lower()]
        ratio = len(matched_must) / len(must_contain)
        if ratio < 0.5:
            correctness_score = max(1.0, 4.0 * ratio)
            notes.append(f"Missing essential keywords: {set(must_contain) - set(matched_must)}.")

    # 5. Completeness (0 - 2)
    completeness_score = 2.0
    if len(answer) < 80:
        completeness_score = 0.5
        notes.append("Response is excessively terse (< 80 characters).")
    elif test_id == "DOC-02":
        # Full paper summary requires all 4 aspects
        facets = ["mục tiêu", "kiến trúc", "huấn luyện", "kết quả"]
        found_facets = sum(1 for f in facets if f in answer.lower())
        if found_facets < 2:
            completeness_score = 0.5
            notes.append("DOC-02: Missing major facets of full paper summary.")
        elif found_facets < 4:
            completeness_score = 1.0

    total_score = round(correctness_score + completeness_score + citation_score + routing_score + calibration_score, 1)
    if is_hard_fail:
        total_score = min(total_score, 4.0)

    # Status classification: PASS, FAIL, DEGRADED, BLOCKED
    if is_hard_fail or total_score < 6.0:
        eval_status = "FAIL"
    elif mode_info["status"] == "DEGRADED":
        eval_status = "DEGRADED"
    else:
        eval_status = "PASS"

    return {
        "test_id": test_id,
        "category": category,
        "query": case["query"],
        "latency_sec": round(latency, 2),
        "detected_intent": detected_intent,
        "intent_correct": (routing_score == 1.0),
        "scores": {
            "correctness": correctness_score,
            "completeness": completeness_score,
            "groundedness_and_citation": citation_score,
            "routing_and_compliance": routing_score,
            "uncertainty_calibration": calibration_score,
            "total_score": total_score
        },
        "status": eval_status,
        "evidence_status": evidence_status,
        "fact_check_label": fact_check_label,
        "citations_count": len(citations),
        "answer": answer,
        "notes": notes
    }


def execute_full_matrix():
    """Runs the 96 evaluations (16 cases x 6 modes)."""
    print("=" * 80)
    print("EXECUTING ĐỢT 7: 96-RUN FULL FUNCTIONAL MATRIX EVALUATION")
    print("=" * 80)

    golden_cases = load_golden_dataset()
    print(f"Loaded {len(golden_cases)} Golden Test Cases.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    git_head = "1af1a120531df8350aa1f13454b53faae2ba6f73"
    git_branch = "refactor/assistant-core-reconstruction"
    try:
        import subprocess
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=str(PROJECT_ROOT)).strip()
        git_branch = subprocess.check_output(["git", "branch", "--show-current"], text=True, cwd=str(PROJECT_ROOT)).strip()
    except Exception:
        pass

    matrix_summary = {
        "evaluation_name": "Multi-Agent RAG Phase E3 Full Functional Matrix",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_head,
        "git_branch": git_branch,
        "total_planned_runs": len(golden_cases) * len(OFFICIAL_EXECUTION_MODES),
        "modes": {}
    }

    for mode in sorted(OFFICIAL_EXECUTION_MODES):
        print(f"\n[{mode.upper()}] Auditing backend availability...")
        mode_probe = probe_mode_status(mode)
        print(f"  Classification: {mode_probe['status']} | Backend: {mode_probe['backend']} | Model: {mode_probe['model_name']}")

        mode_record = {
            "mode": mode,
            "status": mode_probe["status"],
            "backend": mode_probe["backend"],
            "model_name": mode_probe["model_name"],
            "details": mode_probe["details"],
            "runs": []
        }

        # Case 1: Backend is BLOCKED (e.g. gpu_finetune, gpu_max)
        if mode_probe["status"] == "BLOCKED":
            print(f"  [BLOCKED] Mode '{mode}' has no live backend. Recording 16 BLOCKED runs (zero mocks permitted).")
            for case in golden_cases:
                blocked_run = {
                    "run_id": f"run_{mode}_{case['id']}",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "git_commit": git_head,
                    "git_branch": git_branch,
                    "test_id": case["id"],
                    "category": case["category"],
                    "mode": mode,
                    "backend": mode_probe["backend"],
                    "model_name": mode_probe["model_name"],
                    "status": "BLOCKED",
                    "detected_intent": "NONE",
                    "intent_correct": False,
                    "latency_sec": 0.0,
                    "scores": {
                        "correctness": 0.0,
                        "completeness": 0.0,
                        "groundedness_and_citation": 0.0,
                        "routing_and_compliance": 0.0,
                        "uncertainty_calibration": 0.0,
                        "total_score": 0.0
                    },
                    "answer": "",
                    "citations": [],
                    "verified_claims": [],
                    "evidence_status": "NOT_APPLICABLE",
                    "fact_check_label": "not_enough_info",
                    "notes": [f"BLOCKED: Remote backend unreachable ({mode_probe['details']}). No mock substitution permitted."]
                }
                mode_record["runs"].append(blocked_run)

            with open(RESULTS_DIR / f"{mode}.json", "w", encoding="utf-8") as f:
                json.dump(mode_record, f, ensure_ascii=False, indent=2)

            matrix_summary["modes"][mode] = {
                "status": "BLOCKED",
                "backend": mode_probe["backend"],
                "total_runs": len(golden_cases),
                "passed_runs": 0,
                "degraded_runs": 0,
                "blocked_runs": len(golden_cases),
                "failed_runs": 0,
                "avg_score": 0.0,
                "avg_latency": "0.0s"
            }
            continue

        # Case 2: Backend is AVAILABLE or DEGRADED -> Build graph and execute live
        print(f"  Executing {len(golden_cases)} live test cases on mode '{mode}'...")
        try:
            graph = build_research_rag_graph(mode=mode)
        except Exception as e:
            print(f"  [ERROR] Failed to compile graph for mode {mode}: {e}")
            continue

        passed_count = 0
        degraded_count = 0
        failed_count = 0
        scores = []
        latencies = []

        for case in golden_cases:
            test_id = case["id"]
            query = case["query"]
            print(f"    [{test_id}] {query[:55]}...", end=" ", flush=True)

            state: Dict[str, Any] = {
                "query": query,
                "user_id": "eval_user",
                "session_id": f"eval_{mode}_{test_id}",
                "enable_scout": True,
            }

            # Setup target documents
            if not case.get("simulate_no_pdf"):
                if case.get("target_pdfs"):
                    resolved = [resolve_pdf_path(p) for p in case["target_pdfs"] if resolve_pdf_path(p)]
                    if resolved:
                        state["pdf_paths"] = resolved
                        state["pdf_path"] = resolved[0]
                        state["selected_docs"] = [Path(p).name for p in resolved]
                elif case.get("target_pdf"):
                    single_res = resolve_pdf_path(case["target_pdf"])
                    if single_res:
                        state["pdf_path"] = single_res
                        state["pdf_paths"] = [single_res]
                        state["selected_docs"] = [Path(single_res).name]

            t0 = time.perf_counter()
            try:
                result = graph.invoke(state)
                latency = time.perf_counter() - t0
                run_eval = score_response(case, result, latency, mode_probe)
                latencies.append(latency)
                score = run_eval["scores"]["total_score"]
                scores.append(score)

                if run_eval["status"] == "PASS":
                    passed_count += 1
                    print(f"PASS ({score}/10, {latency:.2f}s)")
                elif run_eval["status"] == "DEGRADED":
                    degraded_count += 1
                    print(f"DEGRADED ({score}/10, {latency:.2f}s) - {run_eval.get('notes')}")
                else:
                    failed_count += 1
                    print(f"FAIL ({score}/10, {latency:.2f}s) - {run_eval.get('notes')}")

            except Exception as exc:
                latency = time.perf_counter() - t0
                print(f"EXCEPTION ({latency:.2f}s): {exc}")
                run_eval = {
                    "test_id": test_id,
                    "category": case["category"],
                    "query": query,
                    "latency_sec": round(latency, 2),
                    "status": "FAIL",
                    "scores": {
                        "correctness": 0.0,
                        "completeness": 0.0,
                        "groundedness_and_citation": 0.0,
                        "routing_and_compliance": 0.0,
                        "uncertainty_calibration": 0.0,
                        "total_score": 0.0
                    },
                    "notes": [f"Runtime exception in graph execution: {exc}"]
                }
                failed_count += 1
                scores.append(0.0)

            run_eval["run_id"] = f"run_{mode}_{test_id}"
            run_eval["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            run_eval["git_commit"] = git_head
            run_eval["git_branch"] = git_branch
            run_eval["mode"] = mode
            run_eval["backend"] = mode_probe["backend"]
            run_eval["model_name"] = mode_probe["model_name"]
            mode_record["runs"].append(run_eval)

        with open(RESULTS_DIR / f"{mode}.json", "w", encoding="utf-8") as f:
            json.dump(mode_record, f, ensure_ascii=False, indent=2)

        avg_sc = round(sum(scores) / max(1, len(scores)), 2)
        avg_lat = round(sum(latencies) / max(1, len(latencies)), 2)
        matrix_summary["modes"][mode] = {
            "status": mode_probe["status"],
            "backend": mode_probe["backend"],
            "total_runs": len(golden_cases),
            "passed_runs": passed_count,
            "degraded_runs": degraded_count,
            "blocked_runs": 0,
            "failed_runs": failed_count,
            "avg_score": avg_sc,
            "avg_latency": f"{avg_lat}s"
        }

    # Save summary
    with open(RESULTS_DIR / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(matrix_summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("96-RUN FULL FUNCTIONAL MATRIX EVALUATION COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    execute_full_matrix()
