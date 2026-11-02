# -*- coding: utf-8 -*-
"""
Automated End-to-End Evaluation Runner for All 6 Agentic RAG Modes:
- cpu_min (Ultra-Fast Local Agent on Radeon 780M + AVX-512)
- cpu_max (Deep Local Research Agent on Radeon 780M + Cross-Encoder)
- gemini (Quota-Optimized Cloud Agent - 2 to 3 calls/query)
- kaggle_fn (Fine-Tuned Multi-LoRA Specialist Agent with Local Fallback)
- kaggle_max (Frontier 32B Reasoning Agent with Local Fallback)
- hybrid (Adaptive Cross-Model Adversarial Ensemble - Flagship Mode)
"""

import os
import sys
import time
import json
import argparse
import statistics
from pathlib import Path
from typing import List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.graph import build_graph
from eval.metrics import evaluate_single_prediction
from utils.local_vulkan_setup import probe_all_capabilities

ALL_MODES = ["cpu_min", "cpu_max", "gemini", "kaggle_fn", "kaggle_max", "hybrid"]


def load_benchmark(limit_per_cat: int = 0) -> List[Dict[str, Any]]:
    bench_path = Path(__file__).resolve().parent / "benchmark_120.jsonl"
    items: List[Dict[str, Any]] = []
    with open(bench_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if limit_per_cat <= 0:
        return items

    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        cat = it.get("category", "single_hop")
        by_cat.setdefault(cat, []).append(it)

    sampled: List[Dict[str, Any]] = []
    for cat, group in by_cat.items():
        sampled.extend(group[:limit_per_cat])
    return sampled


def run_mode_evaluation(mode: str, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    app = build_graph(mode)
    agent_modes = {
        "retrieval": mode,
        "reader": mode,
        "summarizer": mode,
        "factchecker": mode,
        "writer": mode,
    }

    records: List[Dict[str, Any]] = []
    for q in questions:
        selected_doc = q.get("selected_doc", "")
        pdf_path = str(ROOT_DIR / "uploads" / selected_doc) if selected_doc else None

        initial_state = {
            "query": q["question"],
            "question": q["question"],
            "original_question": q["question"],
            "mode": mode,
            "agent_modes": agent_modes,
            "selected_doc": selected_doc,
            "pdf_path": pdf_path,
            "retry_count": 0,
            "critic_iterations": 0,
            "reasoning_trace": [],
        }
        t0 = time.perf_counter()
        try:
            final_state = app.invoke(initial_state)
        except Exception as e:
            final_state = {
                "final_response": f"Error: {e}",
                "reranked_chunks": [],
                "extracted_evidence": [],
                "retry_count": 0,
            }
        dt = time.perf_counter() - t0
        rec = evaluate_single_prediction(q, final_state, dt)
        records.append(rec)

    latencies = sorted(r["latency_sec"] for r in records) if records else [0.0]
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]

    return {
        "mode": mode,
        "num_questions": len(records),
        "groundedness": round(statistics.mean(r["groundedness"] for r in records), 4) if records else 0.0,
        "citation_f1": round(statistics.mean(r["citation_f1"] for r in records), 4) if records else 0.0,
        "numeric_exactness": round(statistics.mean(r["numeric_exactness"] for r in records), 4) if records else 0.0,
        "abstention_accuracy": round(statistics.mean(r["abstention_accuracy"] for r in records), 4) if records else 0.0,
        "p50_latency_s": round(p50, 2),
        "p95_latency_s": round(p95, 2),
        "avg_retries": round(statistics.mean(r["retry_count"] for r in records), 2) if records else 0.0,
        "records": records,
    }


def write_markdown_report(summaries: List[Dict[str, Any]], caps: Dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Multi-Agent RAG v4 — Evaluation Benchmark Report",
        "",
        "## 1. Runtime Hardware & Capability Matrix",
        f"- **Local Hardware**: Lenovo ThinkBook 14 G8+ (`AMD Ryzen 7 H 255` + `AMD Radeon 780M` Vulkan iGPU, 32GB LPDDR5X)",
        f"- **Ollama Local Status**: `{caps.get('ollama', {})}`",
        f"- **Gemini API Status**: `{caps.get('gemini', {})}`",
        f"- **Kaggle Tunnel Status**: `{caps.get('kaggle', {})}`",
        "",
        "## 2. Aggregate Metrics Across All 6 Execution Modes",
        "",
        "| Mode | Questions | Groundedness | Citation F1 | Numeric/Date Exactness | Abstention Acc | Latency p50 (s) | Latency p95 (s) | Avg Outer Retries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| `{s['mode']}` | {s['num_questions']} | **{s['groundedness']*100:.1f}%** | "
            f"**{s['citation_f1']*100:.1f}%** | **{s['numeric_exactness']*100:.1f}%** | "
            f"**{s['abstention_accuracy']*100:.1f}%** | {s['p50_latency_s']}s | {s['p95_latency_s']}s | {s['avg_retries']} |"
        )

    lines.extend([
        "",
        "## 3. Architectural Verification Highlights",
        "- **Document-Scoped Retrieval Fix**: All 6 modes dynamically parse and cache `selected_doc` (`load_or_parse_selected_pdf`), eliminating cross-document contamination.",
        "- **Two-Tier Self-Correction Loop**: Inner Generator-Critic loop (`route_after_summarizer`) + Outer Fact-Checker & Gap Re-Query loop (`route_after_fact_check` -> `retry_loop_node`).",
        "- **Cross-Model Adversarial Verification (`Hybrid`)**: Guarantees Generator and Fact-Checker run on distinct model families to eliminate self-confirmation bias.",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=ALL_MODES)
    parser.add_argument("--limit-per-cat", type=int, default=2, help="Questions per category (0 = all 30 per cat = 120 total)")
    args = parser.parse_args()

    caps = probe_all_capabilities()
    questions = load_benchmark(limit_per_cat=args.limit_per_cat)
    print(f"[Benchmark] Loaded {len(questions)} stratified questions across 4 categories.")

    summaries = []
    for mode in args.modes:
        print(f"[Benchmark] Evaluating mode: {mode} ...")
        summary = run_mode_evaluation(mode, questions)
        summaries.append(summary)
        print(
            f"  -> {mode}: Groundedness={summary['groundedness']*100:.1f}%, "
            f"CitationF1={summary['citation_f1']*100:.1f}%, "
            f"NumericExact={summary['numeric_exactness']*100:.1f}%, "
            f"Abstention={summary['abstention_accuracy']*100:.1f}%, "
            f"p50={summary['p50_latency_s']}s"
        )

    report_path = Path(__file__).resolve().parent / "benchmark_report.md"
    write_markdown_report(summaries, caps, report_path)
    json_path = Path(__file__).resolve().parent / "benchmark_results.json"
    json_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Saved benchmark report -> {report_path}")


if __name__ == "__main__":
    main()
