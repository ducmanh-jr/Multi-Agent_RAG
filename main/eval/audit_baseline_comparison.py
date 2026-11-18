# -*- coding: utf-8 -*-
"""
Audits and compares Legacy Metrics vs. Strict Scientific Metrics.
Runs evaluation on benchmark questions and outputs a transparent side-by-side Truth Table.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.graph import build_graph
from eval.metrics import evaluate_single_prediction as eval_legacy
from eval.strict_metrics import (
    compute_strict_citation_f1,
    compute_strict_groundedness,
    compute_strict_abstention_accuracy,
    is_strict_abstention,
)


def run_dual_audit(mode: str = "cpu_min", limit_questions: int = 4):
    print(f"\n==================================================")
    print(f" AUDIT RUN: Mode = '{mode}' | Questions = {limit_questions}")
    print(f"==================================================")

    # 1. Load benchmark questions
    bench_file = ROOT_DIR / "eval" / "benchmark_120.jsonl"
    questions: List[Dict[str, Any]] = []
    with open(bench_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    # Sample 1 from each category: single_hop, multi_hop, table_comparison, adversarial_trap
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for q in questions:
        cat = q.get("category", "single_hop")
        by_cat.setdefault(cat, []).append(q)

    sampled = []
    cats = ["single_hop", "multi_hop", "table_comparison", "adversarial_trap"]
    for c in cats:
        if c in by_cat:
            sampled.append(by_cat[c][0])
    sampled = sampled[:limit_questions]

    app = build_graph(mode)
    agent_modes = {
        "retrieval": mode,
        "reader": mode,
        "summarizer": mode,
        "factchecker": mode,
        "writer": mode,
    }

    comparison_results = []

    for q in sampled:
        qid = q["id"]
        cat = q["category"]
        selected_doc = q.get("selected_doc", "")
        pdf_path = str(ROOT_DIR / "uploads" / selected_doc) if selected_doc else None
        is_unanswerable = bool(q.get("is_unanswerable", False)) or cat == "adversarial_trap"
        gold_keywords = q.get("gold_keywords", [])

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
        final_state = app.invoke(initial_state)
        dt = time.perf_counter() - t0

        answer = (
            final_state.get("answer")
            or final_state.get("final_report")
            or final_state.get("final_response")
            or ""
        )
        evidence = (
            final_state.get("source_units")
            or final_state.get("retrieved_docs")
            or final_state.get("filtered_context")
            or final_state.get("retrieved_chunks")
            or []
        )

        # 1. Compute Legacy Metrics
        legacy_res = eval_legacy(q, final_state, dt)

        # 2. Compute Strict Metrics
        _, _, strict_f1 = compute_strict_citation_f1(answer, evidence, gold_keywords)
        strict_grounded = compute_strict_groundedness(answer, evidence, is_unanswerable)
        strict_abstain = compute_strict_abstention_accuracy(answer, is_unanswerable)

        comparison_results.append({
            "id": qid,
            "category": cat,
            "legacy": {
                "groundedness": legacy_res["groundedness"],
                "citation_f1": legacy_res["citation_f1"],
                "abstention_acc": legacy_res["abstention_accuracy"],
            },
            "strict": {
                "groundedness": strict_grounded,
                "citation_f1": strict_f1,
                "abstention_acc": strict_abstain,
            },
            "answer_preview": answer[:120].replace("\n", " "),
        })

    # Summary Table
    print("\n------------------------------------------------------------------------------------------------------")
    print(f"{'ID':<8} | {'Category':<16} | {'Legacy F1':<10} | {'Strict F1':<10} | {'Legacy Ground':<14} | {'Strict Ground':<14} | {'Strict Abstain'}")
    print("------------------------------------------------------------------------------------------------------")
    for r in comparison_results:
        print(
            f"{r['id']:<8} | {r['category']:<16} | "
            f"{r['legacy']['citation_f1']*100:>8.1f}% | {r['strict']['citation_f1']*100:>8.1f}% | "
            f"{r['legacy']['groundedness']*100:>12.1f}% | {r['strict']['groundedness']*100:>12.1f}% | "
            f"{r['strict']['abstention_acc']*100:>12.1f}%"
        )
    print("------------------------------------------------------------------------------------------------------")

    # Averages
    avg_leg_f1 = sum(r["legacy"]["citation_f1"] for r in comparison_results) / len(comparison_results)
    avg_str_f1 = sum(r["strict"]["citation_f1"] for r in comparison_results) / len(comparison_results)
    avg_leg_gr = sum(r["legacy"]["groundedness"] for r in comparison_results) / len(comparison_results)
    avg_str_gr = sum(r["strict"]["groundedness"] for r in comparison_results) / len(comparison_results)
    avg_leg_ab = sum(r["legacy"]["abstention_acc"] for r in comparison_results) / len(comparison_results)
    avg_str_ab = sum(r["strict"]["abstention_acc"] for r in comparison_results) / len(comparison_results)

    print(
        f"{'AVERAGE':<8} | {'ALL':<16} | "
        f"{avg_leg_f1*100:>8.1f}% | {avg_str_f1*100:>8.1f}% | "
        f"{avg_leg_gr*100:>12.1f}% | {avg_str_gr*100:>12.1f}% | "
        f"{avg_str_ab*100:>12.1f}%"
    )
    print("======================================================================================================\n")

    return comparison_results


if __name__ == "__main__":
    mode_to_test = sys.argv[1] if len(sys.argv) > 1 else "cpu_min"
    run_dual_audit(mode_to_test)

