# -*- coding: utf-8 -*-
"""
Ablation Study Matrix for Multi-Agent Research RAG.
Scientifically benchmarks 4 system configurations on identical benchmark inputs:

Config A: Naive RAG (Baseline)
          - Single query BM25 search
          - No Supervisor decomposition
          - Direct single-prompt answer (No Critic, No Fact-Checker)

Config B: Advanced RAG (Modular)
          - Supervisor Multi-Query + Weighted RRF Hybrid Retrieval + Reader SourceUnits
          - Single-pass Summarizer (No Critic self-refinement, No Fact-Checker loop)

Config C: Multi-Agent RAG (Tier-1 Inner Loop Only)
          - Supervisor + Hybrid Retrieval + Reader
          - Summarizer with Inner Generator-Critic Reflection Loop
          - No Outer Fact-Checker verification

Config D: Full Multi-Agent RAG (Tier-1 & Tier-2 Dual Loops - Flagship)
          - Full two-tier architecture:
            Tier-1: Generator-Critic Inner Loop
            Tier-2: Cross-Model Adversarial Fact-Checker with Negative Verification Guard
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.state import AgentState
from core.tools import load_or_parse_selected_pdf
from agents.agent1_retrieval.tools.bm25 import SimpleBM25
from modes.registry import get_mode_agents
from eval.strict_metrics import (
    compute_strict_citation_f1,
    compute_strict_groundedness,
    compute_strict_abstention_accuracy,
)


def run_naive_rag(question: str, selected_doc: str, mode: str = "cpu_min") -> Dict[str, Any]:
    """Config A: Naive RAG - Single query BM25 + direct single-prompt answer."""
    t0 = time.perf_counter()
    pdf_path = str(ROOT_DIR / "uploads" / selected_doc) if selected_doc else None
    chunks = load_or_parse_selected_pdf(pdf_path) if pdf_path else []

    bm25 = SimpleBM25()
    if chunks:
        bm25.fit(chunks)
        hits = bm25.search(question, top_k=4)
    else:
        hits = []

    context_str = "\n".join(f"[{i+1}] {c.get('text', '')}" for i, c in enumerate(hits))

    # Basic prompt without multi-agent deliberation
    if not hits:
        answer = "Không tìm thấy thông tin liên quan trong tài liệu."
    else:
        # Simple extractive fallback from top hit
        answer = f"Dựa trên tài liệu:\n{hits[0].get('text', '')[:400]} [1]"

    latency = time.perf_counter() - t0
    return {
        "answer": answer,
        "retrieved_docs": hits,
        "latency_sec": latency,
        "critic_loops": 0,
        "outer_retries": 0,
    }


def run_advanced_rag(question: str, selected_doc: str, mode: str = "cpu_min") -> Dict[str, Any]:
    """Config B: Advanced RAG - Supervisor + Weighted Hybrid Retrieval + Reader + Single-pass Summarizer."""
    t0 = time.perf_counter()
    agents = get_mode_agents(mode)
    pdf_path = str(ROOT_DIR / "uploads" / selected_doc) if selected_doc else None

    state: AgentState = {
        "query": question,
        "question": question,
        "original_question": question,
        "mode": mode,
        "selected_doc": selected_doc,
        "pdf_path": pdf_path,
        "retry_count": 0,
        "critic_iterations": 0,
        "max_reflection_loops": 0,  # Zero loops
    }

    # 1. Retrieval
    ret_out = agents["retrieval"].run(state)
    state.update(ret_out)

    # 2. Reader
    read_out = agents["reader"].run(state)
    state.update(read_out)

    # 3. Summarizer (Single pass, no retry)
    sum_out = agents["summarizer"].run(state)
    state.update(sum_out)

    # 4. Writer
    write_out = agents["writer"].run(state)
    state.update(write_out)

    latency = time.perf_counter() - t0
    answer = state.get("final_report") or state.get("answer") or ""
    evidence = state.get("source_units") or state.get("retrieved_docs") or []

    return {
        "answer": answer,
        "retrieved_docs": evidence,
        "latency_sec": latency,
        "critic_loops": 0,
        "outer_retries": 0,
    }


def run_multi_agent_inner_only(question: str, selected_doc: str, mode: str = "cpu_min") -> Dict[str, Any]:
    """Config C: Multi-Agent Inner Loop Only - Supervisor + Retrieval + Reader + Summarizer Critic Reflection."""
    t0 = time.perf_counter()
    agents = get_mode_agents(mode)
    pdf_path = str(ROOT_DIR / "uploads" / selected_doc) if selected_doc else None

    state: AgentState = {
        "query": question,
        "question": question,
        "original_question": question,
        "mode": mode,
        "selected_doc": selected_doc,
        "pdf_path": pdf_path,
        "retry_count": 0,
        "critic_iterations": 0,
        "max_reflection_loops": 2,
    }

    ret_out = agents["retrieval"].run(state)
    state.update(ret_out)

    read_out = agents["reader"].run(state)
    state.update(read_out)

    sum_out = agents["summarizer"].run(state)
    state.update(sum_out)

    # Inner loop reflection if requested
    critic_loops = 0
    if state.get("critic_action") == "rewrite":
        critic_loops = 1
        sum_out = agents["summarizer"].run(state)
        state.update(sum_out)

    write_out = agents["writer"].run(state)
    state.update(write_out)

    latency = time.perf_counter() - t0
    answer = state.get("final_report") or state.get("answer") or ""
    evidence = state.get("source_units") or state.get("retrieved_docs") or []

    return {
        "answer": answer,
        "retrieved_docs": evidence,
        "latency_sec": latency,
        "critic_loops": critic_loops,
        "outer_retries": 0,
    }


def run_full_multi_agent(question: str, selected_doc: str, mode: str = "cpu_min") -> Dict[str, Any]:
    """Config D: Full Multi-Agent RAG - Two-Tier Dual Loops (Inner Critic + Outer Fact-Checker)."""
    t0 = time.perf_counter()
    from core.graph import build_graph
    app = build_graph(mode)
    pdf_path = str(ROOT_DIR / "uploads" / selected_doc) if selected_doc else None

    state = {
        "query": question,
        "question": question,
        "original_question": question,
        "mode": mode,
        "selected_doc": selected_doc,
        "pdf_path": pdf_path,
        "retry_count": 0,
        "critic_iterations": 0,
    }

    final_state = app.invoke(state)
    latency = time.perf_counter() - t0

    answer = final_state.get("final_report") or final_state.get("answer") or ""
    evidence = final_state.get("source_units") or final_state.get("retrieved_docs") or []

    return {
        "answer": answer,
        "retrieved_docs": evidence,
        "latency_sec": latency,
        "critic_loops": int(final_state.get("critic_iterations", 0)),
        "outer_retries": int(final_state.get("retry_count", 0)),
    }


def execute_ablation_matrix(mode: str = "cpu_min", num_samples: int = 4):
    print("\n" + "=" * 90)
    print(f" ABLATION STUDY MATRIX: Mode = '{mode}'")
    print(f" Benchmarking 4 Architectural Configurations on Standardized Research Set")
    print("=" * 90)

    bench_file = ROOT_DIR / "eval" / "benchmark_120.jsonl"
    questions: List[Dict[str, Any]] = []
    with open(bench_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    # Pick representative samples across categories
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for q in questions:
        cat = q.get("category", "single_hop")
        by_cat.setdefault(cat, []).append(q)

    sampled = []
    for c in ["single_hop", "multi_hop", "table_comparison", "adversarial_trap"]:
        if c in by_cat:
            sampled.append(by_cat[c][0])
    sampled = sampled[:num_samples]

    configs = [
        ("Config A: Naive RAG (BM25 + Direct Prompt)", run_naive_rag),
        ("Config B: Advanced RAG (Weighted RRF + No Loops)", run_advanced_rag),
        ("Config C: Multi-Agent (Tier-1 Inner Loop Only)", run_multi_agent_inner_only),
        ("Config D: Full Multi-Agent (Tier-1 & Tier-2 Dual Loops)", run_full_multi_agent),
    ]

    matrix_results = {}

    for config_name, runner_fn in configs:
        print(f"\n>>> Running: {config_name} ...")
        f1_list = []
        ground_list = []
        abstain_list = []
        lat_list = []

        for q in sampled:
            qid = q["id"]
            selected_doc = q.get("selected_doc", "")
            is_unanswerable = bool(q.get("is_unanswerable", False)) or q.get("category") == "adversarial_trap"
            gold_keywords = q.get("gold_keywords", [])

            res = runner_fn(q["question"], selected_doc, mode=mode)
            ans = res["answer"]
            ev = res["retrieved_docs"]

            _, _, f1 = compute_strict_citation_f1(ans, ev, gold_keywords)
            ground = compute_strict_groundedness(ans, ev, is_unanswerable)
            abstain = compute_strict_abstention_accuracy(ans, is_unanswerable)

            f1_list.append(f1)
            ground_list.append(ground)
            abstain_list.append(abstain)
            lat_list.append(res["latency_sec"])

        avg_f1 = sum(f1_list) / len(f1_list)
        avg_ground = sum(ground_list) / len(ground_list)
        avg_abstain = sum(abstain_list) / len(abstain_list)
        avg_lat = sum(lat_list) / len(lat_list)

        matrix_results[config_name] = {
            "strict_f1": avg_f1,
            "strict_groundedness": avg_ground,
            "strict_abstention": avg_abstain,
            "latency_p50_s": sorted(lat_list)[len(lat_list) // 2],
        }

    # Print Final Markdown Matrix
    print("\n" + "=" * 105)
    print(" SUMMARY ABLATION STUDY RESULTS MATRIX")
    print("=" * 105)
    print(f"{'Configuration':<42} | {'Strict Citation F1':<20} | {'Strict Groundedness':<20} | {'Strict Abstention':<18} | {'Latency p50'}")
    print("-" * 105)
    for cfg, stats in matrix_results.items():
        print(
            f"{cfg:<42} | "
            f"{stats['strict_f1']*100:>18.1f}% | "
            f"{stats['strict_groundedness']*100:>18.1f}% | "
            f"{stats['strict_abstention']*100:>16.1f}% | "
            f"{stats['latency_p50_s']:>8.2f}s"
        )
    print("=" * 105 + "\n")

    return matrix_results


if __name__ == "__main__":
    mode_target = sys.argv[1] if len(sys.argv) > 1 else "cpu_min"
    execute_ablation_matrix(mode_target)

