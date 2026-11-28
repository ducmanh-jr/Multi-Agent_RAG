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


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(ordered[idx], 4)


def sample_stratified_questions(
    questions: List[Dict[str, Any]],
    num_samples: int,
    categories: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Round-robin stratified sampling across categories up to `min(num_samples, len(questions))`.
    Eliminates the legacy cap of 1 item per category (max 4).
    """
    target = max(0, int(num_samples))
    if target == 0 or not questions:
        return []
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for q in questions:
        cat = str(q.get("category") or "single_hop")
        by_cat.setdefault(cat, []).append(q)

    preferred = categories or ["single_hop", "multi_hop", "table_comparison", "adversarial_trap"]
    ordered_cats = [c for c in preferred if c in by_cat] + [c for c in by_cat if c not in preferred]

    sampled: List[Dict[str, Any]] = []
    round_idx = 0
    while len(sampled) < target:
        added_any = False
        for cat in ordered_cats:
            bucket = by_cat[cat]
            if round_idx < len(bucket):
                sampled.append(bucket[round_idx])
                added_any = True
                if len(sampled) >= target:
                    break
        if not added_any:
            break
        round_idx += 1
    return sampled


def _resolve_doc_paths(item: Dict[str, Any], pdf_root: Path | None = None) -> List[str]:
    base = pdf_root or (ROOT_DIR / "uploads")
    raw_paths = item.get("pdf_paths") or []
    if isinstance(raw_paths, list) and raw_paths:
        resolved = []
        for p in raw_paths:
            if not isinstance(p, str) or not p.strip():
                continue
            cand = Path(p)
            resolved.append(str(cand if cand.is_absolute() or cand.exists() else (base / p)))
        return resolved
    selected_doc = str(item.get("selected_doc") or "").strip()
    if not selected_doc:
        return []
    cand = Path(selected_doc)
    return [str(cand if cand.is_absolute() or cand.exists() else (base / selected_doc))]


def run_graph_variant(
    question: str,
    pdf_paths: List[str],
    *,
    mode: str = "cpu_min",
    max_reflection_loops: int = 2,
    parallel_retrieval: bool = False,
    auto_scout: bool = False,
) -> Dict[str, Any]:
    """Run the real compiled Research RAG graph with explicit handoff/parallel toggles."""
    from core.graph import build_research_rag_graph

    t0 = time.perf_counter()
    app = build_research_rag_graph(
        mode=mode,
        auto_scout=auto_scout,
        parallel_retrieval=parallel_retrieval,
    )
    state: Dict[str, Any] = {
        "query": question,
        "question": question,
        "original_question": question,
        "mode": mode,
        "pdf_path": pdf_paths[0] if pdf_paths else None,
        "pdf_paths": list(pdf_paths),
        "retry_count": 0,
        "critic_iterations": 0,
        "max_reflection_loops": max(0, int(max_reflection_loops)),
    }
    final_state = app.invoke(state)
    latency = time.perf_counter() - t0

    trace = [e for e in (final_state.get("reasoning_trace") or []) if isinstance(e, dict)]
    handoffs = [e for e in trace if e.get("action") == "agent_handoff"]
    reader_handoffs = [e for e in handoffs if e.get("sender") == "reader"]
    answer = final_state.get("final_report") or final_state.get("answer") or ""
    evidence = final_state.get("source_units") or final_state.get("retrieved_docs") or []

    return {
        "answer": answer,
        "retrieved_docs": evidence,
        "latency_sec": latency,
        "critic_loops": int(final_state.get("critic_iterations", 0)),
        "outer_retries": int(final_state.get("retry_count", 0)),
        "handoff_count": len(handoffs),
        "reader_handoff_count": len(reader_handoffs),
        "handoff_pairs": [f"{h.get('sender')}->{h.get('recipient')}" for h in handoffs],
        "reasoning_trace": trace,
    }


def run_paired_handoff_ablation(
    questions: List[Dict[str, Any]],
    *,
    mode: str = "cpu_min",
    pdf_root: Path | None = None,
    parallel_retrieval: bool = False,
    max_reflection_loops_on: int = 2,
    warm_pdf_cache: bool = True,
) -> Dict[str, Any]:
    """
    Fair graph-level paired ablation toggling ONLY supervised handoff budget:
      - handoff_off: `max_reflection_loops = 0`
      - handoff_on:  `max_reflection_loops = max_reflection_loops_on`
    Runs on identical pinned questions, documents, mode, and graph code.
    Pre-warms PDF parse cache (when `warm_pdf_cache=True`) so `off` vs `on` latencies
    are strictly comparable without cold-parse bias on the first variant.
    Reports per-question deltas (improved/degraded/unchanged), handoff yield,
    and separate p50/p95 latencies for sufficient-evidence vs handoff-triggered queries.
    """
    if warm_pdf_cache:
        warmed_paths: set[str] = set()
        for q in questions:
            for p in _resolve_doc_paths(q, pdf_root=pdf_root):
                if p and p not in warmed_paths and os.path.isfile(p):
                    warmed_paths.add(p)
                    try:
                        load_or_parse_selected_pdf(p)
                    except Exception:
                        pass

    per_question: List[Dict[str, Any]] = []
    off_metrics: List[Dict[str, float]] = []
    on_metrics: List[Dict[str, float]] = []
    lat_off_all: List[float] = []
    lat_on_all: List[float] = []
    lat_on_no_handoff: List[float] = []
    lat_on_with_handoff: List[float] = []

    improved = 0
    degraded = 0
    unchanged = 0
    handoff_triggered = 0
    handoff_improved = 0
    handoff_degraded = 0

    for idx, q in enumerate(questions, 1):
        qid = str(q.get("id") or f"q_{idx}")
        prompt = str(q.get("question") or q.get("query") or "")
        pdf_paths = _resolve_doc_paths(q, pdf_root=pdf_root)
        is_unanswerable = bool(q.get("is_unanswerable", False)) or q.get("category") == "adversarial_trap"
        gold_keywords = list(q.get("gold_keywords") or [])
        gold_sources = q.get("gold_sources")

        res_off = run_graph_variant(
            prompt,
            pdf_paths,
            mode=mode,
            max_reflection_loops=0,
            parallel_retrieval=parallel_retrieval,
        )
        res_on = run_graph_variant(
            prompt,
            pdf_paths,
            mode=mode,
            max_reflection_loops=max_reflection_loops_on,
            parallel_retrieval=parallel_retrieval,
        )

        def _score(res: Dict[str, Any]) -> Dict[str, float]:
            ans = res["answer"]
            ev = res["retrieved_docs"]
            p, r, f1 = compute_strict_citation_f1(
                ans,
                ev,
                gold_keywords,
                is_unanswerable=is_unanswerable,
                gold_sources=gold_sources,
            )
            ground = compute_strict_groundedness(ans, ev, is_unanswerable=is_unanswerable)
            abstain = compute_strict_abstention_accuracy(ans, is_unanswerable)
            kw_hits = (
                sum(1 for kw in gold_keywords if kw and kw.lower() in ans.lower()) / len(gold_keywords)
                if gold_keywords and not is_unanswerable
                else (1.0 if (is_unanswerable and abstain >= 1.0) else 0.0)
            )
            composite = round((f1 + ground + abstain + kw_hits) / 4.0, 4)
            return {
                "citation_precision": p,
                "citation_recall": r,
                "citation_f1": f1,
                "groundedness": ground,
                "abstention": abstain,
                "keyword_recall": round(kw_hits, 4),
                "composite": composite,
            }

        m_off = _score(res_off)
        m_on = _score(res_on)
        off_metrics.append(m_off)
        on_metrics.append(m_on)
        lat_off_all.append(res_off["latency_sec"])
        lat_on_all.append(res_on["latency_sec"])

        delta = round(m_on["composite"] - m_off["composite"], 4)
        if delta > 1e-4:
            outcome = "improved"
            improved += 1
        elif delta < -1e-4:
            outcome = "degraded"
            degraded += 1
        else:
            outcome = "unchanged"
            unchanged += 1

        triggered = res_on["handoff_count"] > 0
        if triggered:
            handoff_triggered += 1
            lat_on_with_handoff.append(res_on["latency_sec"])
            if outcome == "improved":
                handoff_improved += 1
            elif outcome == "degraded":
                handoff_degraded += 1
        else:
            lat_on_no_handoff.append(res_on["latency_sec"])

        per_question.append(
            {
                "id": qid,
                "category": q.get("category", "unspecified"),
                "outcome": outcome,
                "composite_delta": delta,
                "handoff_triggered": triggered,
                "handoff_count": res_on["handoff_count"],
                "handoff_pairs": res_on["handoff_pairs"],
                "off": {**m_off, "latency_sec": round(res_off["latency_sec"], 4)},
                "on": {**m_on, "latency_sec": round(res_on["latency_sec"], 4)},
            }
        )

    def _avg(rows: List[Dict[str, float]], key: str) -> float:
        return round(sum(row[key] for row in rows) / len(rows), 4) if rows else 0.0

    return {
        "mode": mode,
        "parallel_retrieval": parallel_retrieval,
        "total_questions": len(questions),
        "handoff_off": {
            "citation_f1": _avg(off_metrics, "citation_f1"),
            "groundedness": _avg(off_metrics, "groundedness"),
            "abstention": _avg(off_metrics, "abstention"),
            "keyword_recall": _avg(off_metrics, "keyword_recall"),
            "composite": _avg(off_metrics, "composite"),
            "latency_p50_s": _percentile(lat_off_all, 50),
            "latency_p95_s": _percentile(lat_off_all, 95),
        },
        "handoff_on": {
            "citation_f1": _avg(on_metrics, "citation_f1"),
            "groundedness": _avg(on_metrics, "groundedness"),
            "abstention": _avg(on_metrics, "abstention"),
            "keyword_recall": _avg(on_metrics, "keyword_recall"),
            "composite": _avg(on_metrics, "composite"),
            "latency_p50_s": _percentile(lat_on_all, 50),
            "latency_p95_s": _percentile(lat_on_all, 95),
            "latency_no_handoff_p50_s": _percentile(lat_on_no_handoff, 50),
            "latency_no_handoff_p95_s": _percentile(lat_on_no_handoff, 95),
            "latency_with_handoff_p50_s": _percentile(lat_on_with_handoff, 50),
            "latency_with_handoff_p95_s": _percentile(lat_on_with_handoff, 95),
        },
        "outcomes": {
            "improved": improved,
            "degraded": degraded,
            "unchanged": unchanged,
            "handoff_triggered": handoff_triggered,
            "handoff_improved": handoff_improved,
            "handoff_degraded": handoff_degraded,
            "handoff_yield": round(handoff_improved / handoff_triggered, 4) if handoff_triggered else 0.0,
        },
        "per_question": per_question,
    }


def run_full_multi_agent(question: str, selected_doc: str, mode: str = "cpu_min") -> Dict[str, Any]:
    """Config D: Full Multi-Agent RAG - Two-Tier Dual Loops (Inner Critic + Outer Fact-Checker)."""
    pdf_paths = [str(ROOT_DIR / "uploads" / selected_doc)] if selected_doc else []
    return run_graph_variant(question, pdf_paths, mode=mode, max_reflection_loops=2)


def execute_ablation_matrix(mode: str = "cpu_min", num_samples: int = 8):
    print("\n" + "=" * 90)
    print(f" ABLATION STUDY MATRIX: Mode = '{mode}', num_samples = {num_samples}")
    print(f" Benchmarking 4 Architectural Configurations on Standardized Research Set")
    print("=" * 90)

    bench_file = ROOT_DIR / "eval" / "benchmark_120.jsonl"
    questions: List[Dict[str, Any]] = []
    with open(bench_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    sampled = sample_stratified_questions(questions, num_samples)

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
            selected_doc = q.get("selected_doc", "")
            is_unanswerable = bool(q.get("is_unanswerable", False)) or q.get("category") == "adversarial_trap"
            gold_keywords = q.get("gold_keywords", [])
            gold_sources = q.get("gold_sources")

            res = runner_fn(q["question"], selected_doc, mode=mode)
            ans = res["answer"]
            ev = res["retrieved_docs"]

            _, _, f1 = compute_strict_citation_f1(
                ans,
                ev,
                gold_keywords,
                is_unanswerable=is_unanswerable,
                gold_sources=gold_sources,
            )
            ground = compute_strict_groundedness(ans, ev, is_unanswerable)
            abstain = compute_strict_abstention_accuracy(ans, is_unanswerable)

            f1_list.append(f1)
            ground_list.append(ground)
            abstain_list.append(abstain)
            lat_list.append(res["latency_sec"])

        avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
        avg_ground = sum(ground_list) / len(ground_list) if ground_list else 0.0
        avg_abstain = sum(abstain_list) / len(abstain_list) if abstain_list else 0.0

        matrix_results[config_name] = {
            "sample_count": len(sampled),
            "strict_f1": avg_f1,
            "strict_groundedness": avg_ground,
            "strict_abstention": avg_abstain,
            "latency_p50_s": _percentile(lat_list, 50),
            "latency_p95_s": _percentile(lat_list, 95),
        }

    # Print Final Markdown Matrix
    print("\n" + "=" * 118)
    print(" SUMMARY ABLATION STUDY RESULTS MATRIX")
    print("=" * 118)
    print(f"{'Configuration':<42} | {'Strict Citation F1':<18} | {'Strict Groundedness':<19} | {'Strict Abstention':<17} | {'p50 / p95'}")
    print("-" * 118)
    for cfg, stats in matrix_results.items():
        print(
            f"{cfg:<42} | "
            f"{stats['strict_f1']*100:>16.1f}% | "
            f"{stats['strict_groundedness']*100:>17.1f}% | "
            f"{stats['strict_abstention']*100:>15.1f}% | "
            f"{stats['latency_p50_s']:>5.2f}s / {stats['latency_p95_s']:>5.2f}s"
        )
    print("=" * 118 + "\n")

    return matrix_results


if __name__ == "__main__":
    mode_target = sys.argv[1] if len(sys.argv) > 1 else "cpu_min"
    num_target = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    execute_ablation_matrix(mode_target, num_samples=num_target)

