# -*- coding: utf-8 -*-
"""Evaluate the production graph on a fixed, document-scoped question set."""

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import fitz

ROOT_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.config import RunConfig, settings, use_run_config
from core.token_usage import TOKEN_USAGE_ACTION, capture_token_usage
from core.tools import clear_pdf_cache, load_or_parse_selected_pdf
from core.worker_observations import (
    PROVIDER_INFERENCE_ACTION,
    WORKER_INFERENCE_ACTION,
    capture_worker_observations,
    drain_worker_observations,
)
from eval.metrics import evaluate_single_prediction
from modes.registry import MODE_MAP, normalize_mode

ALL_MODES = list(dict.fromkeys(normalize_mode(mode) for mode in MODE_MAP))
CACHE_REGIMES = ("cold", "warm_no_cache", "warm_cached")
LEGACY_DATASET = EVAL_DIR / "benchmark_120.jsonl"
PILOT_DATASET = EVAL_DIR / "benchmark_pilot.json"


def sample_by_category(questions: list[dict], limit: int) -> list[dict]:
    if not limit:
        return questions
    counts: dict[str, int] = {}
    sampled = []
    for question in questions:
        category = question.get("category", "single_hop")
        count = counts.get(category, 0)
        if count < limit:
            sampled.append(question)
            counts[category] = count + 1
    return sampled


def load_benchmark(limit_per_cat: int = 0, dataset: Path = LEGACY_DATASET) -> list[dict]:
    lines = dataset.read_text(encoding="utf-8").splitlines()
    return sample_by_category([json.loads(line) for line in lines if line.strip()], limit_per_cat)


def load_pilot(folder: Path) -> list[dict]:
    fixture = json.loads(PILOT_DATASET.read_text(encoding="utf-8"))
    for filename, pages in fixture["documents"].items():
        with fitz.open() as document:
            for content in pages:
                page = document.new_page()
                for i, line in enumerate(content):
                    page.insert_text((60, 75 + 20 * i), line, fontname="cour", fontsize=11)
            document.save(folder / filename, no_new_id=True)
    return fixture["questions"]


def question_documents(question: dict) -> list[str]:
    selected = question.get("documents") or [question["selected_doc"]]
    return list(dict.fromkeys(selected))


def validate_documents(questions: list[dict], folder: Path) -> dict[str, str]:
    """Verify the selected files and the physical PDF pages labelled in the dataset."""
    hashes = {}
    page_text: dict[str, list[str]] = {}
    seen_ids = set()
    root = folder.resolve()
    for question in questions:
        question_id = question["id"]
        if question_id in seen_ids:
            raise ValueError(f"Duplicate benchmark question id: {question_id}")
        seen_ids.add(question_id)
        selected = question_documents(question)
        for filename in selected:
            path = (root / filename).resolve()
            if not path.is_relative_to(root) or path.suffix.lower() != ".pdf" or not path.is_file():
                raise ValueError(f"{question_id}: selected PDF is unavailable in {root}: {filename}")
            if filename not in hashes:
                with path.open("rb") as pdf:
                    hashes[filename] = hashlib.file_digest(pdf, "sha256").hexdigest()
                with fitz.open(path) as document:
                    page_text[filename] = [" ".join(page.get_text().split()) for page in document]
        labels = question.get("gold_sources", [])
        if not labels and question.get("gold_page"):
            labels = [{"document": question["selected_doc"], "page": question["gold_page"]}]
        for label in labels:
            filename = label["document"]
            page = label["page"]
            if filename not in selected or not isinstance(page, int) or not 1 <= page <= len(page_text[filename]):
                raise ValueError(f"{question_id}: invalid labelled document or page: {filename}, {page}")
            passage = label.get("passage")
            if passage and " ".join(passage.split()) not in page_text[filename][page - 1]:
                raise ValueError(f"{question_id}: gold passage is absent from {filename}, page {page}")
    return hashes


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    def average(key: str) -> float | None:
        values = [row[key] for row in records if row.get(key) is not None]
        return round(statistics.mean(values), 4) if values else None

    latencies = sorted(row["latency_sec"] for row in records)
    return {
        "scored_questions": len(records),
        "groundedness": average("groundedness"),
        "hallucination_rate": average("hallucination_rate"),
        "retrieval_recall": average("retrieval_recall"),
        "citation_f1": average("citation_f1"),
        "citation_provenance_rate": average("citation_provenance_rate"),
        "valid_citation_ratio": average("valid_citation_ratio"),
        "orphan_citation_total": sum(int(row.get("orphan_citation_count", 0)) for row in records),
        "unit_mismatch_total": sum(int(row.get("unit_mismatch_count", 0)) for row in records),
        "numeric_exactness": average("numeric_exactness"),
        "abstention_accuracy": average("abstention_accuracy"),
        "p50_latency_s": round(statistics.median(latencies), 3) if latencies else None,
        "p95_latency_s": latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,
        "avg_retries": average("retry_count"),
    }


def _extract_observation_summary(trace: list[dict[str, Any]]) -> dict[str, Any]:
    worker_calls = [e for e in trace if isinstance(e, dict) and e.get("action") == WORKER_INFERENCE_ACTION]
    provider_calls = [e for e in trace if isinstance(e, dict) and e.get("action") == PROVIDER_INFERENCE_ACTION]
    token_calls = [e for e in trace if isinstance(e, dict) and e.get("action") == TOKEN_USAGE_ACTION]

    served_models: list[str] = []
    for ev in worker_calls:
        if ev.get("model"):
            rev = ev.get("revision") or "unpinned"
            served_models.append(f"kaggle:{ev['model']}@{rev[:12]}")
    for ev in provider_calls:
        if ev.get("model_version") or ev.get("served_model"):
            ver = ev.get("model_version") or ev.get("served_model")
            served_models.append(f"{ev.get('provider', 'cloud')}:{ver}")

    used_fallback = any(
        ev.get("status") in {"fallback", "error", "unavailable", "circuit_open", "http_error", "transport_error"}
        or ev.get("provider") == "heuristic_fallback"
        for ev in [*worker_calls, *provider_calls]
    )
    cache_hits = sum(
        1 for ev in [*worker_calls, *provider_calls] if ev.get("status") == "cache_hit"
    )
    reported_tokens = sum(
        int(ev.get("total_tokens", 0)) for ev in token_calls if ev.get("status") == "reported"
    )
    return {
        "served_models": list(dict.fromkeys(served_models)),
        "used_fallback": used_fallback,
        "cache_hits": cache_hits,
        "reported_tokens": reported_tokens,
        "worker_observations": [
            {k: v for k, v in ev.items() if k != "action"} for ev in worker_calls
        ],
        "provider_observations": [
            {k: v for k, v in ev.items() if k != "action"} for ev in provider_calls
        ],
    }


def run_mode_evaluation(
    mode: str,
    questions: list[dict],
    documents_dir: Path = ROOT_DIR / "uploads",
    *,
    cache_regime: str = "warm_no_cache",
) -> dict[str, Any]:
    from core.graph import build_research_rag_graph

    if cache_regime not in CACHE_REGIMES:
        raise ValueError(f"Unsupported cache_regime: {cache_regime!r}")

    cache_enabled = cache_regime == "warm_cached"
    run_cfg = RunConfig.capture(llm_cache_enabled=cache_enabled)

    if cache_regime == "cold":
        clear_pdf_cache()
    else:
        # Warm parsed PDF chunks into memory prior to timed question loop
        clear_pdf_cache()
        for question in questions:
            for name in question_documents(question):
                load_or_parse_selected_pdf(str((documents_dir / name).resolve()))

    app = build_research_rag_graph(mode, run_config=run_cfg, auto_scout=False)

    # In warm_cached regime, execute one untimed priming pass so genuine model answers populate SQLite cache
    if cache_regime == "warm_cached":
        for question in questions:
            paths = [str((documents_dir / name).resolve()) for name in question_documents(question)]
            prime_state = {
                "query": question["question"],
                "pdf_path": paths[0],
                "pdf_paths": paths,
                "retry_count": 0,
            }
            with use_run_config(run_cfg), capture_worker_observations(), capture_token_usage(None):
                try:
                    app.invoke(prime_state)
                except Exception:
                    pass

    records = []
    errors = []
    for question in questions:
        if cache_regime == "cold":
            clear_pdf_cache()
            app = build_research_rag_graph(mode, run_config=run_cfg, auto_scout=False)
        paths = [str((documents_dir / name).resolve()) for name in question_documents(question)]
        initial_state = {
            "query": question["question"],
            "pdf_path": paths[0],
            "pdf_paths": paths,
            "retry_count": 0,
        }
        start = time.perf_counter()
        try:
            with (
                use_run_config(run_cfg),
                capture_worker_observations(),
                capture_token_usage(None) as ledger,
            ):
                result = app.invoke(initial_state)
                obs_events = drain_worker_observations()
                tok_events = ledger.drain()
        except Exception as exc:
            errors.append({"id": question["id"], "error": type(exc).__name__})
            continue
        elapsed = time.perf_counter() - start
        combined_trace = [*(result.get("reasoning_trace") or []), *obs_events, *tok_events]
        result = {**result, "reasoning_trace": combined_trace}
        scored = evaluate_single_prediction(question, result, elapsed)
        scored["cache_regime"] = cache_regime
        scored["execution_observation"] = _extract_observation_summary(combined_trace)
        records.append(scored)

    overall = _summarize_records(records)
    clean_records = [r for r in records if not (r.get("execution_observation") or {}).get("used_fallback")]
    degraded_records = [r for r in records if (r.get("execution_observation") or {}).get("used_fallback")]
    observed_models = sorted({
        m
        for r in records
        for m in (r.get("execution_observation") or {}).get("served_models", [])
    })
    by_category = {
        cat: _summarize_records([row for row in records if row.get("category") == cat])
        for cat in sorted({str(row.get("category") or "unknown") for row in records})
    }
    by_language = {
        lang: _summarize_records([row for row in records if row.get("language") == lang])
        for lang in sorted({str(row.get("language") or "unknown") for row in records})
    }
    return {
        "mode": mode,
        "cache_regime": cache_regime,
        "num_questions": len(questions),
        **overall,
        "observed_models": observed_models,
        "fallback_question_count": len(degraded_records),
        "cache_hit_total": sum(int((r.get("execution_observation") or {}).get("cache_hits", 0)) for r in records),
        "reported_tokens_total": sum(int((r.get("execution_observation") or {}).get("reported_tokens", 0)) for r in records),
        "by_execution_path": {
            "clean_primary": _summarize_records(clean_records) if clean_records else None,
            "degraded_or_fallback": _summarize_records(degraded_records) if degraded_records else None,
        },
        "by_category": by_category,
        "by_language": by_language,
        "records": records,
        "errors": errors,
    }


def write_results(output: Path, config: dict, summaries: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    result = {"config": config, "summaries": summaries}
    (output / "benchmark_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    headings = [
        "Mode", "Scored", "Errors", "Retrieval recall", "Citation F1",
        "Citation provenance", "Groundedness", "Hallucination", "Numeric exactness",
        "Abstention accuracy", "Orphan cites", "Unit mismatches", "p50 (s)", "p95 (s)",
    ]
    lines = ["# Multi-Agent RAG benchmark", "", "| " + " | ".join(headings) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headings)) + " |")

    def percentage(item: dict, key: str) -> str:
        score = item.get(key)
        return f"{score:.1%}" if score is not None else "—"

    for summary in summaries:
        cells = [
            summary["mode"], f"{summary['scored_questions']}/{summary['num_questions']}",
            str(len(summary["errors"])), percentage(summary, "retrieval_recall"),
            percentage(summary, "citation_f1"), percentage(summary, "citation_provenance_rate"),
            percentage(summary, "groundedness"), percentage(summary, "hallucination_rate"),
            percentage(summary, "numeric_exactness"), percentage(summary, "abstention_accuracy"),
            str(summary.get("orphan_citation_total", 0)), str(summary.get("unit_mismatch_total", 0)),
            str(summary["p50_latency_s"]), str(summary["p95_latency_s"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    for summary in summaries:
        groups = {
            **{f"category:{k}": v for k, v in (summary.get("by_category") or {}).items()},
            **{f"language:{k}": v for k, v in (summary.get("by_language") or {}).items()},
        }
        if not groups:
            continue
        lines.extend([
            "",
            f"## Breakdown — `{summary['mode']}`",
            "",
            "| Group | Scored | Retrieval recall | Citation F1 | Citation provenance | Groundedness | Hallucination | Numeric exactness | Abstention | Orphan cites | Unit mismatches |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for name, item in groups.items():
            lines.append("| " + " | ".join([
                name,
                str(item["scored_questions"]),
                percentage(item, "retrieval_recall"),
                percentage(item, "citation_f1"),
                percentage(item, "citation_provenance_rate"),
                percentage(item, "groundedness"),
                percentage(item, "hallucination_rate"),
                percentage(item, "numeric_exactness"),
                percentage(item, "abstention_accuracy"),
                str(item.get("orphan_citation_total", 0)),
                str(item.get("unit_mismatch_total", 0)),
            ]) + " |")
    (output / "benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", type=normalize_mode, choices=ALL_MODES, default=["cpu_min"])
    parser.add_argument(
        "--cache-regimes",
        nargs="+",
        choices=CACHE_REGIMES,
        default=["warm_no_cache"],
        help="Cache regimes to evaluate separately (cold, warm_no_cache, warm_cached)",
    )
    parser.add_argument("--live", action="store_true", help="Allow modes that can access configured model providers")
    datasets = parser.add_mutually_exclusive_group()
    datasets.add_argument("--pilot", action="store_true", help="Use the self-contained PDF pilot")
    datasets.add_argument("--dataset", type=Path, help="JSONL question set (default: benchmark_120.jsonl)")
    parser.add_argument("--documents-dir", type=Path, default=ROOT_DIR / "uploads")
    parser.add_argument("--limit-per-cat", type=int, default=2, help="Questions per category (0: all)")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "artifacts" / "benchmark")
    args = parser.parse_args()
    if args.limit_per_cat < 0:
        parser.error("--limit-per-cat must be zero or positive")
    if not args.live and any(mode != "cpu_min" for mode in args.modes):
        parser.error("Modes other than cpu_min require --live")

    dataset = PILOT_DATASET if args.pilot else args.dataset or LEGACY_DATASET
    with TemporaryDirectory(prefix="rag-benchmark-") as temp:
        folder = Path(temp) if args.pilot else args.documents_dir
        try:
            questions = (
                sample_by_category(load_pilot(folder), args.limit_per_cat)
                if args.pilot else load_benchmark(args.limit_per_cat, dataset)
            )
            if not questions:
                raise ValueError("The benchmark has no questions")
            hashes = validate_documents(questions, folder)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        config = {
            "dataset": dataset.name,
            # Hash the UTF-8 dataset with normalized line endings across Git checkouts.
            "dataset_sha256": hashlib.sha256(dataset.read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
            "document_sha256": hashes,
            "modes": args.modes,
            "cache_regimes": list(dict.fromkeys(args.cache_regimes)),
            "software": {"python": platform.python_version(), "pymupdf": fitz.VersionBind},
            "limit_per_category": args.limit_per_cat,
            "live": args.live,
            "retrieval_limit": settings.MAX_RETRIEVED_CHUNKS,
            "max_reflection_loops": settings.MAX_ORCHESTRATOR_RETRY,
        }
        # Isolate import-time database initialization and create a dedicated SQLite DB per (mode, cache_regime)
        # so multi-mode runs (e.g., gemini followed by hybrid) can never cross-contaminate via llm_cache.
        original_base_dir = settings.BASE_DIR
        existing_db = sys.modules.get("core.db")
        original_db_path = existing_db.DB_PATH if existing_db else str(Path(original_base_dir) / "data" / "workspace.db")
        original_db_init = existing_db._db_initialized if existing_db else False
        settings.BASE_DIR = temp
        try:
            from core import db
        finally:
            settings.BASE_DIR = original_base_dir

        original_corpus = settings.CORPUS_PATH
        settings.CORPUS_PATH = str(Path(temp) / "empty_corpus.jsonl")
        summaries = []
        try:
            for regime in dict.fromkeys(args.cache_regimes):
                for mode in args.modes:
                    isolated_db = str(Path(temp) / "data" / f"workspace_{mode}_{regime}.db")
                    db.DB_PATH = isolated_db
                    db._db_initialized = False
                    db.init_db()
                    summaries.append(
                        run_mode_evaluation(mode, questions, folder, cache_regime=regime)
                    )
        finally:
            db.DB_PATH = original_db_path
            db._db_initialized = original_db_init
            settings.CORPUS_PATH = original_corpus

    write_results(args.output_dir, config, summaries)
    print(f"Saved benchmark results to {args.output_dir}")
    return int(any(summary["errors"] for summary in summaries))


if __name__ == "__main__":
    sys.exit(main())
