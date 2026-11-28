# -*- coding: utf-8 -*-
"""
Dedicated Held-Out 3-Class Scientific NLI Evaluation Harness (`SUPPORT` | `CONTRADICT` | `NOT_ENOUGH_INFO`).

Evaluates Fact-Checker NLI engines on a disjoint 40-pair bilingual (EN + VI + cross-lingual)
scientific dataset (`main/eval/nli_holdout_40.json`) covering:
  - exact & paraphrased numeric/architectural support
  - numeric swaps, negation reversals, direction reversals, entity swaps, math script mismatches (`x^2` vs `x_2`)
  - unmentioned metrics / out-of-premise NOT_ENOUGH_INFO cases

Produces truthful per-class Precision/Recall/F1, Macro-F1, Accuracy, confusion matrix,
and latency telemetry (never claiming unmeasured F1 = 1.0).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.verifier import review_claims


LABELS: tuple[str, ...] = ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "nli_holdout_40.json"


def normalize_nli_label(raw: Any) -> str:
    val = str(raw or "").strip().upper().replace(" ", "_")
    if val in {"SUPPORT", "SUPPORTED", "SUPPORTS", "ENTAILMENT"}:
        return "SUPPORT"
    if val in {"CONTRADICT", "CONTRADICTED", "CONTRADICTION", "REFUTES", "REFUTED"}:
        return "CONTRADICT"
    return "NOT_ENOUGH_INFO"


def load_nli_holdout_dataset(path: Path | str | None = None) -> list[dict[str, Any]]:
    target = Path(path) if path else DEFAULT_DATASET_PATH
    with open(target, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list) or not data:
        raise ValueError(f"NLI holdout dataset at {target} is empty or invalid.")
    return data


def compute_multiclass_metrics(
    gold_labels: Sequence[str],
    pred_labels: Sequence[str],
    labels: Sequence[str] = LABELS,
) -> dict[str, Any]:
    if len(gold_labels) != len(pred_labels):
        raise ValueError("gold_labels and pred_labels must have equal length.")
    total = len(gold_labels)
    confusion: dict[str, dict[str, int]] = {
        g: {p: 0 for p in labels} for g in labels
    }
    correct = 0
    for g_raw, p_raw in zip(gold_labels, pred_labels):
        g = normalize_nli_label(g_raw)
        p = normalize_nli_label(p_raw)
        confusion[g][p] = confusion[g].get(p, 0) + 1
        if g == p:
            correct += 1

    per_class: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    for lbl in labels:
        tp = confusion[lbl][lbl]
        fp = sum(confusion[other][lbl] for other in labels if other != lbl)
        fn = sum(confusion[lbl][other] for other in labels if other != lbl)
        support = sum(confusion[lbl].values())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[lbl] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        if support > 0:
            f1_values.append(f1)

    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
    accuracy = correct / total if total else 0.0
    return {
        "sample_count": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def predict_with_local_deterministic_verifier(premise: str, claim: str) -> tuple[str, dict[str, Any]]:
    """Run the production `cpu_min` deterministic Fact-Checker verifier (`core.verifier.review_claims`)."""
    unit = {
        "id": "u_holdout_1",
        "text": premise,
        "doc_id": "holdout_doc",
        "page_no": 1,
    }
    state = {
        "query": claim,
        "answer": f"{claim} [1]",
        "source_units": [unit],
        "retrieved_docs": [unit],
        "evidence_spans": [{"id": "ev_1", "unit_id": "u_holdout_1", "quote": premise, "doc_id": "holdout_doc", "page_no": 1}],
        "claims": [{"id": "c_1", "text": claim, "evidence_ids": ["u_holdout_1"]}],
    }
    res = review_claims(state)
    label = normalize_nli_label(res.get("fact_check_label"))
    detail = ""
    rejected = res.get("rejected_claims") or []
    verified = res.get("verified_claims") or []
    if rejected and isinstance(rejected[0], dict):
        label = normalize_nli_label(rejected[0].get("status") or label)
        detail = str(rejected[0].get("reason") or "")
    elif verified and isinstance(verified[0], dict):
        detail = str(verified[0].get("reason") or "")
    return label, {"reason": detail[:180], "engine": "deterministic_guard"}


def predict_with_gemini_fact_checker(premise: str, claim: str) -> tuple[str, dict[str, Any]]:
    """Run the production `gemini` mode Fact-Checker (`GeminiFactCheckerAgent`: Gemini NLI + semantic/math/numeric guards)."""
    from core.config import settings
    from modes.gemini.fact_checker import GeminiFactCheckerAgent

    agent = GeminiFactCheckerAgent()
    unit = {
        "id": "u_holdout_1",
        "text": premise,
        "doc_id": "holdout_doc",
        "page_no": 1,
    }
    state = {
        "query": claim,
        "answer": f"{claim} [1]",
        "source_units": [unit],
        "retrieved_docs": [unit],
        "evidence_spans": [{"id": "ev_1", "unit_id": "u_holdout_1", "quote": premise, "doc_id": "holdout_doc", "page_no": 1}],
        "claims": [{"id": "c_1", "text": claim, "evidence_ids": ["u_holdout_1"]}],
    }
    res = agent.run(state)
    label = normalize_nli_label(res.get("fact_check_label"))
    detail = ""
    rejected = res.get("rejected_claims") or []
    verified = res.get("verified_claims") or []
    if rejected and isinstance(rejected[0], dict):
        label = normalize_nli_label(rejected[0].get("status") or label)
        detail = str(rejected[0].get("reason") or "")
    elif verified and isinstance(verified[0], dict):
        detail = str(verified[0].get("reason") or "")
    return label, {
        "reason": detail[:180],
        "served_model": str(getattr(settings, "DEFAULT_LLM_MODEL", "") or "gemini"),
        "engine": "gemini_fact_checker",
    }


def evaluate_gemini_fact_checker_batched(
    dataset: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 8,
    pause_seconds: float = 0.5,
) -> dict[str, Any]:
    """
    Evaluate the production `GeminiFactCheckerAgent` NLI + deterministic guard stack on the
    40-pair held-out dataset using batched calls (5 API calls for 40 pairs) to stay well within RPM limits.
    """
    from core.config import settings
    from core.verifier import (
        check_entity_value_binding,
        check_math_and_domain_consistency,
        check_numerical_and_date_consistency,
        check_sentence_entailment_guards,
    )
    from modes.gemini.fact_checker import GeminiFactCheckerAgent

    agent = GeminiFactCheckerAgent()
    gold: list[str] = []
    preds: list[str] = []
    latencies_ms: list[float] = []
    errors: list[dict[str, Any]] = []
    served_model = str(getattr(settings, "DEFAULT_LLM_MODEL", "") or "gemini")

    orig_cache = settings.LLM_CACHE_ENABLED
    settings.LLM_CACHE_ENABLED = False
    try:
        for start in range(0, len(dataset), max(1, int(batch_size))):
            chunk = dataset[start : start + max(1, int(batch_size))]
            pairs = [
                {
                    "claim_id": str(item.get("id") or f"c_{start + i}"),
                    "claim": str(item.get("claim") or ""),
                    "evidence": str(item.get("premise") or ""),
                }
                for i, item in enumerate(chunk)
            ]
            t0 = time.perf_counter()
            rows = agent._review(pairs)
            per_item_ms = ((time.perf_counter() - t0) * 1000.0) / max(1, len(chunk))
            by_id = {
                str(r.get("claim_id")): r
                for r in (rows or [])
                if isinstance(r, Mapping) and r.get("claim_id")
            }
            for item, pair in zip(chunk, pairs):
                gold_lbl = normalize_nli_label(item.get("gold_label"))
                row = by_id.get(pair["claim_id"], {})
                status = normalize_nli_label(row.get("status"))
                reason = str(row.get("rationale") or row.get("reason") or "")
                # Apply the same deterministic post-NLI guards used in `review_claims`
                if status == "SUPPORT":
                    sem_ok, sem_verdict, sem_reason = check_sentence_entailment_guards(
                        pair["claim"], pair["evidence"], qualitative_comparatives=False
                    )
                    math_ok, math_reason = check_math_and_domain_consistency(pair["claim"], pair["evidence"])
                    if not sem_ok:
                        status = normalize_nli_label(sem_verdict)
                        reason = f"semantic_guard:{sem_reason}"
                    elif not math_ok:
                        status = "CONTRADICT"
                        reason = f"math_guard:{math_reason}"
                    elif not all(check_numerical_and_date_consistency(pair["claim"], pair["evidence"])[:2]):
                        status = "NOT_ENOUGH_INFO"
                        reason = "numeric_guard_mismatch"
                    elif not check_entity_value_binding(pair["claim"], pair["evidence"])[0]:
                        status = "NOT_ENOUGH_INFO"
                        reason = "entity_binding_mismatch"
                gold.append(gold_lbl)
                preds.append(status)
                latencies_ms.append(per_item_ms)
                if status != gold_lbl:
                    errors.append({
                        "id": item.get("id"),
                        "language": item.get("language"),
                        "category": item.get("category"),
                        "gold": gold_lbl,
                        "pred": status,
                        "reason": reason[:180],
                        "claim": pair["claim"][:140],
                    })
            if start + batch_size < len(dataset) and pause_seconds > 0:
                time.sleep(pause_seconds)
    finally:
        settings.LLM_CACHE_ENABLED = orig_cache

    metrics = compute_multiclass_metrics(gold, preds)
    ordered_lat = sorted(latencies_ms)
    p50_ms = ordered_lat[len(ordered_lat) // 2] if ordered_lat else 0.0
    p95_ms = ordered_lat[min(len(ordered_lat) - 1, int(round(0.95 * (len(ordered_lat) - 1))))] if ordered_lat else 0.0
    by_language: dict[str, dict[str, Any]] = {}
    for lang in sorted({str(x.get("language") or "unknown") for x in dataset}):
        lang_gold = [g for x, g in zip(dataset, gold) if str(x.get("language") or "unknown") == lang]
        lang_pred = [p for x, p in zip(dataset, preds) if str(x.get("language") or "unknown") == lang]
        lm = compute_multiclass_metrics(lang_gold, lang_pred)
        by_language[lang] = {
            "sample_count": lm["sample_count"],
            "accuracy": lm["accuracy"],
            "macro_f1": lm["macro_f1"],
        }
    return {
        "engine": "gemini_fact_checker_batched",
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "served_models": [served_model],
        "served_revisions": [],
        "latency_p50_ms": round(p50_ms, 2),
        "latency_p95_ms": round(p95_ms, 2),
        "metrics": metrics,
        "by_language": by_language,
        "error_count": len(errors),
        "errors": errors,
    }


def predict_with_remote_kaggle_nli(
    premise: str,
    claim: str,
    *,
    remote_url: str,
    remote_token: str = "",
    timeout: float = 30.0,
) -> tuple[str, dict[str, Any]]:
    """Call a live Kaggle `/api/factcheck` NLI worker and return normalized label + served metadata."""
    import httpx

    headers = {"Authorization": f"Bearer {remote_token}"} if remote_token else {}
    resp = httpx.post(
        f"{remote_url.rstrip('/')}/api/factcheck",
        json={"claims": [claim], "context": premise, "adapter": "factcheck"},
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    evals = body.get("evaluations") or []
    first = evals[0] if evals and isinstance(evals[0], Mapping) else {}
    label = normalize_nli_label(first.get("verdict"))
    return label, {
        "confidence": float(first.get("confidence") or 0.0),
        "reason": str(first.get("reason") or "")[:180],
        "served_model": resp.headers.get("X-Research-Model-Name", ""),
        "served_revision": resp.headers.get("X-Research-Model-Revision", ""),
        "engine": "remote_kaggle_nli",
    }


def evaluate_nli_predictor(
    dataset: Sequence[Mapping[str, Any]],
    predictor_fn: Callable[[str, str], tuple[str, dict[str, Any]]],
    *,
    engine_name: str,
) -> dict[str, Any]:
    gold: list[str] = []
    preds: list[str] = []
    latencies_ms: list[float] = []
    errors: list[dict[str, Any]] = []
    served_models: set[str] = set()
    served_revisions: set[str] = set()

    for item in dataset:
        premise = str(item.get("premise") or "")
        claim = str(item.get("claim") or "")
        gold_lbl = normalize_nli_label(item.get("gold_label"))
        t0 = time.perf_counter()
        pred_lbl, meta = predictor_fn(premise, claim)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)
        pred_norm = normalize_nli_label(pred_lbl)
        gold.append(gold_lbl)
        preds.append(pred_norm)
        if meta.get("served_model"):
            served_models.add(str(meta["served_model"]))
        if meta.get("served_revision"):
            served_revisions.add(str(meta["served_revision"]))
        if pred_norm != gold_lbl:
            errors.append({
                "id": item.get("id"),
                "language": item.get("language"),
                "category": item.get("category"),
                "gold": gold_lbl,
                "pred": pred_norm,
                "reason": meta.get("reason", ""),
                "claim": claim[:140],
            })

    metrics = compute_multiclass_metrics(gold, preds)
    ordered_lat = sorted(latencies_ms)
    p50_ms = ordered_lat[len(ordered_lat) // 2] if ordered_lat else 0.0
    p95_ms = ordered_lat[min(len(ordered_lat) - 1, int(round(0.95 * (len(ordered_lat) - 1))))] if ordered_lat else 0.0

    by_language: dict[str, dict[str, Any]] = {}
    for lang in sorted({str(x.get("language") or "unknown") for x in dataset}):
        lang_gold = [g for x, g in zip(dataset, gold) if str(x.get("language") or "unknown") == lang]
        lang_pred = [p for x, p in zip(dataset, preds) if str(x.get("language") or "unknown") == lang]
        lm = compute_multiclass_metrics(lang_gold, lang_pred)
        by_language[lang] = {
            "sample_count": lm["sample_count"],
            "accuracy": lm["accuracy"],
            "macro_f1": lm["macro_f1"],
        }

    return {
        "engine": engine_name,
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "served_models": sorted(served_models),
        "served_revisions": sorted(served_revisions),
        "latency_p50_ms": round(p50_ms, 2),
        "latency_p95_ms": round(p95_ms, 2),
        "metrics": metrics,
        "by_language": by_language,
        "error_count": len(errors),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Fact-Checker NLI on disjoint held-out 40-pair set.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--include-gemini", action="store_true", help="Also evaluate production GeminiFactCheckerAgent (gemini mode).")
    parser.add_argument("--include-remote", action="store_true", help="Also evaluate live Kaggle /api/factcheck.")
    parser.add_argument("--remote-url", default="")
    parser.add_argument("--remote-token", default="")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    dataset = load_nli_holdout_dataset(args.dataset)
    results: dict[str, Any] = {
        "dataset": str(args.dataset),
        "sample_count": len(dataset),
        "engines": {},
    }

    local_res = evaluate_nli_predictor(
        dataset,
        predict_with_local_deterministic_verifier,
        engine_name="local_deterministic_verifier",
    )
    results["engines"]["local_deterministic_verifier"] = local_res

    if args.include_gemini:
        from core.config import settings

        if settings.GEMINI_API_KEY:
            try:
                gem_res = evaluate_gemini_fact_checker_batched(dataset, batch_size=8, pause_seconds=0.5)
                results["engines"]["gemini_fact_checker"] = gem_res
            except Exception as exc:
                results["engines"]["gemini_fact_checker"] = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            results["engines"]["gemini_fact_checker"] = {"error": "GEMINI_API_KEY not set"}

    if args.include_remote:
        from core.config import settings

        r_url = (args.remote_url or settings.REMOTE_INFERENCE_URL or "").strip()
        r_tok = (args.remote_token or settings.REMOTE_INFERENCE_TOKEN or "").strip()
        if r_url:
            try:
                remote_res = evaluate_nli_predictor(
                    dataset,
                    lambda p, c: predict_with_remote_kaggle_nli(p, c, remote_url=r_url, remote_token=r_tok),
                    engine_name="remote_kaggle_nli",
                )
                results["engines"]["remote_kaggle_nli"] = remote_res
            except Exception as exc:
                results["engines"]["remote_kaggle_nli"] = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            results["engines"]["remote_kaggle_nli"] = {"error": "REMOTE_INFERENCE_URL not set"}

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Held-out NLI Evaluation (n={len(dataset)}, dataset={Path(args.dataset).name})")
    for eng, report in results["engines"].items():
        if "error" in report:
            print(f"  - {eng}: ERROR ({report['error']})")
            continue
        m = report["metrics"]
        print(
            f"  - {eng}: Accuracy={m['accuracy']*100:.1f}%, Macro-F1={m['macro_f1']:.4f}, "
            f"p50={report['latency_p50_ms']:.2f}ms, errors={report['error_count']}/{m['sample_count']}"
        )
        for lbl in LABELS:
            pc = m["per_class"][lbl]
            print(
                f"      {lbl:<16} P={pc['precision']:.3f} R={pc['recall']:.3f} "
                f"F1={pc['f1']:.3f} (n={pc['support']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
