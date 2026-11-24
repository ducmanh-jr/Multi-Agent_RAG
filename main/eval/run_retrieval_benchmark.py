"""Compare Sparse, Hybrid and Hybrid with reranking on fixed academic PDFs."""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.request import Request, urlopen

import fitz

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from agents.agent1_retrieval.tools.remote_dense import DEFAULT_DENSE_RRF_K, DEFAULT_DENSE_WEIGHT
from core.config import RunConfig, settings, use_run_config
from core.tools import clear_pdf_cache
from eval.retrieval_metrics import CANDIDATE_LIMIT, METRICS, score_ranked_pages

DATA_DIR = Path(__file__).with_name("retrieval_benchmark")
SPLITS = ("dev", "holdout")
CATEGORIES = (
    "single_document", "multi_document", "vietnamese_english", "numeric",
    "table", "figure", "multi_hop", "insufficient_evidence",
)
MAX_PDF_BYTES = 20_000_000
Retrieve = Callable[[str, tuple[str, ...]], tuple[Sequence[dict[str, Any]], dict[str, Any]]]


def _text_hash(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load_labels(dataset_dir: Path = DATA_DIR) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != 1:
        raise ValueError("Unsupported benchmark manifest version")
    splits = manifest.get("splits", list(SPLITS))
    if (not isinstance(splits, list) or not splits or any(split not in SPLITS for split in splits)
            or len(splits) != len(set(splits))):
        raise ValueError("Invalid benchmark splits")
    categories = manifest.get("categories", list(CATEGORIES))
    if (not isinstance(categories, list) or not categories
            or any(category not in CATEGORIES for category in categories)
            or len(categories) != len(set(categories))):
        raise ValueError("Invalid benchmark categories")

    documents = manifest["documents"]
    keys = set()
    inputs = set()
    by_split = {}
    for name, doc in documents.items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*\.pdf", name):
            raise ValueError(f"Invalid PDF name: {name}")
        if not re.fullmatch(r"https://arxiv\.org/pdf/\d{4}\.\d{5}v\d+", doc["url"]):
            raise ValueError(f"Unversioned or unsupported PDF URL: {name}")
        if doc["split"] not in splits or not re.fullmatch(r"[\da-f]{64}", doc["sha256"]):
            raise ValueError(f"Invalid PDF metadata: {name}")
        if type(doc["pages"]) is not int or doc["pages"] < 1:
            raise ValueError(f"Invalid PDF page count: {name}")
    if len({doc["sha256"] for doc in documents.values()}) != len(documents):
        raise ValueError("A PDF is duplicated in the corpus manifest")

    for split in splits:
        lines = (dataset_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        cases = [json.loads(line) for line in lines if line.strip()]
        for case in cases:
            case_id = case["id"]
            prompt = _normalize(case["question"])
            if case_id in keys or not prompt or prompt in inputs or not case["gold_answer"].strip():
                raise ValueError(f"Duplicate or incomplete question: {case_id}")
            keys.add(case_id)
            inputs.add(prompt)
            scope = case["documents"]
            if (not scope or len(scope) != len(set(scope))
                    or any(doc not in documents or documents[doc]["split"] != split for doc in scope)):
                raise ValueError(f"Invalid document scope for {case_id}; splits must be disjoint")
            if case["category"] not in categories or case["language"] not in {"en", "vi"}:
                raise ValueError(f"Invalid category or language for {case_id}")
            if case["category"] == "vietnamese_english" and case["language"] != "vi":
                raise ValueError(f"Cross-language question must be Vietnamese: {case_id}")
            targets = case["gold_sources"]
            unanswerable = case["category"] == "insufficient_evidence"
            if unanswerable != (not targets):
                raise ValueError(f"Answerability and evidence disagree for {case_id}")
            locations = set()
            for label in targets:
                location = (label["document"], label["page"])
                if (location in locations or location[0] not in scope
                        or type(location[1]) is not int
                        or not 1 <= location[1] <= documents[location[0]]["pages"]
                        or type(label["grade"]) is not int or label["grade"] not in {1, 2}
                        or not label["anchors"] or any(not _normalize(anchor) for anchor in label["anchors"])):
                    raise ValueError(f"Invalid evidence label for {case_id}: {location}")
                locations.add(location)
            if targets and not any(label["grade"] == 2 for label in targets):
                raise ValueError(f"A question needs at least one direct evidence page: {case_id}")
            if case["category"] == "multi_document" and len({doc for doc, _ in locations}) < 2:
                raise ValueError(f"A cross-document question must cite multiple PDFs: {case_id}")
            if case["category"] == "multi_hop" and len(locations) < 2:
                raise ValueError(f"A multi-hop question must cite multiple pages: {case_id}")
            if case["category"] in {"figure", "table"} and not any(label.get("region") for label in targets):
                raise ValueError(f"A table or figure question needs a region label: {case_id}")
        counts = Counter(case["category"] for case in cases)
        minimum = manifest["min_questions_per_category"]
        if any(counts[category] < minimum for category in categories):
            raise ValueError(f"Incomplete category coverage in {split}")
        by_split[split] = cases
    return manifest, by_split


def prepare_documents(
    corpus_dir: Path, documents: dict[str, Any], cases: list[dict[str, Any]], *, fetch: bool = False,
) -> dict[str, str]:
    """Verify exact PDF bytes and that each annotated anchor is on the cited PDF page."""
    root = corpus_dir.resolve()
    page_text = {}
    hashes = {}
    for name in sorted({doc for case in cases for doc in case["documents"]}):
        path = root / name
        expected = documents[name]["sha256"]
        if fetch and not path.exists():
            request = Request(documents[name]["url"], headers={"User-Agent": "ResearchV2Benchmark/1.0"})
            with urlopen(request, timeout=30) as response:
                data = response.read(MAX_PDF_BYTES + 1)
            if len(data) > MAX_PDF_BYTES or not data.startswith(b"%PDF-") or hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Downloaded PDF does not match the manifest: {name}")
            root.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        try:
            with path.open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
        except FileNotFoundError as exc:
            raise ValueError(f"Missing versioned PDF {path}; use --fetch to download missing files") from exc
        if digest != expected:
            raise ValueError(f"PDF checksum mismatch: {path}")
        hashes[name] = digest
        with fitz.open(path) as doc:
            if len(doc) != documents[name]["pages"]:
                raise ValueError(f"PDF page count mismatch: {path}")
            page_text[name] = [_normalize(page.get_text()) for page in doc]

    for case in cases:
        for label in case["gold_sources"]:
            pages = page_text[label["document"]]
            page = label["page"]
            if page > len(pages) or any(_normalize(anchor) not in pages[page - 1] for anchor in label["anchors"]):
                raise ValueError(f"{case['id']}: evidence anchor missing on {label['document']} page {page}: {label['anchors']}")
    return hashes


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = sorted(record["latency_ms"] for record in records)
    result = {
        "questions": len(records),
        "ranked_questions": sum(record["answerable"] for record in records),
        "max_returned_passages": max(len(record["ranking"]) for record in records),
        "dense_fallbacks": sum(record["dense"]["status"] == "fallback" for record in records),
        "dense_only_passages": sum(record["dense"].get("admitted_passages", 0) for record in records),
        "rerank_fallbacks": sum(record["rerank"]["status"] == "fallback" for record in records),
        "rerank_changed": sum(record["rerank"].get("changed", False) for record in records),
        "p95_ms": round(latencies[math.ceil(0.95 * len(latencies)) - 1], 3),
    }
    for metric in METRICS:
        values = [record[metric] for record in records if record[metric] is not None]
        result[metric] = round(statistics.mean(values), 6) if values else None
    return result


def evaluate_ranker(cases: list[dict[str, Any]], retrieve: Retrieve) -> dict[str, Any]:
    records = []
    for case in cases:
        scope = {Path(document).stem.casefold(): document for document in case["documents"]}
        started = time.perf_counter()
        chunks, trace = retrieve(case["question"], tuple(case["documents"]))
        latency = (time.perf_counter() - started) * 1000
        ranked = []
        for chunk in chunks[:CANDIDATE_LIMIT]:
            document = scope.get(Path(str(chunk.get("doc_id") or "")).name.casefold().removesuffix(".pdf"))
            page = chunk.get("page_no") or chunk.get("page")
            if document is None or type(page) is not int or page < 1:
                raise ValueError(f"{case['id']}: retriever returned an invalid or out-of-scope source")
            ranked.append((document, page))
        relevant = {(label["document"], label["page"]): label["grade"] for label in case["gold_sources"]}
        records.append({
            "id": case["id"], "category": case["category"], "language": case["language"],
            "answerable": bool(relevant), "ranking": ranked, "latency_ms": round(latency, 3),
            "dense": trace["dense"], "rerank": trace["rerank"],
            "passage_ids": [chunk.get("id") for chunk in chunks[:CANDIDATE_LIMIT]],
            **score_ranked_pages(ranked, relevant),
        })
    return {
        "overall": _summarize(records),
        "by_category": {
            category: _summarize([record for record in records if record["category"] == category])
            for category in sorted({record["category"] for record in records})
        },
        "by_language": {
            language: _summarize([record for record in records if record["language"] == language])
            for language in sorted({record["language"] for record in records})
        },
        "records": records,
    }


def run_retrieval(
    cases_by_split: dict[str, list[dict[str, Any]]], corpus_dir: Path, *,
    worker_url: str, dense_weight: float, dense_rrf_k: int, page_diverse: bool = False,
    enable_rerank: bool = False, rerank_keep_prefix: int = 0,
    remote_inference_token: str = "",
) -> dict[str, Any]:
    with TemporaryDirectory(prefix="retrieval-baseline-") as directory:
        # The LLM client imports the database module even for a purely local run.
        # Keep its first import from opening or initializing the workspace DB.
        original_base_dir = settings.BASE_DIR
        db_already_imported = "core.db" in sys.modules
        try:
            settings.BASE_DIR = directory
            from modes.gpu_finetune.retrieval import GPUFinetuneRetrievalAgent
        finally:
            settings.BASE_DIR = original_base_dir
            if not db_already_imported and (db := sys.modules.get("core.db")):
                db.DB_PATH = str(Path(original_base_dir) / "data" / "workspace.db")
                db._db_initialized = False

        config = RunConfig.capture(remote_inference_url=worker_url, corpus_path=str(Path(directory) / "empty.jsonl"))
        original_token = settings.REMOTE_INFERENCE_TOKEN
        token_override = remote_inference_token.strip()
        if token_override:
            settings.REMOTE_INFERENCE_TOKEN = token_override
        try:
            with use_run_config(config):
                def retrieve(question: str, documents: tuple[str, ...]):
                    paths = [str(corpus_dir / name) for name in documents]
                    state = {"query": question, "pdf_path": paths[0], "pdf_paths": paths, "top_k": CANDIDATE_LIMIT}
                    result = agent.run(state)
                    return result["retrieved_docs"], result["reasoning_trace"][0]["details"]

                results = {}
                for split, cases in cases_by_split.items():
                    agent = GPUFinetuneRetrievalAgent(
                        enable_dense=bool(worker_url), enable_remote_router=False,
                        enable_rerank=enable_rerank, dense_weight=dense_weight, dense_rrf_k=dense_rrf_k,
                        page_diverse=page_diverse, rerank_keep_prefix=rerank_keep_prefix,
                    )
                    clear_pdf_cache()
                    results[split] = evaluate_ranker(cases, retrieve)
                return results
        finally:
            settings.REMOTE_INFERENCE_TOKEN = original_token


def _write_results(output: Path, report: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "retrieval_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# {report['variant'].title()} retrieval benchmark", "",
        f"Requested passages: {report['candidate_limit']}. Per-branch shortlist limit: "
        f"{report['branch_candidate_limit']}. RRF output limit before selection: {report['rrf_output_limit']}.", "",
        "| Split | Group | Questions | Ranked | Max returned | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Dense-only passages | Dense fallbacks | Rerank fallbacks | Rerank changed | p95 (ms) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not report["complete"]:
        lines[4:4] = ["Comparison incomplete. See checkpoint, skip and fallback details in the JSON report.", ""]
    for split, result in report["splits"].items():
        groups = {
            "overall": result["overall"],
            **result["by_category"],
            **{f"language {language}": values for language, values in result["by_language"].items()},
        }
        for name, item in groups.items():
            scores = ["–" if item[key] is None else f"{item[key]:.4f}" for key in METRICS]
            cells = [
                split, name, str(item["questions"]), str(item["ranked_questions"]),
                str(item["max_returned_passages"]), *scores, str(item["dense_only_passages"]),
                str(item["dense_fallbacks"]), str(item["rerank_fallbacks"]),
                str(item["rerank_changed"]), f"{item['p95_ms']:.2f}",
            ]
            lines.append("| " + " | ".join(cells) + " |")
    (output / "retrieval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=(*SPLITS, "both"), help="Defaults to dev when available")
    parser.add_argument("--dataset-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--strategy", choices=("sparse", "hybrid", "hybrid-rerank"), default="sparse")
    parser.add_argument("--embedding-worker-url", default="")
    parser.add_argument("--remote-inference-token", default="", help="Bearer token for authenticated Kaggle gateway (defaults to REMOTE_INFERENCE_TOKEN env)")
    parser.add_argument("--dense-weight", type=float, default=DEFAULT_DENSE_WEIGHT, help="Dense list weight in the second RRF pass")
    parser.add_argument("--dense-rrf-k", type=int, default=DEFAULT_DENSE_RRF_K, help="Rank offset in the second RRF pass")
    parser.add_argument("--rerank-keep-prefix", type=int, default=0, help="Number of first Hybrid ranks to preserve")
    parser.add_argument("--page-diverse", action="store_true", help="Use deeper shortlists and prefer one hit per PDF page")
    parser.add_argument("--documents-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true", help="Download missing, versioned public PDFs from arXiv")
    args = parser.parse_args()
    if args.strategy != "sparse" and not args.embedding_worker_url.strip():
        parser.error("Hybrid strategies require --embedding-worker-url")
    if not math.isfinite(args.dense_weight) or args.dense_weight <= 0:
        parser.error("--dense-weight must be positive and finite")
    if args.dense_rrf_k < 0:
        parser.error("--dense-rrf-k must be nonnegative")
    if not 0 <= args.rerank_keep_prefix <= CANDIDATE_LIMIT - 2:
        parser.error("--rerank-keep-prefix must leave at least two positions for reranking")
    worker_url = args.embedding_worker_url.strip() if args.strategy != "sparse" else ""
    enable_rerank = args.strategy == "hybrid-rerank"
    try:
        manifest, labels = load_labels(args.dataset_dir)
        if args.split and args.split != "both" and args.split not in labels:
            raise ValueError(f"Dataset {manifest['name']} does not contain split {args.split}")
        default_split = "dev" if "dev" in labels else "holdout"
        splits = tuple(labels) if args.split == "both" else (args.split or default_split,)
        cases = {split: labels[split] for split in splits}
        root = args.documents_dir.resolve()
        hashes = prepare_documents(root, manifest["documents"], [case for group in cases.values() for case in group], fetch=args.fetch)
        results = run_retrieval(
            cases, root, worker_url=worker_url, dense_weight=args.dense_weight, dense_rrf_k=args.dense_rrf_k,
            page_diverse=args.page_diverse, enable_rerank=enable_rerank, rerank_keep_prefix=args.rerank_keep_prefix,
            remote_inference_token=args.remote_inference_token or os.getenv("REMOTE_INFERENCE_TOKEN", ""),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    records = [record for split in results.values() for record in split["records"]]

    def identities(stage: str) -> set[tuple[str, str]]:
        return {(record[stage]["model"], record[stage]["revision"])
                for record in records if record[stage]["status"] == "used"}

    embedding_checkpoints = identities("dense")
    reranker_checkpoints = identities("rerank")
    complete = not worker_url or (len(embedding_checkpoints) == 1 and all(record["dense"]["status"] == "used" for record in records))
    if enable_rerank:
        complete = complete and len(reranker_checkpoints) == 1 and all(record["rerank"]["status"] == "used" for record in records)
    branch_limit = 2 * CANDIDATE_LIMIT if args.page_diverse else CANDIDATE_LIMIT
    report = {
        "benchmark": manifest["name"],
        "variant": args.strategy,
        "strategy": f"gpu_finetune_{args.strategy}_fts5_bm25_rrf",
        "dense_weight": args.dense_weight if worker_url else None,
        "dense_rrf_k": args.dense_rrf_k if worker_url else None,
        "embedding_checkpoints": [
            {"model": model, "revision": revision} for model, revision in sorted(embedding_checkpoints)
        ],
        "reranker_checkpoints": [
            {"model": model, "revision": revision} for model, revision in sorted(reranker_checkpoints)
        ],
        "rerank_keep_prefix": args.rerank_keep_prefix if enable_rerank else None,
        "complete": complete,
        "candidate_limit": CANDIDATE_LIMIT,
        "page_diverse": args.page_diverse,
        "branch_candidate_limit": branch_limit,
        "rrf_output_limit": 2 * branch_limit if args.page_diverse and worker_url else branch_limit,
        "manifest_sha256": _text_hash(args.dataset_dir / "manifest.json"),
        "labels_sha256": {split: _text_hash(args.dataset_dir / f"{split}.jsonl") for split in splits},
        "document_sha256": hashes,
        "software": {"python": platform.python_version(), "pymupdf": fitz.VersionBind},
        "splits": results,
    }
    _write_results(args.output_dir, report)
    print(f"Saved retrieval benchmark in {args.output_dir.resolve()}")
    if not complete:
        print("Each requested model stage must use one checkpoint for every query.", file=sys.stderr)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
