"""Offline tests for the benchmark's scoring and evidence validation."""

import hashlib
import math
from collections import Counter
from pathlib import Path

import fitz
import pytest

from eval.retrieval_metrics import METRICS, score_ranked_pages
from eval.run_retrieval_benchmark import (
    CATEGORIES,
    evaluate_ranker,
    load_labels,
    prepare_documents,
)


def test_relevance_is_scoped_to_document_and_page_and_credited_once():
    direct = ("a.pdf", 2)
    supporting = ("a.pdf", 6)
    wrong_source = ("b.pdf", 2)
    ranking = [wrong_source, direct, direct, wrong_source, wrong_source, supporting]

    result = score_ranked_pages(ranking, {direct: 2, supporting: 1})

    assert result["recall_at_5"] == 0.5
    assert result["recall_at_10"] == 1.0
    assert result["mrr_at_10"] == 0.5
    assert result["ndcg_at_10"] == pytest.approx(
        (3 / math.log2(3) + 1 / math.log2(7)) / (3 + 1 / math.log2(3))
    )


def test_a_hit_after_the_candidate_limit_does_not_score():
    result = score_ranked_pages([("d.pdf", 1)] * 10 + [("d.pdf", 2)], {("d.pdf", 2): 2})
    assert result == dict.fromkeys(METRICS, 0.0)


def test_unanswerable_cases_do_not_supply_ranking_metrics():
    assert score_ranked_pages([("d.pdf", 1)], {}) == dict.fromkeys(METRICS)


def test_benchmark_splits_cover_categories_with_disjoint_documents():
    manifest, splits = load_labels()
    scopes = {
        split: {name for case in cases for name in case["documents"]}
        for split, cases in splits.items()
    }
    assert scopes["dev"].isdisjoint(scopes["holdout"])
    assert scopes["dev"] | scopes["holdout"] == set(manifest["documents"])
    for cases in splits.values():
        assert Counter(case["category"] for case in cases) == dict.fromkeys(CATEGORIES, 3)


@pytest.mark.parametrize(("dataset_name", "minimum"), [
    ("retrieval_holdout_six_pdf", 4), ("retrieval_holdout_pretraining", 3),
])
def test_independent_holdout_uses_six_new_pdfs_for_every_question(dataset_name, minimum):
    directory = Path(__file__).resolve().parents[1] / "eval"
    manifest, splits = load_labels(directory / dataset_name)
    assert tuple(splits) == ("holdout",)
    documents = manifest["documents"]
    assert len(documents) == 6
    for other in {"retrieval_benchmark", "retrieval_holdout_six_pdf", "retrieval_holdout_pretraining"} - {dataset_name}:
        previous, _ = load_labels(directory / other)
        assert not set(documents).intersection(previous["documents"])
        old_hashes = {document["sha256"] for document in previous["documents"].values()}
        assert all(document["sha256"] not in old_hashes for document in documents.values())
    assert Counter(case["category"] for case in splits["holdout"]) == dict.fromkeys(CATEGORIES, minimum)
    assert all(set(case["documents"]) == set(documents) for case in splits["holdout"])


def test_qasper_holdout_keeps_native_paper_scope_and_disjoint_sources():
    directory = Path(__file__).resolve().parents[1] / "eval"
    manifest, splits = load_labels(directory / "retrieval_holdout_qasper")
    assert tuple(splits) == ("holdout",)
    assert manifest["source"]["dataset"] == "allenai/qasper"
    assert manifest["source"]["split"] == "test"
    documents = manifest["documents"]
    cases = splits["holdout"]
    assert len(cases) == 60
    assert len(documents) >= 15
    assert all(doc["pages"] >= 10 for doc in documents.values())
    counts = Counter(case["documents"][0] for case in cases)
    assert max(counts.values()) <= 4
    assert set(counts) == set(documents)
    assert {case["category"] for case in cases} == {"single_document", "multi_hop"}
    for case in cases:
        assert case["documents"] == [f"qasper-{case['qasper']['paper_id']}v1.pdf"]
        assert case["id"] == "qasper-" + case["qasper"]["question_id"]
    arxiv_ids = {doc["url"].rsplit("/", 1)[1].split("v")[0] for doc in documents.values()}
    for name in ("retrieval_benchmark", "retrieval_holdout_six_pdf", "retrieval_holdout_pretraining"):
        earlier, _ = load_labels(directory / name)
        other_ids = {doc["url"].rsplit("/", 1)[1].split("v")[0] for doc in earlier["documents"].values()}
        assert arxiv_ids.isdisjoint(other_ids)


def test_reranker_qasper_holdout_excludes_previous_qasper_papers():
    directory = Path(__file__).resolve().parents[1] / "eval"
    previous, _ = load_labels(directory / "retrieval_holdout_qasper")
    fresh, splits = load_labels(directory / "retrieval_holdout_qasper_rerank")
    assert len(splits["holdout"]) == 80 and len(fresh["documents"]) >= 20
    assert fresh["source"]["parquet_sha256"] == previous["source"]["parquet_sha256"]
    assert fresh["source"]["split"] == "test"
    assert {doc["url"] for doc in fresh["documents"].values()}.isdisjoint(
        doc["url"] for doc in previous["documents"].values()
    )
    for case in splits["holdout"]:
        assert case["documents"] == [f"qasper-{case['qasper']['paper_id']}v1.pdf"]


def test_ranker_receives_only_query_and_scope_and_accepts_dotted_document_ids():
    _, splits = load_labels()
    case = splits["dev"][0]
    name = case["documents"][0]

    def retrieve(question, documents):
        assert question == case["question"]
        assert documents == (name,)
        trace = {"dense": {"status": "disabled"}, "rerank": {"status": "disabled"}}
        return [{"doc_id": name.removesuffix(".pdf"), "page_no": 4}], trace

    result = evaluate_ranker([case], retrieve)
    record = result["records"][0]
    assert record["ranking"] == [(name, 4)]
    assert record["recall_at_5"] == 0.5
    assert record["mrr_at_10"] == 1.0

    with pytest.raises(ValueError, match="out-of-scope"):
        evaluate_ranker([case], lambda *_: ([{"doc_id": "other", "page_no": 4}], {}))


def test_checks_pdf_hash_and_the_exact_evidence_page(tmp_path):
    name = "example.pdf"
    path = tmp_path / name
    with fitz.open() as doc:
        doc.new_page().insert_text((72, 72), "The answer is 68.4.")
        doc.new_page().insert_text((72, 72), "An unrelated page.")
        doc.save(path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    documents = {name: {"sha256": checksum, "pages": 2}}
    label = {"document": name, "page": 1, "anchors": ["68.4"]}
    case = {"id": "sample", "documents": [name], "gold_sources": [label]}
    assert prepare_documents(tmp_path, documents, [case]) == {name: checksum}

    label["page"] = 2
    with pytest.raises(ValueError, match="evidence anchor missing"):
        prepare_documents(tmp_path, documents, [case])

    path.write_bytes(b"wrong PDF")
    with pytest.raises(ValueError, match="PDF checksum mismatch"):
        prepare_documents(tmp_path, documents, [case])


@pytest.mark.parametrize("top_k", [3, 10])
def test_gpu_finetune_fuses_up_to_requested_top_k(tmp_path, top_k):
    from core.config import RunConfig, use_run_config
    from modes.gpu_finetune.retrieval import GPUFinetuneRetrievalAgent

    paths = []
    for name in ("first.pdf", "second.pdf"):
        path = tmp_path / name
        with fitz.open() as doc:
            for page in range(1, 7):
                text = f"Calibration results for {name} at observation {page}."
                doc.new_page().insert_text((72, 72), text)
            doc.save(path)
        paths.append(str(path))

    config = RunConfig.capture(remote_inference_url="", corpus_path=str(tmp_path / "empty.jsonl"))
    with use_run_config(config):
        result = GPUFinetuneRetrievalAgent().run({
            "query": "calibration",
            "pdf_paths": paths,
            "top_k": top_k,
        })

    passages = result["retrieved_docs"]
    assert result["top_k"] == top_k
    assert len(passages) == top_k
    assert all(passage["rrf_score"] > 0 for passage in passages)
    assert len({passage["id"] for passage in passages}) == top_k
