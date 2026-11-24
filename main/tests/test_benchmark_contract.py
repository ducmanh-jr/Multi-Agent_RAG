import pytest

from eval.metrics import evaluate_single_prediction
from eval.run_benchmark import ALL_MODES, load_pilot, validate_documents


def test_retrieval_recall_requires_the_document_and_page_but_does_not_reward_citations():
    case = {
        "gold_sources": [{"document": "orchard.pdf", "page": 2, "passage": "Drip irrigation used 42 liters."}],
    }
    retrieved = {"doc_id": "orchard", "page_no": 2, "text": "Drip irrigation used 42 liters."}
    state = {
        "answer": "Drip irrigation used 42 liters. [1]",
        "retrieved_docs": [retrieved],
        "written_report_data": {"references": [{"ref_id": 1, "doc_id": "orchard", "page_no": 2}]},
        "evidence_spans": [
            {"id": "different-paper", "doc_id": "greenhouse", "page_no": 2, "quote": retrieved["text"]},
        ],
    }

    scored = evaluate_single_prediction(case, state, 0.1)
    assert scored["retrieval_recall"] == 1.0
    assert scored["citation_f1"] == 0.0

    state["retrieved_docs"] = [{**retrieved, "page_no": 1}]
    scored = evaluate_single_prediction(case, state, 0.1)
    assert scored["retrieval_recall"] == 0.0


def test_citation_f1_counts_invalid_markers_and_ignores_bibliography():
    target = {"id": "actual-span", "doc_id": "orchard", "page_no": 2, "quote": "The soil was dry."}
    case = {"gold_sources": [{"document": "orchard.pdf", "page": 2, "passage": target["quote"]}]}
    state = {
        "answer": "The soil was dry. [1]",
        "evidence_spans": [target],
        "final_report": "# Question\n\nThe soil was dry. [4] [99]\n\n## Evidence References\n[7] Source definition",
        "reasoning_trace": [{
            "action": "citation_audit",
            "citation_audit": {
                "references": [{"ref_id": 4, "document_id": "orchard", "page": 2, "evidence_id": "actual-span"}],
            },
        }],
    }
    scored = evaluate_single_prediction(case, state, 0.1)
    assert scored["citation_f1"] == pytest.approx(0.6667)

    state["final_report"] = "The soil was dry. \\[4]\n\n## Evidence References\n[4] Bibliography entry only"
    assert evaluate_single_prediction(case, state, 0.1)["citation_f1"] == 0.0


def test_no_answer_credit_requires_a_real_abstention():
    case = {"is_unanswerable": True}
    assert evaluate_single_prediction(case, {}, 0.1)["abstention_accuracy"] == 0.0
    assert evaluate_single_prediction(case, {"answer": "Wrong but without a number."}, 0.1)["abstention_accuracy"] == 0.0
    result = evaluate_single_prediction(case, {"answer_status": "NOT_FOUND_IN_DOCUMENT"}, 0.1)
    assert result["abstention_accuracy"] == 1.0
    assert result["citation_f1"] is None


def test_missing_gold_numbers_do_not_get_credit_for_a_consistent_wrong_answer():
    case = {"gold_numbers": ["42"]}
    context = {"text": "The orchard trial ran for 421 days and used 42 liters."}
    state = {"answer": "The orchard trial ran for 421 days. [42]", "source_units": [context]}
    assert evaluate_single_prediction(case, state, 0.1)["numeric_exactness"] == 0.0

    state["answer"] = "The orchard trial used 42 liters."
    assert evaluate_single_prediction(case, state, 0.1)["numeric_exactness"] == 1.0


def test_pilot_documents_and_answer_labels_are_reproducible(tmp_path):
    questions = load_pilot(tmp_path)
    assert {q["category"] for q in questions} == {
        "single_hop", "multi_hop", "table_comparison", "adversarial_trap",
    }
    hashes = validate_documents(questions, tmp_path)
    load_pilot(tmp_path)
    assert validate_documents(questions, tmp_path) == hashes

    wrong_page = {**questions[0], "gold_sources": [{**questions[0]["gold_sources"][0], "page": 2}]}
    with pytest.raises(ValueError, match="gold passage is absent"):
        validate_documents([wrong_page], tmp_path)


def test_benchmark_selects_canonical_registry_modes():
    assert set(ALL_MODES) == {"cpu_min", "cpu_max", "gemini", "gpu_finetune", "gpu_max", "hybrid"}
