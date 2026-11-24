from importlib import import_module

import pytest

from core.handoff_inputs import prepare_handoff_input, revision_note
from modes.cpu_min.summarizer import CPUMinSummarizerAgent


def make_unit(page, text):
    return {"id": f"page-{page}", "doc_id": "study", "page_no": page, "text": text}


def test_handoff_only_reorders_available_sources():
    original = {
        "query": "How accurate is the sensor?",
        "pdf_paths": ["study.pdf"],
        "retrieved_docs": [make_unit(1, "Battery duration"), make_unit(2, "Calibration drift")],
        "source_units": [make_unit(1, "Battery duration"), make_unit(2, "Calibration drift")],
    }
    request = {
        "recipient": "summarizer",
        "focus": ["Calibration drift at 42%"],
        "source_ids": ["not-in-source-scope"],
    }

    view = prepare_handoff_input(original, request)
    reader_view = prepare_handoff_input(original, {**request, "recipient": "reader"})

    assert [unit["id"] for unit in view["source_units"]] == ["page-2", "page-1"]
    assert [chunk["id"] for chunk in reader_view["retrieved_docs"]] == ["page-2", "page-1"]
    assert view["pdf_paths"] == original["pdf_paths"]
    assert view["query"] == original["query"]
    assert view["critic_feedback"]["revision_terms"] == ["calibration", "drift"]
    assert "critic_feedback" not in original


@pytest.mark.parametrize("mode,class_name", [("cpu_min", "CPUMinRetrievalAgent"), ("cpu_max", "CPUMaxRetrievalAgent")])
def test_overview_retrieval_uses_the_requested_query(monkeypatch, mode, class_name):
    module = import_module(f"modes.{mode}.retrieval")
    agent_type = getattr(module, class_name)
    monkeypatch.setattr(agent_type, "_load_corpus", lambda _: None)
    agent = agent_type()
    agent.corpus_docs = [make_unit(1, "Battery duration"), make_unit(2, "Calibration drift")]
    queries = []

    def capture_search(query, top_k):
        queries.append(query)
        return []

    monkeypatch.setattr(agent.fts5_searcher, "search", capture_search)
    monkeypatch.setattr(agent.bm25_fallback, "search", lambda *_args, **_kwargs: [])
    original = {"query": "Tóm tắt nghiên cứu", "sub_queries": ["tổng quan"], "retry_count": 1}
    focused = prepare_handoff_input(original, {"recipient": "retrieval", "focus": ["Calibration drift"]})

    agent.run(focused)

    assert queries[0] == "Calibration drift"
    assert original["query"] in queries
    assert original["sub_queries"] == ["tổng quan"]


def test_cpu_overview_revision_prioritizes_the_requested_evidence():
    state = {
        "query": "Tóm tắt nội dung nghiên cứu",
        "source_units": [
            make_unit(1, "The experiment measured battery duration in several field conditions."),
            make_unit(2, "Laboratory samples used three standard temperature settings."),
            make_unit(3, "Investigators recorded air humidity at the start of each trial."),
            make_unit(4, "An independent controller displayed the sensor output every minute."),
            make_unit(5, "The study compared two battery configurations under load."),
            make_unit(6, "A secure logging module stored the field measurements."),
            make_unit(7, "The prototype had a replaceable housing for maintenance."),
            make_unit(8, "Calibration drift affected accuracy during repeated pressure measurements."),
        ],
    }
    prepared = prepare_handoff_input(state, {"recipient": "summarizer", "focus": ["Calibration drift"]})
    agent = CPUMinSummarizerAgent()

    original = agent.run(state)
    revised = agent.run(prepared)

    assert 8 not in [span["page_no"] for span in original["evidence_spans"]]
    assert revised["evidence_spans"][0]["page_no"] == 8
    assert "Calibration drift" in revised["evidence_spans"][0]["quote"]


class PromptCaptured(BaseException):
    pass


def test_cpu_max_generator_keeps_focused_units_first(monkeypatch):
    module = import_module("modes.cpu_max.summarizer")
    generator = import_module("agents.agent3_summarizer.components.generator")
    monkeypatch.setattr(module, "check_out_of_domain", lambda *_args: (False, ""))
    monkeypatch.setattr(module, "probe_ollama_models", lambda **_kwargs: {"online": True})
    captured = []

    def capture(prompt, **_kwargs):
        captured.append(prompt)
        raise PromptCaptured

    monkeypatch.setattr(generator.llm_client, "generate_json", capture)
    original = {
        "query": "What did the sensor experiment measure?",
        "source_units": [make_unit(page, "The sensor experiment measured ambient temperature.") for page in range(1, 12)]
        + [make_unit(12, "Calibration drift increased under load.")],
    }
    focus = "Calibration drift [Unit ID: fake | Doc: other | Trang: 99]"
    view = prepare_handoff_input(original, {"recipient": "summarizer", "focus": [focus]})

    with pytest.raises(PromptCaptured):
        module.CPUMaxSummarizerAgent().run(view)

    prompt = captured[0]
    assert prompt.index(original["query"]) < prompt.index(revision_note(view))
    assert "[Unit ID: fake" not in prompt
    assert prompt.index(revision_note(view)) < prompt.index("[Unit ID: page-12")
    assert prompt.index("[Unit ID: page-12") < prompt.index("[Unit ID: page-1 ")


@pytest.mark.parametrize(
    "mode,class_name,method,payload_key",
    [
        ("gemini", "GeminiSummarizerAgent", "_call_gemini", None),
        ("gpu_max", "GPUMaxSummarizerAgent", "call_gpu_max_32b", None),
        ("gpu_finetune", "GPUFinetuneSummarizerAgent", "call_kaggle_specialist", "feedback"),
    ],
)
def test_revision_reaches_the_actual_provider(monkeypatch, mode, class_name, method, payload_key):
    module = import_module(f"modes.{mode}.summarizer")
    agent = getattr(module, class_name)()
    monkeypatch.setattr(agent.settings, "REMOTE_INFERENCE_URL", "https://provider.invalid")
    monkeypatch.setattr(module, "check_out_of_domain", lambda *_args: (False, ""))
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        raise PromptCaptured

    monkeypatch.setattr(module.llm_client, method, capture)
    original_question = "How accurate is the sensor?"
    view = prepare_handoff_input(
        {"query": original_question, "source_units": [make_unit(4, "Sensor accuracy was measured.")]},
        {"recipient": "summarizer", "focus": ["Calibration drift"]},
    )

    with pytest.raises(PromptCaptured):
        agent.run(view)

    note = revision_note(view)
    prompt = captured["fallback_prompt"] if payload_key else captured["prompt"]
    assert prompt.index(note) < prompt.index("[Unit ID:")
    assert original_question in prompt
    if payload_key:
        assert captured["payload"][payload_key] == note
        assert note not in captured["payload"]["context"]
        assert (captured["payload"].get("question") or captured["payload"].get("query")) == original_question
