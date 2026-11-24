from copy import deepcopy

import pytest

from core.config import settings
from core.coordination import decide_handoff, fact_check_accepted, reflection_limit


def test_checker_requests_unread_material_before_repeating_search():
    state = {
        "query": "What were the results?",
        "fact_check_label": "not_enough_info",
        "source_units": [{"id": "read", "text": "Study methods"}],
        "retrieved_docs": [
            {"id": "read", "text": "Study methods"},
            {"id": "results", "text": "The trial results"},
        ],
        "evidence_gaps": [{"missing_aspect": "outcome", "suggested_query": "trial results"}],
    }
    snapshot = deepcopy(state)
    request = decide_handoff("fact_checker", state)
    assert request.recipient == "reader"
    assert request.source_ids == ("results",)
    assert request.focus == ("trial results",)
    assert state == snapshot
    state["source_units"].append({"id": "results", "text": "The trial results"})
    assert decide_handoff("fact_checker", state).recipient == "retrieval"


def test_rejected_claim_goes_to_summarizer_with_original_evidence():
    state = {
        "query": "How many patients?",
        "fact_check_label": "contradict",
        "rejected_claims": [{"claim_text": "There were 40 patients", "status": "CONTRADICT"}],
        "source_units": [{"id": "read", "text": "The study enrolled 14 patients."}],
        "no_new_evidence": True,
    }
    request = decide_handoff("fact_checker", state)
    assert request.recipient == "summarizer"
    assert request.focus == ("There were 40 patients",)
    assert not fact_check_accepted(state)
    state["reasoning_trace"] = [request.as_trace()]
    assert decide_handoff("fact_checker", state) is None


def test_support_consensus_does_not_hide_rejected_claims():
    state = {
        "fact_check_label": "supported",
        "source_units": [{"id": "u1", "text": "Measured outcome."}],
        "verified_claims": [{"text": "Measured outcome."}],
        "rejected_claims": [{"text": "Unobserved outcome."}],
    }
    assert not fact_check_accepted(state)
    assert decide_handoff("fact_checker", state).recipient == "summarizer"
    state["rejected_claims"] = []
    assert fact_check_accepted(state)
    assert decide_handoff("fact_checker", state) is None


@pytest.mark.parametrize("action", ["approve", "accept", "lower_confidence"])
def test_noncorrective_critic_actions_go_to_verification(action):
    assert decide_handoff("summarizer", {"critic_action": action}) is None


def test_rewrite_is_scoped_and_duplicate_feedback_is_order_insensitive():
    state = {
        "critic_action": "rewrite",
        "critic_feedback": {"unsupported_claims": ["Claim A", "Claim B"]},
        "source_units": [{"id": "u1", "text": "original evidence"}],
    }
    request = decide_handoff("summarizer", state)
    assert request.recipient == "summarizer"
    assert request.focus == ("Claim A", "Claim B")
    state["reasoning_trace"] = [request.as_trace()]
    state["critic_feedback"]["unsupported_claims"].reverse()
    assert decide_handoff("summarizer", state) is None


def test_context_request_stops_after_no_progress():
    state = {"critic_action": "request_more_context", "query": "soil nitrogen"}
    assert decide_handoff("summarizer", state).recipient == "retrieval"
    state["no_new_evidence"] = True
    assert decide_handoff("summarizer", state) is None


def test_verified_abstention_finishes_but_checker_can_challenge_false_refusal():
    state = {
        "answer_status": "NOT_FOUND_IN_DOCUMENT",
        "critic_action": "request_more_context",
        "fact_check_label": "supported",
        "source_units": [{"id": "u1", "text": "An observed result."}],
    }
    assert decide_handoff("summarizer", state) is None
    assert decide_handoff("fact_checker", state) is None
    state.update(fact_check_label="contradict", rejected_claims=[{"claim": "The result is absent."}])
    assert decide_handoff("fact_checker", state).recipient == "summarizer"


def test_explicit_zero_mode_default_and_global_cap(monkeypatch):
    monkeypatch.setattr(settings, "MAX_ORCHESTRATOR_RETRY", 3)
    state = {"critic_action": "request_more_context", "max_reflection_loops": 0}
    assert reflection_limit(state) == 0
    assert decide_handoff("summarizer", state) is None
    assert reflection_limit({"max_iterations": 1}) == 1
    assert reflection_limit({}) == 3
    state.update(max_reflection_loops=100, retry_count=3)
    assert reflection_limit(state) == 3
    assert decide_handoff("summarizer", state) is None
