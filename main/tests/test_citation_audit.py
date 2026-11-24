import hashlib

import pytest

from api.citations import citation_references, format_run_evidence, pin_citation_provenance
from core.citation_audit import audit_claim_citations, get_citation_audit, write_with_citation_audit
from core.state import EvidenceSpan, SourceUnit


@pytest.fixture
def source_state():
    text = "Accuracy was 17.5% as of 2026-11-18."
    return {
        "source_units": [SourceUnit("unit-a", text, "paper-a", 2).to_dict()],
        "evidence_spans": [EvidenceSpan("ev-a", text, "unit-a", "paper-a", 2).to_dict()],
        "claims": [{"id": "summary-id", "text": text, "evidence_ids": ["unit-a"]}],
        "verified_claims": [{"claim_id": "checker-id", "text": text, "status": "SUPPORT"}],
    }


def test_checked_claim_uses_exact_span_id(source_state):
    audit = audit_claim_citations(source_state)
    assert audit["claims"]["checker-id"]["evidence_ids"] == ["ev-a"]
    assert audit["unverified_evidence_count"] == 0


@pytest.mark.parametrize("mutation", [
    {"page_no": 1, "context_hash": ""},
    {"doc_id": "different-paper", "context_hash": ""},
    {"quote": "Accuracy was 70% as of 2026-11-18.", "context_hash": ""},
    {"context_hash": "mismatched-hash"},
])
def test_bad_source_locators_and_quotes_never_become_links(source_state, mutation):
    source_state["evidence_spans"][0].update(mutation)
    audit = audit_claim_citations(source_state)
    assert audit["unverified_evidence_count"] == 1
    assert audit["claims"]["checker-id"]["status"] == "unresolved_evidence"


def test_claim_numbers_and_dates_must_appear_in_its_own_quotes(source_state):
    unrelated = "Accuracy was 75% as of 2026-11-19."
    source_state["source_units"].append(SourceUnit("unit-b", unrelated, "paper-b", 2).to_dict())
    source_state["evidence_spans"].append(EvidenceSpan("ev-b", unrelated, "unit-b", "paper-b", 2).to_dict())
    source_state["claims"][0]["text"] = unrelated
    source_state["verified_claims"][0]["text"] = unrelated
    check = audit_claim_citations(source_state)["claims"]["checker-id"]
    assert check == {
        "status": "missing_numeric_evidence", "evidence_ids": [], "missing_values": ["2026-11-19", "75%"],
    }


def test_legacy_claim_without_links_requires_unique_verbatim_page(source_state):
    source_state["claims"] = []
    text = source_state["source_units"][0]["text"]
    source_state["verified_claims"] = [{"claim": text, "status": "support"}]
    assert audit_claim_citations(source_state)["claims"]["claim_1"]["evidence_ids"] == ["ev-a"]
    source_state["source_units"].append(SourceUnit("unit-b", text, "paper-b", 2).to_dict())
    source_state["evidence_spans"].append(EvidenceSpan("ev-b", text, "unit-b", "paper-b", 2).to_dict())
    assert audit_claim_citations(source_state)["claims"]["claim_1"]["evidence_ids"] == []


def test_raw_report_cannot_adopt_existing_span_numbering(source_state):
    result = write_with_citation_audit(source_state, lambda _: {
        "final_report": "An answer from a remote writer [1].",
        "written_report_data": {"citation_references": [{"ref_id": 1, "evidence_id": "ev-a"}]},
    })
    assert get_citation_audit(result["reasoning_trace"])["references"] == []


def _report_trace(source_state):
    audit = audit_claim_citations(source_state)
    audit["references"] = [{"ref_id": 1, "document_id": "paper-a", "page": 2, "evidence_id": "ev-a"}]
    return [{"action": "citation_audit", "citation_audit": audit}]


def _snapshot(path, identifier):
    version = b"Original paper"
    path.write_bytes(version)
    return {
        "id": identifier, "name": path.name, "path": str(path), "catalogued": True,
        "sha256": hashlib.sha256(version).hexdigest(),
    }


def test_document_link_is_never_reassigned_after_reupload(source_state, tmp_path):
    trace = _report_trace(source_state)
    spans = source_state["evidence_spans"]
    original = _snapshot(tmp_path / "paper-a.pdf", "old")
    replacement = {"id": "new", "name": "paper-a.pdf"}
    assert citation_references(trace, spans, [original, replacement]) == []
    pin_citation_provenance(trace, source_state, [original])
    references = citation_references(trace, spans, [original, replacement])
    assert references == [{"ref_id": 1, "document_id": "old", "page": 2, "evidence_id": "ev-a"}]
    assert format_run_evidence(spans, references)[0]["document_id"] == "old"
    assert citation_references(trace, spans, [replacement]) == []
    assert citation_references([], spans, [original]) == []


@pytest.mark.parametrize(("problem", "expected"), [
    ("changed", "source_changed"),
    ("unavailable", "source_unavailable"),
    ("duplicate name", "ambiguous_document"),
    ("duplicate unit", "unresolved_source_unit"),
    ("other title", "unresolved_document"),
])
def test_provenance_does_not_pin_uncertain_documents(source_state, tmp_path, problem, expected):
    trace = _report_trace(source_state)
    path = tmp_path / ("other.pdf" if problem == "other title" else "paper-a.pdf")
    original = _snapshot(path, "original")
    documents = [original]
    if problem == "changed":
        path.write_bytes(b"Updated paper")
    elif problem == "unavailable":
        path.unlink()
    elif problem == "duplicate name":
        nested = tmp_path / "other"
        nested.mkdir()
        documents.append(_snapshot(nested / path.name, "other"))
    elif problem == "duplicate unit":
        unit = source_state["source_units"][0]
        source_state["source_units"].append(SourceUnit(
            unit["id"], "Results. " + unit["text"], unit["doc_id"], unit["page_no"],
        ).to_dict())
    pin_citation_provenance(trace, source_state, documents)
    audit = get_citation_audit(trace)
    entry = audit["evidence_provenance"]["ev-a"]
    assert entry["status"] == expected
    assert "document" not in entry
    assert citation_references(trace, source_state["evidence_spans"], documents) == []


def test_exact_reader_path_disambiguates_same_named_sources(source_state, tmp_path):
    trace = _report_trace(source_state)
    original = _snapshot(tmp_path / "paper-a.pdf", "actual")
    nested = tmp_path / "other"
    nested.mkdir()
    other = _snapshot(nested / "paper-a.pdf", "other")
    source_state["source_units"][0]["source_path"] = original["path"]
    pin_citation_provenance(trace, source_state, [original, other])
    entry = get_citation_audit(trace)["evidence_provenance"]["ev-a"]
    assert entry["status"] == "linked"
    assert entry["document"]["id"] == "actual"
    assert "source_path" not in entry["source_unit"]
