"""Resolve audited citations to exact document routes in the local catalog."""

from collections import defaultdict
from pathlib import PurePosixPath
from typing import Any

from core.citation_audit import citation_location, evidence_quote, get_citation_audit


def citation_references(
    trace: list[dict[str, Any]], spans: list[dict[str, Any]], documents: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    references = get_citation_audit(trace).get("references") or []
    if not references:
        return []

    aliases = defaultdict(set)
    catalog_ids = set()
    for document in documents:
        document_id = document["id"]
        catalog_ids.add(document_id)
        aliases[document_id].add(document_id)
        for field in ("name", "path"):
            raw = document.get(field)
            if raw:
                path = PurePosixPath(raw.replace("\\", "/"))
                for alias in {raw, path.name, path.stem}:
                    aliases[alias].add(document_id)

    locations = {
        span["id"]: citation_location(span)
        for span in spans
        if isinstance(span, dict) and isinstance(span.get("id"), str)
    }
    resolved = []
    for ref in references:
        if not isinstance(ref, dict) or type(ref.get("ref_id")) is not int or ref["ref_id"] < 1:
            continue
        loc = citation_location(ref)
        evidence_id = ref.get("evidence_id")
        if not loc or not isinstance(evidence_id, str) or locations.get(evidence_id) != loc:
            continue
        source_id, page = loc
        if "catalog_id" in ref:
            document_id = ref["catalog_id"]
        else:
            matches = aliases.get(source_id, set())
            document_id = next(iter(matches)) if len(matches) == 1 else None
        if document_id in catalog_ids:
            resolved.append({"ref_id": ref["ref_id"], "document_id": document_id, "page": page, "evidence_id": evidence_id})
    return resolved


def pin_citation_documents(
    trace: list[dict[str, Any]], spans: list[dict[str, Any]], documents: list[dict[str, Any]]
) -> None:
    """Persist route IDs so a later upload with the same title cannot replace the source."""
    audit = get_citation_audit(trace)
    if audit.get("references"):
        resolved = {ref["ref_id"]: ref["document_id"] for ref in citation_references(trace, spans, documents)}
        audit["references"] = [
            {**ref, "catalog_id": resolved.get(ref["ref_id"])} for ref in audit["references"]
        ]


def format_run_evidence(spans: list[Any], references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans_by_id = {s["id"]: s for s in spans if isinstance(s, dict) and isinstance(s.get("id"), str)}
    canonical_docs = {
        citation_location(spans_by_id[ref["evidence_id"]]): ref["document_id"] for ref in references
    }
    evidence = []
    for i, raw in enumerate(spans, start=1):
        span = raw if isinstance(raw, dict) else {"quote": str(raw)}
        location = citation_location(span)
        doc_id, page = location or ("", None)
        evidence.append({
            "id": span.get("id") or f"legacy_ev_{i}",
            "document_id": canonical_docs.get(location, doc_id),
            "filename": doc_id,
            "quote": evidence_quote(span),
            "page": page,
            "start": span.get("start"),
            "end": span.get("end"),
            "relevance": "Đoạn bằng chứng do hệ thống thu thập",
            "chunk_id": span.get("source_unit_id") or span.get("unit_id") or f"legacy_ch_{i}",
            "regions": span.get("regions") or [],
        })
    return evidence
