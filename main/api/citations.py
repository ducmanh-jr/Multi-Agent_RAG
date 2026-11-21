"""Persist audited evidence and resolve document routes in the local catalog."""

import hashlib
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from core.citation_audit import citation_location, citation_source_units, evidence_quote, get_citation_audit
from core.state import AgentState, compute_context_hash


def file_sha256(path: Path) -> str | None:
    try:
        with path.open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()
    except OSError:
        return None


def _document_aliases(documents: list[dict[str, Any]]) -> dict[str, set[str]]:
    aliases = defaultdict(set)
    for document in documents:
        document_id = document["id"]
        aliases[document_id].add(document_id)
        for field in ("name", "path"):
            raw = document.get(field)
            if raw:
                path = PurePosixPath(raw.replace("\\", "/"))
                for alias in {raw, path.name, path.stem}:
                    aliases[alias].add(document_id)
    return aliases


def citation_references(
    trace: list[dict[str, Any]], spans: list[dict[str, Any]], documents: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    references = get_citation_audit(trace).get("references") or []
    if not references:
        return []

    aliases = _document_aliases(documents)
    catalog_ids = {document["id"] for document in documents}
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


def pin_citation_provenance(
    trace: list[dict[str, Any]], state: AgentState, documents: list[dict[str, Any]]
) -> None:
    """Freeze the Reader content and its run-scoped document association for audited spans."""
    audit = get_citation_audit(trace)
    if not audit:
        return
    evidence_ids = {
        eid for check in audit["claims"].values()
        if check["status"] == "verified" for eid in check["evidence_ids"]
    }
    units = citation_source_units(state, evidence_ids)
    aliases = _document_aliases(documents)
    by_id = {document["id"]: document for document in documents}
    by_path = {Path(document["path"]): document["id"] for document in documents}
    file_versions: dict[str, str | None] = {}
    provenance = {}
    for evidence_id in sorted(evidence_ids):
        entry = provenance[evidence_id] = {"status": "unresolved_source_unit"}
        unit = units.get(evidence_id)
        if unit is None:
            continue
        doc_id, page_no = citation_location(unit)
        text = evidence_quote(unit)
        digest = compute_context_hash(text, doc_id, page_no)
        if unit.get("context_hash") not in (None, "", digest):
            continue
        entry["source_unit"] = {
            "id": unit["id"], "text": text, "doc_id": doc_id, "page_no": page_no,
            "context_hash": digest, "character_offset": unit.get("character_offset"),
            "bbox": unit.get("bbox"),
        }
        path = unit.get("source_path")
        if isinstance(path, str) and path:
            source_id = by_path.get(Path(path))
        else:
            matches = aliases.get(doc_id, set())
            if len(matches) > 1:
                entry["status"] = "ambiguous_document"
                continue
            source_id = next(iter(matches), None)
        if source_id is None:
            entry["status"] = "unresolved_document"
            continue
        document = by_id[source_id]
        if source_id not in file_versions:
            file_versions[source_id] = file_sha256(Path(document["path"]))
        current = file_versions[source_id]
        if current is None:
            entry["status"] = "source_unavailable"
        elif current != document["sha256"]:
            entry["status"] = "source_changed"
        else:
            entry["status"] = "linked"
            entry["document"] = {key: document[key] for key in ("id", "name", "sha256")}
    audit["evidence_provenance"] = provenance

    for ref in audit.get("references") or []:
        source = provenance.get(ref.get("evidence_id"), {}).get("document", {}).get("id")
        ref["catalog_id"] = source if source and by_id[source]["catalogued"] else None


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
