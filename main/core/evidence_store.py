"""
Unified Shared Evidence Store (PR-08).
Provides a single structured view over:
  - SourceUnits (with multi-PDF provenance & content/context hashes)
  - EvidenceSpans (quote -> source unit provenance verification)
  - Claims & verification verdicts (SUPPORT / CONTRADICT / NOT_ENOUGH_INFO)
  - Contradictions (explicitly tracked conflicting claims and sources)
  - EvidenceGaps (missing aspects and suggested follow-up queries)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from core.verifier import verify_evidence_span_provenance


@dataclass
class EvidenceStore:
    """Centralized, serializable evidence ledger shared across Supervisor, FactChecker, and Writer."""

    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    spans: dict[str, dict[str, Any]] = field(default_factory=dict)
    claims: list[dict[str, Any]] = field(default_factory=list)
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)

    def add_source_units(self, units: Sequence[Mapping[str, Any]]) -> int:
        added = 0
        for idx, raw in enumerate(units or [], start=1):
            if not isinstance(raw, Mapping):
                continue
            uid = str(raw.get("id") or f"unit_{idx}")
            record = {
                "id": uid,
                "doc_id": str(raw.get("doc_id") or raw.get("document_id") or "unknown"),
                "page_no": int(raw.get("page_no") or raw.get("page") or 1),
                "section": str(raw.get("section") or ""),
                "context_hash": str(raw.get("context_hash") or ""),
                "content_hash": str(raw.get("content_hash") or ""),
                "provenance": list(raw.get("provenance") or []),
                "text_preview": str(raw.get("text") or raw.get("content") or "")[:240],
            }
            if uid not in self.sources:
                self.sources[uid] = record
                added += 1
            else:
                # Merge any newly discovered multi-PDF provenance entries
                existing_prov = self.sources[uid].get("provenance") or []
                seen = {
                    (p.get("doc_id"), p.get("page_no"), p.get("source_unit_id"))
                    for p in existing_prov
                    if isinstance(p, Mapping)
                }
                for p in record["provenance"]:
                    if isinstance(p, Mapping):
                        sig = (p.get("doc_id"), p.get("page_no"), p.get("source_unit_id"))
                        if sig not in seen:
                            existing_prov.append(dict(p))
                            seen.add(sig)
                self.sources[uid]["provenance"] = existing_prov
        return added

    def add_evidence_spans(
        self,
        spans: Sequence[Mapping[str, Any]],
        source_units: Sequence[Mapping[str, Any]] | None = None,
    ) -> int:
        added = 0
        units_list = [dict(u) for u in (source_units or []) if isinstance(u, Mapping)]
        for idx, raw in enumerate(spans or [], start=1):
            if not isinstance(raw, Mapping):
                continue
            sid = str(raw.get("id") or f"span_{idx}")
            prov_ok = True
            prov_reason = "unverified_no_units"
            if units_list:
                prov_ok, prov_reason = verify_evidence_span_provenance(dict(raw), units_list)
            self.spans[sid] = {
                "id": sid,
                "source_unit_id": str(raw.get("source_unit_id") or ""),
                "doc_id": str(raw.get("doc_id") or ""),
                "page_no": int(raw.get("page_no") or raw.get("page") or 1),
                "quote": str(raw.get("quote") or raw.get("text") or "")[:300],
                "provenance_verified": bool(prov_ok),
                "provenance_reason": prov_reason,
            }
            added += 1
        return added

    def record_verification(
        self,
        verified_claims: Sequence[Mapping[str, Any]],
        rejected_claims: Sequence[Mapping[str, Any]],
        evidence_gaps: Sequence[Mapping[str, Any]],
    ) -> None:
        self.claims = []
        self.contradictions = []
        for item in verified_claims or []:
            if not isinstance(item, Mapping):
                continue
            self.claims.append({
                "id": str(item.get("id") or item.get("claim_id") or ""),
                "text": str(item.get("text") or item.get("claim_text") or item.get("claim") or "")[:300],
                "status": "SUPPORT",
                "evidence_ids": list(item.get("evidence_ids") or []),
                "reason": str(item.get("reason") or ""),
            })
        for item in rejected_claims or []:
            if not isinstance(item, Mapping):
                continue
            status = str(item.get("status") or "NOT_ENOUGH_INFO").upper()
            entry = {
                "id": str(item.get("id") or item.get("claim_id") or ""),
                "text": str(item.get("text") or item.get("claim_text") or item.get("claim") or "")[:300],
                "status": status,
                "evidence_ids": list(item.get("evidence_ids") or []),
                "reason": str(item.get("reason") or ""),
            }
            self.claims.append(entry)
            if status == "CONTRADICT":
                self.contradictions.append({
                    "claim_id": entry["id"],
                    "claim_text": entry["text"],
                    "evidence_ids": entry["evidence_ids"],
                    "reason": entry["reason"],
                })

        self.gaps = []
        for gap in evidence_gaps or []:
            if not isinstance(gap, Mapping):
                continue
            self.gaps.append({
                "missing_aspect": str(gap.get("missing_aspect") or "")[:240],
                "suggested_query": str(gap.get("suggested_query") or "")[:240],
            })

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "EvidenceStore":
        store = cls()
        units = state.get("source_units") or state.get("retrieved_docs") or []
        store.add_source_units(units)
        store.add_evidence_spans(state.get("evidence_spans") or [], units)
        store.record_verification(
            state.get("verified_claims") or [],
            state.get("rejected_claims") or [],
            state.get("evidence_gaps") or [],
        )
        return store

    def summary(self) -> dict[str, Any]:
        verified_count = sum(1 for c in self.claims if c.get("status") == "SUPPORT")
        contradict_count = len(self.contradictions)
        unsupported_count = sum(1 for c in self.claims if c.get("status") != "SUPPORT")
        verified_spans = sum(1 for s in self.spans.values() if s.get("provenance_verified"))
        return {
            "source_count": len(self.sources),
            "unique_documents": sorted({s["doc_id"] for s in self.sources.values() if s.get("doc_id")}),
            "span_count": len(self.spans),
            "verified_span_count": verified_spans,
            "claim_count": len(self.claims),
            "verified_claim_count": verified_count,
            "unsupported_claim_count": unsupported_count,
            "contradiction_count": contradict_count,
            "gap_count": len(self.gaps),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "sources": list(self.sources.values()),
            "spans": list(self.spans.values()),
            "claims": list(self.claims),
            "contradictions": list(self.contradictions),
            "gaps": list(self.gaps),
        }
