"""
Unified Shared Evidence Store (PR-08 / M2 & M4 Upgraded).
Provides a single structured view over:
  - SourceUnits (with multi-PDF provenance & deterministic content/context hashes)
  - EvidenceSpans (quote -> source unit provenance verification, default fail-closed)
  - Claims & verification verdicts (SUPPORT / CONTRADICT / NOT_ENOUGH_INFO)
  - End-to-End Claim Lineage (Claim -> EvidenceSpan -> SourceUnit -> Document/Page/Hash)
  - Multi-Document Source Quality Audit (duplicate/preprint overlap, corroboration, provenance pass rate)
  - Contradictions (explicitly tracked conflicting claims and sources)
  - EvidenceGaps (missing aspects and suggested follow-up queries)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from core.state import compute_content_hash, compute_context_hash
from core.verifier import verify_evidence_span_provenance


@dataclass
class EvidenceStore:
    """Centralized, serializable evidence ledger shared across Supervisor, Reader, Summarizer, FactChecker, and Writer."""

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
            doc_id = str(raw.get("doc_id") or raw.get("document_id") or "unknown")
            try:
                page_no = int(raw.get("page_no") or raw.get("page") or 1)
            except Exception:
                page_no = 1
            full_text = str(raw.get("text") or raw.get("content") or raw.get("text_preview") or "")
            ctx_hash = str(raw.get("context_hash") or "") or (
                compute_context_hash(full_text, doc_id, page_no) if full_text.strip() else ""
            )
            cnt_hash = str(raw.get("content_hash") or "") or (
                compute_content_hash(full_text) if full_text.strip() else ""
            )
            raw_prov = list(raw.get("provenance") or [])
            if not raw_prov:
                raw_prov = [
                    {
                        "doc_id": doc_id,
                        "page_no": page_no,
                        "unit_id": uid,
                        "context_hash": ctx_hash,
                    }
                ]
            record: dict[str, Any] = {
                "id": uid,
                "doc_id": doc_id,
                "page_no": page_no,
                "section": str(raw.get("section") or ""),
                "doi": str(raw.get("doi") or ""),
                "title": str(raw.get("title") or ""),
                "context_hash": ctx_hash,
                "content_hash": cnt_hash,
                "provenance": raw_prov,
                "text": full_text,
                "text_preview": full_text[:240],
            }
            if uid not in self.sources:
                self.sources[uid] = record
                added += 1
            else:
                # Merge any newly discovered multi-PDF provenance entries
                existing_prov = list(self.sources[uid].get("provenance") or [])
                seen = {
                    (p.get("doc_id"), p.get("page_no"), p.get("unit_id") or p.get("source_unit_id"))
                    for p in existing_prov
                    if isinstance(p, Mapping)
                }
                for p in record["provenance"]:
                    if isinstance(p, Mapping):
                        sig = (p.get("doc_id"), p.get("page_no"), p.get("unit_id") or p.get("source_unit_id"))
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
            prov_ok = False
            prov_reason = "unverified_no_units"
            if units_list:
                prov_ok, prov_reason = verify_evidence_span_provenance(dict(raw), units_list)
            unit_id = str(raw.get("source_unit_id") or raw.get("unit_id") or "")
            doc_id = str(raw.get("doc_id") or "")
            try:
                page_no = int(raw.get("page_no") or raw.get("page") or 1)
            except Exception:
                page_no = 1
            quote = str(raw.get("quote") or raw.get("text") or "")[:300]
            ctx_hash = str(raw.get("context_hash") or "") or (
                compute_context_hash(quote, doc_id, page_no) if quote.strip() else ""
            )
            self.spans[sid] = {
                "id": sid,
                "source_unit_id": unit_id,
                "doc_id": doc_id,
                "page_no": page_no,
                "quote": quote,
                "context_hash": ctx_hash,
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
        *,
        draft_claims: Sequence[Mapping[str, Any]] | None = None,
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

        # If verification has not executed yet, retain draft claims as UNVERIFIED for progressive lineage tracking
        if not self.claims and draft_claims:
            for item in draft_claims:
                if not isinstance(item, Mapping):
                    continue
                self.claims.append({
                    "id": str(item.get("id") or item.get("claim_id") or ""),
                    "text": str(item.get("text") or item.get("claim_text") or item.get("claim") or "")[:300],
                    "status": "UNVERIFIED",
                    "evidence_ids": list(item.get("evidence_ids") or []),
                    "reason": "pending_fact_check",
                })

        self.gaps = []
        for gap in evidence_gaps or []:
            if not isinstance(gap, Mapping):
                continue
            self.gaps.append({
                "missing_aspect": str(gap.get("missing_aspect") or "")[:240],
                "suggested_query": str(gap.get("suggested_query") or "")[:240],
            })

    def _resolve_span_and_unit(self, evidence_ref: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Resolves an evidence_id reference to its (EvidenceSpan, SourceUnit) pair."""
        ref = str(evidence_ref or "").strip()
        if not ref:
            return None, None
        span = self.spans.get(ref)
        if span is None:
            # Check if any span points to a source_unit_id matching ref
            span = next((s for s in self.spans.values() if s.get("source_unit_id") == ref), None)
        unit: dict[str, Any] | None = None
        if span is not None:
            uid = str(span.get("source_unit_id") or "")
            unit = self.sources.get(uid)
            if unit is None and span.get("doc_id"):
                unit = next(
                    (
                        u
                        for u in self.sources.values()
                        if u.get("doc_id") == span.get("doc_id") and u.get("page_no") == span.get("page_no")
                    ),
                    None,
                )
        elif ref in self.sources:
            unit = self.sources[ref]
        return span, unit

    def trace_claim_lineage(self, claim_id: str) -> dict[str, Any] | None:
        """
        Traces a specific claim through its complete evidence chain:
          Claim -> EvidenceSpan -> SourceUnit -> Document (doc_id, page_no, context_hash, provenance)
        """
        target = next((c for c in self.claims if c.get("id") == claim_id), None)
        if target is None:
            return None
        return self._build_single_claim_lineage(target)

    def _build_single_claim_lineage(self, claim: Mapping[str, Any]) -> dict[str, Any]:
        chain: list[dict[str, Any]] = []
        for ev_id in claim.get("evidence_ids") or []:
            span, unit = self._resolve_span_and_unit(str(ev_id))
            if span is not None:
                doc_id = (unit.get("doc_id") if unit else None) or span.get("doc_id") or "unknown"
                page_no = int((unit.get("page_no") if unit else None) or span.get("page_no") or 1)
                chain.append({
                    "evidence_ref": str(ev_id),
                    "span_id": span["id"],
                    "quote": span.get("quote", ""),
                    "provenance_verified": bool(span.get("provenance_verified")),
                    "provenance_reason": str(span.get("provenance_reason") or ""),
                    "source_unit_id": unit["id"] if unit else str(span.get("source_unit_id") or ""),
                    "source_unit_found": unit is not None,
                    "doc_id": doc_id,
                    "page_no": page_no,
                    "context_hash": (unit.get("context_hash") if unit else None) or span.get("context_hash", ""),
                    "content_hash": unit.get("content_hash", "") if unit else "",
                    "doi": unit.get("doi", "") if unit else "",
                    "provenance": list(unit.get("provenance") or []) if unit else [],
                })
            elif unit is not None:
                chain.append({
                    "evidence_ref": str(ev_id),
                    "span_id": "",
                    "quote": unit.get("text_preview", ""),
                    "provenance_verified": True,
                    "provenance_reason": "direct_source_unit_link",
                    "source_unit_id": unit["id"],
                    "source_unit_found": True,
                    "doc_id": unit.get("doc_id", "unknown"),
                    "page_no": int(unit.get("page_no") or 1),
                    "context_hash": unit.get("context_hash", ""),
                    "content_hash": unit.get("content_hash", ""),
                    "doi": unit.get("doi", ""),
                    "provenance": list(unit.get("provenance") or []),
                })
            else:
                chain.append({
                    "evidence_ref": str(ev_id),
                    "span_id": "",
                    "quote": "",
                    "provenance_verified": False,
                    "provenance_reason": "unresolved_evidence_ref",
                    "source_unit_id": "",
                    "source_unit_found": False,
                    "doc_id": "",
                    "page_no": 0,
                    "context_hash": "",
                    "content_hash": "",
                    "doi": "",
                    "provenance": [],
                })

        lineage_complete = bool(
            chain
            and all(item["provenance_verified"] and item["source_unit_found"] and item["doc_id"] for item in chain)
        )
        supporting_docs = sorted({item["doc_id"] for item in chain if item.get("doc_id")})
        return {
            "claim_id": str(claim.get("id") or ""),
            "claim_text": str(claim.get("text") or ""),
            "claim_status": str(claim.get("status") or "UNVERIFIED"),
            "evidence_chain": chain,
            "lineage_complete": lineage_complete,
            "supporting_documents": supporting_docs,
        }

    def build_claim_lineage(self) -> list[dict[str, Any]]:
        """Builds the complete Claim -> EvidenceSpan -> SourceUnit -> Document lineage table."""
        return [self._build_single_claim_lineage(c) for c in self.claims]

    def audit_source_quality(self) -> dict[str, Any]:
        """
        Audits multi-document source quality across the EvidenceStore:
          - Detects duplicate/preprint content overlap across distinct doc_ids via content_hash.
          - Counts multi-source corroborated units (provenance across >= 2 distinct documents).
          - Computes provenance_pass_rate and verified_lineage_rate.
        """
        by_content_hash: dict[str, set[str]] = defaultdict(set)
        multi_doc_units = 0
        with_doi_count = 0

        for unit in self.sources.values():
            doc_id = str(unit.get("doc_id") or "")
            cnt_hash = str(unit.get("content_hash") or "")
            if cnt_hash and doc_id:
                by_content_hash[cnt_hash].add(doc_id)
            if unit.get("doi"):
                with_doi_count += 1
            prov_docs = {
                str(p.get("doc_id") or "")
                for p in (unit.get("provenance") or [])
                if isinstance(p, Mapping) and p.get("doc_id")
            }
            if len(prov_docs) >= 2:
                multi_doc_units += 1

        duplicate_groups = [
            {"content_hash": h, "documents": sorted(docs)}
            for h, docs in by_content_hash.items()
            if len(docs) >= 2
        ]

        total_spans = len(self.spans)
        verified_spans = sum(1 for s in self.spans.values() if s.get("provenance_verified"))
        provenance_pass_rate = round(verified_spans / total_spans, 4) if total_spans > 0 else 0.0

        lineage = self.build_claim_lineage()
        supported_lineage = [item for item in lineage if item["claim_status"] == "SUPPORT"]
        complete_supported = sum(1 for item in supported_lineage if item["lineage_complete"])
        verified_lineage_rate = (
            round(complete_supported / len(supported_lineage), 4) if supported_lineage else 0.0
        )

        return {
            "unique_document_count": len({u.get("doc_id") for u in self.sources.values() if u.get("doc_id")}),
            "source_unit_count": len(self.sources),
            "units_with_doi": with_doi_count,
            "multi_source_corroborated_units": multi_doc_units,
            "duplicate_content_groups": duplicate_groups,
            "has_cross_document_duplicates": len(duplicate_groups) > 0,
            "provenance_pass_rate": provenance_pass_rate,
            "verified_lineage_rate": verified_lineage_rate,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "EvidenceStore":
        store = cls()
        schema = state.get("document_schema")
        prior_store = schema.get("evidence_store") if isinstance(schema, Mapping) else None
        prior_sources = (
            list(prior_store.get("sources") or [])
            if isinstance(prior_store, Mapping)
            else []
        )
        current_units = list(state.get("source_units") or state.get("retrieved_docs") or [])
        all_units = [*prior_sources, *current_units]
        store.add_source_units(all_units)
        store.add_evidence_spans(state.get("evidence_spans") or [], all_units)
        store.record_verification(
            state.get("verified_claims") or [],
            state.get("rejected_claims") or [],
            state.get("evidence_gaps") or [],
            draft_claims=state.get("claims") or [],
        )
        return store

    def summary(self) -> dict[str, Any]:
        verified_count = sum(1 for c in self.claims if c.get("status") == "SUPPORT")
        contradict_count = len(self.contradictions)
        unsupported_count = sum(1 for c in self.claims if c.get("status") != "SUPPORT")
        verified_spans = sum(1 for s in self.spans.values() if s.get("provenance_verified"))
        unverified_spans = sum(
            1
            for s in self.spans.values()
            if not s.get("provenance_verified") and s.get("provenance_reason") == "unverified_no_units"
        )
        failed_spans = sum(
            1
            for s in self.spans.values()
            if not s.get("provenance_verified") and s.get("provenance_reason") != "unverified_no_units"
        )
        lineage = self.build_claim_lineage()
        lineage_complete_count = sum(1 for item in lineage if item["lineage_complete"])
        total_spans = len(self.spans)
        return {
            "source_count": len(self.sources),
            "unique_documents": sorted({s["doc_id"] for s in self.sources.values() if s.get("doc_id")}),
            "span_count": total_spans,
            "verified_span_count": verified_spans,
            "unverified_span_count": unverified_spans,
            "failed_verification_span_count": failed_spans,
            "provenance_pass_rate": round(verified_spans / total_spans, 4) if total_spans > 0 else 0.0,
            "claim_count": len(self.claims),
            "verified_claim_count": verified_count,
            "unsupported_claim_count": unsupported_count,
            "contradiction_count": contradict_count,
            "lineage_complete_claim_count": lineage_complete_count,
            "gap_count": len(self.gaps),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "source_quality": self.audit_source_quality(),
            "sources": list(self.sources.values()),
            "spans": list(self.spans.values()),
            "claims": list(self.claims),
            "lineage": self.build_claim_lineage(),
            "contradictions": list(self.contradictions),
            "gaps": list(self.gaps),
        }
