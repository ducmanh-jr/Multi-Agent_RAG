"""
Unified State and Data Contracts for Multi-Agent Research RAG System (L3-L4 Agentic Upgrade).
Combines:
  1. LangGraph StateGraph Reducer compatibility (TypedDict + operator.add).
  2. Research-assist strict contracts (SourceUnits, Claims, Evidence Gaps, Verification Guards).
  3. PaperQA2 Context Hashing & Citation Provenance IDs.
  4. Stanford STORM Multi-Perspective Inquiry attributes.
  5. Two-Tier Critic & Cross-Model Verification metadata for Hybrid & 5 upgraded modes.
"""

from __future__ import annotations

import hashlib
import operator
from dataclasses import asdict, dataclass, field
from typing import Annotated, Any, Literal, Optional, TypedDict

from core.schemas import (
    EvidenceStatus,
    InputStatus,
    PrimaryIntentType,
    ProcessingStatus,
    QueryResolutionStatus,
    VALID_EVIDENCE_STATUSES,
    VALID_INPUT_STATUSES,
    VALID_PRIMARY_INTENTS,
    VALID_PROCESSING_STATUSES,
    VALID_QUERY_RESOLUTION_STATUSES,
)


def compute_context_hash(text: str, doc_id: str = "", page_no: int = 0) -> str:
    """Computes a SHA-256 fingerprint for a chunk/quote to guarantee citation provenance."""
    raw = f"{doc_id}:{page_no}:{text.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def compute_content_hash(text: str) -> str:
    """Computes a document-agnostic SHA-256 hash of normalized text content for multi-source deduplication."""
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()[:16]


def normalize_provenance_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """
    Validates and normalizes a provenance record for multi-source citation tracking.
    Raises ValueError if required fields are missing or invalid.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"Provenance entry must be a dict, got {type(entry).__name__}.")

    doc_id = entry.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("Provenance entry requires a non-empty string 'doc_id'.")

    raw_page = entry.get("page_no")
    if isinstance(raw_page, bool) or not isinstance(raw_page, int) or raw_page < 0:
        raise ValueError(f"Provenance 'page_no' must be a non-negative integer, got {raw_page!r}.")

    raw_offset = entry.get("character_offset", 0)
    if isinstance(raw_offset, bool) or not isinstance(raw_offset, int) or raw_offset < 0:
        raise ValueError(
            f"Provenance 'character_offset' must be a non-negative integer, got {raw_offset!r}."
        )

    unit_id = str(entry.get("unit_id") or "").strip()

    return {
        "doc_id": doc_id.strip(),
        "page_no": raw_page,
        "unit_id": unit_id,
        "character_offset": raw_offset,
    }


@dataclass
class SourceUnit:
    """Verbatim text unit extracted directly from a source document with multi-source provenance."""
    id: str
    text: str
    doc_id: str = "unknown"
    page_no: int = 1
    bbox: Optional[list[float]] = None
    character_offset: int = 0
    context_hash: str = ""
    provenance: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.context_hash:
            self.context_hash = compute_context_hash(self.text, self.doc_id, self.page_no)

        if not self.provenance:
            self.provenance = [
                normalize_provenance_entry(
                    {
                        "doc_id": self.doc_id or "unknown",
                        "page_no": int(self.page_no) if isinstance(self.page_no, int) and not isinstance(self.page_no, bool) and self.page_no >= 0 else 0,
                        "unit_id": self.id,
                        "character_offset": int(self.character_offset) if isinstance(self.character_offset, int) and not isinstance(self.character_offset, bool) and self.character_offset >= 0 else 0,
                    }
                )
            ]
        else:
            normalized: list[dict[str, Any]] = []
            seen: set[tuple[str, int, str]] = set()
            for item in self.provenance:
                rec = normalize_provenance_entry(item)
                key = (rec["doc_id"], rec["page_no"], rec["unit_id"])
                if key not in seen:
                    seen.add(key)
                    normalized.append(rec)
            self.provenance = normalized

    def add_provenance(
        self,
        doc_id: str,
        page_no: int,
        unit_id: str = "",
        character_offset: int = 0,
    ) -> None:
        """Appends a validated provenance record if not already present."""
        rec = normalize_provenance_entry(
            {
                "doc_id": doc_id,
                "page_no": page_no,
                "unit_id": unit_id,
                "character_offset": character_offset,
            }
        )
        key = (rec["doc_id"], rec["page_no"], rec["unit_id"])
        existing = {(p["doc_id"], p["page_no"], p.get("unit_id", "")) for p in self.provenance}
        if key not in existing:
            self.provenance.append(rec)

    def merge_provenance(self, other: "SourceUnit | list[dict[str, Any]]") -> None:
        """Merges provenance entries from another SourceUnit or list of records without losing sources."""
        entries = other.provenance if isinstance(other, SourceUnit) else other
        if not isinstance(entries, list):
            raise ValueError("merge_provenance expects a SourceUnit or list of provenance dicts.")
        for entry in entries:
            rec = normalize_provenance_entry(entry)
            self.add_provenance(
                doc_id=rec["doc_id"],
                page_no=rec["page_no"],
                unit_id=rec["unit_id"],
                character_offset=rec["character_offset"],
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deduplicate_and_merge_source_units(
    units: list[SourceUnit | dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Deduplicates identical text units across multiple documents/chunks while
    merging and preserving all provenance records (`doc_id`, `page_no`, `unit_id`, `character_offset`).
    Shared across ReaderAgent and all 6 mode Reader implementations.
    """
    by_hash: dict[str, SourceUnit] = {}
    ordered: list[SourceUnit] = []

    for idx, item in enumerate(units or []):
        if isinstance(item, SourceUnit):
            su = item
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("content") or "").strip()
            if not text:
                continue
            raw_page = item.get("page_no", 1)
            page_no = (
                int(raw_page)
                if isinstance(raw_page, int) and not isinstance(raw_page, bool) and raw_page >= 0
                else 1
            )
            raw_offset = item.get("character_offset", 0)
            offset = (
                int(raw_offset)
                if isinstance(raw_offset, int) and not isinstance(raw_offset, bool) and raw_offset >= 0
                else 0
            )
            su = SourceUnit(
                id=str(item.get("id") or f"u_{idx + 1}"),
                text=text,
                doc_id=str(item.get("doc_id") or "unknown"),
                page_no=page_no,
                bbox=item.get("bbox"),
                character_offset=offset,
                context_hash=str(item.get("context_hash") or ""),
                provenance=list(item.get("provenance") or []),
            )
        else:
            continue

        if not su.text.strip():
            continue

        c_hash = compute_content_hash(su.text)
        existing = by_hash.get(c_hash)
        if existing is not None:
            existing.merge_provenance(su)
        else:
            by_hash[c_hash] = su
            ordered.append(su)

    return [u.to_dict() for u in ordered]


class EvidenceSpan:
    """A verified quotation extracted by the Reader/Summarizer with offset tracking.
    Flexible constructor supporting unit_id, source_unit_id, quote, text, and optional id.
    """
    def __init__(
        self,
        id: Optional[str] = None,
        quote: str = "",
        source_unit_id: Optional[str] = None,
        doc_id: str = "doc",
        page_no: int = 1,
        context_hash: str = "",
        relevance_score: float = 1.0,
        bbox: Optional[list[float]] = None,
        unit_id: Optional[str] = None,
        text: Optional[str] = None,
        **kwargs: Any,
    ):
        import uuid
        self.id = id or kwargs.get("ev_id") or f"ev_{uuid.uuid4().hex[:8]}"
        self.quote = quote or text or kwargs.get("content", "")
        self.source_unit_id = source_unit_id or unit_id or kwargs.get("evidence_id") or "u1"
        self.unit_id = self.source_unit_id
        self.doc_id = doc_id or "doc"
        try:
            self.page_no = int(page_no)
        except Exception:
            self.page_no = 1
        self.context_hash = context_hash or compute_context_hash(self.quote, self.doc_id, self.page_no)
        self.relevance_score = float(relevance_score) if relevance_score is not None else 1.0
        self.bbox = bbox

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "quote": self.quote,
            "text": self.quote,
            "source_unit_id": self.source_unit_id,
            "unit_id": self.source_unit_id,
            "doc_id": self.doc_id,
            "page_no": self.page_no,
            "context_hash": self.context_hash,
            "relevance_score": self.relevance_score,
            "bbox": self.bbox,
        }


class AtomicClaim:
    """An atomic, independently verifiable proposition generated by the Summarizer.
    Flexible constructor supporting legacy and cross-mode kwargs (unit_id, claim, text).
    """
    def __init__(
        self,
        id: Optional[str] = None,
        text: str = "",
        evidence_ids: Optional[list[str]] = None,
        confidence: float = 1.0,
        unit_id: Optional[str] = None,
        **kwargs: Any,
    ):
        import uuid
        self.id = id or kwargs.get("claim_id") or f"claim_{uuid.uuid4().hex[:8]}"
        self.text = text or kwargs.get("claim") or kwargs.get("claim_text") or ""
        self.claim = self.text
        self.claim_text = self.text
        ev = list(evidence_ids) if evidence_ids is not None else []
        u_id = unit_id or kwargs.get("evidence_unit_id") or kwargs.get("source_unit_id")
        if u_id and u_id not in ev:
            ev.append(u_id)
        self.evidence_ids = ev
        self.confidence = float(confidence) if confidence is not None else 1.0
        self.unit_id = u_id or (ev[0] if ev else None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "claim_id": self.id,
            "text": self.text,
            "claim": self.text,
            "claim_text": self.text,
            "evidence_ids": self.evidence_ids,
            "confidence": self.confidence,
            "unit_id": self.unit_id,
        }


class ClaimCheckResult:
    """Fact-checker decision on an atomic claim.
    Flexible constructor supporting legacy 'claim', 'rationale', and new 'claim_text', 'reason' kwargs.
    """
    def __init__(
        self,
        claim_id: str = "",
        claim_text: str = "",
        claim: str = "",
        text: str = "",
        status: str = "SUPPORT",
        reason: str = "",
        rationale: str = "",
        numerical_passed: bool = True,
        dates_passed: bool = True,
        confidence_score: float = 1.0,
        confidence: float = 1.0,
        contradicting_evidence: Optional[list[str]] = None,
        **kwargs: Any,
    ):
        self.claim_text = claim_text or claim or text or ""
        self.claim = self.claim_text
        self.text = self.claim_text
        self.claim_id = claim_id or f"claim_{abs(hash(self.claim_text)) % 10000}"
        self.status = str(status).upper() if status else "SUPPORT"
        self.reason = reason or rationale or "Verified by Fact-Checker"
        self.rationale = self.reason
        self.numerical_passed = bool(numerical_passed)
        self.dates_passed = bool(dates_passed)
        self.confidence_score = float(
            confidence_score if confidence_score is not None else (confidence or 1.0)
        )
        self.confidence = self.confidence_score
        self.contradicting_evidence = list(contradicting_evidence or [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "claim": self.claim_text,
            "text": self.claim_text,
            "status": self.status,
            "reason": self.reason,
            "rationale": self.rationale,
            "numerical_passed": self.numerical_passed,
            "dates_passed": self.dates_passed,
            "confidence_score": self.confidence_score,
            "confidence": self.confidence,
            "contradicting_evidence": self.contradicting_evidence,
        }


@dataclass
class EvidenceGap:
    """An information void identified by Fact-Checker or Summarizer requiring iterative retrieval."""
    missing_aspect: str
    suggested_query: str
    perspective: str = "general"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentState(TypedDict, total=False):
    """
    Unified LangGraph StateGraph state passed across all Agent nodes.
    Supports incremental state updates and reducer accumulation via Annotated[..., operator.add] (e.g. reasoning_trace).
    Preserves all legacy fields for API/Frontend/Agent compatibility while adding the 4 orthogonal status groups.
    """
    # 1. User & Supervisor Planning Context
    query: str
    intent: str
    primary_intent: PrimaryIntentType
    sub_intent: str
    auto_scout: bool
    needs_scout: bool
    needs_retrieval: bool
    tool_request: Optional[dict[str, Any]]
    tool_result: Optional[dict[str, Any]]
    complexity: str  # "LOW" | "MEDIUM" | "HIGH"
    required_entities: list[str]
    is_numeric_query: bool
    is_comparative_query: bool
    pdf_path: Optional[str]
    pdf_paths: Optional[list[str]]
    perspectives: list[str]

    # 1b. Independent Status Contracts (4 Orthogonal State Groups — AGENT.md Section 9)
    input_status: InputStatus
    query_resolution: QueryResolutionStatus
    processing_status: ProcessingStatus
    evidence_status: EvidenceStatus
    error_stage: Optional[str]
    error_code: Optional[str]
    error_message: Optional[str]

    # 1c. Scout Discovery Fields
    scout_decision: str
    scout_message: Optional[str]

    # 2. Retrieval Stage
    sub_queries: list[str]
    retrieved_docs: list[dict[str, Any]]
    seen_chunks: list[dict[str, Any]]
    retrieval_strategy: str
    alpha: float
    top_k: int

    # 3. Document Reading Stage
    document_schema: dict[str, Any]
    source_units: list[dict[str, Any]]
    context_coverage: dict[str, Any]

    # 4. Summarization & Inner Critic Stage
    answer: str
    evidence_spans: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    critic_action: str  # "approve" | "rewrite" | "request_more_context" | "lower_confidence" | "verification_failed"
    critic_feedback: dict[str, Any]
    thinking_trace: str
    generator_model_family: str

    # 5. Fact-Checking & Cross-Model Verification Stage
    verified_claims: list[dict[str, Any]]
    rejected_claims: list[dict[str, Any]]
    evidence_gaps: list[dict[str, Any]]
    fact_check_label: Literal["support", "contradict", "not_enough_info"]
    statistical_audit: Optional[dict[str, Any]]
    code_verification: Optional[dict[str, Any]]
    verifier_model_family: str

    # 6. Report Generation Stage
    final_report: str
    short_answer: Optional[str]
    answer_status: Optional[str]
    written_report_data: dict[str, Any]
    citations: Optional[list[dict[str, Any]]]

    # 7. Orchestration & Budget Controls
    retry_count: int
    max_reflection_loops: int
    chat_history: list[dict[str, str]]
    query_with_context: Optional[str]
    factcheck_calls: int
    confidence: Literal["high", "low"]
    no_new_evidence: bool
    reasoning_trace: Annotated[list[dict[str, Any]], operator.add]


def create_initial_state(query: str, **overrides: Any) -> AgentState:
    """Creates a contract-compliant initial AgentState with orthogonal default statuses."""
    base: AgentState = {
        "query": query,
        "input_status": "OK",
        "query_resolution": "CLEAR",
        "processing_status": "PENDING",
        "evidence_status": "NOT_APPLICABLE",
        "retry_count": 0,
        "no_new_evidence": False,
        "reasoning_trace": [],
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def validate_state_contract(state: dict[str, Any]) -> list[str]:
    """
    Validates status values present in an AgentState dict against the 4-group contract.
    Returns a list of violation strings (empty if valid).
    """
    errors: list[str] = []
    if "primary_intent" in state and state["primary_intent"] not in VALID_PRIMARY_INTENTS:
        errors.append(f"Invalid primary_intent: {state['primary_intent']!r}")
    if "input_status" in state and state["input_status"] not in VALID_INPUT_STATUSES:
        errors.append(f"Invalid input_status: {state['input_status']!r}")
    if "query_resolution" in state and state["query_resolution"] not in VALID_QUERY_RESOLUTION_STATUSES:
        errors.append(f"Invalid query_resolution: {state['query_resolution']!r}")
    if "processing_status" in state and state["processing_status"] not in VALID_PROCESSING_STATUSES:
        errors.append(f"Invalid processing_status: {state['processing_status']!r}")
    if "evidence_status" in state and state["evidence_status"] not in VALID_EVIDENCE_STATUSES:
        errors.append(f"Invalid evidence_status: {state['evidence_status']!r}")
    return errors
