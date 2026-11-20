"""
Agent 2: Reader Agent (Multi-Source Document Structure & Evidence Unit Parser).
Combines:
  1. Single PDF, Multi-PDF (`pdf_paths`), and `retrieved_docs` unified ingestion.
  2. Multi-source provenance preservation when deduplicating identical text across documents.
  3. Resilient partial-failure handling: one corrupted PDF never discards other valid sources;
     processing_status reflects EXTRACTION_FAILED only when no valid source could be extracted.
"""

from __future__ import annotations

from typing import Any
from core.base_agent import BaseAgent
from core.state import AgentState, SourceUnit, compute_content_hash
from agents.agent2_reader.tools.pdf_parser import parse_pdf_document


class ReaderAgent(BaseAgent):
    """Reader Agent responsible for unified multi-source layout parsing and provenance-preserving SourceUnits."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="ReaderAgent", max_internal_retry=max_internal_retry)

    def _merge_or_append_unit(
        self,
        unit: SourceUnit,
        units_by_content_hash: dict[str, SourceUnit],
        ordered_units: list[SourceUnit],
    ) -> None:
        """Deduplicates identical passages across documents while preserving ALL provenance records."""
        c_hash = compute_content_hash(unit.text)
        existing = units_by_content_hash.get(c_hash)
        if existing is not None:
            existing.merge_provenance(unit)
        else:
            units_by_content_hash[c_hash] = unit
            ordered_units.append(unit)

    def run(self, state: AgentState) -> dict[str, Any]:
        pdf_path = state.get("pdf_path")
        raw_pdf_paths = state.get("pdf_paths") or []
        retrieved_docs = state.get("retrieved_docs", []) or []

        # Collect unique PDF paths preserving order
        pdf_list: list[str] = []
        for p in ([pdf_path] if pdf_path else []) + list(raw_pdf_paths):
            if p and isinstance(p, str) and p not in pdf_list:
                pdf_list.append(p)

        self.logger.info(
            f"ReaderAgent processing: pdf_count={len(pdf_list)}, retrieved_docs={len(retrieved_docs)}"
        )

        units_by_content_hash: dict[str, SourceUnit] = {}
        ordered_units: list[SourceUnit] = []
        document_schema: dict[str, Any] = {}
        schemas_collected: list[dict[str, Any]] = []
        failed_pdfs: list[dict[str, str]] = []

        # 1. Parse all supplied PDF(s) with resilient per-file error handling
        for p in pdf_list:
            try:
                schema, units = parse_pdf_document(p)
                if schema:
                    schemas_collected.append(schema)
                for u in units:
                    if isinstance(u, SourceUnit):
                        su = u
                    elif isinstance(u, dict):
                        su = SourceUnit(
                            id=str(u.get("id") or f"u_{len(ordered_units)}"),
                            text=str(u.get("text") or ""),
                            doc_id=str(u.get("doc_id") or p),
                            page_no=int(u.get("page_no", 1)),
                            bbox=u.get("bbox"),
                            character_offset=int(u.get("character_offset", 0)),
                            context_hash=str(u.get("context_hash") or ""),
                            provenance=list(u.get("provenance") or []),
                        )
                    else:
                        continue
                    if su.text.strip():
                        self._merge_or_append_unit(su, units_by_content_hash, ordered_units)
            except Exception as exc:
                self.logger.warning(f"[ReaderAgent] Failed to parse PDF '{p}': {exc}")
                failed_pdfs.append({"path": p, "error": str(exc)})

        # 2. Always unify retrieved_docs alongside PDF units (multi-source support)
        curr_offset = sum(len(u.text) for u in ordered_units)
        for i, doc in enumerate(retrieved_docs):
            if not isinstance(doc, dict):
                continue
            text = str(doc.get("text") or "").strip()
            if not text:
                continue
            doc_id = str(doc.get("doc_id") or doc.get("id") or f"retrieved_{i}")
            raw_page = doc.get("page_no", 1)
            page_no = int(raw_page) if isinstance(raw_page, int) and not isinstance(raw_page, bool) and raw_page >= 0 else 1

            unit = SourceUnit(
                id=f"{doc_id}_unit_{i}",
                text=text,
                doc_id=doc_id,
                page_no=page_no,
                character_offset=curr_offset,
            )
            self._merge_or_append_unit(unit, units_by_content_hash, ordered_units)
            curr_offset += len(text) + 1

        source_units = [u.to_dict() for u in ordered_units]

        if len(schemas_collected) == 1:
            document_schema = schemas_collected[0]
        elif len(schemas_collected) > 1:
            document_schema = {
                "source": "multi_pdf",
                "documents": schemas_collected,
                "total_units": len(source_units),
                "pages": [p for s in schemas_collected for p in (s.get("pages") or [])],
                "sections": [sec for s in schemas_collected for sec in (s.get("sections") or [])],
            }
        else:
            document_schema = {
                "source": "retrieved_corpus",
                "total_units": len(source_units),
                "sections": [{"unit_id": u["id"], "category": "chunk"} for u in source_units],
            }

        # Determine processing_status based on extraction outcome
        processing_status = "SUCCESS"
        error_stage = None
        error_code = None
        error_message = None

        if failed_pdfs and not source_units:
            processing_status = "EXTRACTION_FAILED"
            error_stage = "reader"
            error_code = "PDF_EXTRACTION_FAILED"
            error_message = "; ".join(f"{f['path']}: {f['error']}" for f in failed_pdfs)
        elif failed_pdfs and source_units:
            # Partial extraction failure: valid sources preserved, error explicitly flagged
            processing_status = "SUCCESS"
            error_stage = "reader"
            error_code = "PARTIAL_EXTRACTION_FAILURE"
            error_message = "; ".join(f"{f['path']}: {f['error']}" for f in failed_pdfs)

        trace = self.log_action(
            "document_reading_completed",
            {
                "has_pdf": bool(pdf_list),
                "pdf_count": len(pdf_list),
                "failed_pdf_count": len(failed_pdfs),
                "failed_pdfs": failed_pdfs,
                "total_source_units": len(source_units),
                "processing_status": processing_status,
                "pages_detected": len(document_schema.get("pages", [])),
                "sections_detected": len(document_schema.get("sections", [])),
            },
        )

        return {
            "document_schema": document_schema,
            "source_units": source_units,
            "processing_status": processing_status,
            "error_stage": error_stage,
            "error_code": error_code,
            "error_message": error_message,
            "reasoning_trace": [trace],
        }
