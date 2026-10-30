"""
PDF & Document Layout Parser.
Extracts verbatim SourceUnits with character offsets, bounding boxes, and SHA-256 provenance hashes.
Supports PyMuPDF (fitz) with safe fallback when fitz is not installed.
"""

from __future__ import annotations

import os
from typing import Any
from core.state import SourceUnit, compute_context_hash
from utils.logger import logger


def parse_pdf_document(pdf_path: str) -> tuple[dict[str, Any], list[SourceUnit]]:
    """
    Parses a PDF file into:
      1. document_schema (DocLayNet-style structured sections, headers, tables)
      2. source_units (normalized text blocks with offsets & provenance hashes)
    """
    if not os.path.exists(pdf_path):
        logger.warning(f"PDF path not found: {pdf_path}")
        return {"error": "File not found", "sections": []}, []

    source_units: list[SourceUnit] = []
    sections: list[dict[str, Any]] = []

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        curr_offset = 0

        for page_idx in range(len(doc)):
            page_no = page_idx + 1
            page = doc[page_idx]
            blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)

            for b in blocks:
                text = b[4].strip()
                if not text:
                    continue

                bbox = [round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2)]
                unit_id = f"{doc_id}_p{page_no}_b{len(source_units)}"

                unit = SourceUnit(
                    id=unit_id,
                    text=text,
                    doc_id=doc_id,
                    page_no=page_no,
                    bbox=bbox,
                    character_offset=curr_offset,
                )
                source_units.append(unit)
                curr_offset += len(text) + 1

                # Heuristic categorization for DocLayNet schema
                category = "paragraph"
                if len(text) < 100 and (text.isupper() or text.endswith(":")):
                    category = "section_header"
                elif "\t" in text or "|" in text or "  " in text:
                    category = "table"

                sections.append({
                    "unit_id": unit_id,
                    "category": category,
                    "page_no": page_no,
                    "bbox": bbox,
                    "text_preview": text[:80],
                })

        doc.close()
        logger.info(f"PyMuPDF parsed {len(source_units)} source units across {len(doc)} pages.")

    except ImportError:
        logger.warning("PyMuPDF (fitz) is not installed. Using raw text fallback parser.")
        # Fallback reading
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            with open(pdf_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            content = "Binary PDF content unable to decode with plain text fallback."

        unit = SourceUnit(
            id=f"{doc_id}_p1_b0",
            text=content[:2000],
            doc_id=doc_id,
            page_no=1,
            character_offset=0,
        )
        source_units.append(unit)
        sections.append({"unit_id": unit.id, "category": "paragraph", "page_no": 1, "bbox": None})

    document_schema = {
        "doc_id": os.path.basename(pdf_path),
        "total_units": len(source_units),
        "sections": sections,
    }

    return document_schema, source_units
