"""
Column-Aware Document & PDF Layout Parser.
Extracts verbatim SourceUnits with character offsets, bounding boxes, and SHA-256 provenance hashes.
Features:
  - Multi-column reading order sorting (prevents horizontal interleaving on 2-column papers).
  - Multi-format support: PDF, TXT, Markdown.
  - Safe fallback when PyMuPDF is not installed.
"""

from __future__ import annotations

import os
from typing import Any
from core.state import SourceUnit
from utils.logger import logger
import re

def is_boilerplate(text: str) -> bool:
    t = text.lower().strip()
    if "quyền riêng tư" in t and ("cookie" in t or "chúng tôi sử dụng" in t):
        return True
    if t in ["từ chối", "đồng ý tất cả", "từ chối\nđồng ý tất cả", "từ chối đồng ý tất cả"]:
        return True
    if t.startswith("http://") or t.startswith("https://"):
        return True
    if re.fullmatch(r"\d+/\d+", t):
        return True
    if re.fullmatch(r"\d{1,2}:\d{2}\s+\d{1,2}/\d{1,2}/\d{2,4}", t):
        return True
    return False



def sort_blocks_column_aware(blocks: list[tuple], page_width: float) -> list[tuple]:
    """
    Sorts PyMuPDF blocks according to standard scientific reading order.
    Detects two-column layouts and orders left column top-to-bottom, then right column.
    """
    if not blocks:
        return []

    x_mid = page_width / 2.0
    left_blocks = []
    right_blocks = []
    full_span_blocks = []

    for b in blocks:
        # b: (x0, y0, x1, y1, text, block_no, block_type)
        x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
        text = b[4].strip() if len(b) > 4 else ""
        if not text:
            continue

        width = x1 - x0
        # If block spans across more than 75% of page (e.g. title or abstract), treat as full span
        if width > (page_width * 0.75):
            full_span_blocks.append(b)
        elif x1 <= (x_mid + 25):
            left_blocks.append(b)
        elif x0 >= (x_mid - 25):
            right_blocks.append(b)
        else:
            # Slightly cross-center block
            if x0 < x_mid:
                left_blocks.append(b)
            else:
                right_blocks.append(b)

    # Sort each group vertically by y0 (top to bottom)
    left_blocks.sort(key=lambda b: b[1])
    right_blocks.sort(key=lambda b: b[1])
    full_span_blocks.sort(key=lambda b: b[1])

    # If two clear columns exist, interleave full-span titles then left column then right column
    if left_blocks and right_blocks:
        # Blocks before columns (like Header/Title)
        min_col_y = min(left_blocks[0][1], right_blocks[0][1])
        headers = [b for b in full_span_blocks if b[1] < min_col_y]
        footers = [b for b in full_span_blocks if b[1] >= min_col_y]
        return headers + left_blocks + right_blocks + footers

    # Single column layout: sort purely by Y
    all_blocks = [b for b in blocks if len(b) > 4 and b[4].strip()]
    all_blocks.sort(key=lambda b: b[1])
    return all_blocks


def parse_pdf_document(file_path: str) -> tuple[dict[str, Any], list[SourceUnit]]:
    """
    Parses a document (PDF, TXT, MD) into:
      1. document_schema (Structured sections, headers, tables)
      2. source_units (Normalized text blocks with offsets & provenance hashes)
    """
    if not os.path.exists(file_path):
        logger.warning(f"Document path not found: {file_path}")
        return {"error": "File not found", "sections": []}, []

    source_units: list[SourceUnit] = []
    sections: list[dict[str, Any]] = []
    doc_id = os.path.splitext(os.path.basename(file_path))[0]
    ext = os.path.splitext(file_path)[1].lower()

    # 1. Plain Text / Markdown Handler
    if ext in [".txt", ".md"]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            curr_offset = 0
            for i, p in enumerate(paragraphs):
                unit_id = f"{doc_id}_p1_b{i}"
                unit = SourceUnit(
                    id=unit_id,
                    text=p,
                    doc_id=doc_id,
                    page_no=1,
                    character_offset=curr_offset,
                )
                source_units.append(unit)
                curr_offset += len(p) + 2
                sections.append({"unit_id": unit_id, "category": "paragraph", "page_no": 1})
            return {"doc_id": doc_id, "total_units": len(source_units), "sections": sections}, source_units
        except Exception as e:
            logger.error(f"Failed to read text file {file_path}: {e}")

    # 2. PDF Handler with Column-Aware Sorting & Heuristic Guardrails
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        curr_offset = 0
        pages_meta = []

        for page_idx in range(len(doc)):
            page_no = page_idx + 1
            page = doc[page_idx]
            raw_blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
            page_text = page.get_text("text").strip()
            has_text_layer = len(page_text) > 20
            images = page.get_images(full=True)
            page_area = max(page.rect.width * page.rect.height, 1.0)

            image_area_ratio = 0.0
            for img in images:
                try:
                    img_bbox = page.get_image_bbox(img)
                    image_area_ratio += (img_bbox.width * img_bbox.height) / page_area
                except Exception:
                    continue
            image_area_ratio = round(min(image_area_ratio, 1.0), 3)

            is_simple = len(raw_blocks) <= 15 and image_area_ratio < 0.1
            layout_type = "simple" if is_simple else ("complex" if has_text_layer else "scanned_image")

            pages_meta.append({
                "page_no": page_no,
                "has_text_layer": has_text_layer,
                "block_count": len(raw_blocks),
                "image_area_ratio": image_area_ratio,
                "layout_type": layout_type,
            })

            if not has_text_layer:
                logger.warning(f"Page {page_no} has no digital text layer (scanned image detected). Flagging for OCR fallback.")

            # Apply Column-Aware Sorting
            sorted_blocks = sort_blocks_column_aware(raw_blocks, page.rect.width)

            for b in sorted_blocks:
                text = b[4].strip()
                if not text or is_boilerplate(text):
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

        total_pages = len(doc)
        doc.close()
        logger.info(f"PyMuPDF parsed {len(source_units)} units across {total_pages} pages with column sorting.")

    except ImportError:
        logger.warning("PyMuPDF (fitz) is not installed. Using raw text fallback parser.")
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            content = "Binary PDF content unable to decode with plain text fallback."

        unit = SourceUnit(
            id=f"{doc_id}_p1_b0",
            text=content[:2500],
            doc_id=doc_id,
            page_no=1,
            character_offset=0,
        )
        source_units.append(unit)
        sections.append({"unit_id": unit.id, "category": "paragraph", "page_no": 1, "bbox": None})

    document_schema = {
        "doc_id": os.path.basename(file_path),
        "total_units": len(source_units),
        "pages": pages_meta if "pages_meta" in locals() else [],
        "sections": sections,
    }

    return document_schema, source_units
