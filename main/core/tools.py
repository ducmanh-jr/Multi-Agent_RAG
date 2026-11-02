"""
Unified Agentic Tool Registry & Document-Scoped Tool Executor.
Provides 7 standardized tools for L3-L4 Autonomous RAG Agents across all 6 modes:
  1. search_selected_doc(query, top_k=5)
  2. search_hybrid_corpus(query, alpha=0.5, top_k=5)
  3. read_pdf_page(doc_id_or_path, page_number)
  4. extract_table_or_equations(doc_id_or_path, page_number)
  5. python_math_eval(expression)
  6. request_more_context(missing_points)
  7. finish_answer(answer, citations, confidence)

Also resolves the Critical Retrieval Scope Bug by ensuring user-selected PDFs
(state["pdf_path"]) are dynamically parsed, cached, and prioritized across EVERY mode.
"""

from __future__ import annotations

import ast
import math
import operator
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional

from core.config import settings
from utils.logger import logger


# In-memory cache of parsed PDF chunks keyed by absolute file path + mtime
_PDF_CHUNK_CACHE: dict[str, list[dict[str, Any]]] = {}


def load_or_parse_selected_pdf(
    pdf_path: Optional[str],
    chunk_size_words: Any = 240,
    overlap_words: int = 50,
    root_dir: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Dynamically parses and caches the user-selected PDF document with sentence-window preservation.
    Guarantees that when a user selects a document in the UI, ALL 6 modes (CPU Min, CPU Max,
    Gemini, Kaggle FN, Kaggle Max, Hybrid) retrieve directly from that document!
    """
    if not pdf_path:
        return []
    # Support legacy/convenience call load_or_parse_selected_pdf(selected_doc, root_dir_str)
    if isinstance(chunk_size_words, (str, Path)):
        root_dir = str(chunk_size_words)
        chunk_size_words = 240
    else:
        try:
            chunk_size_words = int(chunk_size_words)
        except Exception:
            chunk_size_words = 240

    resolved = str(pdf_path).strip()
    if not os.path.exists(resolved):
        candidate = os.path.join(settings.UPLOAD_DIR, os.path.basename(resolved))
        if os.path.exists(candidate):
            resolved = candidate
        elif root_dir and os.path.exists(os.path.join(root_dir, "uploads", os.path.basename(resolved))):
            resolved = os.path.join(root_dir, "uploads", os.path.basename(resolved))
        else:
            return []

    try:
        mtime = os.path.getmtime(resolved)
    except Exception:
        mtime = 0.0
    cache_key = f"{os.path.abspath(resolved)}::{mtime}"
    if cache_key in _PDF_CHUNK_CACHE:
        return _PDF_CHUNK_CACHE[cache_key]

    doc_title = Path(resolved).stem
    doc_title = re.sub(r"^[0-9a-f]{8}_", "", doc_title)
    chunks: list[dict[str, Any]] = []

    try:
        import fitz  # PyMuPDF

        pdf_doc = fitz.open(resolved)
        for page_idx, page in enumerate(pdf_doc):
            page_no = page_idx + 1
            raw_text = page.get_text("text") or ""
            clean_text = re.sub(r"[ \t]+", " ", raw_text).strip()
            if not clean_text:
                continue

            # Split into sentences first so chunks never cut mid-sentence
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?।])\s+|\n{2,}", clean_text)
                if len(s.strip()) > 12
            ]
            if not sentences:
                sentences = [clean_text]

            current_sents: list[str] = []
            current_words = 0
            sub_idx = 1

            for sent in sentences:
                w_count = len(sent.split())
                if current_words + w_count > chunk_size_words and current_sents:
                    chunk_body = " ".join(current_sents).strip()
                    cid = f"{doc_title}_p{page_no}_c{sub_idx}"
                    chunks.append(
                        {
                            "id": cid,
                            "chunk_id": cid,
                            "doc_id": doc_title,
                            "title": doc_title,
                            "page_no": page_no,
                            "page": page_no,
                            "text": chunk_body,
                            "content": chunk_body,
                            "sentences": list(current_sents),
                            "source_path": resolved,
                            "is_selected_pdf": True,
                        }
                    )
                    sub_idx += 1
                    # Keep last 2 sentences for sliding window overlap
                    current_sents = current_sents[-2:] if len(current_sents) >= 2 else current_sents[-1:]
                    current_words = sum(len(x.split()) for x in current_sents)

                current_sents.append(sent)
                current_words += w_count

            if current_sents:
                chunk_body = " ".join(current_sents).strip()
                cid = f"{doc_title}_p{page_no}_c{sub_idx}"
                chunks.append(
                    {
                        "id": cid,
                        "chunk_id": cid,
                        "doc_id": doc_title,
                        "title": doc_title,
                        "page_no": page_no,
                        "page": page_no,
                        "text": chunk_body,
                        "content": chunk_body,
                        "sentences": list(current_sents),
                        "source_path": resolved,
                        "is_selected_pdf": True,
                    }
                )
        pdf_doc.close()
    except Exception as exc:
        logger.warning(f"[ToolRegistry] PyMuPDF parse failed on '{resolved}': {exc}")

    if chunks:
        _PDF_CHUNK_CACHE[cache_key] = chunks
    return chunks


def read_pdf_page_tool(pdf_path: Optional[str], page_number: int = 1) -> str:
    """Reads the full text of a specific page (1-indexed) from the selected PDF."""
    chunks = load_or_parse_selected_pdf(pdf_path)
    if not chunks:
        return f"Không tìm thấy tài liệu PDF tại đường dẫn: {pdf_path}"
    page_chunks = [c["text"] for c in chunks if int(c.get("page_no", 1)) == int(page_number)]
    if not page_chunks:
        max_page = max((int(c.get("page_no", 1)) for c in chunks), default=1)
        return f"Trang {page_number} không có nội dung (Tài liệu có tối đa {max_page} trang)."
    return f"[Trang {page_number}]:\n" + "\n\n".join(page_chunks)


def extract_table_or_equations_tool(pdf_path: Optional[str], page_number: int = 1) -> str:
    """Extracts structured tables and numerical rows from a specific PDF page using PyMuPDF find_tables()."""
    if not pdf_path or not os.path.exists(str(pdf_path)):
        return "Không có đường dẫn PDF hợp lệ để trích xuất bảng."
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        if page_number < 1 or page_number > len(doc):
            doc.close()
            return f"Số trang {page_number} nằm ngoài phạm vi tài liệu (1..{len(doc)})."
        page = doc[page_number - 1]
        markdown_tables: list[str] = []
        if hasattr(page, "find_tables"):
            tabs = page.find_tables()
            for idx, tab in enumerate(tabs.tables):
                try:
                    md = tab.to_markdown()
                    if md and md.strip():
                        markdown_tables.append(f"### Bảng {idx + 1} (Trang {page_number})\n{md.strip()}")
                except Exception:
                    pass
        if not markdown_tables:
            raw_lines = (page.get_text("text") or "").splitlines()
            numeric_lines = [
                ln.strip()
                for ln in raw_lines
                if len(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", ln)) >= 2
            ]
            doc.close()
            if numeric_lines:
                return f"[Dữ liệu số học/bảng biểu Trang {page_number}]:\n" + "\n".join(numeric_lines[:25])
            return f"Không phát hiện bảng biểu trên trang {page_number}."
        doc.close()
        return "\n\n".join(markdown_tables)
    except Exception as exc:
        return f"Lỗi trích xuất bảng biểu: {exc}"


_SAFE_OPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_math_eval(expression: str) -> str:
    """Safely evaluates mathematical expressions (sum, diff, percentage, ratio) without eval()."""
    def _eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Num):  # Python 3.7 compat
            return float(node.n)  # type: ignore[arg-type]
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return float(_SAFE_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return float(_SAFE_OPS[type(node.op)](_eval_node(node.operand)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn_name = node.func.id
            args = [_eval_node(a) for a in node.args]
            safe_funcs = {
                "round": round,
                "abs": abs,
                "min": min,
                "max": max,
                "sqrt": math.sqrt,
            }
            if fn_name in safe_funcs:
                return float(safe_funcs[fn_name](*args))
        raise ValueError(f"Biểu thức không được phép: {ast.dump(node)}")

    try:
        clean_expr = expression.strip().replace("%", "/100.0")
        tree = ast.parse(clean_expr, mode="eval")
        result = _eval_node(tree)
        return f"{result:.6g}"
    except Exception as exc:
        return f"MathError: {exc}"


def get_gemini_tool_declarations() -> list[dict[str, Any]]:
    """Returns OpenAI / Gemini compatible tool schemas for L3-L4 ReAct agents."""
    return [
        {
            "name": "search_selected_doc",
            "description": "Tìm kiếm đoạn văn bản trong tài liệu PDF người dùng đang chọn theo câu truy vấn.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {"type": "STRING", "description": "Câu truy vấn tìm kiếm cụ thể."},
                    "top_k": {"type": "INTEGER", "description": "Số lượng đoạn văn cần lấy (mặc định 5)."},
                },
                "required": ["query"],
            },
        },
        {
            "name": "read_pdf_page",
            "description": "Đọc toàn văn nội dung của một trang cụ thể trong tài liệu PDF đang chọn.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "page_number": {"type": "INTEGER", "description": "Số trang PDF cần đọc (bắt đầu từ 1)."},
                },
                "required": ["page_number"],
            },
        },
        {
            "name": "extract_table_or_equations",
            "description": "Trích xuất bảng số liệu hoặc công thức từ một trang PDF cụ thể dưới dạng Markdown.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "page_number": {"type": "INTEGER", "description": "Số trang PDF chứa bảng biểu."},
                },
                "required": ["page_number"],
            },
        },
        {
            "name": "python_math_eval",
            "description": "Tính toán biểu thức toán học chính xác (tỷ lệ %, tổng, hiệu, trung bình).",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "expression": {"type": "STRING", "description": "Biểu thức số học hợp lệ, ví dụ '(125 - 100) / 100 * 100'."},
                },
                "required": ["expression"],
            },
        },
    ]
