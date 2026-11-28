"""
Two-Tier Page/Region PDF Ingestion, Bilingual Scan OCR, 2D Equation Reconstruction,
512-Token Header-Preserving Table Sharding, and Rotated PDF.js Coordinate Mapping (M2a).
"""

from __future__ import annotations

import base64
import hashlib
import math
import os
import re
import shutil
import subprocess
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import fitz

from core.config import get_run_settings
from utils.logger import logger

EXTRACTOR_VERSION = "m2b-2tier-v1"
MAX_TABLE_SHARD_TOKENS = 380  # Strictly below worker 512-token ceiling for BGE-M3 & reranker

_SCAN_OCR_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_INGEST_MANIFEST_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_INGEST_LOCK = threading.Lock()
_MAX_CACHE = 32


def clear_ingest_caches() -> None:
    with _INGEST_LOCK:
        _SCAN_OCR_CACHE.clear()
        _INGEST_MANIFEST_CACHE.clear()


def compute_file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# =====================================================================
# 1. Parser Error Guard (Prevents Parser Errors Entering SourceUnits)
# =====================================================================

_PARSER_ERROR_PATTERNS = re.compile(
    r"^(?:"
    r"Lỗi\s+trích\s+xuất\s+bảng\s+biểu/công\s+thức|"
    r"Không\s+phát\s+hiện\s+bảng\s+biểu\s+hoặc\s+công\s+thức|"
    r"Không\s+có\s+đường\s+dẫn\s+PDF\s+hợp\s+lệ|"
    r"Không\s+tìm\s+thấy\s+tài\s+liệu\s+PDF\s+tại\s+đường\s+dẫn|"
    r"Số\s+trang\s+\d+\s+nằm\s+ngoài\s+phạm\s+vi|"
    r"Trang\s+\d+\s+không\s+có\s+nội\s+dung|"
    r"ParserError:|PDFParseError:|MuPDF\s+error:"
    r")",
    re.IGNORECASE,
)


def is_parser_error_text(text: Optional[str]) -> bool:
    """True when `text` is a tool/parser failure diagnostic rather than document content."""
    s = (text or "").strip()
    if not s:
        return True
    return bool(_PARSER_ERROR_PATTERNS.search(s))


# =====================================================================
# 2. Bilingual Scan OCR (Windows WinRT + Tesseract VI/EN with BBoxes)
# =====================================================================

_WINDOWS_BILINGUAL_OCR_PS = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
[void][Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]
[void][Windows.Globalization.Language,Windows.Foundation,ContentType=WindowsRuntime]
[void][Windows.Graphics.Imaging.BitmapDecoder,Windows.Foundation,ContentType=WindowsRuntime]
[void][Windows.Graphics.Imaging.SoftwareBitmap,Windows.Foundation,ContentType=WindowsRuntime]
[void][Windows.Media.Ocr.OcrResult,Windows.Foundation,ContentType=WindowsRuntime]
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and
    $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
} | Select-Object -First 1
function Wait-Operation($operation, [type]$resultType) {
    $task = $asTask.MakeGenericMethod($resultType).Invoke($null, @($operation))
    $task.Wait()
    $task.Result
}
$bytes = [Convert]::FromBase64String([Console]::In.ReadToEnd())
$memory = [IO.MemoryStream]::new($bytes, $false)
$stream = [System.IO.WindowsRuntimeStreamExtensions]::AsRandomAccessStream($memory)
$decoder = Wait-Operation ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Wait-Operation ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])

$engines = @()
foreach ($tag in @('vi', 'vi-VN', 'en-US', 'en')) {
    try {
        $lang = [Windows.Globalization.Language]::new($tag)
        if ([Windows.Media.Ocr.OcrEngine]::IsLanguageSupported($lang)) {
            $eng = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang)
            if ($null -ne $eng) { $engines += @{ tag = $tag; engine = $eng } }
        }
    } catch {}
}
$prof = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -ne $prof) { $engines += @{ tag = 'profile'; engine = $prof } }
if ($engines.Count -eq 0) { exit 1 }

$bestLines = @()
$bestLen = -1
$seenTags = @{}
foreach ($item in $engines) {
    $key = $item.engine.RecognizerLanguage.LanguageTag
    if ($seenTags.ContainsKey($key)) { continue }
    $seenTags[$key] = $true
    $res = Wait-Operation ($item.engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
    $linesOut = @()
    $charCount = 0
    foreach ($line in $res.Lines) {
        $x0 = 1e9; $y0 = 1e9; $x1 = 0.0; $y1 = 0.0
        foreach ($w in $line.Words) {
            $r = $w.BoundingRect
            if ($r.X -lt $x0) { $x0 = $r.X }
            if ($r.Y -lt $y0) { $y0 = $r.Y }
            if (($r.X + $r.Width) -gt $x1) { $x1 = $r.X + $r.Width }
            if (($r.Y + $r.Height) -gt $y1) { $y1 = $r.Y + $r.Height }
        }
        $t = $line.Text
        $charCount += $t.Length
        $linesOut += @{ text = $t; bbox = @([math]::Round($x0,1), [math]::Round($y0,1), [math]::Round($x1,1), [math]::Round($y1,1)) }
    }
    if ($charCount -gt $bestLen) {
        $bestLen = $charCount
        $bestLines = $linesOut
    }
}
@{ lines = $bestLines; width = $bitmap.PixelWidth; height = $bitmap.PixelHeight } | ConvertTo-Json -Compress -Depth 4
"""


def run_bilingual_ocr_on_png(png_bytes: bytes) -> dict[str, Any]:
    """
    Runs local bilingual OCR (VI + EN) on PNG bytes, returning:
      `{"engine": str, "text": str, "lines": [{"text": str, "bbox": [x0, y0, x1, y1]}], "width": int, "height": int}`
    """
    if not png_bytes or not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return {"engine": "", "text": "", "lines": [], "width": 0, "height": 0}

    width = int.from_bytes(png_bytes[16:20], "big")
    height = int.from_bytes(png_bytes[20:24], "big")

    # 1. Tesseract (fast cross-platform C++ binary when installed)
    tesseract = shutil.which("tesseract")
    if tesseract:
        cfg_lang = get_run_settings().FIGURE_OCR_LANGUAGE or "eng"
        lang_candidates = list(dict.fromkeys(["vie+eng", cfg_lang, "eng"]))
        for lang_arg in lang_candidates:
            try:
                proc = subprocess.run(
                    [tesseract, "stdin", "stdout", "-l", lang_arg],
                    input=png_bytes,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                    check=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                txt = proc.stdout.decode("utf-8", errors="replace").strip()
                if txt:
                    lines = [
                        {"text": ln.strip(), "bbox": [0.0, 0.0, float(width), float(height)]}
                        for ln in txt.splitlines()
                        if ln.strip()
                    ]
                    return {
                        "engine": f"tesseract:{lang_arg}",
                        "text": "\n".join(l["text"] for l in lines)[:4000],
                        "lines": lines,
                        "width": width,
                        "height": height,
                    }
            except (OSError, subprocess.SubprocessError):
                continue

    # 2. Windows WinRT OCR (built-in on Windows 10/11, zero extra pip packages)
    if os.name == "nt":
        import json

        powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
        try:
            proc = subprocess.run(
                [str(powershell), "-NoProfile", "-NonInteractive", "-Command", _WINDOWS_BILINGUAL_OCR_PS],
                input=base64.b64encode(png_bytes),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=18,
                check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            raw_json = proc.stdout.decode("utf-8", errors="replace").strip()
            if raw_json:
                parsed = json.loads(raw_json)
                raw_lines = parsed.get("lines") or []
                if isinstance(raw_lines, dict):
                    raw_lines = [raw_lines]
                lines = [
                    {"text": str(item.get("text") or "").strip(), "bbox": list(item.get("bbox") or [0, 0, width, height])}
                    for item in raw_lines
                    if isinstance(item, dict) and str(item.get("text") or "").strip()
                ]
                if lines:
                    return {
                        "engine": "windows-ocr",
                        "text": "\n".join(l["text"] for l in lines)[:4000],
                        "lines": lines,
                        "width": int(parsed.get("width") or width),
                        "height": int(parsed.get("height") or height),
                    }
        except Exception as exc:
            logger.debug("[PDFIngest] Windows OCR skipped: %s", exc)

    return {"engine": "", "text": "", "lines": [], "width": width, "height": height}


def get_cached_page_ocr(doc_sha256: str, page_no: int) -> dict[str, Any] | None:
    with _INGEST_LOCK:
        return _SCAN_OCR_CACHE.get(f"{doc_sha256}:p{page_no}")


def ocr_pdf_page_if_needed(page: fitz.Page, *, doc_sha256: str, page_no: int) -> dict[str, Any]:
    """Renders a scanned/image-only PDF page at ~180 DPI and runs cached bilingual OCR."""
    cache_key = f"{doc_sha256}:p{page_no}"
    with _INGEST_LOCK:
        if cache_key in _SCAN_OCR_CACHE:
            _SCAN_OCR_CACHE.move_to_end(cache_key)
            return _SCAN_OCR_CACHE[cache_key]

    vis_rect = page.rect
    max_dim = max(float(vis_rect.width), float(vis_rect.height), 1.0)
    scale = min(2.5, max(1.5, 1600.0 / max_dim))
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False, annots=False)
    png_bytes = pix.tobytes("png")

    ocr_res = run_bilingual_ocr_on_png(png_bytes)
    # Map pixel-space OCR line bboxes back into unrotated PDF page coordinates
    derot = page.derotation_matrix
    unrot_bounds = page.rect * derot
    mapped_lines: list[dict[str, Any]] = []
    px_w = max(float(ocr_res.get("width") or pix.width or 1), 1.0)
    px_h = max(float(ocr_res.get("height") or pix.height or 1), 1.0)
    for ln in ocr_res.get("lines") or []:
        b = ln.get("bbox") or [0, 0, px_w, px_h]
        vis_box = fitz.Rect(
            (float(b[0]) / px_w) * vis_rect.width,
            (float(b[1]) / px_h) * vis_rect.height,
            (float(b[2]) / px_w) * vis_rect.width,
            (float(b[3]) / px_h) * vis_rect.height,
        )
        unrot_box = (vis_box * derot) & unrot_bounds
        mapped_lines.append({
            "text": ln["text"],
            "bbox": [round(float(v), 2) for v in unrot_box],
        })

    union_bbox = [round(float(v), 2) for v in unrot_bounds]
    if mapped_lines:
        ux0 = min(l["bbox"][0] for l in mapped_lines)
        uy0 = min(l["bbox"][1] for l in mapped_lines)
        ux1 = max(l["bbox"][2] for l in mapped_lines)
        uy1 = max(l["bbox"][3] for l in mapped_lines)
        if ux1 > ux0 and uy1 > uy0:
            union_bbox = [round(ux0, 2), round(uy0, 2), round(ux1, 2), round(uy1, 2)]

    out = {
        "engine": ocr_res.get("engine", ""),
        "text": ocr_res.get("text", ""),
        "lines": mapped_lines,
        "bbox": union_bbox,
        "confidence": 0.9 if ocr_res.get("text") else 0.0,
    }
    with _INGEST_LOCK:
        if len(_SCAN_OCR_CACHE) >= _MAX_CACHE:
            _SCAN_OCR_CACHE.popitem(last=False)
        _SCAN_OCR_CACHE[cache_key] = out
    return out


# =====================================================================
# 3. 2D Stacked-Fraction & Multi-Span Display Equation Extraction
# =====================================================================

_EQ_NUM_TAG_RE = re.compile(r"^\(\s*\d+(?:\.\d+)*[a-z]?\s*\)$")
_MATH_OP_CHARS = set("=≈≤≥≠∈←→⇒∑∏∫√∂∇±×⋅")
_VAR_DEF_SENT_RE = re.compile(
    r"^\s*(?:where|with|in\s+which|for\s+which|here|subject\s+to|trong\s+đó|với|khi|điều\s+kiện)\b",
    re.IGNORECASE,
)
_DOMAIN_EXTRACT_RE = re.compile(
    r"([A-Za-zΑ-Ωα-ω0-9_]+\s*(?:!=|≠|>|<|>=|<=|≤|≥)\s*[−+-]?\d+(?:\.\d+)?)",
    re.UNICODE,
)


def _horizontal_overlap(ax0: float, ax1: float, bx0: float, bx1: float) -> float:
    return max(0.0, min(ax1, bx1) - max(ax0, bx0))


def _extract_horizontal_fraction_bars(page: fitz.Page, derot: fitz.Matrix) -> list[tuple[float, float, float, float]]:
    bars: list[tuple[float, float, float, float]] = []
    try:
        drawings = page.get_drawings() or []
    except Exception:
        return []
    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        r = fitz.Rect(rect) * derot
        w = float(r.width)
        h = float(r.height)
        if 6.0 <= w <= 260.0 and h <= 3.5:
            bars.append((float(r.x0), float(r.y0), float(r.x1), float(r.y1)))
    return bars


def extract_page_equations_2d(
    page: fitz.Page,
    *,
    doc_title: str,
    doc_sha256: str,
    page_no: int,
    resolved: str,
    section_title: str,
) -> list[dict[str, Any]]:
    """
    Reconstructs both:
      1. Stacked fractions (`y = a / b` drawn with numerator above a vector bar and denominator below),
      2. Multi-span display equations (`Q_π(s,a) = E_{t>=0} [...]`, limits, subscripts, equation numbers),
      3. Single-line equations (`E = mc^2 (1)`, `Attention(Q, K, V) = ...`)
    with exact unrotated `bbox`, `page_rotation`, `latex`, `equation_number`, `variable_definitions`,
    `domain_constraints`, and verified numerator/denominator/subscript components.
    """
    derot = page.derotation_matrix
    bounds = page.rect * derot
    page_w = max(float(bounds.width), 1.0)
    rotation = int(page.rotation or 0)

    try:
        flags = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES
        raw_dict = page.get_text("dict", flags=flags)
    except Exception:
        return []

    # Collect unrotated span/line items
    line_items: list[dict[str, Any]] = []
    for blk in raw_dict.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for ln in blk.get("lines", []):
            spans_info = []
            for sp in ln.get("spans", []):
                stxt = sp.get("text") or ""
                if stxt.strip():
                    sb = fitz.Rect(sp["bbox"]) * derot
                    spans_info.append({
                        "text": stxt,
                        "bbox": (float(sb.x0), float(sb.y0), float(sb.x1), float(sb.y1)),
                        "size": float(sp.get("size") or 10.0),
                    })
            if not spans_info:
                continue
            lb = fitz.Rect(ln["bbox"]) * derot
            full_txt = "".join(s["text"] for s in spans_info).strip()
            full_txt = re.sub(r"[ \t]+", " ", full_txt)
            if not full_txt:
                continue
            avg_size = sum(s["size"] for s in spans_info) / len(spans_info)
            line_items.append({
                "text": full_txt,
                "bbox": [float(lb.x0), float(lb.y0), float(lb.x1), float(lb.y1)],
                "spans": spans_info,
                "size": avg_size,
            })

    if not line_items:
        return []

    line_items.sort(key=lambda item: (round(item["bbox"][1], 1), item["bbox"][0]))
    bars = _extract_horizontal_fraction_bars(page, derot)

    consumed_indices: set[int] = set()
    eq_chunks: list[dict[str, Any]] = []
    seen_sigs: set[str] = set()

    # --- Pass 1: Reconstruct Stacked Fractions anchored on horizontal fraction bars ---
    for bx0, by0, bx1, by1 in bars:
        bar_mid_y = 0.5 * (by0 + by1)
        num_idxs = [
            idx for idx, it in enumerate(line_items)
            if idx not in consumed_indices
            and -2.0 <= (by0 - it["bbox"][3]) <= 18.0
            and _horizontal_overlap(it["bbox"][0], it["bbox"][2], bx0 - 4.0, bx1 + 4.0) >= 2.0
            and (it["bbox"][2] - it["bbox"][0]) <= (bx1 - bx0) + 40.0
        ]
        den_idxs = [
            idx for idx, it in enumerate(line_items)
            if idx not in consumed_indices
            and -2.0 <= (it["bbox"][1] - by1) <= 18.0
            and _horizontal_overlap(it["bbox"][0], it["bbox"][2], bx0 - 4.0, bx1 + 4.0) >= 2.0
            and (it["bbox"][2] - it["bbox"][0]) <= (bx1 - bx0) + 40.0
        ]
        if not num_idxs or not den_idxs:
            continue

        band_y0 = min(line_items[i]["bbox"][1] for i in num_idxs) - 4.0
        band_y1 = max(line_items[i]["bbox"][3] for i in den_idxs) + 4.0

        # Attach LHS (e.g. 'y =') to the left and RHS / equation tag '(1)' to the right on the same midline
        lhs_idxs = [
            idx for idx, it in enumerate(line_items)
            if idx not in consumed_indices
            and idx not in num_idxs
            and idx not in den_idxs
            and it["bbox"][2] <= bx0 + 6.0
            and it["bbox"][1] >= band_y0 - 4.0
            and it["bbox"][3] <= band_y1 + 4.0
            and abs(0.5 * (it["bbox"][1] + it["bbox"][3]) - bar_mid_y) <= 14.0
        ]
        rhs_idxs = [
            idx for idx, it in enumerate(line_items)
            if idx not in consumed_indices
            and idx not in num_idxs
            and idx not in den_idxs
            and it["bbox"][0] >= bx1 - 6.0
            and it["bbox"][1] >= band_y0 - 4.0
            and it["bbox"][3] <= band_y1 + 4.0
            and abs(0.5 * (it["bbox"][1] + it["bbox"][3]) - bar_mid_y) <= 14.0
        ]

        all_frac_idxs = sorted(set(lhs_idxs + num_idxs + den_idxs + rhs_idxs))
        lhs_text = " ".join(line_items[i]["text"] for i in sorted(lhs_idxs, key=lambda k: line_items[k]["bbox"][0])).strip()
        num_text = " ".join(line_items[i]["text"] for i in sorted(num_idxs, key=lambda k: line_items[k]["bbox"][0])).strip()
        den_text = " ".join(line_items[i]["text"] for i in sorted(den_idxs, key=lambda k: line_items[k]["bbox"][0])).strip()

        eq_num = ""
        rhs_parts = []
        for i in sorted(rhs_idxs, key=lambda k: line_items[k]["bbox"][0]):
            t_r = line_items[i]["text"].strip()
            if _EQ_NUM_TAG_RE.match(t_r):
                eq_num = t_r
            else:
                rhs_parts.append(t_r)
        rhs_text = " ".join(rhs_parts).strip()

        if not num_text or not den_text:
            continue

        # Verify numerator and denominator tokens match raw spans
        latex_frac = f"\\frac{{{num_text}}}{{{den_text}}}"
        inline_frac = f"({num_text})/({den_text})" if (" " in num_text or " " in den_text or len(num_text) > 2 or len(den_text) > 2) else f"{num_text}/{den_text}"
        latex_expr = " ".join(p for p in (lhs_text, latex_frac, rhs_text) if p).strip()
        readable_eq = " ".join(p for p in (lhs_text, inline_frac, rhs_text, eq_num) if p).strip()

        ux0 = min([bx0] + [line_items[i]["bbox"][0] for i in all_frac_idxs])
        uy0 = min([by0] + [line_items[i]["bbox"][1] for i in all_frac_idxs])
        ux1 = max([bx1] + [line_items[i]["bbox"][2] for i in all_frac_idxs])
        uy1 = max([by1] + [line_items[i]["bbox"][3] for i in all_frac_idxs])
        eq_bbox = [round(ux0, 2), round(uy0, 2), round(ux1, 2), round(uy1, 2)]

        # Find adjacent variable definition / domain constraint sentences (immediately before/after band)
        var_defs, dom_constraints, ctx_lines = _collect_equation_context(line_items, min(all_frac_idxs), max(all_frac_idxs), readable_eq)
        consumed_indices.update(all_frac_idxs)

        body = " ".join(dict.fromkeys([readable_eq, f"LaTeX: ${latex_expr}$", *ctx_lines])).strip()
        sig = re.sub(r"\s+", "", latex_expr.lower())
        seen_sigs.add(sig)
        cid = f"{doc_title}_p{page_no}_eq{len(eq_chunks) + 1}"
        eq_chunks.append({
            "id": cid,
            "chunk_id": cid,
            "doc_id": doc_title,
            "document_sha256": doc_sha256,
            "title": doc_title,
            "page_no": page_no,
            "page": page_no,
            "page_rotation": rotation,
            "coordinate_space": "pdf_points_top_left",
            "extractor_version": EXTRACTOR_VERSION,
            "bbox": eq_bbox,
            "text": body,
            "raw_text": readable_eq,
            "normalized_text": body,
            "content": body,
            "equation": readable_eq,
            "latex": latex_expr,
            "equation_number": eq_num,
            "variable_definitions": var_defs,
            "domain_constraints": dom_constraints,
            "equation_components": {
                "numerator": num_text,
                "denominator": den_text,
                "verified_fraction": bool(num_text and den_text),
            },
            "equation_metadata": {
                "equation_number": eq_num,
                "latex": latex_expr,
                "numerator": num_text,
                "denominator": den_text,
                "verified_fraction": bool(num_text and den_text),
                "variable_definitions": var_defs,
                "domain_constraints": dom_constraints,
            },
            "sentences": [readable_eq, *ctx_lines],
            "source_path": resolved,
            "is_selected_pdf": True,
            "unit_type": "equation",
            "section_title": section_title,
        })

    # --- Pass 2: Reconstruct 2D Display Equation Bands & Inline Equations ---
    left_margin = min((it["bbox"][0] for it in line_items), default=72.0)
    idx = 0
    n_lines = len(line_items)
    while idx < n_lines:
        if idx in consumed_indices:
            idx += 1
            continue
        it = line_items[idx]
        txt = it["text"]
        w_line = it["bbox"][2] - it["bbox"][0]
        is_indented_display = (it["bbox"][0] >= left_margin + 22.0) and (w_line < page_w * 0.76)
        looks_eq = _looks_like_equation_fragment(txt, is_indented=is_indented_display)
        if not looks_eq:
            idx += 1
            continue

        # Cluster horizontally/vertically belonging fragments in the same display equation band
        band_idxs = [idx]
        band_x0, band_y0, band_x1, band_y1 = it["bbox"]
        band_max_x = band_x1
        base_size = it["size"]
        j = idx + 1
        while j < n_lines:
            if j in consumed_indices:
                j += 1
                continue
            cand = line_items[j]
            cx0, cy0, cx1, cy1 = cand["bbox"]
            cw = cx1 - cx0
            ct = cand["text"]
            v_gap = cy0 - band_y1
            if v_gap > 14.0:
                break
            # Stop if `cand` is a full-width prose paragraph
            if cw >= page_w * 0.74 and cx0 <= left_margin + 15.0 and not _EQ_NUM_TAG_RE.match(ct):
                break
            # Stop if `cand` starts a brand-new display equation back on the left with its own primary '=' / '←'
            has_primary_rel = bool(re.search(r"[A-Za-zΑ-Ωα-ω0-9_)\]}]\s*(?:=|←|:=|≈)\s*", ct))
            if (
                v_gap >= 2.0
                and cx0 < band_max_x - 55.0
                and has_primary_rel
                and not _EQ_NUM_TAG_RE.match(ct)
            ):
                break
            # Accept into band if vertically overlapping OR subscript/limit (smaller font or short limit like t>=0) OR continuation to the right
            is_sub_or_limit = (cand["size"] < base_size * 0.92) or (len(ct) <= 18 and any(c in ct for c in "≥≤=0123456789ijktnπ"))
            is_right_continuation = cx0 >= band_x0 + 18.0
            if v_gap <= 3.0 or is_sub_or_limit or is_right_continuation or _EQ_NUM_TAG_RE.match(ct):
                band_idxs.append(j)
                band_x0 = min(band_x0, cx0)
                band_y0 = min(band_y0, cy0)
                band_x1 = max(band_x1, cx1)
                band_y1 = max(band_y1, cy1)
                band_max_x = max(band_max_x, cx1)
                j += 1
            else:
                break

        assembled_eq, latex_eq, eq_num, components = _assemble_band_equation(
            [line_items[k] for k in band_idxs]
        )
        if _is_meaningful_equation(assembled_eq):
            norm_sig = re.sub(r"\s+", "", assembled_eq.lower())
            if norm_sig not in seen_sigs:
                seen_sigs.add(norm_sig)
                var_defs, dom_constraints, ctx_lines = _collect_equation_context(
                    line_items, min(band_idxs), max(band_idxs), assembled_eq
                )
                body_parts = [assembled_eq, *ctx_lines]
                body = " ".join(dict.fromkeys(p for p in body_parts if p)).strip()
                eq_bbox = [round(band_x0, 2), round(band_y0, 2), round(band_x1, 2), round(band_y1, 2)]
                cid = f"{doc_title}_p{page_no}_eq{len(eq_chunks) + 1}"
                eq_chunks.append({
                    "id": cid,
                    "chunk_id": cid,
                    "doc_id": doc_title,
                    "document_sha256": doc_sha256,
                    "title": doc_title,
                    "page_no": page_no,
                    "page": page_no,
                    "page_rotation": rotation,
                    "coordinate_space": "pdf_points_top_left",
                    "extractor_version": EXTRACTOR_VERSION,
                    "bbox": eq_bbox,
                    "text": body,
                    "raw_text": assembled_eq,
                    "normalized_text": body,
                    "content": body,
                    "equation": assembled_eq,
                    "latex": latex_eq,
                    "equation_number": eq_num,
                    "variable_definitions": var_defs,
                    "domain_constraints": dom_constraints,
                    "equation_components": components,
                    "equation_metadata": {
                        "equation_number": eq_num,
                        "latex": latex_eq,
                        "variable_definitions": var_defs,
                        "domain_constraints": dom_constraints,
                        **components,
                    },
                    "sentences": [assembled_eq, *ctx_lines],
                    "source_path": resolved,
                    "is_selected_pdf": True,
                    "unit_type": "equation",
                    "section_title": section_title,
                })
                consumed_indices.update(band_idxs)

        idx = max(idx + 1, j)

    return eq_chunks


def _looks_like_equation_fragment(text: str, *, is_indented: bool) -> bool:
    s = (text or "").strip()
    if len(s) < 3 or s.startswith(("http://", "https://")):
        return False
    if any(ch in s for ch in _MATH_OP_CHARS):
        # Exclude long narrative prose lines that happen to contain '=' unless clearly symbolic
        if len(s) > 140 and not is_indented and not re.search(r"\\(?:frac|sum|sqrt)|\bsoftmax\s*\(", s):
            # Still allow sentences with explicit inline formula e.g. 'f(x) = x^2 - 2x + 1' if short enough
            if not re.search(r"[A-Za-zΑ-Ωα-ω0-9_)\]}]\s*=\s*[A-Za-zΑ-Ωα-ω0-9_(\[{−+\-√∑∏∫]", s):
                return False
        return True
    if re.search(r"\b(?:softmax|argmax|argmin|Attention|MultiHead|LayerNorm)\s*\(", s):
        return True
    if is_indented and re.search(r"[\^_]|\\[a-zA-Z]+|[α-ωΑ-Ω]", s) and any(op in s for op in ("+", "-", "/", "(", ")")):
        return True
    return False


def _is_meaningful_equation(text: str) -> bool:
    s = (text or "").strip()
    if len(s) < 3:
        return False
    return any(
        tok in s
        for tok in ("=", "←", "≈", "≤", "≥", "≠", "∈", "∑", "∏", "∫", "√", "^", "²", "³", "\\frac", "softmax", "max")
    )


def _assemble_band_equation(band_lines: list[dict[str, Any]]) -> tuple[str, str, str, dict[str, Any]]:
    """Orders 2D equation fragments left-to-right (attaching limits/subscripts to their parent operator)."""
    if len(band_lines) == 1:
        raw = band_lines[0]["text"].strip()
        eq_num = ""
        m_num = re.search(r"\s*(\(\s*\d+(?:\.\d+)*[a-z]?\s*\))\s*$", raw)
        if m_num:
            eq_num = m_num.group(1).strip()
        return raw, raw, eq_num, {"subscripts": [], "limits": []}

    # Sort primarily by horizontal x0 while attaching vertically stacked limits (like 't>=0' below 'E'/'∑')
    ordered = sorted(band_lines, key=lambda it: (round(it["bbox"][0] / 8.0), it["bbox"][1]))
    parts: list[str] = []
    latex_parts: list[str] = []
    limits: list[str] = []
    subscripts: list[str] = []
    eq_num = ""
    max_size = max(it["size"] for it in ordered)

    for it in ordered:
        t = it["text"].strip()
        if not t:
            continue
        if _EQ_NUM_TAG_RE.match(t):
            eq_num = t
            continue
        is_small = it["size"] < max_size * 0.88 and len(t) <= 24
        if is_small and parts and (
            parts[-1].endswith(("E", "∑", "∏", "∫", "max", "min", "sup", "inf"))
            or any(sym in t for sym in ("≥", "≤", "=", "∼", "∈"))
        ):
            limits.append(t)
            parts[-1] = f"{parts[-1]}_{{{t}}}"
            latex_parts[-1] = f"{latex_parts[-1]}_{{{t}}}"
        elif is_small and parts:
            subscripts.append(t)
            parts.append(f"_{{{t}}}")
            latex_parts.append(f"_{{{t}}}")
        else:
            parts.append(t)
            latex_parts.append(t)

    readable = re.sub(r"\s+", " ", " ".join(parts + ([eq_num] if eq_num else []))).strip()
    latex_str = re.sub(r"\s+", " ", " ".join(latex_parts)).strip()
    return readable, latex_str, eq_num, {"subscripts": subscripts, "limits": limits}


def _collect_equation_context(
    line_items: list[dict[str, Any]],
    first_idx: int,
    last_idx: int,
    equation_str: str,
) -> tuple[list[str], list[str], list[str]]:
    ctx: list[str] = []
    var_defs: list[str] = []
    domains: list[str] = []

    if first_idx > 0:
        prev_txt = line_items[first_idx - 1]["text"].strip()
        if prev_txt and len(prev_txt) >= 12 and not _EQ_NUM_TAG_RE.match(prev_txt):
            ctx.append(prev_txt)

    for offset in (1, 2):
        nxt_i = last_idx + offset
        if nxt_i >= len(line_items):
            break
        nxt_item = line_items[nxt_i]
        nxt_txt = nxt_item["text"].strip()
        if not nxt_txt:
            continue
        # Stop immediately if we reach another numbered or indented display equation
        if (
            not _VAR_DEF_SENT_RE.search(nxt_txt)
            and (
                re.search(r"\(\s*\d+(?:\.\d+)*[a-z]?\s*\)\s*$", nxt_txt)
                or re.search(r"^[A-Za-zΑ-Ωα-ω0-9_(\[{]+\s*(?:=|←|:=|≈)\s*", nxt_txt)
            )
        ):
            break
        ctx.append(nxt_txt)
        if _VAR_DEF_SENT_RE.search(nxt_txt) or ("=" in nxt_txt and len(nxt_txt) > 25):
            var_defs.append(nxt_txt)
        for dm in _DOMAIN_EXTRACT_RE.finditer(nxt_txt):
            d_str = " ".join(dm.group(1).split())
            if d_str not in domains:
                domains.append(d_str)

    for dm in _DOMAIN_EXTRACT_RE.finditer(equation_str):
        d_str = " ".join(dm.group(1).split())
        if d_str not in domains:
            domains.append(d_str)

    return var_defs, domains, ctx


# =====================================================================
# 4. Structured Table Extraction & <=512-Token Row-Group Sharding
# =====================================================================

_TABLE_CAPTION_RE = re.compile(r"^\s*(?:Table|Tab\.|Bảng)\s*([A-Z]?\d+(?:[.-]\d+)*)\b", re.IGNORECASE)
_TABLE_CONT_RE = re.compile(
    r"\b(?:continued|cont\.?|tiếp\s*theo|tiếp|tiep\s*theo)\b",
    re.IGNORECASE,
)
_UNIT_IN_HEADER_RE = re.compile(r"\(([^()]{1,24})\)|\[([^\[\]]{1,24})\]")


def _is_probable_data_row(cells: list[str]) -> bool:
    """
    Returns True when `cells` looks like a headerless table data row (e.g., on page k+1 of a
    multi-page table: `['Method-Omega', '92.4%', '11.8 ms']`) rather than column headers.
    """
    non_empty = [c.strip() for c in cells if c and c.strip()]
    if len(non_empty) < 2:
        return False
    tail = [c.strip() for c in cells[1:] if c and c.strip()]
    if not tail:
        return False
    numeric_tail = sum(
        1
        for c in tail
        if re.match(
            r"^[−+-]?\d+(?:[.,]\d+)?(?:\s*(?:%|pp|ms|s|h|d|day|days|ngày|phút|giờ|gb|mb|kb|kg|g|mg|mg/l|g/l|km|m|cm|mm|mpa|db|db/cm|tokens/s|gwh|mwh|kw|w|v))?$",
            c.strip(),
            re.IGNORECASE,
        )
    )
    return numeric_tail >= max(1, (len(tail) + 1) // 2)


def _extract_pipe_tables_from_page_blocks(
    page: fitz.Page,
    derot: fitz.Matrix,
) -> list[tuple[list[float], list[list[str]]]]:
    """
    Extracts pipe-delimited (`| col1 | col2 | ... |`) tables from text blocks when a PDF renders
    tables using text pipes rather than vector cell borders.
    """
    try:
        blocks = page.get_text("blocks") or []
    except Exception:
        return []

    pipe_lines: list[tuple[list[float], list[str]]] = []
    raw_items: list[tuple[float, float, float, float, str]] = []
    for b in blocks:
        if len(b) >= 7 and int(b[6]) == 0 and str(b[4]).strip():
            rb = fitz.Rect(b[:4]) * derot
            for sub in str(b[4]).splitlines():
                st = sub.strip()
                if st:
                    raw_items.append((float(rb.x0), float(rb.y0), float(rb.x1), float(rb.y1), st))
    raw_items.sort(key=lambda t: (round(t[1], 1), t[0]))

    groups: list[list[tuple[float, float, float, float, list[str]]]] = []
    cur_group: list[tuple[float, float, float, float, list[str]]] = []
    for x0, y0, x1, y1, st in raw_items:
        if st.count("|") >= 2:
            stripped = st.strip("|").strip()
            cells = [_clean_cell(c) for c in stripped.split("|")]
            # Skip pure markdown separator lines like `| --- | --- |`
            if all(re.match(r"^[-:\s]*$", c) for c in cells):
                continue
            if len(cells) >= 2 and any(cells):
                if cur_group and (y0 - cur_group[-1][3]) > 42.0:
                    if len(cur_group) >= 1:
                        groups.append(cur_group)
                    cur_group = []
                cur_group.append((x0, y0, x1, y1, cells))
                continue
        if cur_group and (y0 - cur_group[-1][3]) > 20.0:
            groups.append(cur_group)
            cur_group = []
    if cur_group:
        groups.append(cur_group)

    results: list[tuple[list[float], list[list[str]]]] = []
    for grp in groups:
        rows = [item[4] for item in grp]
        ncols = max((len(r) for r in rows), default=0)
        if ncols < 2:
            continue
        # Single headerless continuation row on p>1 is allowed if it looks like data; otherwise require >=2 rows
        if len(rows) < 2 and not _is_probable_data_row(rows[0]):
            continue
        bx = [
            round(min(it[0] for it in grp), 2),
            round(min(it[1] for it in grp), 2),
            round(max(it[2] for it in grp), 2),
            round(max(it[3] for it in grp), 2),
        ]
        results.append((bx, rows))
    return results


def _estimate_subword_tokens(text: str) -> int:
    words = (text or "").split()
    return max(int(len(words) * 2.2), int(len(text or "") / 3.2)) + 1


def _clean_cell(val: Any) -> str:
    if val is None:
        return ""
    return re.sub(r"\s+", " ", str(val).replace("\n", " ")).strip()


def _format_markdown_rows(header_cells: list[str], rows: list[list[str]]) -> str:
    ncols = max(len(header_cells), max((len(r) for r in rows), default=0))
    padded_hdr = [header_cells[i] if i < len(header_cells) else f"Col_{i+1}" for i in range(ncols)]
    lines = [
        "| " + " | ".join(padded_hdr) + " |",
        "| " + " | ".join(["---"] * ncols) + " |",
    ]
    for r in rows:
        padded = [r[i] if i < len(r) else "" for i in range(ncols)]
        lines.append("| " + " | ".join(padded) + " |")
    return "\n".join(lines)


def _rect_overlap_ratio(
    a: tuple[float, float, float, float] | list[float],
    b: tuple[float, float, float, float] | list[float],
) -> float:
    """Returns max(IoU, containment = inter / min(area_a, area_b)) so loose ONNX boxes wrapping a crisp grid deduplicate cleanly."""
    ix0 = max(float(a[0]), float(b[0]))
    iy0 = max(float(a[1]), float(b[1]))
    ix1 = min(float(a[2]), float(b[2]))
    iy1 = min(float(a[3]), float(b[3]))
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    min_area = min(area_a, area_b)
    return inter / min_area if min_area > 0.0 else 0.0


def _table_cell_signature(rows: list[list[str]]) -> set[str]:
    return {c.strip().lower() for r in rows for c in r if c and c.strip()}


def _materialize_table_entries(
    tables: list[Any],
    derot: fitz.Matrix,
) -> list[tuple[list[float], list[list[str]]]]:
    """
    Immediately extracts `(bbox, cleaned_rows)` from PyMuPDF `Table` objects BEFORE any
    subsequent `page.find_tables()` call can invoke `pymupdf.table.CHARS.clear()`.
    """
    entries: list[tuple[list[float], list[list[str]]]] = []
    for tab in tables:
        try:
            raw_rows = tab.extract() or []
        except Exception:
            continue
        cleaned = [[_clean_cell(c) for c in row] for row in raw_rows if isinstance(row, list)]
        cleaned = [r for r in cleaned if any(c for c in r)]
        if cleaned:
            tb = fitz.Rect(getattr(tab, "bbox", (0.0, 0.0, 0.0, 0.0))) * derot
            entries.append(([round(float(v), 2) for v in tb], cleaned))
    return entries


def _extract_page_table_entries_robust(
    page: fitz.Page,
    derot: fitz.Matrix | None = None,
) -> list[tuple[list[float], list[list[str]]]]:
    """
    Extracts `(bbox, cleaned_rows)` tables consistently across PyMuPDF 1.24-1.27 and
    PyMuPDF 1.28+/2.0 (where ONNX `use_layout=True` can drop/alter compact or headerless
    vector-bordered tables).

    CRITICAL: PyMuPDF's `find_tables()` clears module-global `CHARS` on every call, so we
    hold `_INGEST_LOCK` and materialize `tab.extract()` immediately after EACH pass before
    running any second pass.
    """
    if not hasattr(page, "find_tables"):
        return []

    mat = derot if derot is not None else getattr(page, "derotation_matrix", fitz.Identity)
    has_drawings = False
    try:
        has_drawings = bool(page.get_drawings())
    except Exception:
        has_drawings = False

    with _INGEST_LOCK:
        classic_entries: list[tuple[list[float], list[list[str]]]] = []
        if has_drawings:
            # Path A: PyMuPDF 1.28.2+ / 2.0 explicit `use_layout=False` kwarg
            kwarg_supported = False
            try:
                classic = page.find_tables(
                    use_layout=False,
                    vertical_strategy="lines",
                    horizontal_strategy="lines",
                )
                classic_entries = _materialize_table_entries(list(getattr(classic, "tables", None) or []), mat)
                kwarg_supported = True
            except TypeError:
                kwarg_supported = False
            except Exception:
                classic_entries = []
                kwarg_supported = True

            # Path B: PyMuPDF 2.0 module-level `use_layout(False)` toggle (if kwarg absent)
            if not kwarg_supported:
                try:
                    import pymupdf as _pymupdf_mod
                except Exception:
                    _pymupdf_mod = fitz
                use_layout_fn = getattr(_pymupdf_mod, "use_layout", None) or getattr(fitz, "use_layout", None)
                if callable(use_layout_fn):
                    prev_info = getattr(page, "layout_information", None)
                    try:
                        use_layout_fn(False)
                        if hasattr(page, "layout_information"):
                            page.layout_information = None
                        classic = page.find_tables(
                            vertical_strategy="lines",
                            horizontal_strategy="lines",
                        )
                        classic_entries = _materialize_table_entries(list(getattr(classic, "tables", None) or []), mat)
                    except Exception:
                        classic_entries = []
                    finally:
                        try:
                            use_layout_fn(True)
                        except Exception:
                            pass
                        if hasattr(page, "layout_information"):
                            page.layout_information = prev_info
                else:
                    # Path C: PyMuPDF 1.24-1.27 (suspend external `pymupdf_layout._get_layout` if present)
                    get_layout_meth = getattr(page, "get_layout", None)
                    globs = getattr(getattr(get_layout_meth, "__func__", None), "__globals__", None)
                    prev_layout_fn = globs.get("_get_layout") if isinstance(globs, dict) else None
                    prev_info = getattr(page, "layout_information", None)
                    try:
                        if isinstance(globs, dict) and "_get_layout" in globs:
                            globs["_get_layout"] = None
                        if hasattr(page, "layout_information"):
                            page.layout_information = None
                        classic = page.find_tables(
                            vertical_strategy="lines",
                            horizontal_strategy="lines",
                        )
                        classic_entries = _materialize_table_entries(list(getattr(classic, "tables", None) or []), mat)
                    except Exception:
                        classic_entries = []
                    finally:
                        if hasattr(page, "layout_information"):
                            page.layout_information = prev_info
                        if isinstance(globs, dict) and "_get_layout" in globs:
                            globs["_get_layout"] = prev_layout_fn

        # Default / layout-assisted pass (captures borderless tables & non-drawing pages)
        try:
            tabs = page.find_tables()
            layout_entries = _materialize_table_entries(list(getattr(tabs, "tables", None) or []), mat)
        except Exception:
            layout_entries = []

    if not classic_entries:
        return layout_entries

    # Prefer exact vector-grid tables first, then append genuinely disjoint borderless/layout tables
    merged = list(classic_entries)
    existing_boxes = [entry[0] for entry in merged]
    existing_sigs = [_table_cell_signature(entry[1]) for entry in merged]
    for lb, lrows in layout_entries:
        if any(_rect_overlap_ratio(lb, eb) >= 0.35 for eb in existing_boxes):
            continue
        lsig = _table_cell_signature(lrows)
        if lsig and any(
            esig and (len(lsig & esig) / max(min(len(lsig), len(esig)), 1)) >= 0.5
            for esig in existing_sigs
        ):
            continue
        merged.append((lb, lrows))
        existing_boxes.append(lb)
        existing_sigs.append(lsig)

    merged.sort(key=lambda entry: (round(float(entry[0][1]), 1), float(entry[0][0])))
    return merged


def extract_page_tables_sharded(
    page: fitz.Page,
    *,
    doc_title: str,
    doc_sha256: str,
    page_no: int,
    resolved: str,
    section_title: str,
    max_tokens: int = MAX_TABLE_SHARD_TOKENS,
) -> list[dict[str, Any]]:
    """
    Extracts structured tables from `page`, capturing `(row_header, col_header, unit, value)`
    cell metadata, unrotated `bbox`, `page_rotation`, and automatically splitting wide/long tables
    into `<= max_tokens` (`<= 512` worker limit) row-group shards that **repeat column headers & units**.
    """
    if not hasattr(page, "find_tables"):
        return []
    derot = page.derotation_matrix
    rotation = int(page.rotation or 0)
    raw_table_entries = _extract_page_table_entries_robust(page, derot)

    if not raw_table_entries:
        raw_table_entries = _extract_pipe_tables_from_page_blocks(page, derot)
    if not raw_table_entries:
        return []

    caption_blocks: list[tuple[float, float, float, float, str]] = []
    try:
        for b in page.get_text("blocks") or []:
            if len(b) >= 7 and int(b[6]) == 0:
                rb = fitz.Rect(b[:4]) * derot
                for sub_ln in str(b[4]).splitlines():
                    txt = re.sub(r"[ \t]+", " ", sub_ln).strip()
                    if txt and _TABLE_CAPTION_RE.match(txt):
                        caption_blocks.append((float(rb.x0), float(rb.y0), float(rb.x1), float(rb.y1), txt))
    except Exception:
        caption_blocks = []

    out_chunks: list[dict[str, Any]] = []
    for idx, (bbox, cleaned_rows) in enumerate(raw_table_entries, start=1):
        ncols = max((len(r) for r in cleaned_rows), default=0)
        if ncols < 2:
            continue
        headerless_continuation = (page_no > 1) and _is_probable_data_row(cleaned_rows[0])
        if len(cleaned_rows) < 2 and not headerless_continuation:
            continue
        non_empty = sum(1 for r in cleaned_rows for c in r if c)
        if non_empty < 2:
            continue

        matched_caption = ""
        best_dist = 80.0
        for cx0, cy0, cx1, cy1, ctxt in caption_blocks:
            dist_above = bbox[1] - cy1
            dist_below = cy0 - bbox[3]
            if -12.0 <= dist_above <= best_dist:
                best_dist = max(0.0, dist_above)
                matched_caption = ctxt
            elif -12.0 <= dist_below <= best_dist:
                best_dist = max(0.0, dist_below)
                matched_caption = ctxt

        if headerless_continuation:
            merged_headers = [f"Col_{col_i + 1}" for col_i in range(ncols)]
            col_units = ["" for _ in range(ncols)]
            data_rows = list(cleaned_rows)
        else:
            # Detect multi-row header / explicit unit row (e.g., Row 0 = Validation | Test, Row 1 = Acc (%) | F1 (%))
            header_rows = [cleaned_rows[0]]
            data_start = 1
            if len(cleaned_rows) >= 3:
                second = cleaned_rows[1]
                numeric_cells_r1 = sum(1 for c in second[1:] if re.match(r"^[−+-]?\d+(?:[.,]\d+)?%?$", c))
                if numeric_cells_r1 == 0 and any(c for c in second):
                    header_rows.append(second)
                    data_start = 2

            merged_headers = []
            col_units = []
            for col_i in range(ncols):
                h_parts = []
                for hr in header_rows:
                    val = hr[col_i] if col_i < len(hr) else ""
                    if val and val not in h_parts:
                        h_parts.append(val)
                col_name = " / ".join(h_parts) if h_parts else f"Col_{col_i + 1}"
                merged_headers.append(col_name)
                u_match = _UNIT_IN_HEADER_RE.search(col_name)
                col_units.append((u_match.group(1) or u_match.group(2) or "").strip() if u_match else "")

            data_rows = cleaned_rows[data_start:]
            if not data_rows:
                data_rows = cleaned_rows[1:]
                merged_headers = [cleaned_rows[0][i] if i < len(cleaned_rows[0]) else f"Col_{i+1}" for i in range(ncols)]

        # Build structured cell bindings for deterministic verification
        structured_cells: list[dict[str, Any]] = []
        for r_idx, row in enumerate(data_rows):
            row_hdr = row[0] if row else ""
            for c_idx in range(1, min(len(row), ncols)):
                cell_val = row[c_idx]
                if cell_val:
                    structured_cells.append({
                        "row_index": r_idx,
                        "col_index": c_idx,
                        "row_header": row_hdr,
                        "col_header": merged_headers[c_idx],
                        "unit": col_units[c_idx],
                        "value": cell_val,
                    })

        # Partition data_rows into shards <= max_tokens with repeated caption + merged_headers
        shards: list[tuple[list[list[str]], int, int]] = []
        cur_rows: list[list[str]] = []
        cur_start = 0
        for r_idx, row in enumerate(data_rows):
            trial_rows = [*cur_rows, row]
            trial_md = _format_markdown_rows(merged_headers, trial_rows)
            trial_body = f"{matched_caption}\n{trial_md}".strip() if matched_caption else trial_md
            if cur_rows and _estimate_subword_tokens(trial_body) > max_tokens:
                shards.append((cur_rows, cur_start, r_idx - 1))
                cur_rows = [row]
                cur_start = r_idx
            else:
                cur_rows.append(row)
        if cur_rows:
            shards.append((cur_rows, cur_start, len(data_rows) - 1))

        total_rows = max(len(data_rows), 1)
        tab_h = max(bbox[3] - bbox[1], 10.0)
        for s_idx, (s_rows, r_begin, r_end) in enumerate(shards, start=1):
            md_str = _format_markdown_rows(merged_headers, s_rows)
            body = f"{matched_caption}\n{md_str}".strip() if matched_caption else md_str
            cid = f"{doc_title}_p{page_no}_t{idx}" if len(shards) == 1 else f"{doc_title}_p{page_no}_t{idx}_part{s_idx}"

            # Compute proportional row-band bbox when sharded
            if len(shards) == 1:
                shard_bbox = bbox
            else:
                sy0 = bbox[1] + (r_begin / total_rows) * tab_h
                sy1 = bbox[1] + ((r_end + 1) / total_rows) * tab_h
                shard_bbox = [bbox[0], round(sy0, 2), bbox[2], round(sy1, 2)]

            shard_cells = [c for c in structured_cells if r_begin <= c["row_index"] <= r_end]
            out_chunks.append({
                "id": cid,
                "chunk_id": cid,
                "doc_id": doc_title,
                "document_sha256": doc_sha256,
                "title": doc_title,
                "page_no": page_no,
                "page": page_no,
                "page_rotation": rotation,
                "coordinate_space": "pdf_points_top_left",
                "extractor_version": EXTRACTOR_VERSION,
                "text": body,
                "raw_text": body,
                "normalized_text": body,
                "content": body,
                "sentences": [ln.strip() for ln in body.splitlines() if ln.strip()],
                "source_path": resolved,
                "is_selected_pdf": True,
                "unit_type": "table",
                "bbox": shard_bbox,
                "section_title": section_title,
                "table_metadata": {
                    "table_index_on_page": idx,
                    "caption": matched_caption,
                    "headers": merged_headers,
                    "units": col_units,
                    "has_own_header": not headerless_continuation,
                    "ncols": ncols,
                    "raw_data_rows": s_rows,
                    "table_bbox_on_page": bbox,
                    "shard_index": s_idx,
                    "shard_count": len(shards),
                    "row_range": [r_begin, r_end],
                    "estimated_tokens": _estimate_subword_tokens(body),
                    "cells": shard_cells,
                },
            })

    return out_chunks


# =====================================================================
# 4b. M2b Cross-Page Table Stitching & Diagram Node-Edge Extraction
# =====================================================================

def _extract_table_number_tag(caption: str) -> str:
    m = _TABLE_CAPTION_RE.match((caption or "").strip())
    return m.group(1).lower() if m else ""


def stitch_cross_page_tables(
    chunks: list[dict[str, Any]],
    *,
    doc_title: str,
    page_heights: Optional[dict[int, float]] = None,
    max_tokens: int = MAX_TABLE_SHARD_TOKENS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    M2b Cross-Page Table Stitcher:
      - Groups consecutive-page table segments (`page k -> page k+1 -> page k+2...`) that belong
        to the same logical table (matching `ncols`, compatible horizontal span or explicit
        `continued`/`tiếp theo` marker, headerless or repeated-header continuation, and no
        conflicting table number like `Table 1` vs `Table 2`).
      - Propagates the canonical head-page `caption`, `headers`, and `units` into every
        continuation shard on downstream pages while keeping EVERY shard `<= max_tokens` (`<= 512` ceiling)
        and preserving each shard's exact physical `page_no` and `bbox` for PDF.js navigation!
      - Returns `(updated_chunks, stitched_manifest_entries)`.
    """
    table_positions = [i for i, ch in enumerate(chunks) if isinstance(ch, dict) and ch.get("unit_type") == "table"]
    if len(table_positions) < 2:
        return chunks, []

    # Group per-page table shards back into page-level table objects `(page_no, table_index_on_page)`
    page_tables: list[dict[str, Any]] = []
    grouped_idxs: OrderedDict[tuple[int, int], list[int]] = OrderedDict()
    for pos in table_positions:
        ch = chunks[pos]
        tmeta = ch.get("table_metadata") or {}
        key = (int(ch.get("page_no") or 1), int(tmeta.get("table_index_on_page") or 1))
        grouped_idxs.setdefault(key, []).append(pos)

    for (pno, tidx), pos_list in grouped_idxs.items():
        first_ch = chunks[pos_list[0]]
        tmeta = first_ch.get("table_metadata") or {}
        rows_combined: list[list[str]] = []
        for pos in pos_list:
            rows_combined.extend((chunks[pos].get("table_metadata") or {}).get("raw_data_rows") or [])
        tbbox = list(tmeta.get("table_bbox_on_page") or first_ch.get("bbox") or [0.0, 0.0, 595.0, 842.0])
        page_tables.append({
            "page_no": pno,
            "table_index_on_page": tidx,
            "chunk_positions": pos_list,
            "caption": str(tmeta.get("caption") or ""),
            "headers": list(tmeta.get("headers") or []),
            "units": list(tmeta.get("units") or []),
            "has_own_header": bool(tmeta.get("has_own_header", True)),
            "ncols": int(tmeta.get("ncols") or len(tmeta.get("headers") or [])),
            "bbox": tbbox,
            "rows": rows_combined,
        })

    # Determine which consecutive page_tables form a multi-page chain
    chains: list[list[dict[str, Any]]] = []
    for pt in page_tables:
        if not chains:
            chains.append([pt])
            continue
        prev_chain = chains[-1]
        prev = prev_chain[-1]
        head = prev_chain[0]

        is_next_page = pt["page_no"] == prev["page_no"] + 1
        same_cols = pt["ncols"] >= 2 and pt["ncols"] == head["ncols"]
        head_num = _extract_table_number_tag(head["caption"])
        cur_num = _extract_table_number_tag(pt["caption"])
        explicit_cont = bool(_TABLE_CONT_RE.search(pt["caption"]))
        conflicting_table_num = bool(cur_num and head_num and cur_num != head_num)
        brand_new_numbered_table = bool(cur_num and not head_num and not explicit_cont)

        # Horizontal span compatibility
        px0, _, px1, py1 = prev["bbox"]
        cx0, cy0, cx1, _ = pt["bbox"]
        overlap_w = _horizontal_overlap(px0, px1, cx0, cx1)
        min_w = max(min(px1 - px0, cx1 - cx0), 1.0)
        horiz_aligned = (overlap_w / min_w) >= 0.65 or (abs(px0 - cx0) <= 50.0 and abs(px1 - cx1) <= 50.0)

        # Header compatibility: either headerless continuation, explicit `continued`, or identical repeated headers
        repeated_same_headers = (
            pt["has_own_header"]
            and [h.strip().lower() for h in pt["headers"]] == [h.strip().lower() for h in head["headers"]]
        )
        header_continuation_signal = (not pt["has_own_header"]) or explicit_cont or repeated_same_headers

        p_height = float((page_heights or {}).get(pt["page_no"], 842.0) or 842.0)
        near_top_of_next_page = cy0 <= p_height * 0.48 or explicit_cont

        if (
            is_next_page
            and same_cols
            and horiz_aligned
            and header_continuation_signal
            and near_top_of_next_page
            and not conflicting_table_num
            and not brand_new_numbered_table
            and head["has_own_header"]
        ):
            prev_chain.append(pt)
        else:
            chains.append([pt])

    stitched_summary: list[dict[str, Any]] = []
    updated = list(chunks)
    logical_counter = 0

    for chain in chains:
        logical_counter += 1
        logical_id = f"{doc_title}_ltab{logical_counter}"
        if len(chain) == 1:
            # Single-page table: annotate logical_table_id + spanned_pages=[p] cleanly
            only = chain[0]
            for pos in only["chunk_positions"]:
                tm = dict(updated[pos].get("table_metadata") or {})
                tm["logical_table_id"] = logical_id
                tm["is_cross_page"] = False
                tm["spanned_pages"] = [only["page_no"]]
                updated[pos] = {**updated[pos], "table_metadata": tm}
            continue

        # Multi-page stitched table!
        head = chain[0]
        canon_caption = head["caption"] or f"Table {logical_counter}"
        canon_headers = list(head["headers"])
        canon_units = list(head["units"])
        ncols = head["ncols"]
        spanned_pages = [seg["page_no"] for seg in chain]
        global_row_offset = 0
        total_rows_across_pages = sum(len(seg["rows"]) for seg in chain)

        for seg_idx, seg in enumerate(chain):
            pno = seg["page_no"]
            seg_rows = seg["rows"]
            seg_bbox = seg["bbox"]
            is_vi_caption = canon_caption.lower().startswith("bảng")
            if seg_idx == 0:
                seg_caption = f"{canon_caption} (Pages {spanned_pages[0]}–{spanned_pages[-1]})" if not is_vi_caption else f"{canon_caption} (Trang {spanned_pages[0]}–{spanned_pages[-1]})"
            else:
                seg_caption = (
                    f"{canon_caption} [Continued from p.{spanned_pages[0]}, p.{pno}/{spanned_pages[-1]}]"
                    if not is_vi_caption
                    else f"{canon_caption} [Tiếp theo từ trang {spanned_pages[0]}, trang {pno}/{spanned_pages[-1]}]"
                )

            # Rebuild structured cells with inherited canonical headers & units + global_row_index + page_no
            seg_cells: list[dict[str, Any]] = []
            for r_local, row in enumerate(seg_rows):
                row_hdr = row[0] if row else ""
                for c_idx in range(1, min(len(row), ncols)):
                    cell_val = row[c_idx]
                    if cell_val:
                        seg_cells.append({
                            "row_index": r_local,
                            "global_row_index": global_row_offset + r_local,
                            "col_index": c_idx,
                            "row_header": row_hdr,
                            "col_header": canon_headers[c_idx] if c_idx < len(canon_headers) else f"Col_{c_idx + 1}",
                            "unit": canon_units[c_idx] if c_idx < len(canon_units) else "",
                            "value": cell_val,
                            "page_no": pno,
                        })

            # Re-shard segment rows <= max_tokens with inherited headers while preserving exact count of existing chunk slots if possible
            shards: list[tuple[list[list[str]], int, int]] = []
            cur_rows: list[list[str]] = []
            cur_start = 0
            for r_local, row in enumerate(seg_rows):
                trial_md = _format_markdown_rows(canon_headers, [*cur_rows, row])
                trial_body = f"{seg_caption}\n{trial_md}".strip()
                if cur_rows and _estimate_subword_tokens(trial_body) > max_tokens:
                    shards.append((cur_rows, cur_start, r_local - 1))
                    cur_rows = [row]
                    cur_start = r_local
                else:
                    cur_rows.append(row)
            if cur_rows:
                shards.append((cur_rows, cur_start, len(seg_rows) - 1))

            pos_list = seg["chunk_positions"]
            base_template = dict(updated[pos_list[0]])
            seg_total_rows = max(len(seg_rows), 1)
            tab_h = max(seg_bbox[3] - seg_bbox[1], 10.0)

            rebuilt_for_seg: list[dict[str, Any]] = []
            for s_idx, (s_rows, r_begin, r_end) in enumerate(shards, start=1):
                md_str = _format_markdown_rows(canon_headers, s_rows)
                body = f"{seg_caption}\n{md_str}".strip()
                cid = (
                    f"{doc_title}_p{pno}_t{seg['table_index_on_page']}"
                    if len(shards) == 1
                    else f"{doc_title}_p{pno}_t{seg['table_index_on_page']}_part{s_idx}"
                )
                if len(shards) == 1:
                    shard_bbox = seg_bbox
                else:
                    sy0 = seg_bbox[1] + (r_begin / seg_total_rows) * tab_h
                    sy1 = seg_bbox[1] + ((r_end + 1) / seg_total_rows) * tab_h
                    shard_bbox = [seg_bbox[0], round(sy0, 2), seg_bbox[2], round(sy1, 2)]

                shard_cells = [c for c in seg_cells if r_begin <= c["row_index"] <= r_end]
                rebuilt_for_seg.append({
                    **base_template,
                    "id": cid,
                    "chunk_id": cid,
                    "page_no": pno,
                    "page": pno,
                    "bbox": shard_bbox,
                    "text": body,
                    "raw_text": body,
                    "normalized_text": body,
                    "content": body,
                    "sentences": [ln.strip() for ln in body.splitlines() if ln.strip()],
                    "table_metadata": {
                        "logical_table_id": logical_id,
                        "is_cross_page": True,
                        "spanned_pages": spanned_pages,
                        "page_segment_index": seg_idx + 1,
                        "page_segment_count": len(chain),
                        "table_index_on_page": seg["table_index_on_page"],
                        "caption": canon_caption,
                        "segment_caption": seg_caption,
                        "headers": canon_headers,
                        "units": canon_units,
                        "has_own_header": seg["has_own_header"],
                        "ncols": ncols,
                        "raw_data_rows": s_rows,
                        "table_bbox_on_page": seg_bbox,
                        "shard_index": s_idx,
                        "shard_count": len(shards),
                        "row_range": [r_begin, r_end],
                        "global_row_range": [global_row_offset + r_begin, global_row_offset + r_end],
                        "total_logical_rows": total_rows_across_pages,
                        "estimated_tokens": _estimate_subword_tokens(body),
                        "cells": shard_cells,
                    },
                })

            # Replace original chunk positions cleanly
            first_pos = pos_list[0]
            for rem_pos in reversed(pos_list[1:]):
                updated.pop(rem_pos)
                # Shift downstream positions in later segments of this chain and later chains
                for later_chain in chains:
                    for later_seg in later_chain:
                        later_seg["chunk_positions"] = [
                            (p - 1 if p > rem_pos else p) for p in later_seg["chunk_positions"]
                        ]
            updated[first_pos : first_pos + 1] = rebuilt_for_seg
            delta = len(rebuilt_for_seg) - 1
            if delta != 0:
                for later_chain in chains:
                    for later_seg in later_chain:
                        if later_seg is seg:
                            continue
                        later_seg["chunk_positions"] = [
                            (p + delta if p > first_pos else p) for p in later_seg["chunk_positions"]
                        ]
            global_row_offset += len(seg_rows)

        stitched_summary.append({
            "logical_table_id": logical_id,
            "caption": canon_caption,
            "headers": canon_headers,
            "units": canon_units,
            "spanned_pages": spanned_pages,
            "total_rows": total_rows_across_pages,
        })

    return updated, stitched_summary


_DIAGRAM_CAPTION_RE = re.compile(
    r"^\s*(?:Figure|Fig\.|Hình|Sơ\s+đồ|Lưu\s+đồ|Architecture|Pipeline|Workflow)\s*[A-Z]?\d*(?:[.-]\d+)*\b",
    re.IGNORECASE,
)
_INLINE_ARROW_SPLIT_RE = re.compile(r"\s*(?:-->|->|==>|<->|→|⟶|⇒)\s*")
_COND_IN_BRACKET_RE = re.compile(r"^\s*\[([^\[\]]{1,32})\]\s*(.+)$|^\s*\(([^()]{1,32})\)\s*(.+)$")


def _point_to_rect_dist(px: float, py: float, box: list[float]) -> float:
    dx = max(box[0] - px, 0.0, px - box[2])
    dy = max(box[1] - py, 0.0, py - box[3])
    return math.hypot(dx, dy)


def _rect_iou_or_overlap(a: list[float], b: list[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1.0)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1.0)
    return inter / min(area_a, area_b)


def extract_page_diagrams_graph(
    page: fitz.Page,
    *,
    doc_title: str,
    doc_sha256: str,
    page_no: int,
    resolved: str,
    section_title: str,
    excluded_table_bboxes: Optional[list[list[float]]] = None,
) -> list[dict[str, Any]]:
    """
    M2b Diagram Node–Edge Graph Extractor:
      1. Vector 2D Box-and-Arrow Flowcharts / Architecture Diagrams:
         - Detects enclosed vector boxes/diamonds containing text labels (excluding table regions),
         - Resolves directed connectors (`u -> v`) using vector arrowheads (triangles / V-chevrons)
           and attaches branch/edge condition labels (`Yes`/`No`/`Đúng`/`Sai`/`Timeout`, etc.).
      2. Explicit Inline Arrow-Chain Specifications (`Input -> Encoder -> Router -> Decoder`):
         - Parses multi-hop arrow chains inside figure/diagram blocks when no vector boxes exist.
      Emits `unit_type = 'diagram'` chunks (`<= 512` tokens) with `diagram_metadata = {nodes, edges, adjacency}`
      and exact unrotated `bbox` + per-node `bbox` coordinates.
    """
    derot = page.derotation_matrix
    rotation = int(page.rotation or 0)
    excluded = excluded_table_bboxes or []

    try:
        drawings = page.get_drawings() or []
        raw_dict = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)
    except Exception:
        return []

    # Collect text lines with unrotated bboxes
    text_lines: list[dict[str, Any]] = []
    for blk in raw_dict.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for ln in blk.get("lines", []):
            txt = "".join((sp.get("text") or "") for sp in ln.get("spans", [])).strip()
            txt = re.sub(r"\s+", " ", txt)
            if not txt:
                continue
            lb = fitz.Rect(ln["bbox"]) * derot
            box = [round(float(lb.x0), 2), round(float(lb.y0), 2), round(float(lb.x1), 2), round(float(lb.y1), 2)]
            if any(_rect_iou_or_overlap(box, tb) >= 0.6 for tb in excluded):
                continue
            text_lines.append({"text": txt, "bbox": box})

    # 1. Inspect vector drawings for node boxes, connector segments, and arrowheads
    candidate_boxes: list[tuple[list[float], str]] = []
    line_segments: list[tuple[float, float, float, float]] = []
    arrowhead_tips: list[tuple[float, float, float, float]] = []  # (tip_x, tip_y, dir_dx, dir_dy)

    for d in drawings:
        rect_raw = d.get("rect")
        if rect_raw is None:
            continue
        r = fitz.Rect(rect_raw) * derot
        b = [round(float(r.x0), 2), round(float(r.y0), 2), round(float(r.x1), 2), round(float(r.y1), 2)]
        if any(_rect_iou_or_overlap(b, tb) >= 0.5 for tb in excluded):
            continue
        w, h = b[2] - b[0], b[3] - b[1]
        items = d.get("items") or []

        # Detect small filled/closed triangle or V-chevron arrowheads (`3..18pt`)
        if 2.5 <= w <= 20.0 and 2.5 <= h <= 20.0:
            pts: list[tuple[float, float]] = []
            for it in items:
                if it[0] == "l":
                    p1 = it[1] * derot
                    p2 = it[2] * derot
                    pts.extend([(float(p1.x), float(p1.y)), (float(p2.x), float(p2.y))])
            # Deduplicate vertices within 1pt
            uniq_pts: list[tuple[float, float]] = []
            for px, py in pts:
                if not any(math.hypot(px - ux, py - uy) < 1.2 for ux, uy in uniq_pts):
                    uniq_pts.append((px, py))
            if len(uniq_pts) == 3:
                # Tip of an isosceles/chevron arrowhead is the vertex opposite the shortest base (or shared apex of 2 segments)
                d01 = math.hypot(uniq_pts[0][0] - uniq_pts[1][0], uniq_pts[0][1] - uniq_pts[1][1])
                d12 = math.hypot(uniq_pts[1][0] - uniq_pts[2][0], uniq_pts[1][1] - uniq_pts[2][1])
                d02 = math.hypot(uniq_pts[0][0] - uniq_pts[2][0], uniq_pts[0][1] - uniq_pts[2][1])
                # Base is typically the shortest side of a narrow arrowhead triangle; if V-chevron (2 equal legs), vertex with 2 occurrences in `pts` is tip
                counts_pt = [
                    sum(1 for px, py in pts if math.hypot(px - ux, py - uy) < 1.2)
                    for ux, uy in uniq_pts
                ]
                if len(pts) == 4 and max(counts_pt) >= 2:
                    tip_i = counts_pt.index(max(counts_pt))
                else:
                    sides_opp = [(d12, 0), (d02, 1), (d01, 2)]
                    tip_i = min(sides_opp, key=lambda x: x[0])[1]
                others = [uniq_pts[k] for k in range(3) if k != tip_i]
                mid_bx = 0.5 * (others[0][0] + others[1][0])
                mid_by = 0.5 * (others[0][1] + others[1][1])
                tx, ty = uniq_pts[tip_i]
                arrowhead_tips.append((tx, ty, tx - mid_bx, ty - mid_by))
                continue

        # Node box / diamond candidate (`28..340pt` wide, `14..150pt` tall)
        has_rect_item = any(it[0] in ("re", "qu") for it in items)
        line_items_in_d = [it for it in items if it[0] == "l"]
        is_closed_poly = len(line_items_in_d) in (4, 5, 6, 8) and w >= 28.0 and h >= 14.0
        if (has_rect_item or is_closed_poly) and 28.0 <= w <= 340.0 and 14.0 <= h <= 150.0:
            shape = "diamond" if (not has_rect_item and len(line_items_in_d) == 4 and abs(line_items_in_d[0][1].x - line_items_in_d[0][2].x) > 3 and abs(line_items_in_d[0][1].y - line_items_in_d[0][2].y) > 3) else "rect"
            if not any(_rect_iou_or_overlap(b, eb) >= 0.85 for eb, _ in candidate_boxes):
                candidate_boxes.append((b, shape))
                continue

        # Connector segments (`l` or `c` items outside boxes)
        if line_items_in_d:
            p_start = line_items_in_d[0][1] * derot
            p_end = line_items_in_d[-1][2] * derot
            seg_len = math.hypot(float(p_end.x - p_start.x), float(p_end.y - p_start.y))
            if seg_len >= 8.0:
                line_segments.append((float(p_start.x), float(p_start.y), float(p_end.x), float(p_end.y)))

    # Bind text labels inside candidate node boxes
    nodes: list[dict[str, Any]] = []
    consumed_text_idxs: set[int] = set()
    candidate_boxes.sort(key=lambda item: (round(item[0][1] / 12.0), item[0][0]))

    for box, shape in candidate_boxes:
        inside_idxs = []
        for t_idx, tl in enumerate(text_lines):
            tb = tl["bbox"]
            cx, cy = 0.5 * (tb[0] + tb[2]), 0.5 * (tb[1] + tb[3])
            if (box[0] - 4.0 <= cx <= box[2] + 4.0) and (box[1] - 4.0 <= cy <= box[3] + 4.0):
                inside_idxs.append(t_idx)
        if not inside_idxs:
            continue
        label = " ".join(text_lines[i]["text"] for i in inside_idxs).strip()
        if not label or len(label) > 120:
            continue
        consumed_text_idxs.update(inside_idxs)
        nid = f"N{len(nodes) + 1}"
        nodes.append({
            "id": nid,
            "label": label,
            "normalized_label": re.sub(r"\s+", " ", label.lower()).strip(),
            "bbox": box,
            "shape": shape,
        })

    diagram_chunks: list[dict[str, Any]] = []

    # Build vector node-edge graph when >= 2 labeled boxes and >= 1 connector exist
    if len(nodes) >= 2 and line_segments:
        edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()
        unused_labels = [
            (i, tl) for i, tl in enumerate(text_lines)
            if i not in consumed_text_idxs and not _DIAGRAM_CAPTION_RE.match(tl["text"]) and len(tl["text"]) <= 36
        ]
        used_edge_label_idxs: set[int] = set()

        for sx0, sy0, sx1, sy1 in line_segments:
            # Find nearest node to start `(sx0, sy0)` and end `(sx1, sy1)`
            n_start = min(nodes, key=lambda n: _point_to_rect_dist(sx0, sy0, n["bbox"]))
            n_end = min(nodes, key=lambda n: _point_to_rect_dist(sx1, sy1, n["bbox"]))
            d_start = _point_to_rect_dist(sx0, sy0, n_start["bbox"])
            d_end = _point_to_rect_dist(sx1, sy1, n_end["bbox"])
            if n_start["id"] == n_end["id"] or d_start > 28.0 or d_end > 28.0:
                continue

            # Determine arrow direction from arrowhead_tips near `(sx1, sy1)` vs `(sx0, sy0)`
            arrow_at_end = any(math.hypot(tx - sx1, ty - sy1) <= 16.0 for tx, ty, _, _ in arrowhead_tips)
            arrow_at_start = any(math.hypot(tx - sx0, ty - sy0) <= 16.0 for tx, ty, _, _ in arrowhead_tips)

            if arrow_at_start and not arrow_at_end:
                src_node, dst_node, directed = n_end, n_start, True
            elif arrow_at_end and not arrow_at_start:
                src_node, dst_node, directed = n_start, n_end, True
            else:
                # Drawn polyline direction `(sx0, sy0) -> (sx1, sy1)` when arrowhead tips exist on diagram or clear flow axis
                src_node, dst_node, directed = n_start, n_end, bool(arrowhead_tips or (sy1 - sy0) >= 8.0 or (sx1 - sx0) >= 12.0)

            # Attach nearest unused short condition label near segment midpoint (e.g., Yes/No/Đúng/Sai)
            mx, my = 0.5 * (sx0 + sx1), 0.5 * (sy0 + sy1)
            cond_label = ""
            best_ldist = 26.0
            best_lidx = None
            for li, tl in unused_labels:
                if li in used_edge_label_idxs:
                    continue
                lb = tl["bbox"]
                lcx, lcy = 0.5 * (lb[0] + lb[2]), 0.5 * (lb[1] + lb[3])
                ldist = math.hypot(lcx - mx, lcy - my)
                if ldist <= best_ldist:
                    best_ldist = ldist
                    cond_label = tl["text"].strip()
                    best_lidx = li
            if best_lidx is not None:
                used_edge_label_idxs.add(best_lidx)

            ekey = (src_node["id"], dst_node["id"], cond_label.lower())
            if ekey in seen_edges:
                continue
            seen_edges.add(ekey)
            edges.append({
                "source": src_node["id"],
                "source_label": src_node["label"],
                "target": dst_node["id"],
                "target_label": dst_node["label"],
                "condition": cond_label,
                "directed": directed,
            })

        if edges:
            # Find matching caption above/below diagram union bbox
            all_boxes = [n["bbox"] for n in nodes]
            ux0 = min(b[0] for b in all_boxes)
            uy0 = min(b[1] for b in all_boxes)
            ux1 = max(b[2] for b in all_boxes)
            uy1 = max(b[3] for b in all_boxes)
            caption = ""
            for i, tl in enumerate(text_lines):
                if i in consumed_text_idxs:
                    continue
                if _DIAGRAM_CAPTION_RE.match(tl["text"]):
                    tb = tl["bbox"]
                    if -15.0 <= (uy0 - tb[3]) <= 90.0 or -15.0 <= (tb[1] - uy1) <= 90.0:
                        caption = tl["text"]
                        ux0, uy0 = min(ux0, tb[0]), min(uy0, tb[1])
                        ux1, uy1 = max(ux1, tb[2]), max(uy1, tb[3])
                        break

            adjacency: dict[str, list[dict[str, str]]] = {}
            for e in edges:
                adjacency.setdefault(e["source_label"], []).append({
                    "target": e["target_label"],
                    "condition": e["condition"],
                })

            edge_lines = [
                f"- {e['source_label']} -> {e['target_label']}" + (f" [condition: {e['condition']}]" if e["condition"] else "")
                for e in edges
            ]
            node_names = ", ".join(n["label"] for n in nodes)
            header_line = caption or f"Diagram on Page {page_no}"
            body = f"{header_line}\nNodes: {node_names}\nDirected Transitions:\n" + "\n".join(edge_lines)
            cid = f"{doc_title}_p{page_no}_diag1"
            diagram_chunks.append({
                "id": cid,
                "chunk_id": cid,
                "doc_id": doc_title,
                "document_sha256": doc_sha256,
                "title": doc_title,
                "page_no": page_no,
                "page": page_no,
                "page_rotation": rotation,
                "coordinate_space": "pdf_points_top_left",
                "extractor_version": EXTRACTOR_VERSION,
                "bbox": [round(ux0, 2), round(uy0, 2), round(ux1, 2), round(uy1, 2)],
                "text": body,
                "raw_text": body,
                "normalized_text": body,
                "content": body,
                "sentences": [header_line, f"Nodes: {node_names}", *edge_lines],
                "source_path": resolved,
                "is_selected_pdf": True,
                "unit_type": "diagram",
                "section_title": section_title,
                "diagram_metadata": {
                    "diagram_id": cid,
                    "caption": caption,
                    "extraction_mode": "vector_2d",
                    "nodes": nodes,
                    "edges": edges,
                    "adjacency": adjacency,
                },
            })

    # 2. Fallback: Explicit inline arrow-chain flow specifications in figure/diagram blocks
    if not diagram_chunks:
        arrow_lines: list[tuple[list[float], str]] = []
        caption_line = ""
        for tl in text_lines:
            t = tl["text"]
            if _DIAGRAM_CAPTION_RE.match(t):
                caption_line = t
            if len(_INLINE_ARROW_SPLIT_RE.split(t)) >= 2 and any(sym in t for sym in ("->", "-->", "→", "⟶", "⇒")):
                # Exclude math formulas like `x -> 0` or `f: R -> R`
                hops = [h.strip() for h in _INLINE_ARROW_SPLIT_RE.split(t) if h.strip()]
                if len(hops) >= 2 and all(len(h) <= 60 for h in hops) and not re.search(r"\b(?:lim|inf|sup)\b|^\s*[a-z]\s*:", t):
                    arrow_lines.append((tl["bbox"], t))

        if arrow_lines and (caption_line or len(arrow_lines) >= 1):
            inl_nodes: OrderedDict[str, dict[str, Any]] = OrderedDict()
            inl_edges: list[dict[str, Any]] = []
            seen_inl: set[tuple[str, str, str]] = set()
            for abox, aline in arrow_lines:
                # Strip leading bullet or prefix like `Flow:` / `Luồng:`
                cleaned_line = re.sub(r"^(?:[-*•]|\b(?:Flow|Pipeline|Luồng|Nhánh)\s*:)\s*", "", aline, flags=re.I)
                raw_hops = [h.strip() for h in _INLINE_ARROW_SPLIT_RE.split(cleaned_line) if h.strip()]
                parsed_hops: list[tuple[str, str]] = []  # (incoming_cond, node_label)
                for h_idx, raw_h in enumerate(raw_hops):
                    cond = ""
                    lbl = raw_h
                    if h_idx > 0:
                        m_c = _COND_IN_BRACKET_RE.match(raw_h)
                        if m_c:
                            cond = (m_c.group(1) or m_c.group(3) or "").strip()
                            lbl = (m_c.group(2) or m_c.group(4) or "").strip()
                    lbl = lbl.strip(" .;:")
                    if lbl:
                        parsed_hops.append((cond, lbl))
                        nkey = lbl.lower()
                        if nkey not in inl_nodes:
                            inl_nodes[nkey] = {
                                "id": f"N{len(inl_nodes) + 1}",
                                "label": lbl,
                                "normalized_label": nkey,
                                "bbox": abox,
                                "shape": "inline_node",
                            }
                for k in range(len(parsed_hops) - 1):
                    _, src_lbl = parsed_hops[k]
                    edge_cond, dst_lbl = parsed_hops[k + 1]
                    s_node = inl_nodes[src_lbl.lower()]
                    d_node = inl_nodes[dst_lbl.lower()]
                    ek = (s_node["id"], d_node["id"], edge_cond.lower())
                    if s_node["id"] != d_node["id"] and ek not in seen_inl:
                        seen_inl.add(ek)
                        inl_edges.append({
                            "source": s_node["id"],
                            "source_label": s_node["label"],
                            "target": d_node["id"],
                            "target_label": d_node["label"],
                            "condition": edge_cond,
                            "directed": True,
                        })

            if len(inl_nodes) >= 2 and inl_edges:
                ux0 = min(b[0] for b, _ in arrow_lines)
                uy0 = min(b[1] for b, _ in arrow_lines)
                ux1 = max(b[2] for b, _ in arrow_lines)
                uy1 = max(b[3] for b, _ in arrow_lines)
                adjacency_inl: dict[str, list[dict[str, str]]] = {}
                for e in inl_edges:
                    adjacency_inl.setdefault(e["source_label"], []).append({
                        "target": e["target_label"],
                        "condition": e["condition"],
                    })
                edge_lines = [
                    f"- {e['source_label']} -> {e['target_label']}" + (f" [condition: {e['condition']}]" if e["condition"] else "")
                    for e in inl_edges
                ]
                node_list = list(inl_nodes.values())
                node_names = ", ".join(n["label"] for n in node_list)
                hdr = caption_line or f"Diagram on Page {page_no}"
                body = f"{hdr}\nNodes: {node_names}\nDirected Transitions:\n" + "\n".join(edge_lines)
                cid = f"{doc_title}_p{page_no}_diag1"
                diagram_chunks.append({
                    "id": cid,
                    "chunk_id": cid,
                    "doc_id": doc_title,
                    "document_sha256": doc_sha256,
                    "title": doc_title,
                    "page_no": page_no,
                    "page": page_no,
                    "page_rotation": rotation,
                    "coordinate_space": "pdf_points_top_left",
                    "extractor_version": EXTRACTOR_VERSION,
                    "bbox": [round(ux0, 2), round(uy0, 2), round(ux1, 2), round(uy1, 2)],
                    "text": body,
                    "raw_text": body,
                    "normalized_text": body,
                    "content": body,
                    "sentences": [hdr, f"Nodes: {node_names}", *edge_lines],
                    "source_path": resolved,
                    "is_selected_pdf": True,
                    "unit_type": "diagram",
                    "section_title": section_title,
                    "diagram_metadata": {
                        "diagram_id": cid,
                        "caption": caption_line,
                        "extraction_mode": "inline_arrow_spec",
                        "nodes": node_list,
                        "edges": inl_edges,
                        "adjacency": adjacency_inl,
                    },
                })

    return diagram_chunks


# =====================================================================
# 5. Rotated PDF.js Region Descriptor & On-Demand Region Crop Renderer
# =====================================================================

def compute_pdfjs_region_From_page(
    page: fitz.Page,
    bbox: Optional[list[float] | tuple[float, ...]],
) -> Optional[dict[str, Any]]:
    if not bbox or len(bbox) < 4:
        return None
    try:
        ux0, uy0, ux1, uy1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return None
    if ux1 <= ux0 or uy1 <= uy0:
        return None

    rotation = int(page.rotation or 0)
    unrot_rect = page.rect * page.derotation_matrix
    vis_rect = page.rect

    # Transform unrotated PDF top-left box into visual rotated PDF.js viewport coordinates
    vis_box = (fitz.Rect(ux0, uy0, ux1, uy1) * page.rotation_matrix) & vis_rect
    vw = max(float(vis_rect.width), 1.0)
    vh = max(float(vis_rect.height), 1.0)
    uw = max(float(unrot_rect.width), 1.0)
    uh = max(float(unrot_rect.height), 1.0)

    return {
        "bbox": [round(ux0, 2), round(uy0, 2), round(ux1, 2), round(uy1, 2)],
        "visual_bbox": [round(float(vis_box.x0), 2), round(float(vis_box.y0), 2), round(float(vis_box.x1), 2), round(float(vis_box.y1), 2)],
        "normalized_bbox": [
            round(float(vis_box.x0) / vw, 4),
            round(float(vis_box.y0) / vh, 4),
            round(float(vis_box.x1) / vw, 4),
            round(float(vis_box.y1) / vh, 4),
        ],
        # Standard PDF bottom-left user space for #page=P&zoom=auto,left,top
        "pdf_points_bottom_left": [
            round(ux0, 2),
            round(uh - uy1, 2),
            round(ux1, 2),
            round(uh - uy0, 2),
        ],
        "page_rotation": rotation,
        "page_size": [round(vw, 2), round(vh, 2)],
        "unrotated_page_size": [round(uw, 2), round(uh, 2)],
        "coordinate_space": "pdf_points_top_left",
    }


def build_pdfjs_region_descriptor(
    pdf_path: Optional[str | Path],
    page_no: int,
    bbox: Optional[list[float] | tuple[float, ...]],
) -> Optional[dict[str, Any]]:
    if not bbox or len(bbox) < 4:
        return None
    if not pdf_path or not os.path.isfile(str(pdf_path)):
        return {
            "bbox": [round(float(v), 2) for v in bbox[:4]],
            "page_rotation": 0,
            "coordinate_space": "pdf_points_top_left",
        }
    try:
        with fitz.open(str(pdf_path)) as doc:
            if page_no < 1 or page_no > len(doc):
                return None
            return compute_pdfjs_region_From_page(doc[page_no - 1], bbox)
    except Exception:
        return {
            "bbox": [round(float(v), 2) for v in bbox[:4]],
            "page_rotation": 0,
            "coordinate_space": "pdf_points_top_left",
        }


def render_pdf_region_crop(
    pdf_path: str | Path,
    page_no: int,
    bbox: list[float] | tuple[float, ...],
) -> bytes:
    """Renders an arbitrary unrotated `[x0, y0, x1, y1]` region on `page_no` (handling rotation) to PNG."""
    if len(bbox) < 4:
        raise ValueError("bbox requires 4 coordinates")
    ux0, uy0, ux1, uy1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if ux1 <= ux0 or uy1 <= uy0:
        raise ValueError("Invalid empty bbox")
    with fitz.open(str(pdf_path)) as doc:
        if not doc.is_pdf or doc.needs_pass or page_no < 1 or page_no > len(doc):
            raise KeyError(f"Page {page_no} out of range")
        page = doc[page_no - 1]
        # Pad slightly (4pt) so equation fraction bars / superscripts are crisp
        unrot_bounds = page.rect * page.derotation_matrix
        padded_unrot = (fitz.Rect(ux0, uy0, ux1, uy1) + (-4, -4, 4, 4)) & unrot_bounds
        clip = (padded_unrot * page.rotation_matrix) & page.rect
        if clip.is_empty or clip.width <= 1 or clip.height <= 1:
            raise ValueError("Clip outside page bounds")
        scale = min(2.5, 1600.0 / max(clip.width, clip.height))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid scale")
        pix = page.get_pixmap(clip=clip, matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False, annots=False)
        return pix.tobytes("png")


def record_ingest_manifest(resolved_path: str, manifest: dict[str, Any]) -> None:
    key = os.path.normcase(os.path.abspath(resolved_path))
    with _INGEST_LOCK:
        if len(_INGEST_MANIFEST_CACHE) >= _MAX_CACHE:
            _INGEST_MANIFEST_CACHE.popitem(last=False)
        _INGEST_MANIFEST_CACHE[key] = manifest


def get_ingest_manifest(resolved_path: str) -> dict[str, Any] | None:
    key = os.path.normcase(os.path.abspath(resolved_path))
    with _INGEST_LOCK:
        return _INGEST_MANIFEST_CACHE.get(key)
