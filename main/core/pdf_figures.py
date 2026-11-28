"""Locate figures in selected PDFs and render their original page regions on demand."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from pathlib import Path
import re
import threading

import fitz

_FIGURE_INDEX_LOCK = threading.RLock()


_CAPTION = re.compile(
    r"^\s*(?P<label>(?:fig(?:ure)?\.?|hình(?:\s+vẽ)?|biểu\s+đồ|sơ\s+đồ)"
    r"\s*(?:[A-Z]?\d+(?:[.-]\d+)*[a-z]?|[IVX]+))(?=\s|[:.()\-–—]|$)", re.I,
)
_FIGURE_ID = re.compile(r"fig-[0-9a-f]{24}-p[1-9]\d*-f[1-9]\d*")


def _text_blocks(page):
    flags = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES
    groups = []
    for block in page.get_text("dict", flags=flags)["blocks"]:
        lines = []
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text:
                continue
            box = fitz.Rect(line["bbox"])
            # MuPDF can place both captions of a two-column page in one text block.
            if lines and (_CAPTION.match(text) or box.x0 > bounds.x1 or box.x1 < bounds.x0):
                groups.append((bounds, " ".join(lines)))
                lines = []
            bounds = bounds | box if lines else box
            lines.append(text)
        if lines:
            groups.append((bounds, " ".join(lines)))
    return groups


def _vector_regions(page, bounds):
    # Paths inside a PDF form can extend well outside its visible clip (e.g. cropped slides).
    clips = {}
    boxes = []
    for path in page.get_cdrawings(extended=True):
        level = path["level"]
        while clips and next(reversed(clips)) >= level:
            clips.popitem()
        visible = next(reversed(clips.values())) if clips else bounds
        if path["type"] == "clip":
            clips[level] = visible & fitz.Rect(path["scissor"])
        elif path["type"] in {"s", "f", "fs"}:
            box = fitz.Rect(path["rect"]) + (-0.5, -0.5, 0.5, 0.5)
            box &= visible
            # Full-page backgrounds and page frames are layout, not figure regions.
            if box and box.get_area() < 0.85 * bounds.get_area():
                boxes.append(box)

    # Dense scatterplots may contain thousands of tiny paths. Coalesce nearby marks in
    # the same tile before clustering so the cost depends mainly on page dimensions.
    if len(boxes) > 800:
        tiles = {}
        for box in boxes:
            cell = (int(box.x0 // 12), int(box.y0 // 12))
            tiles[cell] = tiles[cell] | box if cell in tiles else box
        boxes = list(tiles.values())

    clusters = []
    for box in sorted(boxes, key=lambda r: (r.y0, r.x0)):
        touching = []
        vicinity = box + (-7, -7, 7, 7)
        for i, region in enumerate(clusters):
            if region.intersects(vicinity):
                touching.append(i)
        for i in reversed(touching):
            box |= clusters.pop(i)
        clusters.append(box)
    return [box for box in clusters if box.width >= 28 and box.height >= 20]


def _closest_caption(box, captions):
    best = None
    for index, (location, _, _) in enumerate(captions):
        overlap = max(0, min(box.x1, location.x1) - max(box.x0, location.x0))
        if overlap < 0.25 * min(box.width, location.width):
            continue
        below = location.y0 - box.y1
        above = box.y0 - location.y1
        if below >= -3:
            distance = max(0, below)
        elif above >= -3:
            distance = max(0, above) + 12  # scholarly captions are usually below the figure
        else:
            continue
        if distance <= 78 and (best is None or distance < best[0]):
            best = distance, index
    return best[1] if best is not None else None


def _page_figures(page):
    bounds = page.rect * page.derotation_matrix
    blocks = _text_blocks(page)
    captions = [(rect, text, match["label"]) for rect, text in blocks if (match := _CAPTION.match(text))]
    regions = []
    full_images = []
    for image in page.get_image_info():
        box = fitz.Rect(image["bbox"]) & bounds
        if box.get_area() >= 0.85 * bounds.get_area():
            full_images.append(box)
        elif box.width >= 28 and box.height >= 20 and box.get_area() >= 1000:
            regions.append((box, "image"))

    for box in _vector_regions(page, bounds):
        if not any((box & image_box).get_area() >= 0.90 * box.get_area() for image_box, _ in regions):
            regions.append((box, "vector"))

    # Several panel images may share a single caption. Render their union, including the labels.
    groups = {}
    for i, (box, kind) in enumerate(regions):
        caption_index = _closest_caption(box, captions)
        key = ("caption", caption_index) if caption_index is not None else ("region", i)
        if key in groups:
            groups[key][0] |= box
            if kind == "image":
                groups[key][1] = kind
        else:
            groups[key] = [box, kind]
    if not groups and full_images:
        # A full-page scan has no reliable intra-page figure boundary.
        groups[("scan", 0)] = [max(full_images, key=lambda box: box.get_area()), "scan"]

    figures = []
    for (group_type, index), (box, kind) in groups.items():
        label = caption = ""
        caption_bbox = None
        if group_type == "caption":
            caption_rect, caption, label = captions[index]
            caption_bbox = tuple(round(v, 2) for v in caption_rect)
        # Include searchable axis labels/titles adjacent to the drawing, while excluding
        # captions and paragraphs outside the drawing. The PNG keeps raster labels verbatim.
        nearby = (box + (-28, -16, 28, 16)) & bounds
        text = []
        for rect, line in blocks:
            if _CAPTION.match(line):
                continue
            if rect in box or (rect in nearby and len(line) <= 110):
                text.append(line)
                box |= rect
        box = (box + (-2, -2, 2, 2)) & bounds
        figures.append({
            "bbox": tuple(round(v, 2) for v in box), "kind": kind, "label": label,
            "caption": caption[:1800], "caption_bbox": caption_bbox, "embedded_text": "\n".join(text)[:3500],
        })
    return sorted(figures, key=lambda row: (row["bbox"][1], row["bbox"][0]))


def _file_version(path: Path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=12)
def _figure_index(path: Path, version: tuple[int, int, int]):
    with path.open("rb") as source:
        content_hash = hashlib.sha256()
        for chunk in iter(lambda: source.read(1 << 20), b""):
            content_hash.update(chunk)
        digest = content_hash.hexdigest()[:24]
    doc_id = re.sub(r"^[0-9a-f]{8}_", "", path.stem)
    figures = {}
    with fitz.open(path) as doc:
        if not doc.is_pdf or doc.needs_pass:
            raise ValueError("The file must be an accessible PDF.")
        for page_no, page in enumerate(doc, 1):
            for ordinal, item in enumerate(_page_figures(page), 1):
                figure_id = f"fig-{digest}-p{page_no}-f{ordinal}"
                figures[figure_id] = {
                    **item, "figure_id": figure_id, "page_no": page_no,
                    "doc_id": doc_id, "source_path": str(path),
                }
    if _file_version(path) != version:
        raise ValueError("PDF changed while its figures were being indexed.")
    return figures


def extract_pdf_figures(pdf_path: str | Path) -> list[dict]:
    """Return descriptors with one-based pages and unrotated PDF point coordinates."""
    path = Path(pdf_path).resolve()
    with _FIGURE_INDEX_LOCK:
        indexed = _figure_index(path, _file_version(path))
    return [
        {**item, "bbox": list(item["bbox"])}
        for item in indexed.values()
    ]


def figure_reference(figure: dict, query_terms: set[str]) -> dict | None:
    """Find an author's sentence that explicitly refers to this figure, with its own page and box."""
    label = _CAPTION.match(figure["label"])
    if not label:
        return None
    number = re.search(r"(?:[A-Z]?\d+(?:[.-]\d+)*[a-z]?|[IVX]+)$", label["label"], re.I)
    if not number:
        return None
    mention = re.compile(r"\b(?:fig(?:ure)?\.?|hình|biểu đồ|sơ đồ)\s*" + re.escape(number[0]) + r"(?![\w.-]\d|\w)", re.I)
    candidates = []
    with fitz.open(figure["source_path"]) as doc:
        origin = figure["page_no"] - 1
        for page_index in range(max(0, origin - 2), min(len(doc), origin + 3)):
            for block in doc[page_index].get_text("blocks"):
                text = " ".join(block[4].split())
                if not mention.search(text) or (figure["caption"] and text.startswith(figure["caption"])):
                    continue
                # Keep an initial 'Fig.' abbreviation attached to its number.
                for sentence in re.split(r"(?<!Fig\.)(?<!fig\.)(?<=[.!?])\s+(?=[A-ZÀ-Ỹ])", text):
                    if not 28 <= len(sentence) <= 700 or not mention.search(sentence):
                        continue
                    words = set(re.findall(r"\w+", sentence.casefold()))
                    score = (len(words & query_terms), -abs(page_index - origin))
                    candidates.append((score, {"text": sentence, "page_no": page_index + 1, "bbox": list(block[:4])}))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def render_pdf_figure(pdf_path: str | Path, figure_id: str) -> bytes:
    """Render the catalogued region (including vector marks, overlays and image masks)."""
    if not isinstance(figure_id, str) or not _FIGURE_ID.fullmatch(figure_id):
        raise KeyError(figure_id)
    path = Path(pdf_path).resolve()
    version = _file_version(path)
    item = _figure_index(path, version)[figure_id]
    with fitz.open(path) as doc:
        page = doc[item["page_no"] - 1]
        clip = fitz.Rect(item["bbox"]) * page.rotation_matrix
        # Keep large scans/posters bounded while retaining fine labels at up to 144 dpi.
        scale = min(2.0, 1800 / max(clip.width, clip.height))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid figure bounds.")
        image = page.get_pixmap(clip=clip, matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False, annots=False)
        png = image.tobytes("png")
    if _file_version(path) != version:
        raise KeyError(figure_id)
    return png
