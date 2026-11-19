"""Conservative pixel measurements for one unstacked, upright bar chart."""

import re
import unicodedata

import fitz
import numpy as np


def _runs(mask):
    edges = np.flatnonzero(np.diff(mask.astype(np.int8), prepend=0, append=0))
    return zip(edges[::2], edges[1::2])


def _bar_height(rgb, row, start, end):
    width = end - start
    columns = [start + width // 4, (start + end) // 2, end - 1 - width // 4]
    color = np.median(rgb[row, columns], axis=0)
    # Three vertical slices must reach the same flat top through a uniform solid fill.
    matches = np.max(np.abs(rgb[:row + 1, columns] - color), axis=2) < 22
    heights = np.argmax(~matches[::-1], axis=0)
    if min(heights) < 5 or np.ptp(heights) > 2:
        return None
    top = row - int(np.median(heights))
    # A different colored region above the bar indicates a stack or a patterned plot.
    if top < 3 or np.any(rgb[top - 2, columns].min(axis=1) < 215):
        return None
    return row - top


def _chart_at_axis(rgb, dark, foreground, baseline, left, right):
    width = right - left
    row = baseline - 3
    spans = [(left + a, left + b) for a, b in _runs(foreground[row, left:right]) if b - a >= max(5, width * .018)]
    if not 2 <= len(spans) <= 16:
        return None
    widths = [b - a for a, b in spans]
    if max(widths) > min(widths) * 1.25:
        return None
    gaps = [c - b for (_, b), (c, _) in zip(spans, spans[1:])]
    if min(gaps) < 3 or max(gaps) > min(gaps) * 1.55:
        return None
    heights = [_bar_height(rgb, row, a, b) for a, b in spans]
    if any(h is None for h in heights):
        return None
    # Require a visible y-axis and a clear plot background immediately below the baseline.
    tallest = max(heights)
    if not dark[baseline - tallest:baseline, max(0, left - 2):left + 3].any(axis=1).mean() > .9:
        return None
    if any(foreground[baseline + 3, a + 1:b - 1].mean() > .25 for a, b in spans):
        return None
    return heights


def read_bar_comparison(png: bytes, question: str) -> str:
    """Identify only an unambiguous tallest/shortest bar, in left-to-right order."""
    normalized = "".join(c for c in unicodedata.normalize("NFKD", question.casefold()) if not unicodedata.combining(c))
    if not re.search(r"\b(bars?|columns?|cot|thanh)\b", normalized):
        return ""
    high = bool(re.search(r"\b(tallest|highest|largest|longest|max(?:imum)?|(?:cao|lon|nhieu|dai) nhat)\b", normalized))
    low = bool(re.search(r"\b(shortest|lowest|smallest|min(?:imum)?|(?:thap|nho|it|ngan) nhat)\b", normalized))
    if high == low:
        return ""
    pix = fitz.Pixmap(png)
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)
    if pix.colorspace != fitz.csRGB:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    while max(pix.width, pix.height) > 1024:
        pix.shrink(1)
    if min(pix.width, pix.height) < 24:
        return ""
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    dark = rgb.max(axis=2) < 160
    foreground = rgb.min(axis=2) < 210
    # Collect unique charts so that a multi-panel figure does not select the wrong panel.
    measurements = {}
    for baseline in range(max(pix.height // 4, 7), pix.height - 3):
        for left, right in _runs(dark[baseline]):
            if right - left < pix.width * .30:
                continue
            heights = _chart_at_axis(rgb, dark, foreground, baseline, left, right)
            if heights:
                order = np.argsort(heights)[::-1] if high else np.argsort(heights)
                if abs(heights[order[0]] - heights[order[1]]) < max(4, max(heights) * .05):
                    return ""
                # The same axis can occupy multiple raster rows.
                previous = next((key for key in measurements if abs(key[0] - baseline) <= 4 and abs(key[1] - left) <= 3), None)
                measurements[previous or (baseline, left)] = (int(order[0]) + 1, len(heights))
    if len(measurements) != 1:
        return ""
    index, count = next(iter(measurements.values()))
    if re.search(r"\b(cot|thanh)\b", normalized):
        return f"Trong {count} cột, cột thứ {index} từ trái sang {'cao nhất' if high else 'thấp nhất'} theo chiều cao hiển thị."
    return f"Of the {count} bars, bar {index} from the left is the {'tallest' if high else 'shortest'} by visible height."
