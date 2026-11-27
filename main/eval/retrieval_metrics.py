"""Ranking metrics over ordered passages, document/page judgments, and PDF region/bbox targets."""

import math
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

Page = tuple[str, int]
METRICS = ("recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10")
REGION_METRICS = ("region_recall_at_5", "region_recall_at_10", "region_mrr_at_10", "region_ndcg_at_10")
CANDIDATE_LIMIT = 10


def _norm_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text or "")).casefold().split())


def bbox_iou(box_a: Sequence[float] | None, box_b: Sequence[float] | None) -> float:
    """Computes Intersection-over-Union for two `[x0, y0, x1, y1]` boxes."""
    if not box_a or not box_b or len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    try:
        ax0, ay0, ax1, ay1 = (float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3]))
        bx0, by0, bx1, by1 = (float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3]))
    except (TypeError, ValueError):
        return 0.0
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(0.0, (bx1 - bx0) * (by1 - by0))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_containment(pred_box: Sequence[float] | None, gold_box: Sequence[float] | None) -> float:
    """Fraction of `gold_box` covered by `pred_box` (handles region/shard containing gold target)."""
    if not pred_box or not gold_box or len(pred_box) < 4 or len(gold_box) < 4:
        return 0.0
    try:
        ax0, ay0, ax1, ay1 = (float(pred_box[0]), float(pred_box[1]), float(pred_box[2]), float(pred_box[3]))
        bx0, by0, bx1, by1 = (float(gold_box[0]), float(gold_box[1]), float(gold_box[2]), float(gold_box[3]))
    except (TypeError, ValueError):
        return 0.0
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    gold_area = max(0.0, (bx1 - bx0) * (by1 - by0))
    return ((ix1 - ix0) * (iy1 - iy0)) / gold_area if gold_area > 0 else 0.0


def score_ranked_pages(ranked: Sequence[Page], relevant: Mapping[Page, int]) -> dict[str, float | None]:
    """A page gains credit once. Duplicate passages keep their original rank slots."""
    if not relevant:
        return dict.fromkeys(METRICS)

    credited = set()
    gains = []
    for location in ranked[:CANDIDATE_LIMIT]:
        grade = relevant.get(location, 0)
        gains.append(grade if location not in credited else 0)
        credited.add(location)

    def recall(cutoff: int) -> float:
        return sum(grade > 0 for grade in gains[:cutoff]) / len(relevant)

    def dcg(grades: Sequence[int]) -> float:
        return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))

    ideal = sorted(relevant.values(), reverse=True)[:CANDIDATE_LIMIT]
    first_hit = next((rank for rank, grade in enumerate(gains, start=1) if grade), None)
    return {
        "recall_at_5": recall(5),
        "recall_at_10": recall(CANDIDATE_LIMIT),
        "mrr_at_10": 1 / first_hit if first_hit else 0.0,
        "ndcg_at_10": dcg(gains) / dcg(ideal),
    }


def _chunk_matches_gold_region(
    chunk: Mapping[str, Any],
    chunk_doc: str,
    gold: Mapping[str, Any],
) -> bool:
    if chunk_doc != gold.get("document"):
        return False
    page = chunk.get("page_no") or chunk.get("page")
    if page != gold.get("page"):
        return False

    gold_unit_type = str(gold.get("unit_type") or "").strip().lower()
    chunk_unit_type = str(chunk.get("unit_type") or "text").strip().lower()
    if gold_unit_type and chunk_unit_type != gold_unit_type:
        # Allow ocr_scan/figure interchangeably when gold expects figure or scan
        if not ({gold_unit_type, chunk_unit_type} <= {"figure", "ocr_scan"}):
            return False

    gold_bbox = gold.get("bbox")
    chunk_bbox = chunk.get("bbox")
    if isinstance(gold_bbox, (list, tuple)) and len(gold_bbox) >= 4:
        if not (bbox_iou(chunk_bbox, gold_bbox) >= 0.25 or bbox_containment(chunk_bbox, gold_bbox) >= 0.55):
            return False

    anchors = [a for a in (gold.get("anchors") or []) if isinstance(a, str) and a.strip()]
    if anchors:
        chunk_txt = _norm_text(chunk.get("text") or chunk.get("content") or "")
        if not all(_norm_text(anchor) in chunk_txt for anchor in anchors):
            return False

    return True


def score_ranked_regions(
    ranked_chunks: Sequence[tuple[str, Mapping[str, Any]]],
    gold_regions: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    """
    Region-level retrieval scoring: credits a target region `(document, page, unit_type, bbox, anchors)`
    once at its first matching rank slot within `CANDIDATE_LIMIT`.
    """
    if not gold_regions:
        return dict.fromkeys(REGION_METRICS)

    credited_indices: set[int] = set()
    gains: list[int] = []
    for doc_name, chunk in ranked_chunks[:CANDIDATE_LIMIT]:
        best_idx = None
        best_grade = 0
        for g_idx, gold in enumerate(gold_regions):
            if g_idx in credited_indices:
                continue
            if _chunk_matches_gold_region(chunk, doc_name, gold):
                grade = int(gold.get("grade") or 2)
                if grade > best_grade:
                    best_grade = grade
                    best_idx = g_idx
        if best_idx is not None:
            credited_indices.add(best_idx)
            gains.append(best_grade)
        else:
            gains.append(0)

    def recall(cutoff: int) -> float:
        return sum(g > 0 for g in gains[:cutoff]) / len(gold_regions)

    def dcg(grades: Sequence[int]) -> float:
        return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))

    ideal = sorted((int(g.get("grade") or 2) for g in gold_regions), reverse=True)[:CANDIDATE_LIMIT]
    first_hit = next((rank for rank, grade in enumerate(gains, start=1) if grade), None)
    return {
        "region_recall_at_5": recall(5),
        "region_recall_at_10": recall(CANDIDATE_LIMIT),
        "region_mrr_at_10": 1 / first_hit if first_hit else 0.0,
        "region_ndcg_at_10": dcg(gains) / dcg(ideal) if dcg(ideal) > 0 else 0.0,
    }
