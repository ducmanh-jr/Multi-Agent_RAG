"""Ranking metrics over ordered passages and document/page relevance judgments."""

import math
from collections.abc import Mapping, Sequence

Page = tuple[str, int]
METRICS = ("recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10")
CANDIDATE_LIMIT = 10


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
