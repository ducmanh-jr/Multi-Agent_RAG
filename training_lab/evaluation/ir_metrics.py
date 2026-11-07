# -*- coding: utf-8 -*-
"""
Information Retrieval (IR) Evaluation Metrics Module.
Implements standard academic metrics: Recall@K, nDCG@K, MRR@K, MAP@K.
Follows TREC / BEIR / MoAI benchmark evaluation standards.
"""

import math
from typing import List, Dict, Any, Union, Optional


def compute_recall_at_k(relevant_ids: List[Union[str, int]], ranked_retrieved_ids: List[Union[str, int]], k: int = 10) -> float:
    """
    Computes Recall@K: Proportion of relevant documents retrieved within the top-K ranking.
    Recall@K = |Relevant ∩ Retrieved@K| / |Relevant|
    """
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = set(ranked_retrieved_ids[:max(0, int(k))])
    rel_set = set(relevant_ids)
    hits = len(top_k.intersection(rel_set))
    return hits / len(rel_set)


def compute_mrr_at_k(relevant_ids: List[Union[str, int]], ranked_retrieved_ids: List[Union[str, int]], k: int = 10) -> float:
    """
    Computes Mean Reciprocal Rank @ K (MRR@K):
    Reciprocal rank of the FIRST relevant document found within top-K.
    """
    rel_set = set(relevant_ids)
    for rank_idx, doc_id in enumerate(ranked_retrieved_ids[:k], start=1):
        if doc_id in rel_set:
            return 1.0 / rank_idx
    return 0.0


def compute_dcg_at_k(relevance_gains: List[float], k: int = 10) -> float:
    """
    Computes Discounted Cumulative Gain (DCG@K):
    DCG@K = sum_{i=1..K} (2^{rel_i} - 1) / log2(i + 1)
    """
    if k <= 0:
        return 0.0
    dcg = 0.0
    for i, rel in enumerate(relevance_gains[:max(0, int(k))], start=1):
        rel_val = min(10.0, max(0.0, float(rel)))
        dcg += (2.0 ** rel_val - 1.0) / math.log2(i + 1.0)
    return dcg


def compute_ndcg_at_k(relevance_gains: List[float], ideal_relevance_gains: List[float] = None, k: int = 10) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (nDCG@K):
    nDCG@K = DCG@K / IDCG@K
    """
    actual_dcg = compute_dcg_at_k(relevance_gains, k=k)
    if ideal_relevance_gains is None:
        ideal_relevance_gains = sorted(relevance_gains, reverse=True)
    ideal_dcg = compute_dcg_at_k(ideal_relevance_gains, k=k)
    if ideal_dcg <= 0.0:
        return 0.0
    return min(1.0, max(0.0, float(actual_dcg / ideal_dcg)))


def compute_average_precision_at_k(relevant_ids: List[Union[str, int]], ranked_retrieved_ids: List[Union[str, int]], k: int = 10) -> float:
    """
    Computes Average Precision @ K (AP@K):
    AP@K = sum_{i=1..K} (Precision@i * rel_i) / min(|Relevant|, K)
    """
    if not relevant_ids or k <= 0:
        return 0.0
    rel_set = set(relevant_ids)
    hits = 0.0
    sum_precisions = 0.0
    for rank_idx, doc_id in enumerate(ranked_retrieved_ids[:k], start=1):
        if doc_id in rel_set:
            hits += 1.0
            sum_precisions += hits / rank_idx
    denom = min(len(rel_set), k)
    return (sum_precisions / denom) if denom > 0 else 0.0


def evaluate_retrieval_system(
    query_eval_records: List[Dict[str, Any]], 
    k_list: Optional[List[int]] = None
) -> Dict[str, float]:
    if k_list is None:
        k_list = [1, 5, 10]
    """
    Evaluates an entire retrieval or reranking run across multiple queries.
    Each record must have:
      - 'relevant_doc_ids': list of true matching doc/chunk ids.
      - 'retrieved_doc_ids': ordered list of doc ids returned by retrieval/reranking.
      - (optional) 'binary_relevance': [1, 0, 0, 1, ...] for explicit nDCG calculation.
    """
    if not query_eval_records:
        return {f"Recall@{k}": 0.0 for k in k_list} | {f"nDCG@{k}": 0.0 for k in k_list} | {f"MRR@{k}": 0.0 for k in k_list} | {f"MAP@{k}": 0.0 for k in k_list}

    num_queries = len(query_eval_records)
    aggregated_metrics = {}

    for k in k_list:
        total_recall = 0.0
        total_mrr = 0.0
        total_ndcg = 0.0
        total_map = 0.0

        for row in query_eval_records:
            rel = row.get("relevant_doc_ids", [])
            ret = row.get("retrieved_doc_ids", [])

            # 1. Recall@K
            total_recall += compute_recall_at_k(rel, ret, k=k)

            # 2. MRR@K
            total_mrr += compute_mrr_at_k(rel, ret, k=k)

            # 3. nDCG@K (Standard TREC/BEIR ideal DCG calculation)
            rel_set = set(rel)
            gains = [1.0 if d in rel_set else 0.0 for d in ret[:k]]
            ideal_gains = [1.0] * min(len(rel), k) + [0.0] * max(0, k - len(rel))
            total_ndcg += compute_ndcg_at_k(gains, ideal_relevance_gains=ideal_gains, k=k)
            total_map += compute_average_precision_at_k(rel, ret, k=k)

        aggregated_metrics[f"Recall@{k}"] = round(total_recall / num_queries, 4)
        aggregated_metrics[f"MRR@{k}"] = round(total_mrr / num_queries, 4)
        aggregated_metrics[f"nDCG@{k}"] = round(total_ndcg / num_queries, 4)
        aggregated_metrics[f"MAP@{k}"] = round(total_map / num_queries, 4)

    return aggregated_metrics
