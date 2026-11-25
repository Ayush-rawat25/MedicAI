import math


def calculate_retrieval_metrics(
    retrieved_record_ids: list[str],
    relevant_record_ids: list[str],
    *,
    calculate_ranking: bool = True,
) -> dict[str, float | None]:
    """Calculate set-based metrics and optional ranking metrics."""
    retrieved = list(dict.fromkeys(retrieved_record_ids))
    relevant = set(relevant_record_ids)
    true_positives = sum(1 for record_id in retrieved if record_id in relevant)

    precision = true_positives / len(retrieved) if retrieved else 0.0
    recall = true_positives / len(relevant) if relevant else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    if not calculate_ranking:
        mrr = ndcg = None
    else:
        first_relevant_rank = next(
            (rank for rank, record_id in enumerate(retrieved, start=1)
             if record_id in relevant),
            None,
        )
        mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, record_id in enumerate(retrieved, start=1)
            if record_id in relevant
        )
        ideal_count = min(len(relevant), len(retrieved))
        ideal_dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_count + 1)
        )
        ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "mrr": round(mrr, 6) if mrr is not None else None,
        "ndcg": round(ndcg, 6) if ndcg is not None else None,
    }
