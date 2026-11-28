import math
from typing import Any


# ============================================================
# 1. RETRIEVAL EVALUATION METRICS
# ============================================================

def calculate_retrieval_metrics(
    retrieved_record_ids: list[str],
    relevant_record_ids: list[str],
    *,
    calculate_ranking: bool = True,
) -> dict[str, float | None]:
    """
    Calculate retrieval precision, recall, F1, MRR and nDCG.

    Ranking metrics should only be enabled when the retrieval
    output has a meaningful relevance-based ordering.
    """

    # Remove duplicate IDs while preserving retrieval order.
    retrieved = list(dict.fromkeys(retrieved_record_ids))
    relevant = set(relevant_record_ids)

    true_positives = sum(
        1 for record_id in retrieved
        if record_id in relevant
    )

    precision = (
        true_positives / len(retrieved)
        if retrieved else 0.0
    )

    recall = (
        true_positives / len(relevant)
        if relevant else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    if not calculate_ranking:
        mrr = None
        ndcg = None

    else:
        # Mean Reciprocal Rank (single-query reciprocal rank).
        first_relevant_rank = next(
            (
                rank
                for rank, record_id in enumerate(retrieved, start=1)
                if record_id in relevant
            ),
            None,
        )

        mrr = (
            1.0 / first_relevant_rank
            if first_relevant_rank else 0.0
        )

        # Binary Discounted Cumulative Gain.
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

        ndcg = (
            dcg / ideal_dcg
            if ideal_dcg else 0.0
        )

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "mrr": round(mrr, 6) if mrr is not None else None,
        "ndcg": round(ndcg, 6) if ndcg is not None else None,
    }


# ============================================================
# 2. ANSWER COVERAGE METRICS
# ============================================================

VALID_ASSESSMENT_STATUSES = {
    "supported",
    "partially_supported",
    "missing",
    "contradicted",
    "not_applicable",
}


def calculate_answer_coverage(
    fact_assessments: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate answer coverage using evaluator-assigned
    reference fact assessments.

    Scoring:
        supported           = 1.0
        partially_supported = 0.5
        missing             = 0.0
        contradicted        = 0.0
        not_applicable      = excluded

    Coverage denominator includes only applicable facts.
    """

    counts = {
        "supported": 0,
        "partially_supported": 0,
        "missing": 0,
        "contradicted": 0,
        "not_applicable": 0,
    }

    for assessment in fact_assessments:
        status = assessment.get("status")

        if status not in VALID_ASSESSMENT_STATUSES:
            raise ValueError(
                f"Invalid fact assessment status: {status}"
            )

        counts[status] += 1

    applicable_count = (
        counts["supported"]
        + counts["partially_supported"]
        + counts["missing"]
        + counts["contradicted"]
    )

    if applicable_count == 0:
        coverage_score = None
        fully_supported_percentage = None

    else:
        weighted_score = (
            counts["supported"]
            + 0.5 * counts["partially_supported"]
        )

        coverage_score = (
            weighted_score / applicable_count
        ) * 100

        fully_supported_percentage = (
            counts["supported"] / applicable_count
        ) * 100

    return {
        "applicable_fact_count": applicable_count,
        "supported_count": counts["supported"],
        "partially_supported_count": counts["partially_supported"],
        "missing_count": counts["missing"],
        "contradicted_count": counts["contradicted"],
        "not_applicable_count": counts["not_applicable"],
        "coverage_score": (
            round(coverage_score, 2)
            if coverage_score is not None else None
        ),
        "fully_supported_percentage": (
            round(fully_supported_percentage, 2)
            if fully_supported_percentage is not None else None
        ),
    }


# ============================================================
# 3. UNSUPPORTED CLAIM RATE
# ============================================================

def calculate_unsupported_claim_rate(
    unsupported_claim_count: int,
    total_answer_claims: int,
) -> float | None:
    """
    Calculate the percentage of generated claims identified
    as unsupported, incorrect, contradictory or invented.

    total_answer_claims must be the total number of claims
    assessed by the evaluator, not just unsupported claims.
    """

    if unsupported_claim_count < 0 or total_answer_claims < 0:
        raise ValueError("Claim counts cannot be negative.")

    if unsupported_claim_count > total_answer_claims:
        raise ValueError(
            "Unsupported claims cannot exceed total assessed claims."
        )

    if total_answer_claims == 0:
        return None

    rate = (
        unsupported_claim_count / total_answer_claims
    ) * 100

    return round(rate, 2)


# ============================================================
# 4. CONTEXT GROUNDING SCORE
# ============================================================

def calculate_context_grounding_score(
    fact_assessments: list[dict[str, Any]],
) -> float | None:
    """
    Calculate how many assessed facts are grounded in the
    actual context supplied to the generation model.

    Only assessments with an explicit context_grounded value
    are included in the denominator.

    This distinguishes globally correct facts from facts
    that the strategy was actually given evidence to support.
    """

    grounding_assessments = [
        assessment
        for assessment in fact_assessments
        if assessment.get("context_grounded") is not None
        and assessment.get("status") != "not_applicable"
    ]

    if not grounding_assessments:
        return None

    grounded_count = sum(
        1
        for assessment in grounding_assessments
        if assessment.get("context_grounded") is True
    )

    score = (
        grounded_count / len(grounding_assessments)
    ) * 100

    return round(score, 2)


# ============================================================
# 5. CRITICAL FACT ERROR COUNT
# ============================================================

def calculate_critical_fact_errors(
    fact_assessments: list[dict[str, Any]],
    reference_facts: list[dict[str, Any]],
) -> int:
    """
    Count missing or contradicted critical/high-priority facts.

    Not-applicable facts are excluded.

    A critical fact error is counted once per reference fact.
    """

    priority_by_id = {
        fact["fact_id"]: fact.get("priority", "medium")
        for fact in reference_facts
        if fact.get("fact_id")
    }

    critical_priorities = {"critical", "high"}

    error_count = 0

    for assessment in fact_assessments:
        fact_id = assessment.get("fact_id")
        if not fact_id:
            continue

        priority = priority_by_id.get(fact_id, "medium")

        if priority not in critical_priorities:
            continue

        if assessment.get("status") in {
            "missing",
            "contradicted",
        }:
            error_count += 1

    return error_count


# ============================================================
# 6. COMPLETE ANSWER EVALUATION
# ============================================================

def calculate_answer_metrics(
    fact_assessments: list[dict[str, Any]],
    unsupported_claims: list[dict[str, Any]],
    reference_facts: list[dict[str, Any]],
    *,
    total_answer_claims: int,
) -> dict[str, Any]:
    """
    Calculate all answer-level evaluation metrics.

    Inputs are evaluator-produced assessments, not raw LLM
    judgments. The evaluator must first assess each reference
    fact and generated claim.
    """

    coverage = calculate_answer_coverage(
        fact_assessments
    )

    unsupported_claim_rate = calculate_unsupported_claim_rate(
        unsupported_claim_count=len(unsupported_claims),
        total_answer_claims=total_answer_claims,
    )

    context_grounding_score = calculate_context_grounding_score(
        fact_assessments
    )

    critical_fact_errors = calculate_critical_fact_errors(
        fact_assessments,
        reference_facts,
    )

    return {
        **coverage,
        "unsupported_claim_rate": unsupported_claim_rate,
        "context_grounding_score": context_grounding_score,
        "critical_fact_errors": critical_fact_errors,
    }