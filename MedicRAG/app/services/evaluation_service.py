import logging
import time
from typing import Any

from app.retrieval.structured_retriver import (
    StructuredRetriever,
    StructuredRetrievalError,
)
from app.schemas.evaluation_schemas import EvaluationRequest, EvaluationStrategyResult
from app.schemas.retrieval_input import RetrievalRequest
from app.services.evaluation_metrics import calculate_retrieval_metrics
from app.services.retrieval_service import RetrievalService, RetrievalServiceError

logger = logging.getLogger(__name__)


class EvaluationServiceError(Exception):
    """Raised when a retrieval evaluation cannot be completed."""


class EvaluationService:
    """Runs comparable consultation-only, full-history and selective tests."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        structured_retriever: StructuredRetriever,
    ):
        self.retrieval_service = retrieval_service
        self.structured_retriever = structured_retriever

    def run(self, request: EvaluationRequest) -> dict[str, Any]:
        excluded_ids = [request.consultation_id]
        ground_truth = request.relevant_record_ids
        results: list[EvaluationStrategyResult] = []

        try:
            # Sequential execution keeps the same ChromaDB collection and dataset
            # in place for all strategies.
            for strategy in request.strategies:
                if strategy == "consultation_only":
                    result = self._run_consultation_only(ground_truth)
                elif strategy == "full_history":
                    result = self._run_full_history(
                        request.patient_id,
                        request.cutoff_timestamp,
                        excluded_ids,
                        ground_truth,
                    )
                elif strategy == "selective":
                    result = self._run_selective(
                        request,
                        request.cutoff_timestamp,
                        excluded_ids,
                        ground_truth,
                    )
                else:
                    raise EvaluationServiceError(
                        f"Unsupported evaluation strategy: {strategy}"
                    )
                results.append(result)

            response = {
                "status": "success",
                "patient_id": request.patient_id,
                "consultation_id": request.consultation_id,
                "cutoff_timestamp": request.cutoff_timestamp,
                "query": request.query,
                "ground_truth_record_count": len(ground_truth),
                "results": [result.model_dump() for result in results],
            }
            logger.info(
                "Evaluation completed | Patient: %s | Strategies: %s",
                request.patient_id, ", ".join(request.strategies)
            )
            return response
        except (RetrievalServiceError, StructuredRetrievalError) as exc:
            logger.exception("Evaluation retrieval failed.")
            raise EvaluationServiceError(
                "Unable to complete retrieval evaluation."
            ) from exc
        except EvaluationServiceError:
            raise
        except Exception as exc:
            logger.exception("Unexpected evaluation error.")
            raise EvaluationServiceError(
                "Unexpected evaluation service failure."
            ) from exc

    @staticmethod
    def _run_consultation_only(
        ground_truth: list[str],
    ) -> EvaluationStrategyResult:
        return EvaluationStrategyResult(
            strategy="consultation_only",
            retrieved_record_ids=[],
            relevant_record_ids_retrieved=[],
            missed_relevant_record_ids=ground_truth.copy(),
            record_count=0,
            retrieval_time_seconds=0.0,
            retrieval_method="none",
            metrics={
                "precision": None,
                "recall": None,
                "f1_score": None,
                "mrr": None,
                "ndcg": None,
            },
            note=(
                "No historical records are supplied by this strategy. Retrieval "
                "metrics are not applicable. The consultation transcript is "
                "accepted as case context but is not searched by this retrieval-only endpoint."
            ),
            metadata={"historical_records_supplied": 0},
        )

    def _run_full_history(
        self,
        patient_id: str,
        cutoff: str,
        excluded_ids: list[str],
        ground_truth: list[str],
    ) -> EvaluationStrategyResult:
        start = time.perf_counter()
        retrieval_result = self.structured_retriever.retrieve_all(
            patient_id=patient_id,
            record_date_before=cutoff,
            exclude_record_ids=excluded_ids,
        )
        elapsed = time.perf_counter() - start
        records = retrieval_result.get("retrieved_records", [])
        retrieved_ids = self._record_ids(records)
        retrieved_set = set(retrieved_ids)
        relevant_retrieved = [rid for rid in ground_truth if rid in retrieved_set]
        missed = [rid for rid in ground_truth if rid not in retrieved_set]

        metrics = calculate_retrieval_metrics(
            retrieved_ids, ground_truth, calculate_ranking=False
        )
        return EvaluationStrategyResult(
            strategy="full_history",
            retrieved_record_ids=retrieved_ids,
            relevant_record_ids_retrieved=relevant_retrieved,
            missed_relevant_record_ids=missed,
            record_count=len(retrieved_ids),
            retrieval_time_seconds=round(elapsed, 6),
            retrieval_method="structured_full_history",
            metrics=metrics,
            note=(
                "Precision, recall and F1 measure query-relevant record coverage "
                "within the full eligible history. MRR and nDCG are omitted "
                "because full history is chronologically ordered, not ranked by query relevance."
            ),
            metadata={
                "cutoff_operator": "strictly_before",
                "cutoff_applied_before_retrieval_limit": True,
                "excluded_record_ids": excluded_ids,
                "store_retrieval_time_seconds": retrieval_result.get(
                    "metadata", {}
                ).get("retrieval_time_seconds"),
            },
        )

    def _run_selective(
        self,
        request: EvaluationRequest,
        cutoff: str,
        excluded_ids: list[str],
        ground_truth: list[str],
    ) -> EvaluationStrategyResult:
        retrieval_request = RetrievalRequest(
            patient_id=request.patient_id,
            conversation_id=f"evaluation-{request.consultation_id}",
            consultation_id=request.consultation_id,
            query=request.query,
        )
        start = time.perf_counter()
        retrieval_result = self.retrieval_service.retrieve(
            retrieval_request,
            eligible_before=cutoff,
            exclude_record_ids=excluded_ids,
        )
        elapsed = time.perf_counter() - start
        records = retrieval_result.get("retrieved_records", [])
        retrieved_ids = self._record_ids(records)
        retrieved_set = set(retrieved_ids)
        relevant_retrieved = [rid for rid in ground_truth if rid in retrieved_set]
        missed = [rid for rid in ground_truth if rid not in retrieved_set]
        actual_method = retrieval_result.get("metadata", {}).get(
            "retrieval_method", "unknown"
        )
        has_relevance_ranking = actual_method == "semantic"

        return EvaluationStrategyResult(
            strategy="selective",
            retrieved_record_ids=retrieved_ids,
            relevant_record_ids_retrieved=relevant_retrieved,
            missed_relevant_record_ids=missed,
            record_count=len(retrieved_ids),
            retrieval_time_seconds=round(elapsed, 6),
            retrieval_method=actual_method,
            metrics=calculate_retrieval_metrics(
                retrieved_ids,
                ground_truth,
                calculate_ranking=has_relevance_ranking,
            ),
            note=(
                "Selective retrieval uses the existing intent classifier and "
                "retrieval path with the evaluation cutoff and current-consultation "
                "exclusion applied. MRR/nDCG are reported only when the selected "
                "path returns relevance-ranked semantic results."
            ),
            metadata={
                **retrieval_result.get("metadata", {}),
                "cutoff_operator": "strictly_before",
                "cutoff_applied_before_retrieval_limit": True,
                "excluded_record_ids": excluded_ids,
            },
        )

    @staticmethod
    def _record_ids(records: list[dict[str, Any]]) -> list[str]:
        return list(dict.fromkeys(
            record["record_id"]
            for record in records
            if record.get("record_id")
        ))
