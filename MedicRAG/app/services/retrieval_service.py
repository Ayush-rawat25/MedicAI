import logging
import time
from typing import Any

from app.config import settings
from app.retrieval.intent_classifier import IntentClassifier
from app.retrieval.retrival_router import RetrievalRouter, RetrievalRouterError

logger = logging.getLogger(__name__)


class RetrievalServiceError(Exception):
    """Raised when the retrieval service fails."""


class RetrievalService:
    def __init__(
        self,
        intent_classifier: IntentClassifier,
        retrieval_router: RetrievalRouter,
    ):
        self.intent_classifier = intent_classifier
        self.retrieval_router = retrieval_router

    def retrieve(
        self,
        request: Any,
        *,
        eligible_before: str | None = None,
        exclude_record_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Classify and retrieve. Optional filters are used by evaluation only."""
        start_time = time.perf_counter()
        try:
            intent = self.intent_classifier.classify(
                query=request.query,
                context=getattr(request, "retrieval_context", None),
            )
            logger.info(
                "Query classified | Patient: %s | Strategy: %s | Method: %s",
                request.patient_id, intent.strategy, intent.method
            )
            retrieval_result = self.retrieval_router.retrieve(
                patient_id=request.patient_id,
                query=request.query,
                intent=intent,
                eligible_before=eligible_before,
                exclude_record_ids=exclude_record_ids,
            )
            records = retrieval_result.get("retrieved_records", [])
            actual_method = retrieval_result.get("retrieval_method", intent.method)
            if records:
                result_status = "success"
            elif actual_method == "structured":
                result_status = "no_records_found"
            else:
                result_status = "no_relevant_records"

            elapsed_time = time.perf_counter() - start_time
            router_metadata = retrieval_result.get("metadata", {})
            response = {
                "status": result_status,
                "patient_id": request.patient_id,
                "conversation_id": getattr(request, "conversation_id", None),
                "query": request.query,
                "retrieved_records": records,
                "metadata": {
                    "retrieval_strategy": intent.strategy,
                    "retrieval_method": actual_method,
                    "embedding_model": (
                        settings.EMBEDDING_MODEL if actual_method == "semantic" else None
                    ),
                    "vector_database": "ChromaDB",
                    "candidates_retrieved": router_metadata.get("candidates_retrieved", 0),
                    "records_returned": len(records),
                    "records_filtered": router_metadata.get("records_filtered", 0),
                    "retrieval_time_seconds": round(elapsed_time, 4),
                },
            }
            logger.info(
                "Retrieval completed | Patient: %s | Strategy: %s | Method: %s | Records: %d | Time: %.4fs",
                request.patient_id, intent.strategy, actual_method, len(records), elapsed_time
            )
            return response
        except (RetrievalRouterError, ValueError) as exc:
            logger.exception("Retrieval service execution failed.")
            raise RetrievalServiceError(
                "Unable to retrieve patient history."
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected retrieval service error.")
            raise RetrievalServiceError(
                "Unexpected retrieval service failure."
            ) from exc
