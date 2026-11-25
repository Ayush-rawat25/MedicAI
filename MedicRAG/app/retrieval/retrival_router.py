import logging
from typing import Any

from app.retrieval.intent_classifier import RetrievalIntent
from app.retrieval.retriever import Retriever
from app.retrieval.ranking import RecordRanker
from app.retrieval.structured_retriver import StructuredRetriever

logger = logging.getLogger(__name__)


class RetrievalRouterError(Exception):
    """Raised when a retrieval strategy cannot be executed."""


class RetrievalRouter:
    """Routes classified queries to semantic or structured retrieval."""

    RECORD_TYPE_MAP = {
        "medication": "medication_record",
        "lab": "lab_report",
        "consultation": "consultation",
    }

    def __init__(
        self,
        semantic_retriever: Retriever,
        structured_retriever: StructuredRetriever,
        record_ranker: RecordRanker,
    ):
        self.semantic_retriever = semantic_retriever
        self.structured_retriever = structured_retriever
        self.record_ranker = record_ranker

    def retrieve(
        self,
        patient_id: str,
        query: str,
        intent: RetrievalIntent,
        eligible_before: str | None = None,
        exclude_record_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        if not patient_id or not patient_id.strip():
            raise RetrievalRouterError("Patient ID cannot be empty.")
        try:
            logger.info(
                "Routing retrieval | Patient: %s | Strategy: %s | Method: %s",
                patient_id, intent.strategy, intent.method
            )
            if intent.strategy == "patient_overview":
                result = self._retrieve_overview(
                    patient_id, eligible_before, exclude_record_ids
                )
            elif intent.strategy == "chronological":
                result = self._retrieve_chronological(
                    patient_id, intent, eligible_before, exclude_record_ids
                )
            elif intent.strategy == "category_specific":
                result = self._retrieve_category(
                    patient_id, query, intent, eligible_before, exclude_record_ids
                )
            else:
                raise RetrievalRouterError(
                    f"Unsupported strategy: {intent.strategy}"
                )

            result["retrieval_strategy"] = intent.strategy
            logger.info(
                "Routing completed | Strategy: %s | Actual method: %s | Records: %d",
                intent.strategy, result["retrieval_method"],
                len(result["retrieved_records"])
            )
            return result
        except RetrievalRouterError:
            raise
        except Exception as exc:
            logger.exception("Retrieval routing failed.")
            raise RetrievalRouterError(
                "Unable to execute the selected retrieval strategy."
            ) from exc

    def _retrieve_overview(
        self,
        patient_id: str,
        eligible_before: str | None,
        exclude_record_ids: list[str] | None,
    ) -> dict[str, Any]:
        result = self.structured_retriever.retrieve_all(
            patient_id=patient_id,
            record_date_before=eligible_before,
            exclude_record_ids=exclude_record_ids,
        )
        return self._normalize_structured_result(result)

    def _retrieve_chronological(
        self,
        patient_id: str,
        intent: RetrievalIntent,
        eligible_before: str | None,
        exclude_record_ids: list[str] | None,
    ) -> dict[str, Any]:
        record_type = self._map_record_type(intent.record_category)
        result = self.structured_retriever.retrieve_by_date(
            patient_id=patient_id,
            record_type=record_type,
            date_from=intent.date_from,
            date_to=intent.date_to,
            limit=intent.record_limit,
            record_date_before=eligible_before,
            exclude_record_ids=exclude_record_ids,
        )
        return self._normalize_structured_result(result)

    def _retrieve_category(
        self,
        patient_id: str,
        query: str,
        intent: RetrievalIntent,
        eligible_before: str | None,
        exclude_record_ids: list[str] | None,
    ) -> dict[str, Any]:
        category = intent.record_category
        if intent.method == "structured":
            record_type = self._map_record_type(category)
            if record_type is not None:
                result = self.structured_retriever.retrieve_by_category(
                    patient_id=patient_id,
                    record_type=record_type,
                    date_from=intent.date_from,
                    date_to=intent.date_to,
                    limit=intent.record_limit,
                    record_date_before=eligible_before,
                    exclude_record_ids=exclude_record_ids,
                )
                return self._normalize_structured_result(result)
            logger.warning(
                "No structured record type mapping for category '%s'. Using semantic retrieval.",
                category
            )
        elif intent.method == "hybrid":
            logger.warning(
                "Hybrid retrieval is not implemented yet. Using semantic retrieval as a temporary fallback."
            )
        elif intent.method != "semantic":
            raise RetrievalRouterError(
                f"Unsupported retrieval method: {intent.method}"
            )

        return self._retrieve_semantic(
            patient_id, query, eligible_before, exclude_record_ids
        )

    def _retrieve_semantic(
        self,
        patient_id: str,
        query: str,
        eligible_before: str | None,
        exclude_record_ids: list[str] | None,
    ) -> dict[str, Any]:
        chunks = self.semantic_retriever.retrieve(
            patient_id=patient_id,
            query=query,
            record_date_before=eligible_before,
            exclude_record_ids=exclude_record_ids,
        )
        records = self.record_ranker.rank(chunks)
        return {
            "patient_id": patient_id,
            "retrieved_records": records,
            "retrieval_method": "semantic",
            "metadata": {
                "candidates_retrieved": len(chunks),
                "records_returned": len(records),
                "retrieval_method": "semantic",
                "vector_database": "ChromaDB",
            },
        }

    @staticmethod
    def _normalize_structured_result(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "patient_id": result.get("patient_id"),
            "retrieved_records": result.get("retrieved_records", []),
            "retrieval_method": "structured",
            "metadata": result.get("metadata", {}),
        }

    @classmethod
    def _map_record_type(cls, category: str | None) -> str | None:
        if not category:
            return None
        return cls.RECORD_TYPE_MAP.get(category)
