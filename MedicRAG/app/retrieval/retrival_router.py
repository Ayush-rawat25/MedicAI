
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
    """
    Routes classified queries to semantic or structured retrieval.

    All retrieval paths return a consistent result structure.
    """

    # Map clinical categories to actual stored record types.
    # Categories without a reliable metadata mapping use semantic retrieval.
    RECORD_TYPE_MAP = {
        "medication": "medication_record",
        "lab": "lab_report",
        "consultation": "consultation"
    }

    def __init__(
        self,
        semantic_retriever: Retriever,
        structured_retriever: StructuredRetriever,
        record_ranker: RecordRanker
    ):
        self.semantic_retriever = semantic_retriever
        self.structured_retriever = structured_retriever
        self.record_ranker = record_ranker

    def retrieve(
        self,
        patient_id: str,
        query: str,
        intent: RetrievalIntent
    ) -> dict[str, Any]:

        if not patient_id or not patient_id.strip():
            raise RetrievalRouterError("Patient ID cannot be empty.")

        try:
            logger.info(
                "Routing retrieval | Patient: %s | Strategy: %s | Method: %s",
                patient_id,
                intent.strategy,
                intent.method
            )

            if intent.strategy == "patient_overview":
                result = self._retrieve_overview(patient_id)

            elif intent.strategy == "chronological":
                result = self._retrieve_chronological(
                    patient_id,
                    intent
                )

            elif intent.strategy == "category_specific":
                result = self._retrieve_category(
                    patient_id,
                    query,
                    intent
                )

            else:
                raise RetrievalRouterError(
                    f"Unsupported strategy: {intent.strategy}"
                )

            result["retrieval_strategy"] = intent.strategy

            logger.info(
                "Routing completed | Strategy: %s | Actual method: %s | Records: %d",
                intent.strategy,
                result["retrieval_method"],
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
        patient_id: str
    ) -> dict[str, Any]:

        logger.info(
            "Executing patient overview | Patient: %s",
            patient_id
        )

        result = self.structured_retriever.retrieve_all(
            patient_id=patient_id
        )

        return self._normalize_structured_result(result)

    def _retrieve_chronological(
        self,
        patient_id: str,
        intent: RetrievalIntent
    ) -> dict[str, Any]:

        logger.info(
            "Executing chronological retrieval | Patient: %s",
            patient_id
        )

        record_type = self._map_record_type(
            intent.record_category
        )

        result = self.structured_retriever.retrieve_by_date(
            patient_id=patient_id,
            record_type=record_type,
            date_from=intent.date_from,
            date_to=intent.date_to,
            limit=intent.record_limit
        )

        return self._normalize_structured_result(result)

    def _retrieve_category(
        self,
        patient_id: str,
        query: str,
        intent: RetrievalIntent
    ) -> dict[str, Any]:

        category = intent.record_category

        logger.info(
            "Executing category retrieval | Patient: %s | Category: %s",
            patient_id,
            category
        )

        if intent.method == "structured":
            record_type = self._map_record_type(category)

            # Avoid filtering on a clinical category that is not
            # represented by an exact stored record_type.
            if record_type is not None:
                result = self.structured_retriever.retrieve_by_category(
                    patient_id=patient_id,
                    record_type=record_type,
                    date_from=intent.date_from,
                    date_to=intent.date_to,
                    limit=intent.record_limit
                )

                return self._normalize_structured_result(result)

            logger.warning(
                "No structured record type mapping for category '%s'. "
                "Using semantic retrieval instead.",
                category
            )

        elif intent.method == "hybrid":
            logger.warning(
                "Hybrid retrieval is not implemented yet. "
                "Using semantic retrieval as a temporary fallback."
            )

        elif intent.method != "semantic":
            raise RetrievalRouterError(
                f"Unsupported retrieval method: {intent.method}"
            )

        return self._retrieve_semantic(
            patient_id=patient_id,
            query=query
        )

    def _retrieve_semantic(
        self,
        patient_id: str,
        query: str
    ) -> dict[str, Any]:

        chunks = self.semantic_retriever.retrieve(
            patient_id=patient_id,
            query=query
        )

        records = self.record_ranker.rank(chunks)

        return {
            "patient_id": patient_id,
            "retrieved_records": records,
            "retrieval_method": "semantic",
            "metadata": {
                "records_returned": len(records),
                "retrieval_method": "semantic",
                "vector_database": "ChromaDB"
            }
        }

    @staticmethod
    def _normalize_structured_result(
        result: dict[str, Any]
    ) -> dict[str, Any]:

        metadata = result.get("metadata", {})

        return {
            "patient_id": result.get("patient_id"),
            "retrieved_records": result.get("retrieved_records", []),
            "retrieval_method": "structured",
            "metadata": metadata
        }

    @classmethod
    def _map_record_type(
        cls,
        category: str | None
    ) -> str | None:

        if not category:
            return None

        return cls.RECORD_TYPE_MAP.get(category)
