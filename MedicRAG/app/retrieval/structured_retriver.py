
import logging
import time
from typing import Any, Optional

from app.vectorStore.chroma_store import (
    ChromaStore,
    ChromaStoreError
)

logger = logging.getLogger(__name__)


class StructuredRetrievalError(Exception):
    """Raised when structured historical retrieval fails."""


class StructuredRetriever:
    """
    Retrieves patient records using metadata and chronological filters.

    No embedding generation or similarity threshold is used here.
    """

    def __init__(self, chroma_store: ChromaStore):
        self.chroma_store = chroma_store

    def retrieve_all(
        self,
        patient_id: str
    ) -> dict[str, Any]:
        """Retrieve all available records for a patient."""

        start_time = time.perf_counter()

        try:
            self._validate_patient_id(patient_id)

            chunks = self.chroma_store.get_patient_records(
                patient_id=patient_id
            )

            records = self._group_chunks(chunks)

            records.sort(
                key=lambda record: record["date"],
                reverse=True
            )

            return self._build_response(
                patient_id=patient_id,
                records=records,
                start_time=start_time
            )

        except Exception as exc:
            self._handle_error(exc)

    def retrieve_by_category(
        self,
        patient_id: str,
        record_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: Optional[int] = None
    ) -> dict[str, Any]:
        """Retrieve records matching a specific category."""

        start_time = time.perf_counter()

        try:
            self._validate_patient_id(patient_id)
            self._validate_dates(date_from, date_to)
            self._validate_limit(limit)

            chunks = self.chroma_store.get_patient_records(
                patient_id=patient_id,
                record_type=record_type,
                date_from=date_from,
                date_to=date_to
            )

            records = self._group_chunks(chunks)

            records.sort(
                key=lambda record: record["date"],
                reverse=True
            )

            if limit is not None:
                records = records[:limit]

            return self._build_response(
                patient_id=patient_id,
                records=records,
                start_time=start_time
            )

        except Exception as exc:
            self._handle_error(exc)

    def retrieve_by_date(
        self,
        patient_id: str,
        record_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: Optional[int] = None
    ) -> dict[str, Any]:
        """Retrieve records ordered by date, newest first."""

        return self.retrieve_by_category(
            patient_id=patient_id,
            record_type=record_type,
            date_from=date_from,
            date_to=date_to,
            limit=limit
        )

    @staticmethod
    def _group_chunks(
        chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Combine chunks belonging to the same historical record."""

        grouped = {}

        for chunk in chunks:
            metadata = chunk.get("metadata", {})

            record_id = metadata.get("record_id")
            patient_id = metadata.get("patient_id")

            if not record_id or not patient_id:
                continue

            key = (patient_id, record_id)

            if key not in grouped:
                grouped[key] = {
                    "record_id": record_id,
                    "record_type": metadata.get("record_type"),
                    "date": metadata.get("record_date"),
                    "source": metadata.get("source"),
                    "chunks": []
                }

            grouped[key]["chunks"].append({
                "chunk_index": metadata.get("chunk_index", 0),
                "content": chunk.get("text", "")
            })

        records = []

        for record in grouped.values():
            ordered_chunks = sorted(
                record["chunks"],
                key=lambda item: item["chunk_index"]
            )

            record_content = "\n".join(
                chunk["content"].strip()
                for chunk in ordered_chunks
                if chunk["content"].strip()
            )

            records.append({
                "record_id": record["record_id"],
                "record_type": record["record_type"],
                "date": record["date"],
                "content": record_content,
                "source": record["source"],
                "relevance_score": None
            })

        return records

    @staticmethod
    def _build_response(
        patient_id: str,
        records: list[dict[str, Any]],
        start_time: float
    ) -> dict[str, Any]:

        return {
            "patient_id": patient_id,
            "retrieved_records": records,
            "metadata": {
                "records_returned": len(records),
                "retrieval_time_seconds": round(
                    time.perf_counter() - start_time,
                    4
                ),
                "retrieval_method": "structured",
                "vector_database": "ChromaDB"
            }
        }

    @staticmethod
    def _validate_patient_id(patient_id: str) -> None:

        if not patient_id or not patient_id.strip():
            raise StructuredRetrievalError(
                "Patient ID cannot be empty."
            )

    @staticmethod
    def _validate_dates(
        date_from: Optional[str],
        date_to: Optional[str]
    ) -> None:

        from datetime import date

        try:
            start = date.fromisoformat(date_from) if date_from else None
            end = date.fromisoformat(date_to) if date_to else None

            if start and end and start > end:
                raise StructuredRetrievalError(
                    "date_from cannot be later than date_to."
                )

        except ValueError as exc:
            raise StructuredRetrievalError(
                "Dates must use YYYY-MM-DD format."
            ) from exc

    @staticmethod
    def _validate_limit(limit: Optional[int]) -> None:

        if limit is not None and limit <= 0:
            raise StructuredRetrievalError(
                "Record limit must be greater than zero."
            )

    @staticmethod
    def _handle_error(exc: Exception) -> None:

        if isinstance(exc, StructuredRetrievalError):
            raise exc

        if isinstance(exc, ChromaStoreError):
            logger.exception("ChromaDB structured retrieval failed.")
            raise StructuredRetrievalError(
                "Unable to retrieve historical patient records."
            ) from exc

        logger.exception("Unexpected structured retrieval error.")
        raise StructuredRetrievalError(
            "Structured retrieval failed."
        ) from exc
