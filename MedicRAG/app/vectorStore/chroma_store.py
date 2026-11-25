import logging
from datetime import datetime, timezone
from typing import Any

import chromadb

from app.config import settings

logger = logging.getLogger(__name__)


class ChromaStoreError(Exception):
    """Raised when a ChromaDB operation fails."""


class ChromaStore:
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIRECTORY
            )
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": settings.CHROMA_DISTANCE_METRIC},
            )
            logger.info(
                "ChromaDB initialized successfully. Collection: %s",
                settings.CHROMA_COLLECTION_NAME,
            )
        except Exception as exc:
            logger.exception("Failed to initialize ChromaDB")
            raise ChromaStoreError("Unable to initialize vector database.") from exc

    @staticmethod
    def _validate_embedded_chunks(chunks: list[dict[str, Any]]) -> None:
        for chunk in chunks:
            if not chunk.get("chunk_id"):
                raise ChromaStoreError("Chunk ID is missing.")
            if not chunk.get("text"):
                raise ChromaStoreError("Chunk text is missing.")
            if not chunk.get("embedding"):
                raise ChromaStoreError("Chunk embedding is missing.")

            metadata = chunk.get("metadata", {})
            if not metadata.get("patient_id"):
                raise ChromaStoreError("Patient ID missing from chunk metadata.")
            if not metadata.get("record_id"):
                raise ChromaStoreError("Record ID missing from chunk metadata.")
            if not isinstance(chunk["embedding"], list):
                raise ChromaStoreError("Embedding must be a list of numbers.")
            if not all(isinstance(value, (int, float)) for value in chunk["embedding"]):
                raise ChromaStoreError("Embedding contains invalid values.")

    @staticmethod
    def _validate_utc_timestamp(value: str | None) -> None:
        """Require an ISO-8601 timestamp with an explicit timezone."""
        if value is None:
            return
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("Timestamp must include a timezone.")
        except ValueError as exc:
            raise ChromaStoreError(
                "Timestamp must be ISO-8601 with a timezone, e.g. 2020-08-28T11:13:21Z."
            ) from exc

    @staticmethod
    def _timestamp_is_before(record_date: str | None, cutoff: str) -> bool:
        if not record_date:
            return False
        try:
            record_dt = datetime.fromisoformat(record_date.replace("Z", "+00:00"))
            cutoff_dt = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
            if record_dt.tzinfo is None or cutoff_dt.tzinfo is None:
                return False
            return record_dt < cutoff_dt
        except ValueError:
            return False

    @staticmethod
    def _build_where_filter(
        patient_id: str,
        record_type: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        record_date_before: str | None = None,
        exclude_record_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        conditions: list[dict[str, Any]] = [{"patient_id": patient_id}]

        if record_type:
            conditions.append({"record_type": record_type})
        if date_from:
            conditions.append({"record_date": {"$gte": date_from}})
        if date_to:
            conditions.append({"record_date": {"$lte": date_to}})
        # ISO-8601 string timestamps are filtered in Python. Chroma metadata
        # comparison operators are not relied on for string ordering.
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    def add_chunks(self, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0
        try:
            self._validate_embedded_chunks(chunks)
            self.collection.upsert(
                ids=[chunk["chunk_id"] for chunk in chunks],
                documents=[chunk["text"] for chunk in chunks],
                embeddings=[chunk["embedding"] for chunk in chunks],
                metadatas=[chunk["metadata"] for chunk in chunks],
            )
            logger.info("Inserted or updated %d chunks in ChromaDB", len(chunks))
            return len(chunks)
        except ChromaStoreError:
            raise
        except Exception as exc:
            logger.exception("Failed to insert chunks")
            raise ChromaStoreError("Unable to insert embedded chunks.") from exc

    def delete_record(self, patient_id: str, record_id: str) -> None:
        if not patient_id or not record_id:
            raise ChromaStoreError("Patient ID and record ID are required.")
        try:
            self.collection.delete(
                where={"$and": [{"patient_id": patient_id}, {"record_id": record_id}]}
            )
            logger.info("Deleted record %s for patient %s", record_id, patient_id)
        except Exception as exc:
            logger.exception("Failed to delete historical record")
            raise ChromaStoreError("Unable to delete historical record.") from exc

    def delete_patient_records(self, patient_id: str) -> None:
        if not patient_id:
            raise ChromaStoreError("Patient ID is required.")
        try:
            self.collection.delete(where={"patient_id": patient_id})
            logger.info("Deleted all indexed records for patient %s", patient_id)
        except Exception as exc:
            logger.exception("Failed to delete patient records")
            raise ChromaStoreError("Unable to delete patient records.") from exc

    def search(
        self,
        patient_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        record_date_before: str | None = None,
        exclude_record_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search, optionally restricted to records before a UTC cutoff."""
        if not patient_id:
            raise ChromaStoreError("Patient ID is required for retrieval.")
        if not query_embedding:
            raise ChromaStoreError("Query embedding cannot be empty.")
        if top_k < 1:
            raise ChromaStoreError("top_k must be greater than zero.")
        self._validate_utc_timestamp(record_date_before)

        try:
            collection_count = self.collection.count()
            if collection_count == 0:
                return []

            # When eligibility filters are active, request all patient chunks,
            # filter by timestamp/record ID, then apply top_k. This prevents
            # ineligible future chunks from consuming the semantic candidate limit.
            needs_python_filter = bool(record_date_before or exclude_record_ids)
            effective_limit = (
                collection_count if needs_python_filter
                else min(top_k, collection_count)
            )
            where_filter = self._build_where_filter(patient_id=patient_id)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=effective_limit,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            retrieved_chunks = []
            ids = results.get("ids", [[]])[0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances
            ):
                if metadata.get("patient_id") != patient_id:
                    logger.error(
                        "Patient isolation validation failed for chunk %s", chunk_id
                    )
                    raise ChromaStoreError(
                        "Retrieved chunk belongs to another patient."
                    )
                if exclude_record_ids and metadata.get("record_id") in set(exclude_record_ids):
                    continue
                if record_date_before and not self._timestamp_is_before(
                    metadata.get("record_date"), record_date_before
                ):
                    logger.debug(
                        "Skipping chunk %s outside timestamp cutoff %s",
                        chunk_id, record_date_before
                    )
                    continue
                retrieved_chunks.append({
                    "chunk_id": chunk_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": float(distance),
                })

            # Chroma returns query results in distance order; slicing here means
            # only eligible candidates consume the configured candidate limit.
            retrieved_chunks = retrieved_chunks[:top_k]

            logger.info(
                "Retrieved %d semantic candidate chunks for patient %s",
                len(retrieved_chunks), patient_id
            )
            return retrieved_chunks
        except ChromaStoreError:
            raise
        except Exception as exc:
            logger.exception("Vector search failed")
            raise ChromaStoreError("Unable to retrieve patient records.") from exc

    def get_record_chunks(
        self, patient_id: str, record_id: str
    ) -> list[dict[str, Any]]:
        if not patient_id or not record_id:
            raise ChromaStoreError("Patient ID and record ID are required.")
        try:
            results = self.collection.get(
                where={"$and": [{"patient_id": patient_id}, {"record_id": record_id}]},
                include=["documents", "metadatas"],
            )
            return [
                {"chunk_id": chunk_id, "text": document, "metadata": metadata}
                for chunk_id, document, metadata in zip(
                    results["ids"], results["documents"], results["metadatas"]
                )
            ]
        except Exception as exc:
            logger.exception("Failed to fetch record chunks")
            raise ChromaStoreError(
                "Unable to fetch historical record chunks."
            ) from exc

    def get_patient_records(
        self,
        patient_id: str,
        record_type: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        record_date_before: str | None = None,
        exclude_record_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch all patient chunks, with optional exact timestamp restrictions."""
        from datetime import date

        if not patient_id or not patient_id.strip():
            raise ChromaStoreError("Patient ID is required.")
        try:
            start_date = date.fromisoformat(date_from) if date_from else None
            end_date = date.fromisoformat(date_to) if date_to else None
            if start_date and end_date and start_date > end_date:
                raise ChromaStoreError("date_from cannot be later than date_to.")
        except ValueError as exc:
            raise ChromaStoreError("Dates must use YYYY-MM-DD format.") from exc
        self._validate_utc_timestamp(record_date_before)

        try:
            where_filter = self._build_where_filter(
                patient_id=patient_id,
                record_type=record_type,
                date_from=date_from,
                date_to=date_to,
            )
            all_chunks = []
            offset = 0
            batch_size = 500
            while True:
                results = self.collection.get(
                    where=where_filter,
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas"],
                )
                ids = results.get("ids", [])
                documents = results.get("documents") or []
                metadatas = results.get("metadatas") or []
                if not ids:
                    break
                for chunk_id, document, metadata in zip(ids, documents, metadatas):
                    if metadata.get("patient_id") != patient_id:
                        logger.error(
                            "Patient isolation validation failed for chunk %s", chunk_id
                        )
                        raise ChromaStoreError(
                            "Retrieved chunk belongs to another patient."
                        )
                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "text": document,
                        "metadata": metadata,
                    })
                offset += len(ids)
                if len(ids) < batch_size:
                    break

            excluded = set(exclude_record_ids or [])
            if excluded:
                all_chunks = [
                    chunk for chunk in all_chunks
                    if chunk.get("metadata", {}).get("record_id") not in excluded
                ]

            if record_date_before:
                # Enforce cutoff at record level: all chunks for a record must
                # carry a valid timestamp strictly earlier than the cutoff.
                grouped_dates: dict[str, list[bool]] = {}
                for chunk in all_chunks:
                    metadata = chunk.get("metadata", {})
                    record_id = metadata.get("record_id")
                    if record_id:
                        grouped_dates.setdefault(record_id, []).append(
                            self._timestamp_is_before(
                                metadata.get("record_date"), record_date_before
                            )
                        )
                eligible_record_ids = {
                    record_id
                    for record_id, chunk_checks in grouped_dates.items()
                    if chunk_checks and all(chunk_checks)
                }
                all_chunks = [
                    chunk for chunk in all_chunks
                    if chunk.get("metadata", {}).get("record_id") in eligible_record_ids
                ]

            logger.info(
                "Retrieved %d eligible structured chunks for patient %s",
                len(all_chunks), patient_id
            )
            return all_chunks
        except ChromaStoreError:
            raise
        except Exception as exc:
            logger.exception("Failed to retrieve structured patient records.")
            raise ChromaStoreError(
                "Unable to retrieve structured patient records."
            ) from exc

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception as exc:
            logger.exception("Failed to retrieve collection count")
            raise ChromaStoreError("Unable to retrieve collection count.") from exc
