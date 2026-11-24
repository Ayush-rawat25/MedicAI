import logging
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
        """
        Validates chunks before inserting them into ChromaDB.
        """

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

    def add_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """
        Inserts or updates embedded chunks in ChromaDB.

        Uses upsert so that repeated ingestion of the same
        chunk IDs does not create duplicate entries.
        """

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
        """
        Deletes all chunks belonging to a specific
        historical record of a specific patient.
        """

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
        """
        Deletes all indexed historical records
        belonging to a specific patient.

        Use only when explicitly required.
        """

        if not patient_id:
            raise ChromaStoreError("Patient ID is required.")

        try:
            self.collection.delete(where={"patient_id": patient_id})

            logger.info("Deleted all indexed records for patient %s", patient_id)

        except Exception as exc:
            logger.exception("Failed to delete patient records")

            raise ChromaStoreError("Unable to delete patient records.") from exc

    def search(
        self, patient_id: str, query_embedding: list[float], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Searches historical records belonging to one patient.

        Patient filtering is mandatory and happens inside
        the ChromaDB query.
        """

        if not patient_id:
            raise ChromaStoreError("Patient ID is required for retrieval.")

        if not query_embedding:
            raise ChromaStoreError("Query embedding cannot be empty.")

        if top_k < 1:
            raise ChromaStoreError("top_k must be greater than zero.")

        try:
            collection_count = self.collection.count()

            if collection_count == 0:
                return []

            effective_limit = min(top_k, collection_count)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=effective_limit,
                where={"patient_id": patient_id},
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

                retrieved_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": document,
                        "metadata": metadata,
                        "distance": float(distance),
                    }
                )

            logger.info(
                "Retrieved %d chunks for patient %s", len(retrieved_chunks), patient_id
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
        """
        Fetches all indexed chunks for a specific record
        belonging to a specific patient.
        """

        if not patient_id or not record_id:
            raise ChromaStoreError("Patient ID and record ID are required.")

        try:
            results = self.collection.get(
                where={"$and": [{"patient_id": patient_id}, {"record_id": record_id}]},
                include=["documents", "metadatas"],
            )

            chunks = []

            for chunk_id, document, metadata in zip(
                results["ids"], results["documents"], results["metadatas"]
            ):
                chunks.append(
                    {"chunk_id": chunk_id, "text": document, "metadata": metadata}
                )

            return chunks

        except Exception as exc:
            logger.exception("Failed to fetch record chunks")

            raise ChromaStoreError("Unable to fetch historical record chunks.") from exc

    def get_patient_records(
        self,
        patient_id: str,
        record_type: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieves all chunks belonging to a patient's records.

        Supports optional record-type and date filtering.
        Uses pagination to avoid silently truncating large histories.
        """

        from datetime import date

        if not patient_id or not patient_id.strip():
            raise ChromaStoreError("Patient ID is required.")

        # Validate dates before querying ChromaDB.
        try:
            start_date = date.fromisoformat(date_from) if date_from else None
            end_date = date.fromisoformat(date_to) if date_to else None

            if start_date and end_date and start_date > end_date:
                raise ChromaStoreError("date_from cannot be later than date_to.")

        except ValueError as exc:
            raise ChromaStoreError("Dates must use YYYY-MM-DD format.") from exc

        try:
            conditions = [{"patient_id": patient_id}]

            if record_type:
                conditions.append({"record_type": record_type})

            if date_from:
                conditions.append({"record_date": {"$gte": date_from}})

            if date_to:
                conditions.append({"record_date": {"$lte": date_to}})

            where_filter = (
                conditions[0] if len(conditions) == 1 else {"$and": conditions}
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

                    all_chunks.append(
                        {"chunk_id": chunk_id, "text": document, "metadata": metadata}
                    )

                offset += len(ids)

                if len(ids) < batch_size:
                    break

            logger.info(
                "Retrieved %d structured chunks for patient %s",
                len(all_chunks),
                patient_id,
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
        """
        Returns the total number of indexed chunks.
        """

        try:
            return self.collection.count()

        except Exception as exc:
            logger.exception("Failed to retrieve collection count")

            raise ChromaStoreError("Unable to retrieve collection count.") from exc
