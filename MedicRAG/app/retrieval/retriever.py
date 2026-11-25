import logging
from datetime import datetime
from typing import Any, Optional

from app.config import settings
from app.vectorStore.chroma_store import ChromaStore, ChromaStoreError
from app.ingestion.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class RetrieverError(Exception):
    """Raised when semantic retrieval fails."""


class Retriever:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        chroma_store: ChromaStore
    ):
        self.embedding_service = embedding_service
        self.chroma_store = chroma_store
        self.candidate_limit = settings.RETRIEVAL_CANDIDATE_LIMIT
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD

        if self.candidate_limit < 1:
            raise ValueError("Candidate limit must be greater than zero.")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1.")

    @staticmethod
    def _timestamp_is_before(value: str | None, cutoff: str) -> bool:
        if not value:
            return False
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
            cutoff_dt = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
            if timestamp.tzinfo is None or cutoff_dt.tzinfo is None:
                return False
            return timestamp < cutoff_dt
        except ValueError:
            return False

    def retrieve(
        self,
        patient_id: str,
        query: str,
        top_k: Optional[int] = None,
        record_date_before: str | None = None,
        exclude_record_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve complete matching records using semantic similarity."""
        if not patient_id or not patient_id.strip():
            raise RetrieverError("Patient ID cannot be empty.")
        if not query or not query.strip():
            raise RetrieverError("Query cannot be empty.")

        candidate_limit = top_k if top_k is not None else self.candidate_limit
        if candidate_limit < 1:
            raise RetrieverError("Candidate limit must be greater than zero.")

        try:
            query_embedding = self.embedding_service.embed_query(query.strip())
            candidate_chunks = self.chroma_store.search(
                patient_id=patient_id,
                query_embedding=query_embedding,
                top_k=candidate_limit,
                record_date_before=record_date_before,
                exclude_record_ids=exclude_record_ids,
            )

            qualifying_chunks = []
            for chunk in candidate_chunks:
                distance = chunk.get("distance")
                if distance is None:
                    continue
                similarity = 1.0 - float(distance)
                if similarity >= self.similarity_threshold:
                    chunk["relevance_score"] = similarity
                    qualifying_chunks.append(chunk)

            if not qualifying_chunks:
                logger.info(
                    "No chunks crossed similarity threshold %.3f for patient %s",
                    self.similarity_threshold, patient_id
                )
                return []

            record_scores: dict[str, float] = {}
            for chunk in qualifying_chunks:
                metadata = chunk.get("metadata", {})
                record_id = metadata.get("record_id")
                if not record_id:
                    continue
                record_scores[record_id] = max(
                    record_scores.get(record_id, 0.0),
                    chunk["relevance_score"]
                )

            excluded = set(exclude_record_ids or [])
            complete_chunks = []
            for record_id, relevance_score in record_scores.items():
                if record_id in excluded:
                    continue
                record_chunks = self.chroma_store.get_record_chunks(
                    patient_id=patient_id,
                    record_id=record_id
                )

                # Defend against inconsistent chunk metadata: when a cutoff is
                # active, every chunk in the record must have a valid date before it.
                if record_date_before:
                    all_chunks_eligible = bool(record_chunks) and all(
                        chunk.get("metadata", {}).get("record_id") not in excluded
                        and self._timestamp_is_before(
                            chunk.get("metadata", {}).get("record_date"),
                            record_date_before,
                        )
                        for chunk in record_chunks
                    )
                    if not all_chunks_eligible:
                        logger.warning(
                            "Skipping record %s because one or more chunks do not satisfy cutoff %s",
                            record_id, record_date_before
                        )
                        continue

                for record_chunk in record_chunks:
                    metadata = record_chunk.get("metadata", {})
                    if metadata.get("patient_id") != patient_id:
                        raise RetrieverError(
                            "Retrieved record belongs to another patient."
                        )
                    record_chunk["relevance_score"] = relevance_score
                    complete_chunks.append(record_chunk)

            logger.info(
                "Semantic retrieval completed | Patient: %s | Candidates: %d | Matching records: %d | Returned chunks: %d",
                patient_id, len(candidate_chunks), len(record_scores), len(complete_chunks)
            )
            return complete_chunks
        except (RetrieverError, ChromaStoreError):
            raise
        except Exception as exc:
            logger.exception("Semantic retrieval failed.")
            raise RetrieverError(
                "Unable to retrieve relevant patient records."
            ) from exc
