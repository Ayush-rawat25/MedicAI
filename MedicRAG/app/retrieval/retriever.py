
import logging
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

    def retrieve(
        self,
        patient_id: str,
        query: str,
        top_k: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """
        Retrieves relevant historical records using semantic similarity.

        top_k is retained as an optional compatibility parameter.
        It controls candidate chunks, NOT the final number of records.
        """

        if not patient_id or not patient_id.strip():
            raise RetrieverError("Patient ID cannot be empty.")

        if not query or not query.strip():
            raise RetrieverError("Query cannot be empty.")

        candidate_limit = (
            top_k if top_k is not None else self.candidate_limit
        )

        if candidate_limit < 1:
            raise RetrieverError(
                "Candidate limit must be greater than zero."
            )

        try:
            # 1. Generate query embedding.
            query_embedding = self.embedding_service.embed_query(
                query.strip()
            )

            # 2. Search only within this patient's records.
            candidate_chunks = self.chroma_store.search(
                patient_id=patient_id,
                query_embedding=query_embedding,
                top_k=candidate_limit
            )

            # 3. Apply the similarity threshold.
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
                    "No chunks crossed similarity threshold %.3f "
                    "for patient %s",
                    self.similarity_threshold,
                    patient_id
                )
                return []

            # 4. Find the best matching score for each record.
            record_scores = {}

            for chunk in qualifying_chunks:
                record_id = chunk["metadata"]["record_id"]

                record_scores[record_id] = max(
                    record_scores.get(record_id, 0.0),
                    chunk["relevance_score"]
                )

            # 5. Fetch ALL chunks belonging to qualifying records.
            complete_chunks = []

            for record_id, relevance_score in record_scores.items():

                record_chunks = self.chroma_store.get_record_chunks(
                    patient_id=patient_id,
                    record_id=record_id
                )

                for record_chunk in record_chunks:
                    metadata = record_chunk.get("metadata", {})

                    # Defensive patient-isolation check.
                    if metadata.get("patient_id") != patient_id:
                        raise RetrieverError(
                            "Retrieved record belongs to another patient."
                        )

                    record_chunk["relevance_score"] = relevance_score
                    complete_chunks.append(record_chunk)

            logger.info(
                "Semantic retrieval completed | Patient: %s | "
                "Candidates: %d | Matching records: %d | "
                "Returned chunks: %d",
                patient_id,
                len(candidate_chunks),
                len(record_scores),
                len(complete_chunks)
            )

            return complete_chunks

        except (RetrieverError, ChromaStoreError):
            raise

        except Exception as exc:
            logger.exception("Semantic retrieval failed.")

            raise RetrieverError(
                "Unable to retrieve relevant patient records."
            ) from exc
