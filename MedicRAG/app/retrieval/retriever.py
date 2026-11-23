
import logging
import time
from typing import Any

from app.config import settings
from app.ingestion.embedding_service import (
    EmbeddingService,
    EmbeddingServiceError,
)
from app.vectorStore.chroma_store import (
    ChromaStore,
    ChromaStoreError,
)


logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Raised when historical record retrieval fails."""


class Retriever:

    def __init__(
        self,
        embedding_service: EmbeddingService,
        chroma_store: ChromaStore
    ):
        self.embedding_service = embedding_service
        self.chroma_store = chroma_store

        self.top_k = settings.RETRIEVAL_TOP_K
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD

    def retrieve(
        self,
        patient_id: str,
        query: str,
        top_k: int | None = None
    ) -> dict[str, Any]:
        """
        Retrieve relevant historical record chunks for a patient.

        Similarity filtering is optional. When a threshold is configured,
        chunks below that threshold are discarded.
        """

        start_time = time.perf_counter()

        try:
            # Validate input
            if not patient_id or not patient_id.strip():
                raise RetrievalError("Patient ID cannot be empty.")

            if not query or not query.strip():
                raise RetrievalError("Query cannot be empty.")

            # Resolve top_k
            result_limit = self.top_k if top_k is None else top_k

            if result_limit <= 0:
                raise RetrievalError("top_k must be greater than zero.")

            patient_id = patient_id.strip()
            query = query.strip()

            logger.info(
                "Starting retrieval | Patient: %s | top_k: %d",
                patient_id,
                result_limit
            )

            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(query)

            # Search only within the selected patient's records
            search_results = self.chroma_store.search(
                patient_id=patient_id,
                query_embedding=query_embedding,
                top_k=result_limit
            )

            retrieved_chunks = []

            for result in search_results:

                metadata = result.get("metadata", {})

                # Additional patient isolation check
                if metadata.get("patient_id") != patient_id:
                    logger.warning(
                        "Discarding chunk belonging to another patient."
                    )
                    continue

                distance = result.get("distance")

                if distance is None:
                    logger.warning(
                        "Discarding chunk with missing distance: %s",
                        result.get("chunk_id")
                    )
                    continue

                # ChromaDB cosine distance -> cosine similarity
                relevance_score = 1.0 - float(distance)

                # Apply optional similarity threshold
                if (
                    self.similarity_threshold is not None
                    and relevance_score < self.similarity_threshold
                ):
                    continue

                retrieved_chunks.append({
                    "chunk_id": result["chunk_id"],
                    "record_id": metadata["record_id"],
                    "patient_id": metadata["patient_id"],
                    "record_type": metadata["record_type"],
                    "date": metadata["record_date"],
                    "chunk_index": metadata["chunk_index"],
                    "total_chunks": metadata["total_chunks"],
                    "content": result["text"],
                    "relevance_score": relevance_score,
                    "source": metadata.get("source")
                })

            retrieval_time = time.perf_counter() - start_time

            logger.info(
                "Retrieval completed | Patient: %s | Retrieved chunks: %d "
                "| Retrieval time: %.4f seconds",
                patient_id,
                len(retrieved_chunks),
                retrieval_time
            )

            return {
                "patient_id": patient_id,
                "query": query,
                "retrieved_chunks": retrieved_chunks,
                "metadata": {
                    "retrieved_count": len(retrieved_chunks),
                    "top_k": result_limit,
                    "retrieval_time_seconds": round(
                        retrieval_time, 4
                    )
                }
            }

        except RetrievalError:
            raise

        except EmbeddingServiceError as exc:
            logger.exception("Query embedding generation failed.")
            raise RetrievalError(
                "Unable to generate query embedding."
            ) from exc

        except ChromaStoreError as exc:
            logger.exception("Vector database search failed.")
            raise RetrievalError(
                "Unable to search historical records."
            ) from exc

        except Exception as exc:
            logger.exception("Unexpected retrieval error.")
            raise RetrievalError(
                "Historical record retrieval failed."
            ) from exc
