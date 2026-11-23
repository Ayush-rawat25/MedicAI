
import logging
import time
from typing import Any

from sentence_transformers import SentenceTransformer

from app.config import settings


logger = logging.getLogger(__name__)


class EmbeddingServiceError(Exception):
    """Raised when an embedding operation fails."""


class EmbeddingService:

    def __init__(self):

        self.model_name = settings.EMBEDDING_MODEL
        self.device = settings.EMBEDDING_DEVICE
        self.batch_size = settings.EMBEDDING_BATCH_SIZE

        logger.info(
            "Loading embedding model: %s on %s",
            self.model_name,
            self.device
        )

        start_time = time.perf_counter()

        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )

            logger.info(
                "Embedding model loaded successfully in %.2f seconds",
                time.perf_counter() - start_time
            )

        except Exception as exc:

            logger.exception("Failed to load embedding model")

            raise EmbeddingServiceError(
                "Unable to initialize embedding model."
            ) from exc

    def embed_documents(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Generates embeddings for historical clinical records.

        Uses the passage prefix required by E5 models.
        """

        if not texts:
            return []

        if any(not isinstance(text, str) or not text.strip()
               for text in texts):
            raise EmbeddingServiceError(
                "Document texts must be non-empty strings."
            )

        try:

            start_time = time.perf_counter()

            prefixed_texts = [
                f"passage: {text}"
                for text in texts
            ]

            embeddings = self.model.encode(
                prefixed_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            duration = time.perf_counter() - start_time

            logger.info(
                "Generated %d document embeddings in %.2f seconds",
                len(texts),
                duration
            )

            return embeddings.tolist()

        except Exception as exc:

            logger.exception(
                "Failed to generate document embeddings"
            )

            raise EmbeddingServiceError(
                "Unable to generate document embeddings."
            ) from exc

    def embed_query(self, query: str) -> list[float]:
        """
        Generates an embedding for a doctor's question.

        Uses the query prefix required by E5 models.
        """

        if not isinstance(query, str) or not query.strip():
            raise EmbeddingServiceError(
                "Query must be a non-empty string."
            )

        try:

            start_time = time.perf_counter()

            prefixed_query = f"query: {query}"

            embedding = self.model.encode(
                prefixed_query,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            duration = time.perf_counter() - start_time

            logger.info(
                "Generated query embedding in %.4f seconds",
                duration
            )

            return embedding.tolist()

        except Exception as exc:

            logger.exception(
                "Failed to generate query embedding"
            )

            raise EmbeddingServiceError(
                "Unable to generate query embedding."
            ) from exc

    def embed_chunks(
        self,
        chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Generates embeddings for processed record chunks.

        Preserves chunk IDs, original text and metadata
        for subsequent vector database insertion.
        """

        if not chunks:
            return []

        try:

            texts = []
            validated_chunks = []

            for chunk in chunks:

                if not chunk.get("chunk_id"):
                    raise EmbeddingServiceError(
                        "Chunk ID is missing."
                    )

                if not chunk.get("metadata"):
                    raise EmbeddingServiceError(
                        "Chunk metadata is missing."
                    )

                metadata = chunk["metadata"]

                if not metadata.get("patient_id"):
                    raise EmbeddingServiceError(
                        "Patient ID missing from chunk metadata."
                    )

                if not metadata.get("record_id"):
                    raise EmbeddingServiceError(
                        "Record ID missing from chunk metadata."
                    )

                if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
                    raise EmbeddingServiceError(
                        "Chunk text must be non-empty."
                    )

                texts.append(chunk["text"])
                validated_chunks.append(chunk)

            embeddings = self.embed_documents(texts)

            embedded_chunks = []

            for chunk, embedding in zip(
                validated_chunks,
                embeddings
            ):

                embedded_chunks.append({
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "embedding": embedding
                })

            logger.info(
                "Successfully embedded %d processed chunks",
                len(embedded_chunks)
            )

            return embedded_chunks

        except EmbeddingServiceError:
            raise

        except Exception as exc:

            logger.exception(
                "Failed to embed processed chunks"
            )

            raise EmbeddingServiceError(
                "Unable to embed processed chunks."
            ) from exc
