
import logging
import time

from app.config import settings

from app.schemas.retrieval_input import RetrievalInput
from app.schemas.retrieval_response import (
    RetrievalResponse,
    RetrievedRecord,
    RetrievalMetadata
)

from app.ingestion.embedding_service import EmbeddingService
from app.vectorStore.chroma_store import ChromaStore
from app.retrieval.retriever import Retriever, RetrievalError
from app.retrieval.ranking import RecordRanker, RankingError


logger = logging.getLogger(__name__)


class RetrievalServiceError(Exception):
    """Raised when the retrieval service fails."""
    pass


class RetrievalService:

    def __init__(
        self,
        retriever: Retriever,
        record_ranker: RecordRanker
    ):
        self.retriever = retriever
        self.record_ranker = record_ranker

        self.top_k = settings.RETRIEVAL_TOP_K

    def retrieve(
        self,
        request: RetrievalInput
    ) -> RetrievalResponse:
        """
        Retrieve relevant historical records for a patient.

        Args:
            request: Validated retrieval request.

        Returns:
            Structured retrieval response containing ranked records.
        """

        start_time = time.perf_counter()

        try:
            logger.info(
                "Starting retrieval | Patient: %s | Conversation: %s",
                request.patient_id,
                request.conversation_id
            )

            # Step 1: Retrieve relevant chunks for the patient
            retrieval_result = self.retriever.retrieve(
                patient_id=request.patient_id,
                query=request.query,
                top_k=self.top_k * 3
            )

            retrieved_chunks = retrieval_result["retrieved_chunks"]

            # Step 2: Group chunks into complete historical records
            # and rank records by relevance
            ranked_records = self.record_ranker.rank(
                retrieved_chunks=retrieved_chunks,
                top_k=self.top_k
            )

            # Step 3: Convert ranked records into response schema
            response_records = [
                RetrievedRecord(
                    record_id=record["record_id"],
                    date=record["date"],
                    record_type=record["record_type"],
                    content=record["content"],
                    relevance_score=record["relevance_score"],
                    source=record.get("source")
                )
                for record in ranked_records
            ]

            total_time = time.perf_counter() - start_time

            # Step 4: Construct retrieval metadata
            metadata = RetrievalMetadata(
                retrieved_count=len(response_records),
                top_k=self.top_k,
                retrieval_time_seconds=round(total_time, 4),
                embedding_model=settings.EMBEDDING_MODEL,
                vector_database="ChromaDB"
            )

            # Step 5: Construct final response
            response = RetrievalResponse(
                patient_id=request.patient_id,
                conversation_id=request.conversation_id,
                consultation_id=request.consultation_id,
                case_id=request.case_id,
                status="success",
                retrieved_records=response_records,
                metadata=metadata
            )

            logger.info(
                "Retrieval completed | Patient: %s | Records: %d | Time: %.3fs",
                request.patient_id,
                len(response_records),
                total_time
            )

            return response

        except (RetrievalError, RankingError) as exc:
            logger.exception(
                "Retrieval pipeline failed for patient %s",
                request.patient_id
            )

            raise RetrievalServiceError(
                "Unable to retrieve patient history."
            ) from exc

        except Exception as exc:
            logger.exception(
                "Unexpected retrieval service error for patient %s",
                request.patient_id
            )

            raise RetrievalServiceError(
                "An unexpected error occurred during retrieval."
            ) from exc
