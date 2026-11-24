
import logging
import time
from typing import Any

from app.generation.answer_generation import (
    AnswerGenerator,
    AnswerGenerationError
)
from app.services.retrieval_service import RetrievalService


logger = logging.getLogger(__name__)


class RAGServiceError(Exception):
    """Raised when the complete RAG pipeline fails."""
    pass


class RAGService:

    def __init__(
        self,
        retrieval_service: RetrievalService,
        answer_generator: AnswerGenerator
    ):

        self.retrieval_service = retrieval_service
        self.answer_generator = answer_generator

    def answer(self, request: Any) -> dict[str, Any]:
        """
        Retrieve relevant records and generate an answer.
        """

        total_start_time = time.perf_counter()

        try:
            # Step 1: Retrieve patient records
            retrieval_result = self.retrieval_service.retrieve(
                request
            )

            # Support either a dictionary or a Pydantic response.
            if hasattr(retrieval_result, "model_dump"):
                retrieval_result = retrieval_result.model_dump()
            elif hasattr(retrieval_result, "dict"):
                retrieval_result = retrieval_result.dict()

            records = retrieval_result.get(
                "retrieved_records", []
            )

            retrieval_metadata = retrieval_result.get(
                "metadata", {}
            )

            # Step 2: Stop if no records were found.
            if not records:

                return {
                    "status": "no_relevant_records",
                    "patient_id": request.patient_id,
                    "conversation_id": getattr(
                        request, "conversation_id", None
                    ),
                    "query": request.query,
                    "answer": (
                        "No relevant information was found "
                        "in the available patient records."
                    ),
                    "supporting_records": [],
                    "retrieval_metadata": retrieval_metadata,
                    "generation_metadata": {
                        "model": None,
                        "generation_time_seconds": 0.0
                    }
                }

            # Step 3: Generate an answer using retrieved records.
            generation_result = self.answer_generator.generate(
                query=request.query,
                records=records
            )

            supporting_records = list(dict.fromkeys(
                record["record_id"]
                for record in records
                if record.get("record_id")
            ))

            total_time = (
                time.perf_counter() - total_start_time
            )

            logger.info(
                "RAG request completed in %.4f seconds",
                total_time
            )

            return {
                "status": "success",
                "patient_id": request.patient_id,
                "conversation_id": getattr(
                    request, "conversation_id", None
                ),
                "query": request.query,
                "answer": generation_result["answer"],
                "supporting_records": supporting_records,
                "retrieval_metadata": retrieval_metadata,
                "generation_metadata": {
                    "model": generation_result["model"],
                    "generation_time_seconds": (
                        generation_result[
                            "generation_time_seconds"
                        ]
                    )
                }
            }

        except AnswerGenerationError as exc:
            logger.exception("RAG answer generation failed")
            raise RAGServiceError(str(exc)) from exc

        except Exception as exc:
            logger.exception("RAG pipeline failed")
            raise RAGServiceError(
                f"RAG pipeline failed: {exc}"
            ) from exc
