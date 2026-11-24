
import json
import logging
import time
from pathlib import Path
from typing import Any

from app.generation.llm_service import (
    LLMService,
    LLMServiceError
)


logger = logging.getLogger(__name__)


class AnswerGenerationError(Exception):
    """Raised when answer generation fails."""
    pass


class AnswerGenerator:

    def __init__(self, llm_service: LLMService):

        self.llm_service = llm_service

        prompt_path = (
            Path(__file__).parent
            / "prompts"
            / "answer_prompt.txt"
        )

        try:
            self.system_prompt = prompt_path.read_text(
                encoding="utf-8"
            )
        except OSError as exc:
            raise AnswerGenerationError(
                f"Could not load answer prompt: {exc}"
            ) from exc

    def generate(
        self,
        query: str,
        records: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Generate an answer using retrieved patient records.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not records:
            raise ValueError(
                "Cannot generate an answer without records."
            )

        start_time = time.perf_counter()

        try:
            evidence = self._prepare_evidence(records)

            user_prompt = f"""
DOCTOR'S QUESTION:

{query}

RETRIEVED PATIENT RECORDS:

{evidence}

Answer the doctor's question using only the records above.
"""

            answer = self.llm_service.generate(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt
            )

            generation_time = (
                time.perf_counter() - start_time
            )

            return {
                "answer": answer,
                "model": self.llm_service.model,
                "generation_time_seconds": round(
                    generation_time, 4
                )
            }

        except LLMServiceError:
            raise

        except Exception as exc:
            logger.exception("Answer generation failed")

            raise AnswerGenerationError(
                f"Answer generation failed: {exc}"
            ) from exc

    @staticmethod
    def _prepare_evidence(
        records: list[dict[str, Any]]
    ) -> str:
        """
        Convert retrieved records into readable evidence.
        """

        evidence_records = []

        for record in records:

            evidence_records.append({
                "record_id": record.get("record_id"),
                "record_type": record.get("record_type"),
                "date": record.get("date"),
                "content": record.get("content")
            })

        return json.dumps(
            evidence_records,
            ensure_ascii=False,
            indent=2
        )
