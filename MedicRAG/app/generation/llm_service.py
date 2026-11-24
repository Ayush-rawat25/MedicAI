
import logging
from typing import Optional

import ollama
from ollama import ResponseError

from app.config import settings


logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """Raised when Ollama generation fails."""
    pass


class LLMService:

    def __init__(self):
        self.client = ollama.Client(
            host=settings.OLLAMA_HOST
        )

        self.model = settings.OLLAMA_MODEL

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1
    ) -> str:
        """
        Generate a response using the configured Ollama model.
        """

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                options={
                    "temperature": temperature
                }
            )

            answer = response.message.content.strip()

            if not answer:
                raise LLMServiceError(
                    "Ollama returned an empty response."
                )

            return answer

        except ResponseError as exc:
            logger.exception("Ollama response error")
            raise LLMServiceError(
                f"Ollama model error: {exc}"
            ) from exc

        except Exception as exc:
            logger.exception("Ollama generation failed")
            raise LLMServiceError(
                f"Failed to generate answer: {exc}"
            ) from exc
