import logging
from typing import Any, Optional

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
        temperature: float = 0.1,
        *,
        json_mode: bool = False,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        """
        Generate a response using the configured Ollama model.
        """
        options: dict[str, Any] = {
            "temperature": temperature,
        }

        if num_ctx is not None:
            options["num_ctx"] = num_ctx

        if num_predict is not None:
            options["num_predict"] = num_predict

        chat_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "options": options,
        }

        if json_mode:
            chat_kwargs["format"] = "json"

        try:
            response = self.client.chat(**chat_kwargs)

            prompt_eval_sec = (
                (getattr(response, "prompt_eval_duration", 0) or 0) / 1e9
            )
            eval_sec = (
                (getattr(response, "eval_duration", 0) or 0) / 1e9
            )
            total_sec = (
                (getattr(response, "total_duration", 0) or 0) / 1e9
            )

            logger.info(
                "[OLLAMA] model=%s json_mode=%s prompt_eval=%.2fs "
                "eval=%.2fs total=%.2fs",
                self.model,
                json_mode,
                prompt_eval_sec,
                eval_sec,
                total_sec,
            )

            content = (
                response.message.content
                if getattr(response, "message", None)
                else ""
            )
            answer = (content or "").strip()

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