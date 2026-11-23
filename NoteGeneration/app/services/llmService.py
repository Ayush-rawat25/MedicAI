
import ollama
import logging

from app.config import settings


logger = logging.getLogger(__name__)


class LLMService:

    def __init__(self):
        self.client = ollama.Client(
            host=settings.OLLAMA_HOST
        )

        self.model = settings.OLLAMA_MODEL

    def generate(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """
        Sends system and user prompts to Ollama
        and returns the generated response.
        """

        try:
            logger.info(
                "Sending request to Ollama using model: %s",
                self.model
            )

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
                format="json",
                options={
                    "temperature": 0.1,
                    "num_ctx": 8192
                }
            )

            generated_text = response["message"]["content"]

            if not generated_text:
                raise ValueError("Ollama returned an empty response")

            logger.info("LLM response received successfully")

            return generated_text.strip()

        except Exception:
            logger.exception("Ollama generation failed")
            raise


# Create a reusable service instance
llm_service = LLMService()
