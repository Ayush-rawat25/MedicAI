
import json
import logging
from pathlib import Path

from app.schema.noteSchema import ClinicalNote
from app.schema.inputSchema import NoteGenerationInput
from app.services.llmService import llm_service


logger = logging.getLogger(__name__)


# Resolve prompt files independently of the working directory.
BASE_DIR = Path(__file__).resolve().parents[1]
PROMPTS_DIR = BASE_DIR / "prompts"


class NoteGenerationError(Exception):
    """Raised when clinical note generation or validation fails."""


class NoteGenerationService:

    def __init__(self):
        self.system_prompt = self._load_prompt("../prompt/systemPrompt.txt")
        self.clinical_note_prompt = self._load_prompt(
            "../prompt/clinicalPrompt.txt"
        )

    @staticmethod
    def _load_prompt(filename: str) -> str:
        """
        Loads a prompt from the prompts directory.
        """

        prompt_path = PROMPTS_DIR / filename

        try:
            return prompt_path.read_text(encoding="utf-8")

        except FileNotFoundError as exc:
            logger.exception("Prompt file not found: %s", prompt_path)
            raise NoteGenerationError(
                f"Required prompt file not found: {filename}"
            ) from exc

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """
        Converts seconds into MM:SS format.
        """

        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)

        return f"{minutes:02d}:{remaining_seconds:02d}"

    def _format_transcript(self, transcript) -> str:
        """
        Formats the transcript while preserving speaker labels
        and timestamps when diarized segments are available.
        """

        if transcript.segments:
            formatted_segments = []

            for segment in transcript.segments:
                timestamp = self._format_timestamp(segment.start)

                speaker = (
                    segment.speaker
                    if segment.speaker
                    else "Unknown Speaker"
                )

                formatted_segments.append(
                    f"[{timestamp}] {speaker}: {segment.text}"
                )

            return "\n".join(formatted_segments)

        return transcript.text.strip()

    @staticmethod
    def _format_historical_records(context) -> str:
        """
        Formats historical patient records supplied by the caller.

        This service does not retrieve records from a database
        or vector database.
        """

        records = context.historical_records

        if not records:
            return "No historical patient records were provided."

        formatted_records = []

        for record in records:
            formatted_records.append(
                f"""
Record ID: {record.record_id}
Date: {record.date}
Record Type: {record.record_type}
Content:
{record.content}
""".strip()
            )

            if record.relevance_score is not None:
                formatted_records[-1] += (
                    f"\nRelevance Score: {record.relevance_score}"
                )

        return "\n\n".join(formatted_records)

    def _build_user_prompt(
        self,
        request: NoteGenerationInput
    ) -> str:
        """
        Builds the final user prompt using the existing
        clinical note prompt template.
        """

        transcript_text = self._format_transcript(
            request.transcript
        )

        historical_context = self._format_historical_records(
            request.context
        )

        output_schema = json.dumps(
            ClinicalNote.model_json_schema(),
            indent=2,
            ensure_ascii=False
        )

        strategy = request.context.strategy

        prompt_context = f"""
CONSULTATION INFORMATION

Patient ID: {request.patient_id}
Consultation ID: {request.consultation_id}
Context Strategy: {strategy}

CURRENT CONSULTATION TRANSCRIPT

{transcript_text}


HISTORICAL PATIENT CONTEXT

{historical_context}
"""

        return self.clinical_note_prompt.format(
            output_schema=output_schema,
            transcript=prompt_context
        )

    def generate_note(
        self,
        request: NoteGenerationInput
    ) -> ClinicalNote:
        """
        Generates and validates a structured clinical note.

        Flow:
        1. Prepare consultation and historical context.
        2. Build the LLM prompt.
        3. Generate JSON using Ollama.
        4. Parse the JSON response.
        5. Validate it against ClinicalNote.
        """

        try:
            logger.info(
                "Starting note generation | Patient: %s | Consultation: %s",
                request.patient_id,
                request.consultation_id
            )

            user_prompt = self._build_user_prompt(request)

            logger.info("Sending clinical context to LLM")

            generated_response = llm_service.generate(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt
            )

            logger.info("LLM response received. Validating JSON.")

            # Parse and validate the generated JSON.
            generated_note = ClinicalNote.model_validate_json(
                generated_response
            )

            logger.info(
                "Clinical note generated successfully | Patient: %s",
                request.patient_id
            )

            return generated_note

        except Exception as exc:
            logger.exception(
                "Clinical note generation failed | Patient: %s",
                request.patient_id
            )

            raise NoteGenerationError(
                "Failed to generate or validate clinical note."
            ) from exc


# Reusable service instance
note_generation_service = NoteGenerationService()
