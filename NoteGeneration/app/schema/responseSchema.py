
from typing import Optional

from pydantic import BaseModel, Field

from app.schema.noteSchema import ClinicalNote
from app.schema.inputSchema import ContextStrategy


class GenerationMetadata(BaseModel):
    """
    Metadata about the note-generation execution.
    """

    model: str

    generation_time_seconds: float = Field(
        ge=0,
        description="Time taken by the LLM to generate the note."
    )

    input_tokens: Optional[int] = Field(
        default=None,
        ge=0
    )

    output_tokens: Optional[int] = Field(
        default=None,
        ge=0
    )


class NoteGenerationResponse(BaseModel):
    """
    Standard API response for clinical note generation.
    """

    case_id: Optional[str] = None

    patient_id: str

    consultation_id: str

    strategy: ContextStrategy

    status: str = "success"

    note: ClinicalNote

    metadata: GenerationMetadata
