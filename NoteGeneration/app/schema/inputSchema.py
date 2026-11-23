
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------
# 1. Context Strategy
# --------------------------------------------------

class ContextStrategy(str, Enum):
    CONSULTATION_ONLY = "consultation_only"
    FULL_HISTORY = "full_history"
    SELECTIVE_RETRIEVAL = "selective_retrieval"


# --------------------------------------------------
# 2. Transcript Segment
# --------------------------------------------------

class TranscriptSegment(BaseModel):
    speaker: Optional[str] = Field(
        default=None,
        description="Speaker identifier, e.g. SPEAKER_00 or Doctor"
    )

    start: float = Field(
        ge=0,
        description="Segment start time in seconds"
    )

    end: float = Field(
        ge=0,
        description="Segment end time in seconds"
    )

    text: str = Field(
        min_length=1,
        description="Spoken dialogue"
    )

    @model_validator(mode="after")
    def validate_timestamps(self):
        if self.end < self.start:
            raise ValueError("Segment end cannot be before start")
        return self


# --------------------------------------------------
# 3. Transcript
# --------------------------------------------------

class TranscriptInput(BaseModel):
    language: Optional[str] = None

    text: Optional[str] = None

    segments: list[TranscriptSegment] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_transcript(self):
        if not self.text and not self.segments:
            raise ValueError(
                "Transcript must contain text or at least one segment"
            )
        return self


# --------------------------------------------------
# 4. Historical Patient Record
# --------------------------------------------------

class HistoricalRecord(BaseModel):
    record_id: str = Field(
        min_length=1
    )

    date: date

    record_type: str = Field(
        min_length=1,
        description="Diagnosis, medication, consultation, etc."
    )

    content: str = Field(
        min_length=1,
        description="Actual historical record content"
    )

    relevance_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Retrieval relevance score, when available"
    )


# --------------------------------------------------
# 5. Patient Context
# --------------------------------------------------

class PatientContext(BaseModel):
    strategy: ContextStrategy = ContextStrategy.CONSULTATION_ONLY

    historical_records: list[HistoricalRecord] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_context(self):

        if (
            self.strategy == ContextStrategy.CONSULTATION_ONLY
            and self.historical_records
        ):
            raise ValueError(
                "consultation_only strategy cannot contain historical records"
            )

        return self


# --------------------------------------------------
# 6. Main Note Generation Input
# --------------------------------------------------

class NoteGenerationInput(BaseModel):

    case_id: Optional[str] = None

    patient_id: str = Field(
        min_length=1
    )

    consultation_id: str = Field(
        min_length=1
    )

    transcript: TranscriptInput

    context: PatientContext = Field(
        default_factory=PatientContext
    )
