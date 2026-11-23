
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class RetrievalInput(BaseModel):
    """
    Request schema for querying a patient's
    historical medical records.
    """

    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(
        ...,
        min_length=1,
        description="Unique patient identifier."
    )

    query: str = Field(
        ...,
        min_length=1,
        description="Doctor's natural language question."
    )

    conversation_id: str = Field(
        ...,
        min_length=1,
        description="Identifier for the ongoing doctor-patient conversation."
    )

    consultation_id: Optional[str] = Field(
        default=None,
        description="Identifier of the current consultation."
    )

    case_id: Optional[str] = Field(
        default=None,
        description="Research case identifier, used during experiments."
    )
