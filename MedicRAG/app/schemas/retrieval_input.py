
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class RetrievalContext(BaseModel):
    """
    Optional context supplied by the chat/orchestration service.
    Helps interpret conversational references and retrieval constraints.
    """

    previous_topic: Optional[str] = Field(
        default=None,
        description="Previously discussed topic, if relevant."
    )

    resolved_references: Optional[dict[str, str]] = Field(
        default=None,
        description="Resolved references from conversation, such as 'that medication'."
    )

    record_category: Optional[str] = Field(
        default=None,
        description="Requested category, e.g. consultation, medication, allergy or lab."
    )

    date_from: Optional[str] = Field(
        default=None,
        description="Optional start date for historical record retrieval (YYYY-MM-DD)."
    )

    date_to: Optional[str] = Field(
        default=None,
        description="Optional end date for historical record retrieval (YYYY-MM-DD)."
    )

    record_limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of historical records requested."
    )


class RetrievalRequest(BaseModel):
    """
    Input contract for the conversational RAG retrieval service.

    The query may be conversational, but should be resolved by the
    orchestration service before being submitted.
    """

    patient_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier of the patient whose history is requested."
    )

    conversation_id: Optional[str] = Field(
        default=None,
        description="Identifier of the ongoing doctor-patient conversation."
    )

    consultation_id: Optional[str] = Field(
        default=None,
        description="Identifier of the current consultation, if applicable."
    )

    query: str = Field(
        ...,
        min_length=1,
        description="Conversational retrieval query, resolved using conversation context when necessary."
    )

    retrieval_context: Optional[RetrievalContext] = Field(
        default=None,
        description="Optional structured context and constraints from the orchestrator."
    )

    @field_validator("patient_id", "query")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Field cannot be empty or contain only whitespace.")

        return value
