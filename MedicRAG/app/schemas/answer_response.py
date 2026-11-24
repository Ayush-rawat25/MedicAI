
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class GenerationMetadata(BaseModel):

    model: Optional[str] = None
    generation_time_seconds: float = 0.0


class RAGResponse(BaseModel):

    status: Literal[
        "success",
        "no_relevant_records",
        "generation_failed"
    ]

    patient_id: str
    conversation_id: Optional[str] = None
    query: str

    answer: Optional[str] = None

    supporting_records: list[str] = Field(
        default_factory=list
    )

    retrieval_metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    generation_metadata: GenerationMetadata = Field(
        default_factory=GenerationMetadata
    )
