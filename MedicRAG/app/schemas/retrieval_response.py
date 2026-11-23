
from typing import Optional

from pydantic import BaseModel, Field


class RetrievedRecord(BaseModel):
    """
    Represents one historical patient record
    retrieved from the vector database.
    """

    record_id: str

    date: str

    record_type: str

    content: str

    relevance_score: float

    source: Optional[str] = None


class RetrievalMetadata(BaseModel):
    """
    Metadata for retrieval monitoring and evaluation.
    """

    retrieved_count: int = Field(
        ...,
        ge=0
    )

    top_k: int = Field(
        ...,
        ge=1
    )

    retrieval_time_seconds: float = Field(
        ...,
        ge=0
    )

    embedding_model: str

    vector_database: str = "ChromaDB"


class RetrievalResponse(BaseModel):
    """
    Response returned by the RAG service.
    """

    patient_id: str

    conversation_id: str

    consultation_id: Optional[str] = None

    case_id: Optional[str] = None

    status: str = "success"

    retrieved_records: list[RetrievedRecord] = Field(
        default_factory=list
    )

    metadata: RetrievalMetadata
