
from typing import Optional, Literal
from pydantic import BaseModel, Field


class RetrievedRecord(BaseModel):
    """Represents one retrieved historical patient record."""

    record_id: str
    record_type: str
    date: str

    content: str = Field(
        ...,
        description="Retrieved clinical information from the historical record."
    )

    relevance_score: Optional[float] = Field(
        default=None,
        description="Similarity or reranking score. None for chronological retrieval."
    )

    source: Optional[str] = None


class RetrievalMetadata(BaseModel):
    """Execution details of the retrieval operation."""

    retrieval_strategy: Literal[
        "patient_overview",
        "category_specific",
        "chronological"
    ]

    retrieval_method: Literal[
        "semantic",
        "structured",
        "hybrid"
    ]

    embedding_model: Optional[str] = None

    vector_database: Optional[str] = None

    candidates_retrieved: int = 0

    records_returned: int = 0

    records_filtered: int = 0

    retrieval_time_seconds: float = 0.0


class RetrievalResponse(BaseModel):
    """Standard response returned by the conversational RAG service."""

    status: Literal[
        "success",
        "no_relevant_records",
        "no_records_found"
    ]

    patient_id: str

    conversation_id: Optional[str] = None

    query: str

    retrieved_records: list[RetrievedRecord] = Field(
        default_factory=list
    )

    metadata: RetrievalMetadata
