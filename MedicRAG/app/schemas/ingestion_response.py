
from pydantic import BaseModel, Field


class IngestionResponse(BaseModel):
    status: str = "success"
    records_received: int = Field(ge=0)
    records_processed: int = Field(ge=0)
    chunks_created: int = Field(ge=0)
    embedding_model: str
    vector_database: str = "ChromaDB"
