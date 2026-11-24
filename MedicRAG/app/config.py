
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # Application
    APP_NAME: str = "MedScribe AI - Conversational RAG Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Embedding Configuration
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-base"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 16

    # ChromaDB Configuration
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma"
    CHROMA_COLLECTION_NAME: str = "patient_records"
    CHROMA_DISTANCE_METRIC: str = "cosine"

    # Retrieval Configuration
    RETRIEVAL_CANDIDATE_LIMIT: int = 50
    SIMILARITY_THRESHOLD: float = 0.82

    # Document Processing
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Ollama Configuration
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
