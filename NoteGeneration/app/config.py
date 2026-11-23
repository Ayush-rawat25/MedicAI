
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # Application
    APP_NAME: str = "MedScribe AI - Note Generation Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Ollama Configuration
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"

    # LLM Generation Configuration
    LLM_TEMPERATURE: float = 0.1
    LLM_NUM_CTX: int = 8192

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
