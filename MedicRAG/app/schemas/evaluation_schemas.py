from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

EvaluationStrategy = Literal["consultation_only", "full_history", "selective"]
DEFAULT_EVALUATION_STRATEGIES: list[EvaluationStrategy] = [
    "consultation_only", "full_history", "selective"
]


class EvaluationRequest(BaseModel):
    """Inputs shared by all retrieval evaluation strategies."""

    patient_id: str = Field(..., min_length=1)
    consultation_id: str = Field(
        ..., min_length=1,
        description="Current consultation/record ID to exclude from historical retrieval."
    )
    cutoff_timestamp: str = Field(
        ...,
        description="UTC ISO-8601 timestamp. Only records strictly before this time are eligible."
    )
    consultation_text: str = Field(
        ..., min_length=1,
        description="Current consultation transcript. This retrieval-only endpoint does not embed it."
    )
    query: str = Field(..., min_length=1, description="Doctor's follow-up question.")
    relevant_record_ids: list[str] = Field(
        ..., min_length=1,
        description="Query-specific ground-truth historical record IDs."
    )
    strategies: list[EvaluationStrategy] = Field(
        default_factory=lambda: DEFAULT_EVALUATION_STRATEGIES.copy(),
        min_length=1,
        description="Strategies to run. Defaults to all three."
    )

    @field_validator("patient_id", "consultation_id", "query", "consultation_text")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be empty or contain only whitespace.")
        return value

    @field_validator("cutoff_timestamp")
    @classmethod
    def validate_cutoff_timestamp(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("Timezone is required.")
            parsed = parsed.astimezone(timezone.utc)
            normalized = parsed.isoformat(timespec="microseconds")
            if normalized.endswith(".000000+00:00"):
                normalized = normalized.replace(".000000+00:00", "+00:00")
            return normalized.replace("+00:00", "Z")
        except ValueError as exc:
            raise ValueError(
                "cutoff_timestamp must be ISO-8601 with timezone, e.g. 2020-08-28T11:13:21Z."
            ) from exc

    @field_validator("relevant_record_ids")
    @classmethod
    def validate_ground_truth_ids(cls, values: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(v.strip() for v in values if v.strip()))
        if not cleaned:
            raise ValueError("At least one relevant_record_id is required.")
        return cleaned

    @field_validator("strategies")
    @classmethod
    def validate_unique_strategies(cls, values: list[EvaluationStrategy]) -> list[EvaluationStrategy]:
        if len(values) != len(set(values)):
            raise ValueError("Strategies must not contain duplicates.")
        return values


class EvaluationMetrics(BaseModel):
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    mrr: float | None = None
    ndcg: float | None = None


class EvaluationStrategyResult(BaseModel):
    strategy: EvaluationStrategy
    retrieved_record_ids: list[str] = Field(default_factory=list)
    relevant_record_ids_retrieved: list[str] = Field(default_factory=list)
    missed_relevant_record_ids: list[str] = Field(default_factory=list)
    record_count: int = 0
    retrieval_time_seconds: float = 0.0
    retrieval_method: str
    metrics: EvaluationMetrics
    note: str | None = None
    metadata: dict = Field(default_factory=dict)


class EvaluationResponse(BaseModel):
    status: Literal["success"] = "success"
    patient_id: str
    consultation_id: str
    cutoff_timestamp: str
    query: str
    ground_truth_record_count: int
    results: list[EvaluationStrategyResult]
