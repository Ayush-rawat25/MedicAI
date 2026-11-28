from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# 1. STRATEGY DEFINITIONS
# ============================================================

EvaluationStrategy = Literal[
    "consultation_only",
    "full_history",
    "selective_retrieval",
]

DEFAULT_EVALUATION_STRATEGIES: list[EvaluationStrategy] = [
    "consultation_only",
    "full_history",
    "selective_retrieval",
]


# ============================================================
# 2. EXISTING RAG RESPONSE SNAPSHOTS
# ============================================================

class RetrievedRecordSnapshot(BaseModel):
    record_id: str
    record_type: str
    date: Optional[str] = None
    content: str
    relevance_score: Optional[float] = None
    source: Optional[str] = None


class RetrievalMetadataSnapshot(BaseModel):
    retrieval_strategy: Optional[str] = None

    retrieval_method: Optional[
        Literal["semantic", "structured", "hybrid"]
    ] = None

    embedding_model: Optional[str] = None
    vector_database: Optional[str] = None

    candidates_retrieved: int = 0
    records_returned: int = 0
    records_filtered: int = 0

    retrieval_time_seconds: float = 0.0


class RetrievalSnapshot(BaseModel):
    status: Literal[
        "success",
        "no_relevant_records",
        "no_records_found",
    ]

    patient_id: str
    conversation_id: Optional[str] = None
    query: str

    retrieved_records: list[RetrievedRecordSnapshot] = Field(
        default_factory=list
    )

    metadata: RetrievalMetadataSnapshot


class GenerationMetadataSnapshot(BaseModel):
    model: Optional[str] = None
    generation_time_seconds: float = 0.0


class GenerationSnapshot(BaseModel):
    status: Literal[
        "success",
        "no_relevant_records",
        "generation_failed",
    ]

    patient_id: str
    conversation_id: Optional[str] = None
    query: str

    # Generated answer may be JSON-formatted SOAP or plain text.
    answer: Optional[str] = None

    supporting_records: list[str] = Field(
        default_factory=list
    )

    retrieval_metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    generation_metadata: GenerationMetadataSnapshot = Field(
        default_factory=GenerationMetadataSnapshot
    )

    @model_validator(mode="after")
    def validate_answer_status(self):
        if self.status == "success" and not self.answer:
            raise ValueError(
                "A successful generation must contain an answer."
            )

        if self.status == "generation_failed" and self.answer:
            raise ValueError(
                "A failed generation cannot contain an answer."
            )

        return self


# ============================================================
# 3. EXACT CONTEXT SUPPLIED TO THE GENERATION MODEL
# ============================================================

class ContextRecord(BaseModel):
    record_id: str
    record_type: Optional[str] = None
    date: Optional[str] = None
    content: str


class ContextSnapshot(BaseModel):
    """
    Stores the exact evidence supplied to the generation model.

    Retrieved records are not automatically considered context.
    A record must actually be included in this snapshot.
    """

    consultation_id: Optional[str] = None
    consultation_text: str

    current_measurements: list[ContextRecord] = Field(
        default_factory=list
    )

    historical_records: list[ContextRecord] = Field(
        default_factory=list
    )

    context_record_ids: list[str] = Field(
        default_factory=list
    )

    context_available: bool = True

    @model_validator(mode="after")
    def validate_context_records(self):
        actual_ids = list(
            dict.fromkeys(
                record.record_id
                for record in (
                    self.current_measurements
                    + self.historical_records
                )
            )
        )

        if not self.context_record_ids:
            self.context_record_ids = actual_ids

        elif not set(actual_ids).issubset(
            set(self.context_record_ids)
        ):
            raise ValueError(
                "context_record_ids must include every record "
                "present in the context snapshot."
            )

        if not self.context_available and (
            self.consultation_text
            or self.current_measurements
            or self.historical_records
        ):
            raise ValueError(
                "Unavailable context cannot contain context data."
            )

        return self


# ============================================================
# 4. REFERENCE DATASET SCHEMAS
# ============================================================

FactCategory = Literal[
    "condition",
    "allergy",
    "measurement",
    "medication",
    "consultation",
    "negative_constraint",
    "other",
]

FactPriority = Literal[
    "critical",
    "high",
    "medium",
    "low",
]

VALID_FACT_PRIORITIES: set[str] = {
    "critical",
    "high",
    "medium",
    "low",
}


# Keys used in CASE_001.json strategy_context_availability.
DATASET_STRATEGY_MAPPING = {
    "A_consultation_only": "consultation_only",
    "B_full_history": "full_history",
    "C_selective_retrieval": "selective_retrieval",
}


class ReferenceFact(BaseModel):
    fact_id: str
    description: str
    category: FactCategory
    priority: FactPriority = "medium"

    source_record_ids: list[str] = Field(
        default_factory=list
    )

    available_in: list[EvaluationStrategy] = Field(
        default_factory=list
    )

    is_negative_constraint: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    # Original reference dataset annotations.
    reference_statement: Optional[str] = None
    importance: Optional[str] = None
    scoring_role: Optional[str] = None
    expected_representation: Optional[str] = None

    strategy_context_availability: dict[str, Any] = Field(
        default_factory=dict
    )

    source_type: Optional[str] = None
    source_dates: list[str] = Field(
        default_factory=list
    )
    temporal_status: Optional[str] = None
    source_verification_status: Optional[str] = None
    clinical_review_status: Optional[str] = None
    annotation_notes: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_reference_fact(cls, data):
        if not isinstance(data, dict):
            return data

        data = dict(data)

        # ----------------------------------------------------
        # 1. Map importance to priority safely.
        # ----------------------------------------------------
        if "priority" not in data or data["priority"] is None:
            raw_importance = data.get("importance")
            if isinstance(raw_importance, str):
                normalized = raw_importance.strip().lower()
                if normalized in VALID_FACT_PRIORITIES:
                    data["priority"] = normalized
                elif normalized == "urgent":
                    data["priority"] = "critical"
                else:
                    data["priority"] = "medium"
            else:
                data["priority"] = "medium"

        # ----------------------------------------------------
        # 2. Map dataset strategy availability.
        # ----------------------------------------------------
        strategy_availability = data.get(
            "strategy_context_availability", {}
        )

        if "available_in" not in data:
            available_strategies = []

            for dataset_key, strategy_name in (
                DATASET_STRATEGY_MAPPING.items()
            ):
                availability = strategy_availability.get(
                    dataset_key,
                    False,
                )

                # "only_if_retrieved" means the strategy
                # may access the fact only if its source
                # record is actually in the generation context.
                if (
                    availability is True
                    or availability == "only_if_retrieved"
                ):
                    available_strategies.append(strategy_name)

            data["available_in"] = available_strategies

        # ----------------------------------------------------
        # 3. Identify negative constraints.
        # ----------------------------------------------------
        if "is_negative_constraint" not in data:
            data["is_negative_constraint"] = (
                data.get("scoring_role") == "negative_constraint"
                or data.get("category") == "negative_constraint"
            )

        # ----------------------------------------------------
        # 4. Use reference_statement as description if needed.
        # ----------------------------------------------------
        if (
            not data.get("description")
            and data.get("reference_statement")
        ):
            data["description"] = data["reference_statement"]

        return data

    @field_validator("source_record_ids", "available_in")
    @classmethod
    def validate_unique_values(cls, values):
        if len(values) != len(set(values)):
            raise ValueError(
                "Duplicate values are not allowed."
            )

        return values

    @model_validator(mode="after")
    def validate_strategy_availability(self):
        valid_strategies = set(
            DEFAULT_EVALUATION_STRATEGIES
        )

        invalid_strategies = (
            set(self.available_in) - valid_strategies
        )

        if invalid_strategies:
            raise ValueError(
                f"Invalid evaluation strategies: "
                f"{invalid_strategies}"
            )

        return self


class EvaluationReferenceDataset(BaseModel):
    schema_version: str = "1.0"

    case_id: str
    patient_id: str

    source_dataset: Optional[str] = None
    index_encounter_id: str
    knowledge_cutoff: str

    purpose: Optional[str] = None
    annotation_status: Optional[str] = None
    important_status_note: Optional[str] = None

    reference_facts: list[ReferenceFact]

    scoring_protocol: dict[str, Any] = Field(
        default_factory=dict
    )

    strategy_context_definitions: dict[str, Any] = Field(
        default_factory=dict
    )

    review_checklist: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @field_validator("knowledge_cutoff")
    @classmethod
    def validate_knowledge_cutoff(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )

            if parsed.tzinfo is None:
                raise ValueError(
                    "Timezone is required."
                )

            return parsed.astimezone(
                timezone.utc
            ).isoformat()

        except ValueError as exc:
            raise ValueError(
                "knowledge_cutoff must be a timezone-aware "
                "ISO-8601 timestamp."
            ) from exc

    @model_validator(mode="after")
    def validate_reference_facts(self):
        fact_ids = [
            fact.fact_id
            for fact in self.reference_facts
        ]

        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError(
                "Reference dataset contains duplicate fact IDs."
            )

        return self


# ============================================================
# 5. EXPERIMENT INPUT
# ============================================================

class StrategyExperimentInput(BaseModel):
    strategy: EvaluationStrategy

    retrieval: RetrievalSnapshot
    generation: GenerationSnapshot

    context: ContextSnapshot

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_strategy_outputs(self):
        if self.retrieval.query != self.generation.query:
            raise ValueError(
                "Retrieval and generation queries must match."
            )

        if (
            self.retrieval.conversation_id
            and self.generation.conversation_id
            and self.retrieval.conversation_id
            != self.generation.conversation_id
        ):
            raise ValueError(
                "Retrieval and generation conversation IDs differ."
            )

        if (
            self.context.consultation_id
            and self.generation.conversation_id
            and self.context.consultation_id
            != self.generation.conversation_id
        ):
            raise ValueError(
                "Context consultation ID and generation "
                "conversation ID differ."
            )

        return self


class ExperimentEvaluationRequest(BaseModel):
    """
    One evaluation case containing the outputs of all strategies.

    Reference facts and relevant record IDs are evaluator-only.
    They must never be passed to the original generation model.
    """

    case_id: str
    patient_id: str
    consultation_id: str

    cutoff_timestamp: str

    query: str

    relevant_record_ids: list[str] = Field(
        default_factory=list
    )

    reference_dataset: EvaluationReferenceDataset

    strategy_runs: list[StrategyExperimentInput] = Field(
        min_length=1
    )

    @field_validator("cutoff_timestamp")
    @classmethod
    def validate_cutoff_timestamp(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )

            if parsed.tzinfo is None:
                raise ValueError(
                    "Timezone is required."
                )

            return parsed.astimezone(
                timezone.utc
            ).isoformat()

        except ValueError as exc:
            raise ValueError(
                "cutoff_timestamp must be an ISO-8601 "
                "timezone-aware timestamp."
            ) from exc

    @field_validator("relevant_record_ids")
    @classmethod
    def validate_relevant_record_ids(cls, values):
        if len(values) != len(set(values)):
            raise ValueError(
                "Duplicate relevant record IDs."
            )

        return values

    @model_validator(mode="after")
    def validate_case_consistency(self):

        if self.reference_dataset.case_id != self.case_id:
            raise ValueError(
                "Reference dataset case_id mismatch."
            )

        if self.reference_dataset.patient_id != self.patient_id:
            raise ValueError(
                "Reference dataset patient_id mismatch."
            )

        if (
            self.reference_dataset.index_encounter_id
            != self.consultation_id
        ):
            raise ValueError(
                "Reference dataset encounter ID mismatch."
            )

        strategies = [
            run.strategy for run in self.strategy_runs
        ]

        if len(strategies) != len(set(strategies)):
            raise ValueError(
                "Each evaluation strategy can appear only once."
            )

        for run in self.strategy_runs:

            if run.retrieval.patient_id != self.patient_id:
                raise ValueError(
                    f"Retrieval patient mismatch: {run.strategy}"
                )

            if run.generation.patient_id != self.patient_id:
                raise ValueError(
                    f"Generation patient mismatch: {run.strategy}"
                )

            if run.retrieval.query != self.query:
                raise ValueError(
                    f"Retrieval query mismatch: {run.strategy}"
                )

            if run.generation.query != self.query:
                raise ValueError(
                    f"Generation query mismatch: {run.strategy}"
                )

            if (
                run.context.consultation_id
                and run.context.consultation_id
                != self.consultation_id
            ):
                raise ValueError(
                    f"Context consultation mismatch: {run.strategy}"
                )

        return self


# ============================================================
# 6. ANSWER EVALUATION
# ============================================================

FactAssessmentStatus = Literal[
    "supported",
    "partially_supported",
    "missing",
    "contradicted",
    "not_applicable",
]


class FactAssessment(BaseModel):
    fact_id: str

    status: FactAssessmentStatus

    context_available: bool

    context_grounded: Optional[bool] = None

    answer_evidence: Optional[str] = None
    context_evidence: Optional[str] = None

    explanation: Optional[str] = None

    @model_validator(mode="after")
    def validate_assessment(self):
        if self.status == "not_applicable":
            if self.context_grounded is not None:
                raise ValueError(
                    "not_applicable facts cannot have "
                    "a context_grounded assessment."
                )

        if (
            self.context_grounded is True
            and not self.context_available
        ):
            raise ValueError(
                "A fact cannot be context-grounded when "
                "its context was unavailable."
            )

        return self


class UnsupportedClaim(BaseModel):
    claim: str

    severity: Literal[
        "critical",
        "high",
        "medium",
        "low",
    ] = "medium"

    claim_type: Literal[
        "factually_incorrect",
        "context_unsupported",
        "contradictory",
        "invented",
    ]

    explanation: Optional[str] = None


class AnswerEvaluationResult(BaseModel):
    fact_assessments: list[FactAssessment] = Field(
        default_factory=list
    )

    unsupported_claims: list[UnsupportedClaim] = Field(
        default_factory=list
    )

    total_answer_claims: int = 0

    applicable_fact_count: int = 0
    supported_count: int = 0
    partially_supported_count: int = 0
    missing_count: int = 0
    contradicted_count: int = 0
    not_applicable_count: int = 0

    coverage_score: Optional[float] = None

    fully_supported_percentage: Optional[float] = None

    unsupported_claim_rate: Optional[float] = None

    critical_fact_errors: int = 0

    context_grounding_score: Optional[float] = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_answer_metrics(self):

        if self.total_answer_claims < len(self.unsupported_claims):
            raise ValueError(
                "Total answer claims cannot be fewer than "
                "unsupported claims."
            )

        if self.applicable_fact_count != (
            self.supported_count
            + self.partially_supported_count
            + self.missing_count
            + self.contradicted_count
        ):
            raise ValueError(
                "Applicable fact count does not match "
                "the individual assessment counts."
            )

        if self.not_applicable_count < 0:
            raise ValueError(
                "not_applicable_count cannot be negative."
            )

        return self


# ============================================================
# 7. RETRIEVAL EVALUATION
# ============================================================

class RetrievalEvaluationMetrics(BaseModel):
    # Undefined metrics should remain None, not zero.
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None

    mrr: Optional[float] = None
    ndcg: Optional[float] = None


class RetrievalEvaluationResult(BaseModel):
    retrieved_record_ids: list[str] = Field(
        default_factory=list
    )

    relevant_record_ids: list[str] = Field(
        default_factory=list
    )

    relevant_record_ids_retrieved: list[str] = Field(
        default_factory=list
    )

    missed_relevant_record_ids: list[str] = Field(
        default_factory=list
    )

    record_count: int = 0

    retrieval_time_seconds: float = 0.0

    retrieval_method: Optional[str] = None

    metrics: RetrievalEvaluationMetrics = Field(
        default_factory=RetrievalEvaluationMetrics
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================
# 8. STRATEGY EVALUATION OUTPUT
# ============================================================

class StrategyEvaluationResult(BaseModel):
    strategy: EvaluationStrategy

    retrieval_evaluation: RetrievalEvaluationResult

    answer_evaluation: AnswerEvaluationResult

    generation_status: Literal[
        "success",
        "no_relevant_records",
        "generation_failed",
    ]

    generation_time_seconds: float = 0.0

    total_time_seconds: Optional[float] = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================
# 9. FINAL CASE EVALUATION RESPONSE
# ============================================================

class ExperimentEvaluationResponse(BaseModel):
    status: Literal[
        "success",
        "partial_success",
        "failed",
    ]

    case_id: str
    patient_id: str
    consultation_id: str

    query: str

    evaluated_strategy_count: int

    results: list[StrategyEvaluationResult] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_result_count(self):
        if self.evaluated_strategy_count != len(self.results):
            raise ValueError(
                "evaluated_strategy_count must match the number of results."
            )

        return self