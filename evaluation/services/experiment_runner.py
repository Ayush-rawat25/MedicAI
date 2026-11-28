import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from MedicRAG.app.retrieval.structured_retriver import StructuredRetriever
from MedicRAG.app.services.retrieval_service import RetrievalService
from MedicRAG.app.generation.answer_generation import AnswerGenerator

from evaluation.schemas.evaluation_schemas import (
    EvaluationReferenceDataset,
    ExperimentEvaluationRequest,
    StrategyExperimentInput,
    RetrievalSnapshot,
    GenerationSnapshot,
    ContextSnapshot,
    ContextRecord,
)

from evaluation.services.evaluation_service import EvaluationService


logger = logging.getLogger(__name__)

# Project/
# ├── MedicRAG/
# └── evaluation/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ExperimentRunnerError(Exception):
    """Raised when experiment execution fails."""


class ExperimentRunner:

    STRATEGIES = [
        "consultation_only",
        "full_history",
        "selective_retrieval",
    ]

    def __init__(
        self,
        structured_retriever: StructuredRetriever,
        retrieval_service: RetrievalService,
        answer_generator: AnswerGenerator,
        evaluation_service: EvaluationService,
        results_dir: str | Path = "evaluation/results/individual",
    ):
        self.structured_retriever = structured_retriever
        self.retrieval_service = retrieval_service
        self.answer_generator = answer_generator
        self.evaluation_service = evaluation_service

        self.results_dir = self._resolve_path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Experiment results directory: %s",
            self.results_dir,
        )

    # ========================================================
    # PATH RESOLUTION
    # ========================================================

    @staticmethod
    def _resolve_path(path: str | Path) -> Path:
        resolved_path = Path(path)

        if not resolved_path.is_absolute():
            resolved_path = PROJECT_ROOT / resolved_path

        return resolved_path.resolve()

    # ========================================================
    # RECORD COMPACTION
    # ========================================================

    @staticmethod
    def _compact_historical_record(record: dict[str, Any]) -> dict[str, Any]:
        content = str(record.get("content") or "").strip()
        if len(content) > 800:
            content = content[:797].rstrip() + "..."

        compacted: dict[str, Any] = {
            "record_id": str(record.get("record_id", "")),
            "record_type": record.get("record_type"),
            "date": record.get("date"),
            "content": content,
        }

        if record.get("source") is not None:
            compacted["source"] = record["source"]

        if record.get("relevance_score") is not None:
            compacted["relevance_score"] = record["relevance_score"]

        return compacted

    # ========================================================
    # MAIN EXPERIMENT ENTRY POINT
    # ========================================================

    def run_experiment(
        self,
        *,
        reference_file: str | Path,
        patient_id: str,
        consultation_id: str,
        query: str,
        consultation_text: str,
        current_measurements: list[dict[str, Any]],
        relevant_record_ids: list[str] | None = None,
        retrieval_gold_file: str | Path | None = None,
        retrieval_context: Any = None,
        exclude_record_ids: list[str] | None = None,
    ):

        try:
            # Load answer reference dataset.
            reference_dataset = self._load_reference(reference_file)

            # Load retrieval ground-truth dataset.
            retrieval_gold_path = self._get_retrieval_gold_path(
                reference_file=reference_file,
                retrieval_gold_file=retrieval_gold_file,
            )

            retrieval_gold = self._load_retrieval_gold(
                retrieval_gold_path
            )

            gold_record_ids = self._extract_relevant_record_ids(
                retrieval_gold
            )

            # Validate experiment inputs and dataset consistency.
            self._validate_inputs(
                reference_dataset=reference_dataset,
                retrieval_gold=retrieval_gold,
                patient_id=patient_id,
                consultation_id=consultation_id,
                query=query,
                consultation_text=consultation_text,
                relevant_record_ids=gold_record_ids,
            )

            # Retrieval-gold file is the source of truth.
            if (
                relevant_record_ids is not None
                and set(relevant_record_ids) != set(gold_record_ids)
            ):
                logger.warning(
                    "Manually supplied relevant_record_ids differ from "
                    "the retrieval-gold file. Using retrieval-gold IDs."
                )

            relevant_record_ids = gold_record_ids

            # Normalize knowledge cutoff into a timezone-aware UTC datetime.
            raw_cutoff = reference_dataset.knowledge_cutoff
            if isinstance(raw_cutoff, str):
                cutoff = datetime.fromisoformat(
                    raw_cutoff.replace("Z", "+00:00")
                )
            elif isinstance(raw_cutoff, datetime):
                cutoff = raw_cutoff
            else:
                raise ExperimentRunnerError(
                    "Answer reference knowledge_cutoff must be an ISO timestamp."
                )

            if cutoff.tzinfo is None:
                raise ExperimentRunnerError(
                    "Answer reference knowledge_cutoff must include timezone information."
                )

            cutoff = cutoff.astimezone(timezone.utc)

            # ISO string used by the existing retrieval methods.
            cutoff_for_retrieval = cutoff.isoformat()

            excluded_ids = list(
                dict.fromkeys(
                    (exclude_record_ids or []) + [consultation_id]
                )
            )

            logger.info(
                "Starting experiment: case=%s patient=%s cutoff=%s",
                reference_dataset.case_id,
                patient_id,
                cutoff_for_retrieval,
            )

            logger.info(
                "Retrieval gold loaded: %s relevant records",
                len(relevant_record_ids),
            )

            logger.info(
                "Retrieval gold file: %s",
                retrieval_gold_path,
            )

            # Run all three strategies.
            strategy_runs = []

            for strategy in self.STRATEGIES:
                logger.info(
                    "Running strategy: %s",
                    strategy,
                )

                strategy_run = self._run_strategy(
                    strategy=strategy,
                    patient_id=patient_id,
                    consultation_id=consultation_id,
                    query=query,
                    consultation_text=consultation_text,
                    current_measurements=current_measurements,
                    cutoff=cutoff_for_retrieval,
                    exclude_record_ids=excluded_ids,
                    retrieval_context=retrieval_context,
                )

                strategy_runs.append(strategy_run)

            # Build evaluation request.
            evaluation_request = ExperimentEvaluationRequest(
                case_id=reference_dataset.case_id,
                patient_id=patient_id,
                consultation_id=consultation_id,
                cutoff_timestamp=cutoff_for_retrieval,
                query=query,
                relevant_record_ids=relevant_record_ids,
                reference_dataset=reference_dataset,
                strategy_runs=strategy_runs,
            )

            logger.info(
                "All strategy runs completed. Starting evaluation."
            )

            evaluation_response = self.evaluation_service.run(
                evaluation_request
            )

            # Save evaluation results.
            output_path = self._save_results(
                case_id=reference_dataset.case_id,
                response=evaluation_response,
            )

            logger.info(
                "Experiment completed. Results saved to %s",
                output_path,
            )

            return evaluation_response

        except Exception as exc:
            logger.exception("Experiment execution failed.")

            if isinstance(exc, ExperimentRunnerError):
                raise

            raise ExperimentRunnerError(
                f"Experiment execution failed: {exc}"
            ) from exc

    # ========================================================
    # RETRIEVAL GOLD FILE PATH
    # ========================================================

    @staticmethod
    def _get_retrieval_gold_path(
        *,
        reference_file: str | Path,
        retrieval_gold_file: str | Path | None = None,
    ) -> Path:

        if retrieval_gold_file is not None:
            return ExperimentRunner._resolve_path(
                retrieval_gold_file
            )

        reference_path = ExperimentRunner._resolve_path(
            reference_file
        )

        gold_filename = (
            f"{reference_path.stem}_retrieval_gold.json"
        )

        return reference_path.with_name(gold_filename)

    # ========================================================
    # LOAD RETRIEVAL GOLD
    # ========================================================

    @staticmethod
    def _load_retrieval_gold(
        retrieval_gold_file: str | Path,
    ) -> dict[str, Any]:

        path = (
            retrieval_gold_file
            if isinstance(retrieval_gold_file, Path)
            else ExperimentRunner._resolve_path(retrieval_gold_file)
        )

        if not path.exists():
            raise ExperimentRunnerError(
                f"Retrieval gold file not found: {path}"
            )

        if not path.is_file():
            raise ExperimentRunnerError(
                f"Retrieval gold path is not a file: {path}"
            )

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if not isinstance(data, dict):
                raise ExperimentRunnerError(
                    "Retrieval gold file must contain a JSON object."
                )

            if not isinstance(data.get("gold_records"), list):
                raise ExperimentRunnerError(
                    "Retrieval gold file must contain a gold_records list."
                )

            logger.info(
                "Retrieval gold dataset loaded successfully: %s",
                path,
            )

            return data

        except json.JSONDecodeError as exc:
            raise ExperimentRunnerError(
                f"Invalid JSON in retrieval gold file: {exc}"
            ) from exc

    # ========================================================
    # EXTRACT RELEVANT RECORD IDS
    # ========================================================

    @staticmethod
    def _extract_relevant_record_ids(
        retrieval_gold: dict[str, Any],
    ) -> list[str]:

        relevant_ids = []

        for record in retrieval_gold["gold_records"]:
            record_id = record.get("record_id")

            if not record_id:
                raise ExperimentRunnerError(
                    "A gold record is missing its record_id."
                )

            binary_relevant = record.get(
                "binary_relevant_for_primary_precision_recall"
            )

            if binary_relevant is None:
                relevance_grade = record.get("relevance_grade")
                binary_relevant = relevance_grade in (1, 2)

            if binary_relevant:
                relevant_ids.append(str(record_id))

        relevant_ids = list(dict.fromkeys(relevant_ids))

        if not relevant_ids:
            raise ExperimentRunnerError(
                "Retrieval gold file contains no relevant record IDs."
            )

        declared_count = retrieval_gold.get(
            "retrieval_scoring", {}
        ).get("relevant_record_count")

        if (
            declared_count is not None
            and declared_count != len(relevant_ids)
        ):
            raise ExperimentRunnerError(
                "Retrieval gold count mismatch: "
                f"declared {declared_count}, "
                f"but found {len(relevant_ids)} relevant unique IDs."
            )

        logger.info(
            "Extracted %s unique relevant record IDs from retrieval gold.",
            len(relevant_ids),
        )

        return relevant_ids

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    @staticmethod
    def _validate_inputs(
        *,
        reference_dataset: EvaluationReferenceDataset,
        retrieval_gold: dict[str, Any],
        patient_id: str,
        consultation_id: str,
        query: str,
        consultation_text: str,
        relevant_record_ids: list[str],
    ):

        # Validate patient ID.
        if reference_dataset.patient_id != patient_id:
            raise ExperimentRunnerError(
                "Patient ID does not match answer reference dataset."
            )

        if retrieval_gold.get("patient_id") != patient_id:
            raise ExperimentRunnerError(
                "Patient ID does not match retrieval gold dataset."
            )

        # Validate encounter ID.
        if reference_dataset.index_encounter_id != consultation_id:
            raise ExperimentRunnerError(
                "Consultation ID does not match answer reference dataset."
            )

        if retrieval_gold.get("index_encounter_id") != consultation_id:
            raise ExperimentRunnerError(
                "Consultation ID does not match retrieval gold dataset."
            )

        # Validate case ID.
        if retrieval_gold.get("case_id") != reference_dataset.case_id:
            raise ExperimentRunnerError(
                "Case ID does not match retrieval gold dataset."
            )

        # Validate knowledge cutoff.
        gold_cutoff_value = retrieval_gold.get("knowledge_cutoff")

        if not gold_cutoff_value:
            raise ExperimentRunnerError(
                "Retrieval gold file is missing knowledge_cutoff."
            )

        def normalize_cutoff(
            value: Any,
            source: str,
        ) -> datetime:

            try:
                if isinstance(value, str):
                    value = datetime.fromisoformat(
                        value.replace("Z", "+00:00")
                    )

                if not isinstance(value, datetime):
                    raise ValueError(
                        "Expected a datetime or ISO timestamp."
                    )

                if value.tzinfo is None:
                    raise ValueError(
                        "Timezone information is missing."
                    )

                return value.astimezone(timezone.utc)

            except (ValueError, TypeError, AttributeError) as exc:
                raise ExperimentRunnerError(
                    f"Invalid knowledge_cutoff in {source}: {value!r}"
                ) from exc

        reference_cutoff = normalize_cutoff(
            reference_dataset.knowledge_cutoff,
            "answer reference dataset",
        )

        gold_cutoff = normalize_cutoff(
            gold_cutoff_value,
            "retrieval gold dataset",
        )

        logger.info(
            "Answer reference cutoff: %s",
            reference_cutoff.isoformat(),
        )

        logger.info(
            "Retrieval gold cutoff: %s",
            gold_cutoff.isoformat(),
        )

        if reference_cutoff != gold_cutoff:
            raise ExperimentRunnerError(
                "Knowledge cutoff differs between answer reference "
                "and retrieval gold datasets. "
                f"Answer reference: {reference_cutoff.isoformat()}, "
                f"Retrieval gold: {gold_cutoff.isoformat()}"
            )

        # Validate query.
        if not query or not query.strip():
            raise ExperimentRunnerError(
                "Query cannot be empty."
            )

        # Validate consultation.
        if not consultation_text or not consultation_text.strip():
            raise ExperimentRunnerError(
                "Consultation text cannot be empty."
            )

        # Validate relevant record IDs.
        if not relevant_record_ids:
            raise ExperimentRunnerError(
                "No relevant record IDs found in retrieval gold."
            )

    # ========================================================
    # EXECUTE ONE STRATEGY
    # ========================================================

    def _run_strategy(
        self,
        *,
        strategy: str,
        patient_id: str,
        consultation_id: str,
        query: str,
        consultation_text: str,
        current_measurements: list[dict[str, Any]],
        cutoff: str,
        exclude_record_ids: list[str],
        retrieval_context: Any,
    ) -> StrategyExperimentInput:

        retrieval_start = time.perf_counter()

        historical_records = []
        retrieval_status = "no_relevant_records"
        retrieval_method = None
        retrieval_metadata = {}

        # Strategy A: consultation only.
        if strategy == "consultation_only":
            logger.info(
                "Consultation-only: historical retrieval skipped."
            )

        # Strategy B: full history.
        elif strategy == "full_history":
            logger.info(
                "Full-history: retrieving all eligible historical records."
            )

            full_history = self.structured_retriever.retrieve_all(
                patient_id=patient_id,
                record_date_before=cutoff,
                exclude_record_ids=exclude_record_ids,
            )

            raw_historical = full_history.get(
                "retrieved_records", []
            )
            historical_records = [
                self._compact_historical_record(rec)
                if isinstance(rec, dict)
                else self._compact_historical_record(rec.model_dump())
                for rec in raw_historical
            ]

            retrieval_metadata = full_history.get(
                "metadata", {}
            )

            retrieval_status = (
                "success" if historical_records else "no_records_found"
            )

            retrieval_method = "structured"

        # Strategy C: selective retrieval.
        elif strategy == "selective_retrieval":
            logger.info(
                "Selective retrieval: invoking existing RetrievalService."
            )

            retrieval_request = SimpleNamespace(
                patient_id=patient_id,
                query=query,
                retrieval_context=retrieval_context,
                conversation_id=consultation_id,
            )

            selective_result = self.retrieval_service.retrieve(
                retrieval_request,
                eligible_before=cutoff,
                exclude_record_ids=exclude_record_ids,
            )

            historical_records = selective_result.get(
                "retrieved_records", []
            )

            retrieval_metadata = selective_result.get(
                "metadata", {}
            )

            retrieval_status = selective_result.get(
                "status",
                "no_relevant_records",
            )

            retrieval_method = retrieval_metadata.get(
                "retrieval_method"
            )

        else:
            raise ExperimentRunnerError(
                f"Unsupported strategy: {strategy}"
            )

        retrieval_elapsed = time.perf_counter() - retrieval_start

        # Build consultation record.
        consultation_record = {
            "record_id": consultation_id,
            "record_type": "consultation",
            "date": cutoff,
            "content": consultation_text,
            "source": "current_consultation",
            "relevance_score": None,
        }

        normalized_measurements = [
            self._normalize_measurement(record, index)
            for index, record in enumerate(current_measurements)
        ]

        # All strategies receive the consultation and current measurements.
        # Only historical records differ between strategies.
        generation_records = (
            [consultation_record]
            + normalized_measurements
            + historical_records
        )

        # Generate answer.
        generation_start = time.perf_counter()

        generation_result = None
        generation_error = None
        answer = None
        generation_status = "generation_failed"

        try:
            generation_result = self.answer_generator.generate(
                query=query,
                records=generation_records,
            )

            answer = generation_result.get("answer")

            generation_status = (
                "success" if answer else "generation_failed"
            )

        except Exception as exc:
            generation_error = str(exc)

            logger.exception(
                "Answer generation failed for strategy %s",
                strategy,
            )

        generation_elapsed = time.perf_counter() - generation_start

        generation_model = (
            generation_result.get("model")
            if generation_result
            else None
        )

        # Build retrieval snapshot.
        retrieval_snapshot = RetrievalSnapshot(
            status=retrieval_status,
            patient_id=patient_id,
            conversation_id=consultation_id,
            query=query,
            retrieved_records=historical_records,
            metadata={
                "retrieval_method": retrieval_method,
                "vector_database": "ChromaDB",
                "candidates_retrieved": retrieval_metadata.get(
                    "candidates_retrieved", 0
                ),
                "records_returned": len(historical_records),
                "records_filtered": retrieval_metadata.get(
                    "records_filtered", 0
                ),
                "retrieval_time_seconds": round(
                    retrieval_elapsed, 4
                ),
                "retrieval_strategy": strategy,
            },
        )

        # Build generation snapshot.
        supporting_records = [
            record["record_id"]
            for record in generation_records
            if record.get("record_id")
        ]

        generation_snapshot = GenerationSnapshot(
            status=generation_status,
            patient_id=patient_id,
            conversation_id=consultation_id,
            query=query,
            answer=answer,
            supporting_records=supporting_records,
            retrieval_metadata=retrieval_metadata,
            generation_metadata={
                "model": generation_model,
                "generation_time_seconds": round(
                    generation_elapsed, 4
                ),
            },
        )

        # Build exact context snapshot.
        measurement_context = [
            ContextRecord(
                record_id=record["record_id"],
                record_type=record.get("record_type"),
                date=record.get("date"),
                content=record["content"],
            )
            for record in normalized_measurements
        ]

        historical_context = [
            ContextRecord(
                record_id=record["record_id"],
                record_type=record.get("record_type"),
                date=record.get("date"),
                content=record.get("content", ""),
            )
            for record in historical_records
        ]

        context_snapshot = ContextSnapshot(
            consultation_id=consultation_id,
            consultation_text=consultation_text,
            current_measurements=measurement_context,
            historical_records=historical_context,
            context_available=True,
        )

        return StrategyExperimentInput(
            strategy=strategy,
            retrieval=retrieval_snapshot,
            generation=generation_snapshot,
            context=context_snapshot,
            metadata={
                "generation_record_count": len(generation_records),
                "historical_record_count": len(historical_records),
                "generation_error": generation_error,
                "retrieval_time_seconds": round(
                    retrieval_elapsed, 4
                ),
                "generation_time_seconds": round(
                    generation_elapsed, 4
                ),
            },
        )

    # ========================================================
    # NORMALIZE CURRENT MEASUREMENTS
    # ========================================================

    @staticmethod
    def _normalize_measurement(
        measurement: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:

        record_id = (
            measurement.get("record_id")
            or measurement.get("observation_id")
            or measurement.get("id")
        )

        if not record_id:
            raise ExperimentRunnerError(
                "A current measurement is missing its record ID. "
                "Provide the actual observation ID so reference facts "
                "can be matched against the generated context."
            )

        content = (
            measurement.get("content")
            or measurement.get("description")
            or ""
        )

        return {
            "record_id": str(record_id),
            "record_type": measurement.get(
                "record_type", "observation"
            ),
            "date": (
                measurement.get("date")
                or measurement.get("timestamp")
            ),
            "content": str(content),
            "source": measurement.get(
                "source", "current_measurement"
            ),
            "relevance_score": None,
        }

    # ========================================================
    # LOAD ANSWER REFERENCE DATASET
    # ========================================================

    @staticmethod
    def _load_reference(
        reference_file: str | Path,
    ) -> EvaluationReferenceDataset:

        path = ExperimentRunner._resolve_path(reference_file)

        if not path.exists():
            raise ExperimentRunnerError(
                f"Reference file not found: {path}"
            )

        if not path.is_file():
            raise ExperimentRunnerError(
                f"Reference path is not a file: {path}"
            )

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            reference_dataset = (
                EvaluationReferenceDataset.model_validate(data)
            )

            logger.info(
                "Answer reference dataset loaded successfully: %s",
                path,
            )

            return reference_dataset

        except (json.JSONDecodeError, ValueError) as exc:
            raise ExperimentRunnerError(
                f"Unable to load answer reference dataset: {exc}"
            ) from exc

    # ========================================================
    # SAVE EVALUATION RESULTS
    # ========================================================

    def _save_results(
        self,
        case_id: str,
        response,
    ) -> Path:

        output_path = (
            self.results_dir / f"{case_id}_evaluation.json"
        )

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                response.model_dump(mode="json"),
                file,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(
            "Results saved: %s",
            output_path,
        )

        return output_path