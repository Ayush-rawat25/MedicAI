
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from evaluation.schemas.evaluation_schemas import (
    ExperimentEvaluationRequest,
    ExperimentEvaluationResponse,
)

from evaluation.services.evaluation_service import (
    EvaluationService,
    EvaluationServiceError,
)

from evaluation.services.experiment_runner import (
    ExperimentRunner,
    ExperimentRunnerError,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/evaluation",
    tags=["Evaluation"],
)


# ============================================================
# 1. EXPERIMENT RUN REQUEST SCHEMA
# ============================================================

class ExperimentRunRequest(BaseModel):

    reference_file: str

    # Optional override. The runner loads the authoritative IDs
    # from the retrieval-gold JSON file.
    retrieval_gold_file: Optional[str] = None

    patient_id: str
    consultation_id: str

    query: str
    consultation_text: str

    current_measurements: list[dict[str, Any]] = Field(
        default_factory=list
    )

    relevant_record_ids: list[str] = Field(
        default_factory=list
    )

    retrieval_context: Optional[Any] = None

    exclude_record_ids: Optional[list[str]] = None


# ============================================================
# 2. EVALUATION HEALTH CHECK
# ============================================================

@router.get("/health")
def evaluation_health():

    return {
        "status": "success",
        "message": "Evaluation service is available.",
    }


# ============================================================
# 3. RUN EVALUATION ON PREPARED STRATEGY RESULTS
# ============================================================

@router.post(
    "/run",
    response_model=ExperimentEvaluationResponse,
    status_code=status.HTTP_200_OK,
)
def run_evaluation(
    payload: ExperimentEvaluationRequest,
    http_request: Request,
):

    try:
        evaluation_service: EvaluationService = (
            http_request.app.state.evaluation_service
        )

        return evaluation_service.run(payload)

    except AttributeError as exc:
        logger.exception("Evaluation service is not initialized.")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evaluation service is not initialized.",
        ) from exc

    except EvaluationServiceError as exc:
        logger.exception("Evaluation execution failed.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected evaluation route error.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during evaluation.",
        ) from exc


# ============================================================
# 4. RUN COMPLETE THREE-STRATEGY EXPERIMENT
# ============================================================

@router.post(
    "/run-experiment",
    status_code=status.HTTP_200_OK,
)
def run_experiment(
    payload: ExperimentRunRequest,
    http_request: Request,
):

    try:
        experiment_runner: ExperimentRunner = (
            http_request.app.state.experiment_runner
        )

        result = experiment_runner.run_experiment(
            reference_file=payload.reference_file,
            retrieval_gold_file=payload.retrieval_gold_file,
            patient_id=payload.patient_id,
            consultation_id=payload.consultation_id,
            query=payload.query,
            consultation_text=payload.consultation_text,
            current_measurements=payload.current_measurements,
            relevant_record_ids=payload.relevant_record_ids,
            retrieval_context=payload.retrieval_context,
            exclude_record_ids=payload.exclude_record_ids,
        )

        return result

    except AttributeError as exc:
        logger.exception("Experiment runner is not initialized.")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Experiment runner is not initialized.",
        ) from exc

    except ExperimentRunnerError as exc:
        logger.exception("Experiment execution failed.")

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except EvaluationServiceError as exc:
        logger.exception("Experiment evaluation failed.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected experiment execution error.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during experiment execution.",
        ) from exc
