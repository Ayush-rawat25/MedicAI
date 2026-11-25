import logging

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.evaluation_schemas import EvaluationRequest, EvaluationResponse
from app.services.evaluation_service import EvaluationServiceError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation", tags=["Retrieval Evaluation"])


@router.post("/run", response_model=EvaluationResponse)
def run_retrieval_evaluation(
    evaluation_request: EvaluationRequest,
    http_request: Request,
):
    evaluation_service = http_request.app.state.evaluation_service
    try:
        return evaluation_service.run(evaluation_request)
    except EvaluationServiceError as exc:
        logger.exception("Evaluation endpoint failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
