
import logging

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.retrieval_input import RetrievalInput
from app.schemas.retrieval_response import RetrievalResponse
from app.services.retrieval_service import RetrievalServiceError


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/retrieval",
    tags=["Historical Record Retrieval"]
)


@router.post(
    "/search",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK
)
def retrieve_patient_history(
    request: RetrievalInput,
    http_request: Request
) -> RetrievalResponse:
    """
    Retrieve relevant historical records for a patient.
    """

    retrieval_service = http_request.app.state.retrieval_service

    try:
        return retrieval_service.retrieve(request)

    except RetrievalServiceError as exc:
        logger.exception(
            "Retrieval failed for patient %s",
            request.patient_id
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve patient history."
        ) from exc
