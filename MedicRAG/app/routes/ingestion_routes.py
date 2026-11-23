
import logging

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.ingestion_input import IngestionRequest
from app.schemas.ingestion_response import IngestionResponse

from app.services.ingestion_service import (
    IngestionServiceError,
    DuplicateRecordError
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/ingestion",
    tags=["Historical Record Ingestion"]
)


@router.post(
    "/records",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED
)
def ingest_records(
    request: IngestionRequest,
    http_request: Request
) -> IngestionResponse:

    ingestion_service = http_request.app.state.ingestion_service

    try:
        return ingestion_service.ingest(request)

    except DuplicateRecordError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc)
        ) from exc

    except IngestionServiceError as exc:
        logger.exception("Record ingestion endpoint failed")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to ingest historical records."
        ) from exc
