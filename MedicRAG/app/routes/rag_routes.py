
import logging

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.retrieval_input import RetrievalRequest
from app.schemas.answer_response import RAGResponse
from app.services.rag_service import RAGServiceError


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/rag",
    tags=["RAG Answer Generation"]
)


@router.post(
    "/answer",
    response_model=RAGResponse
)
def generate_answer(
    request: RetrievalRequest,
    http_request: Request
):

    rag_service = http_request.app.state.rag_service

    try:
        return rag_service.answer(request)

    except RAGServiceError as exc:
        logger.exception("RAG endpoint failed")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        )
