
import logging

from fastapi import APIRouter, HTTPException, status

from app.schema.inputSchema import NoteGenerationInput
from app.schema.noteSchema import ClinicalNote
from app.services.noteGeneration import (
    note_generation_service,
    NoteGenerationError
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/notes",
    tags=["Clinical Notes"]
)


@router.post(
    "/generate",
    response_model=ClinicalNote,
    status_code=status.HTTP_200_OK
)
async def generate_note(
    request: NoteGenerationInput
) -> ClinicalNote:
    """
    Generate a structured clinical note from a consultation
    transcript and optional historical patient context.
    """

    try:
        logger.info(
            "Received note generation request | Patient: %s | Consultation: %s",
            request.patient_id,
            request.consultation_id
        )

        generated_note = note_generation_service.generate_note(
            request
        )

        logger.info(
            "Note generation completed | Patient: %s",
            request.patient_id
        )

        return generated_note

    except NoteGenerationError as exc:
        logger.exception(
            "Note generation failed | Patient: %s",
            request.patient_id
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to generate clinical note."
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected error in note generation route"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred."
        ) from exc
