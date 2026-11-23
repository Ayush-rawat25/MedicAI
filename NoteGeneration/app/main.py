
import logging

from fastapi import FastAPI

from app.routes.noteRoutes import router as note_router


# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="MedScribe AI - Note Generation Service",
    description=(
        "A microservice responsible for generating "
        "structured clinical notes from consultation "
        "transcripts and patient context."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# Register API Routes
# --------------------------------------------------

app.include_router(note_router)


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get(
    "/health",
    tags=["Health"]
)
def health_check():
    """
    Check whether the note-generation service is running.
    """

    return {
        "status": "healthy",
        "service": "note-generation"
    }


# --------------------------------------------------
# Root Endpoint
# --------------------------------------------------

@app.get(
    "/",
    tags=["Root"]
)
def root():
    return {
        "message": "MedScribe AI Note Generation Service",
        "docs": "/docs",
        "health": "/health"
    }


logger.info("Note Generation Service initialized")
