
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings

from app.ingestion.record_processor import RecordProcessor
from app.ingestion.embedding_service import EmbeddingService

from app.vectorStore.chroma_store import ChromaStore

from app.retrieval.retriever import Retriever
from app.retrieval.ranking import RecordRanker

from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService

from app.routes.ingestion_routes import router as ingestion_router
from app.routes.retrieval_routes import router as retrieval_router


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Application Lifespan
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting %s", settings.APP_NAME)

    try:
        # Initialize shared infrastructure
        embedding_service = EmbeddingService()
        chroma_store = ChromaStore()

        # Initialize ingestion components
        record_processor = RecordProcessor()

        ingestion_service = IngestionService(
            record_processor=record_processor,
            embedding_service=embedding_service,
            chroma_store=chroma_store
        )

        # Initialize retrieval components
        retriever = Retriever(
            embedding_service=embedding_service,
            chroma_store=chroma_store
        )

        record_ranker = RecordRanker()

        retrieval_service = RetrievalService(
            retriever=retriever,
            record_ranker=record_ranker
        )

        # Store services in application state
        app.state.embedding_service = embedding_service
        app.state.chroma_store = chroma_store
        app.state.ingestion_service = ingestion_service
        app.state.retrieval_service = retrieval_service

        logger.info("RAG service initialized successfully")

        yield

    except Exception:
        logger.exception("RAG service initialization failed")
        raise

    finally:
        logger.info("Shutting down %s", settings.APP_NAME)


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Patient-specific historical record ingestion "
        "and conversational retrieval service."
    ),
    debug=settings.DEBUG,
    lifespan=lifespan
)

# Register routes before startup
app.include_router(ingestion_router)
app.include_router(retrieval_router)


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health", tags=["Health"])
def health_check():

    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION
    }
