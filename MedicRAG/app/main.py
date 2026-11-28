
import logging
from contextlib import asynccontextmanager
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from fastapi import FastAPI

from app.config import settings

# Evaluation
from evaluation.services.evaluation_service import EvaluationService
from evaluation.services.experiment_runner import ExperimentRunner
from evaluation.routes.evaluation_routes import router as evaluation_router

# Ingestion
from app.ingestion.record_processor import RecordProcessor
from app.ingestion.embedding_service import EmbeddingService
from app.services.ingestion_service import IngestionService

# Vector Store
from app.vectorStore.chroma_store import ChromaStore

# Retrieval
from app.retrieval.retriever import Retriever
from app.retrieval.ranking import RecordRanker
from app.retrieval.structured_retriver import StructuredRetriever
from app.retrieval.intent_classifier import IntentClassifier
from app.retrieval.retrival_router import RetrievalRouter
from app.services.retrieval_service import RetrievalService

# Generation
from app.generation.llm_service import LLMService
from app.generation.answer_generation import AnswerGenerator
from app.services.rag_service import RAGService

# Routes
from app.routes.ingestion_routes import router as ingestion_router
from app.routes.retrieval_routes import router as retrieval_router
from app.routes.rag_routes import router as rag_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting MedScribe RAG Service...")

    try:

        # -----------------------------------------
        # 1. Initialize Embedding Service
        # -----------------------------------------

        logger.info("Initializing embedding service...")

        embedding_service = EmbeddingService()

        # -----------------------------------------
        # 2. Initialize ChromaDB
        # -----------------------------------------

        logger.info("Initializing ChromaDB...")

        chroma_store = ChromaStore()

        # -----------------------------------------
        # 3. Initialize Ingestion Service
        # -----------------------------------------

        logger.info("Initializing ingestion service...")

        record_processor = RecordProcessor()

        ingestion_service = IngestionService(
            record_processor=record_processor,
            embedding_service=embedding_service,
            chroma_store=chroma_store
        )

        # -----------------------------------------
        # 4. Initialize Semantic Retriever
        # -----------------------------------------

        logger.info("Initializing semantic retriever...")

        semantic_retriever = Retriever(
            embedding_service=embedding_service,
            chroma_store=chroma_store
        )

        # -----------------------------------------
        # 5. Initialize Structured Retriever
        # -----------------------------------------

        logger.info("Initializing structured retriever...")

        structured_retriever = StructuredRetriever(
            chroma_store=chroma_store
        )

        # -----------------------------------------
        # 6. Initialize Intent Classifier
        # -----------------------------------------

        logger.info("Initializing intent classifier...")

        intent_classifier = IntentClassifier()

        # -----------------------------------------
        # 7. Initialize Record Ranker
        # -----------------------------------------

        record_ranker = RecordRanker()

        # -----------------------------------------
        # 8. Initialize Retrieval Router
        # -----------------------------------------

        logger.info("Initializing retrieval router...")

        retrieval_router_service = RetrievalRouter(
            semantic_retriever=semantic_retriever,
            structured_retriever=structured_retriever,
            record_ranker=record_ranker
        )

        # -----------------------------------------
        # 9. Initialize Retrieval Service
        # -----------------------------------------

        logger.info("Initializing retrieval service...")

        retrieval_service = RetrievalService(
            intent_classifier=intent_classifier,
            retrieval_router=retrieval_router_service
        )

        # -----------------------------------------
        # 10. Initialize LLM Service
        # -----------------------------------------

        logger.info("Initializing LLM service...")

        llm_service = LLMService()

        # -----------------------------------------
        # 11. Initialize Evaluation Service
        # -----------------------------------------

        logger.info("Initializing evaluation service...")

        evaluation_service = EvaluationService(
            llm_service=llm_service
        )

        # -----------------------------------------
        # 12. Initialize Answer Generator
        # -----------------------------------------

        logger.info("Initializing answer generator...")

        answer_generator = AnswerGenerator(
            llm_service=llm_service
        )

        # -----------------------------------------
        # 13. Initialize RAG Service
        # -----------------------------------------

        logger.info("Initializing complete RAG service...")

        rag_service = RAGService(
            retrieval_service=retrieval_service,
            answer_generator=answer_generator
        )

        # -----------------------------------------
        # 14. Initialize Experiment Runner
        # -----------------------------------------

        logger.info("Initializing evaluation experiment runner...")

        experiment_runner = ExperimentRunner(
            structured_retriever=structured_retriever,
            retrieval_service=retrieval_service,
            answer_generator=answer_generator,
            evaluation_service=evaluation_service
        )

        # -----------------------------------------
        # 15. Store Components in Application State
        # -----------------------------------------

        app.state.embedding_service = embedding_service
        app.state.chroma_store = chroma_store

        app.state.record_processor = record_processor
        app.state.ingestion_service = ingestion_service

        app.state.semantic_retriever = semantic_retriever
        app.state.structured_retriever = structured_retriever
        app.state.record_ranker = record_ranker
        app.state.intent_classifier = intent_classifier
        app.state.retrieval_router = retrieval_router_service
        app.state.retrieval_service = retrieval_service

        app.state.llm_service = llm_service
        app.state.answer_generator = answer_generator
        app.state.rag_service = rag_service

        app.state.evaluation_service = evaluation_service
        app.state.experiment_runner = experiment_runner

        logger.info("MedScribe RAG Service started successfully.")

        yield

    except Exception:
        logger.exception("Failed to initialize MedScribe RAG Service.")
        raise

    finally:
        logger.info("Shutting down MedScribe RAG Service...")


# ---------------------------------------------
# FastAPI Application
# ---------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan
)


# ---------------------------------------------
# Register API Routers
# ---------------------------------------------

app.include_router(ingestion_router)
app.include_router(retrieval_router)
app.include_router(rag_router)
app.include_router(evaluation_router)


# ---------------------------------------------
# Health Check
# ---------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION
    }
