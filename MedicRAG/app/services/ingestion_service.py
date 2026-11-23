
import logging

from app.config import settings

from app.schemas.ingestion_input import IngestionRequest
from app.schemas.ingestion_response import IngestionResponse

from app.ingestion.record_processor import (
    RecordProcessor,
    RecordProcessingError
)

from app.ingestion.embedding_service import (
    EmbeddingService,
    EmbeddingServiceError
)

from app.vectorStore.chroma_store import (
    ChromaStore,
    ChromaStoreError
)


logger = logging.getLogger(__name__)


class IngestionServiceError(Exception):
    """Raised when historical record ingestion fails."""
    pass


class DuplicateRecordError(IngestionServiceError):
    """Raised when an existing historical record is submitted again."""
    pass


class IngestionService:

    def __init__(
        self,
        record_processor: RecordProcessor,
        embedding_service: EmbeddingService,
        chroma_store: ChromaStore
    ):
        self.record_processor = record_processor
        self.embedding_service = embedding_service
        self.chroma_store = chroma_store

    def ingest(
        self,
        request: IngestionRequest
    ) -> IngestionResponse:

        try:
            records = [
                record.model_dump()
                for record in request.records
            ]

            logger.info(
                "Starting ingestion of %d historical records",
                len(records)
            )

            # Step 1: Check whether any submitted record already exists.
            # This initial ingestion workflow is append-only.
            for record in records:
                existing_chunks = self.chroma_store.get_record_chunks(
                    patient_id=record["patient_id"],
                    record_id=record["record_id"]
                )

                if existing_chunks:
                    raise DuplicateRecordError(
                        "Record already exists: "
                        f"{record['record_id']}"
                    )

            # Step 2: Convert records into chunks.
            processed_chunks = self.record_processor.process_records(
                records
            )

            # Step 3: Generate embeddings for all chunks.
            embedded_chunks = self.embedding_service.embed_chunks(
                processed_chunks
            )

            # Step 4: Store chunks and embeddings in ChromaDB.
            chunks_created = self.chroma_store.add_chunks(
                embedded_chunks
            )

            logger.info(
                "Ingestion completed | Records: %d | Chunks: %d",
                len(records),
                chunks_created
            )

            return IngestionResponse(
                status="success",
                records_received=len(records),
                records_processed=len(records),
                chunks_created=chunks_created,
                embedding_model=settings.EMBEDDING_MODEL
            )

        except DuplicateRecordError:
            raise

        except (
            RecordProcessingError,
            EmbeddingServiceError,
            ChromaStoreError
        ) as exc:
            logger.exception("Historical record ingestion failed")

            raise IngestionServiceError(
                "Unable to ingest historical records."
            ) from exc

        except Exception as exc:
            logger.exception(
                "Unexpected error during historical record ingestion"
            )

            raise IngestionServiceError(
                "An unexpected error occurred during ingestion."
            ) from exc
