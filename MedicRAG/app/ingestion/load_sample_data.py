
import json
import logging
from pathlib import Path

from app.config import settings
from app.ingestion.record_processor import RecordProcessor
from app.ingestion.embedding_service import EmbeddingService
from app.vectorStore.chroma_store import ChromaStore
from app.services.ingestion_service import IngestionService
from app.schemas.ingestion_input import IngestionRequest


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def load_sample_data():

    project_root = Path(__file__).resolve().parents[2]

    data_path = project_root / "data" / "p001_ingestion_request.json"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Sample data file not found: {data_path}"
        )

    with open(data_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        print("Data type:", type(data).__name__)

        if isinstance(data, list):
            print("Total items:", len(data))
            print("First record:")
            print(json.dumps(data[0], indent=4))
        elif isinstance(data, dict):
            print("Keys:", data.keys())

    request = IngestionRequest(**data)

    logger.info(
        "Loaded %d sample records",
        len(request.records)
    )

    # Initialize ingestion dependencies
    record_processor = RecordProcessor()
    embedding_service = EmbeddingService()
    chroma_store = ChromaStore()

    ingestion_service = IngestionService(
        record_processor=record_processor,
        embedding_service=embedding_service,
        chroma_store=chroma_store
    )

    response = ingestion_service.ingest(request)

    logger.info("Sample data ingestion completed")

    print(response.model_dump_json(indent=4))


if __name__ == "__main__":
    load_sample_data()
