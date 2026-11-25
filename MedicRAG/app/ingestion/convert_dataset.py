
import json
import logging
from pathlib import Path

from app.schemas.ingestion_input import IngestionRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def convert_dataset():

    project_root = Path(__file__).resolve().parents[2]

    input_path = (
        project_root / "data" / "p001_chroma_import.json"
    )

    output_path = (
        project_root / "data" / "p001_ingestion_request.json"
    )

    with open(input_path, "r", encoding="utf-8") as file:
        raw_records = json.load(file)

    if not isinstance(raw_records, list):
        raise ValueError("Expected a JSON list of records.")

    converted_records = []

    for item in raw_records:

        metadata = item["metadata"]

        converted_records.append({
            "patient_id": metadata["patient_id"],
            "record_id": metadata["record_id"],
            "record_type": metadata["record_type"],
            "record_date": metadata["record_date"],
            "content": item["document"],
            "source": metadata.get("source")
        })

    # Validate against the actual ingestion schema
    request = IngestionRequest(
        records=converted_records
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            request.model_dump(),
            file,
            indent=4,
            ensure_ascii=False
        )

    logger.info(
        "Successfully converted %d records",
        len(request.records)
    )

    logger.info("Output: %s", output_path)


if __name__ == "__main__":
    convert_dataset()
