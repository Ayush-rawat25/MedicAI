
import hashlib
import logging
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


logger = logging.getLogger(__name__)


class RecordProcessingError(Exception):
    """Raised when a historical record cannot be processed."""


class RecordProcessor:

    def __init__(self):

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ", ",
                " ",
                ""
            ]
        )

    @staticmethod
    def _generate_content_hash(content: str) -> str:
        """
        Generates a stable hash for detecting record content changes.
        """

        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _validate_record(record: dict[str, Any]) -> None:
        """
        Validates the minimum information required
        to process a historical patient record.
        """

        required_fields = [
            "patient_id",
            "record_id",
            "record_type",
            "record_date",
            "content"
        ]

        for field in required_fields:

            if field not in record or record[field] is None:
                raise RecordProcessingError(
                    f"Missing required field: {field}"
                )

            if isinstance(record[field], str) and not record[field].strip():
                raise RecordProcessingError(
                    f"Empty required field: {field}"
                )

    @staticmethod
    def _format_record(record: dict[str, Any]) -> str:
        """
        Converts a structured historical record
        into searchable clinical text.
        """

        return (
            f"Record Type: {record['record_type']}\n"
            f"Record Date: {record['record_date']}\n"
            f"Clinical Information: {record['content']}"
        )

    def process_record(
        self,
        record: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Processes one historical patient record
        and returns its searchable chunks with metadata.
        """

        try:

            self._validate_record(record)

            patient_id = str(record["patient_id"])
            record_id = str(record["record_id"])

            formatted_text = self._format_record(record)

            content_hash = self._generate_content_hash(
                formatted_text
            )

            chunks = self.text_splitter.split_text(
                formatted_text
            )

            processed_chunks = []

            for index, chunk in enumerate(chunks):

                chunk_id = (
                    f"{patient_id}_{record_id}_"
                    f"{content_hash[:12]}_{index}"
                )

                metadata = {
                    "patient_id": patient_id,
                    "record_id": record_id,
                    "record_type": str(record["record_type"]),
                    "record_date": str(record["record_date"]),
                    "chunk_index": index,
                    "total_chunks": len(chunks),
                    "content_hash": content_hash
                }

                if record.get("source"):
                    metadata["source"] = str(record["source"])

                processed_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "metadata": metadata
                })

            logger.info(
                "Processed record %s for patient %s into %d chunks",
                record_id,
                patient_id,
                len(processed_chunks)
            )

            return processed_chunks

        except RecordProcessingError:
            raise

        except Exception as exc:

            logger.exception(
                "Failed to process historical record"
            )

            raise RecordProcessingError(
                "Unable to process historical record."
            ) from exc

    def process_records(
        self,
        records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Processes multiple historical patient records.
        """

        processed_chunks = []

        for record in records:

            chunks = self.process_record(record)

            processed_chunks.extend(chunks)

        logger.info(
            "Processed %d historical records into %d chunks",
            len(records),
            len(processed_chunks)
        )

        return processed_chunks
