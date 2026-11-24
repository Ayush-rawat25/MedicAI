
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RecordRanker:

    @staticmethod
    def rank(
        chunks: list[dict[str, Any]],
        top_k: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """
        Groups chunks into complete records and sorts by relevance.

        top_k is retained for compatibility, but is intentionally
        not applied. Semantic threshold filtering determines which
        records are returned.
        """

        if not chunks:
            return []

        grouped_records = {}

        for chunk in chunks:

            metadata = chunk.get("metadata", {})

            patient_id = metadata.get("patient_id")
            record_id = metadata.get("record_id")

            if not patient_id or not record_id:
                continue

            key = (patient_id, record_id)

            if key not in grouped_records:
                grouped_records[key] = {
                    "record_id": record_id,
                    "patient_id": patient_id,
                    "record_type": metadata.get("record_type"),
                    "date": metadata.get("record_date"),
                    "source": metadata.get("source"),
                    "relevance_score": 0.0,
                    "chunks": {}
                }

            record = grouped_records[key]

            chunk_index = metadata.get("chunk_index", 0)

            # Avoid duplicate chunks.
            chunk_id = chunk.get("chunk_id")

            if chunk_id in record["chunks"]:
                continue

            record["chunks"][chunk_id] = {
                "chunk_index": chunk_index,
                "content": chunk.get("text", "")
            }

            record["relevance_score"] = max(
                record["relevance_score"],
                float(chunk.get("relevance_score", 0.0))
            )

        records = []

        for record in grouped_records.values():

            ordered_chunks = sorted(
                record["chunks"].values(),
                key=lambda item: item["chunk_index"]
            )

            content = "\n".join(
                chunk["content"].strip()
                for chunk in ordered_chunks
                if chunk["content"].strip()
            )

            records.append({
                "record_id": record["record_id"],
                "record_type": record["record_type"],
                "date": record["date"],
                "content": content,
                "relevance_score": record["relevance_score"],
                "source": record["source"]
            })

        records.sort(
            key=lambda record: record["relevance_score"],
            reverse=True
        )

        logger.info(
            "Ranked %d qualifying historical records.",
            len(records)
        )

        return records
