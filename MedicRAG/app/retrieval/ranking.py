
import logging
from collections import defaultdict
from typing import Any


logger = logging.getLogger(__name__)


class RankingError(Exception):
    """Raised when retrieved chunks cannot be ranked."""


class RecordRanker:

    @staticmethod
    def _validate_chunk(chunk: dict[str, Any]) -> None:
        """Validate the fields required for record grouping."""

        required_fields = [
            "record_id",
            "patient_id",
            "record_type",
            "date",
            "chunk_index",
            "content",
            "relevance_score"
        ]

        for field in required_fields:
            if field not in chunk or chunk[field] is None:
                raise RankingError(
                    f"Missing required chunk field: {field}"
                )

        if not isinstance(chunk["content"], str):
            raise RankingError("Chunk content must be a string.")

        if not chunk["content"].strip():
            raise RankingError("Chunk content cannot be empty.")

        if not isinstance(chunk["relevance_score"], (int, float)):
            raise RankingError(
                "Chunk relevance score must be numeric."
            )

        if not isinstance(chunk["chunk_index"], int):
            raise RankingError("Chunk index must be an integer.")

    def group_chunks(
        self,
        retrieved_chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Combine chunks belonging to the same historical record.

        A record's relevance score is determined by its highest-scoring
        retrieved chunk.
        """

        try:
            grouped_chunks = defaultdict(list)

            for chunk in retrieved_chunks:

                self._validate_chunk(chunk)

                # Group by both patient and record ID
                group_key = (
                    chunk["patient_id"],
                    chunk["record_id"]
                )

                grouped_chunks[group_key].append(chunk)

            records = []

            for (patient_id, record_id), chunks in grouped_chunks.items():

                # Sort chunks into their original order
                chunks.sort(
                    key=lambda item: item["chunk_index"]
                )

                # Avoid accidentally duplicating a chunk
                unique_chunks = []
                seen_chunk_ids = set()

                for chunk in chunks:
                    chunk_id = chunk.get("chunk_id")

                    if chunk_id is not None:
                        if chunk_id in seen_chunk_ids:
                            continue

                        seen_chunk_ids.add(chunk_id)

                    unique_chunks.append(chunk)

                if not unique_chunks:
                    continue

                first_chunk = unique_chunks[0]

                # Combine retrieved chunks into one record
                content = "\n".join(
                    chunk["content"].strip()
                    for chunk in unique_chunks
                )

                # Highest chunk relevance score represents the record
                record_score = max(
                    chunk["relevance_score"]
                    for chunk in unique_chunks
                )

                records.append({
                    "patient_id": patient_id,
                    "record_id": record_id,
                    "record_type": first_chunk["record_type"],
                    "date": first_chunk["date"],
                    "content": content,
                    "relevance_score": record_score,
                    "source": first_chunk.get("source")
                })

            logger.info(
                "Grouped %d chunks into %d historical records.",
                len(retrieved_chunks),
                len(records)
            )

            return records

        except RankingError:
            raise

        except Exception as exc:
            logger.exception("Failed to group retrieved chunks.")
            raise RankingError(
                "Unable to group retrieved chunks."
            ) from exc

    @staticmethod
    def rank_records(
        records: list[dict[str, Any]],
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Rank historical records by relevance score.

        Records with higher similarity scores appear first.
        """

        if top_k is not None and top_k <= 0:
            raise RankingError("top_k must be greater than zero.")

        ranked_records = sorted(
            records,
            key=lambda record: record["relevance_score"],
            reverse=True
        )

        if top_k is not None:
            ranked_records = ranked_records[:top_k]

        return ranked_records

    def rank(
        self,
        retrieved_chunks: list[dict[str, Any]],
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Complete ranking pipeline:
        1. Group chunks into records.
        2. Rank records by relevance.
        3. Return the requested number of records.
        """

        try:
            if not retrieved_chunks:
                logger.info(
                    "No chunks available for record ranking."
                )
                return []

            records = self.group_chunks(retrieved_chunks)

            ranked_records = self.rank_records(
                records,
                top_k=top_k
            )

            logger.info(
                "Ranking completed | Records returned: %d",
                len(ranked_records)
            )

            return ranked_records

        except RankingError:
            raise

        except Exception as exc:
            logger.exception("Record ranking failed.")
            raise RankingError(
                "Unable to rank historical records."
            ) from exc
