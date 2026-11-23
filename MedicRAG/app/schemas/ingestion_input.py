
from pydantic import BaseModel, ConfigDict, Field, field_validator


class HistoricalRecordInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    record_type: str = Field(min_length=1)
    record_date: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source: str | None = None

    @field_validator(
        "patient_id",
        "record_id",
        "record_type",
        "record_date",
        "content"
    )
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Field cannot be empty.")

        return value


class IngestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[HistoricalRecordInput] = Field(min_length=1)

    @field_validator("records")
    @classmethod
    def validate_unique_records(
        cls,
        records: list[HistoricalRecordInput]
    ) -> list[HistoricalRecordInput]:

        record_keys = [
            (record.patient_id, record.record_id)
            for record in records
        ]

        if len(record_keys) != len(set(record_keys)):
            raise ValueError(
                "Duplicate patient_id and record_id combinations found."
            )

        return records
