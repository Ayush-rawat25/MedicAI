
from pydantic import BaseModel, Field
from typing import List


class ClinicalNote(BaseModel):
    patient_complaint: List[str] = Field(default_factory=list)

    history_of_present_illness: str = "Not mentioned"

    symptoms: List[str] = Field(default_factory=list)

    medical_history: List[str] = Field(default_factory=list)

    medications: List[str] = Field(default_factory=list)

    allergies: List[str] = Field(default_factory=list)

    examination_findings: List[str] = Field(default_factory=list)

    assessment: str = "Not mentioned"

    diagnosis: List[str] = Field(default_factory=list)

    treatment_plan: List[str] = Field(default_factory=list)

    follow_up: str = "Not mentioned"

    missing_information: List[str] = Field(default_factory=list)
