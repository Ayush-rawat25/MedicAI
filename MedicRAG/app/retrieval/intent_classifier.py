
import logging
import re
from typing import Literal, Optional

from pydantic import BaseModel

from app.schemas.retrieval_input import RetrievalContext

logger = logging.getLogger(__name__)


class RetrievalIntent(BaseModel):
    """Structured result of conversational query classification."""

    strategy: Literal[
        "patient_overview",
        "category_specific",
        "chronological"
    ]

    method: Literal[
        "semantic",
        "structured",
        "hybrid"
    ]

    record_category: Optional[str] = None

    date_from: Optional[str] = None
    date_to: Optional[str] = None

    record_limit: Optional[int] = None


class IntentClassifier:
    """
    Classifies conversational retrieval queries.

    This classifier does not generate clinical answers.
    It only determines how historical records should be retrieved.
    """

    OVERVIEW_PATTERNS = [
        r"\b(patient|medical)\s+(overview|summary|history)\b",
        r"\b(give|provide|show|tell)\s+me\s+(a\s+)?(brief|summary|overview)\b",
        r"\b(summarize|summarise)\s+(the\s+)?patient\b",
        r"\b(all|complete|entire)\s+(medical\s+)?history\b",
        r"\bfull\s+patient\s+history\b",
    ]

    CHRONOLOGICAL_PATTERNS = [
        r"\blast\s+(session|consultation|visit|appointment)\b",
        r"\bprevious\s+(session|consultation|visit|appointment)\b",
        r"\bmost\s+recent\s+(session|consultation|visit|appointment)\b",
        r"\blatest\s+(session|consultation|visit|appointment)\b",
        r"\b(last|previous)\s+\d+\s+(sessions|consultations|visits|appointments)\b",
        r"\bchronological(ly)?\b",
        r"\bwhat\s+happened\s+(during|in)\s+the\s+last\b",
    ]

    CATEGORY_PATTERNS = {
        "allergy": [
            r"\ballerg(y|ies|ic)\b",
            r"\badverse\s+reaction(s)?\b",
        ],
        "medication": [
            r"\bmedication(s)?\b",
            r"\bmedicine(s)?\b",
            r"\bdrug(s)?\b",
            r"\bprescription(s)?\b",
        ],
        "symptoms": [
            r"\bsymptom(s)?\b",
            r"\bcomplaint(s)?\b",
        ],
        "medical_history": [
            r"\bmedical\s+history\b",
            r"\bmedical\s+condition(s)?\b",
            r"\bchronic\s+condition(s)?\b",
        ],
        "lab": [
            r"\blab(s)?\b",
            r"\blaboratory\s+result(s)?\b",
            r"\bblood\s+test(s)?\b",
            r"\btest\s+result(s)?\b",
        ],
        "consultation": [
            r"\bconsultation(s)?\b",
            r"\bvisit(s)?\b",
            r"\bappointment(s)?\b",
        ],
        "treatment": [
            r"\btreatment(s)?\b",
            r"\btherapy\b",
            r"\bfollow[\s-]?up(s)?\b",
        ],
    }

    STRUCTURED_PATTERNS = [
        r"\b(show|list|retrieve|get|fetch|display)\s+(all|every)\b",
        r"\ball\s+(previous|historical|recorded|available)\b",
        r"\b(show|list)\s+all\b",
    ]

    def classify(
        self,
        query: str,
        context: Optional[RetrievalContext] = None
    ) -> RetrievalIntent:
        """
        Determine the retrieval strategy from the query and
        optional structured context.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        normalized_query = self._normalize(query)

        # Explicit context supplied by the orchestrator
        context_category = None

        if context:
            context_category = self._normalize_category(
                context.record_category
            )

        # 1. Patient overview
        if self._matches(
            normalized_query,
            self.OVERVIEW_PATTERNS
        ):
            return RetrievalIntent(
                strategy="patient_overview",
                method="structured",
                record_limit=None
            )

        # 2. Chronological or session-specific retrieval
        if self._matches(
            normalized_query,
            self.CHRONOLOGICAL_PATTERNS
        ):
            return RetrievalIntent(
                strategy="chronological",
                method="structured",
                record_category=context_category,
                date_from=context.date_from if context else None,
                date_to=context.date_to if context else None,
                record_limit=context.record_limit if context else None
            )

        # 3. Category-specific retrieval
        detected_category = (
            context_category
            or self._detect_category(normalized_query)
        )

        if detected_category:
            is_broad_category_request = self._matches(
                normalized_query,
                self.STRUCTURED_PATTERNS
            )

            method = (
                "structured"
                if is_broad_category_request
                else "hybrid"
            )

            return RetrievalIntent(
                strategy="category_specific",
                method=method,
                record_category=detected_category,
                date_from=context.date_from if context else None,
                date_to=context.date_to if context else None,
                record_limit=context.record_limit if context else None
            )

        # 4. Fallback for queries that do not match known patterns
        logger.info(
            "No explicit retrieval intent detected. "
            "Using semantic retrieval."
        )

        return RetrievalIntent(
            strategy="category_specific",
            method="semantic",
            record_category=None,
            date_from=context.date_from if context else None,
            date_to=context.date_to if context else None,
            record_limit=context.record_limit if context else None
        )

    @staticmethod
    def _normalize(query: str) -> str:
        return re.sub(
            r"\s+",
            " ",
            query.strip().lower()
        )

    @staticmethod
    def _matches(
        query: str,
        patterns: list[str]
    ) -> bool:
        return any(
            re.search(pattern, query)
            for pattern in patterns
        )

    def _detect_category(
        self,
        query: str
    ) -> Optional[str]:

        for category, patterns in self.CATEGORY_PATTERNS.items():
            if self._matches(query, patterns):
                return category

        return None

    @staticmethod
    def _normalize_category(
        category: Optional[str]
    ) -> Optional[str]:

        if not category:
            return None

        normalized = category.strip().lower()

        aliases = {
            "allergies": "allergy",
            "medications": "medication",
            "medicines": "medication",
            "labs": "lab",
            "laboratory": "lab",
            "symptom": "symptoms",
            "consultations": "consultation",
            "visits": "consultation",
            "appointments": "consultation",
            "medical": "medical_history",
            "history": "medical_history",
        }

        return aliases.get(normalized, normalized)
