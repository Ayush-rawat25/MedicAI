import json
import logging
import math
import re
from typing import Any

from MedicRAG.app.generation.llm_service import (
    LLMService,
    LLMServiceError,
)

from evaluation.schemas.evaluation_schemas import (
    ExperimentEvaluationRequest,
    ExperimentEvaluationResponse,
    StrategyEvaluationResult,
    RetrievalEvaluationResult,
    RetrievalEvaluationMetrics,
    AnswerEvaluationResult,
    FactAssessment,
    UnsupportedClaim,
    EvaluationStrategy,
)

logger = logging.getLogger(__name__)


class EvaluationServiceError(Exception):
    """Raised when an evaluation operation fails."""


class _ApplicableNotApplicableError(Exception):
    """Internal signal used to trigger a targeted retry when an applicable fact is marked not_applicable."""

    def __init__(self, fact_ids: list[str]):
        super().__init__(
            f"Applicable facts marked as not_applicable: {', '.join(fact_ids)}"
        )
        self.fact_ids = fact_ids


class EvaluationService:
    REQUIRED_RESPONSE_KEYS = {
        "fact_assessments",
        "unsupported_claims",
        "answer_claims",
    }

    def __init__(
        self,
        llm_service: LLMService | None = None,
    ):
        # Reuse the existing application's LLM service.
        self.llm_service = llm_service or LLMService()

    # ========================================================
    # MAIN EVALUATION ENTRY POINT
    # ========================================================

    def run(
        self,
        request: ExperimentEvaluationRequest,
    ) -> ExperimentEvaluationResponse:

        try:
            results = []

            if not request.relevant_record_ids:
                raise EvaluationServiceError(
                    "Retrieval ground truth is empty. "
                    "Check that ExperimentRunner loaded the retrieval gold file."
                )

            for strategy_run in request.strategy_runs:
                logger.info(
                    "Evaluating strategy: %s | Case: %s",
                    strategy_run.strategy,
                    request.case_id,
                )

                retrieval_result = self._evaluate_retrieval(
                    strategy=strategy_run.strategy,
                    retrieval=strategy_run.retrieval,
                    relevant_record_ids=request.relevant_record_ids,
                )

                answer_result = self._evaluate_answer(
                    strategy=strategy_run.strategy,
                    query=request.query,
                    generation=strategy_run.generation,
                    context=strategy_run.context,
                    reference_facts=(request.reference_dataset.reference_facts),
                    consultation_id=request.consultation_id,
                )

                generation_time = (
                    strategy_run.generation.generation_metadata.generation_time_seconds
                )

                retrieval_time = strategy_run.retrieval.metadata.retrieval_time_seconds

                # Excludes separate LLM evaluator execution time.
                total_pipeline_time = retrieval_time + generation_time

                results.append(
                    StrategyEvaluationResult(
                        strategy=strategy_run.strategy,
                        retrieval_evaluation=retrieval_result,
                        answer_evaluation=answer_result,
                        generation_status=(strategy_run.generation.status),
                        generation_time_seconds=generation_time,
                        total_time_seconds=total_pipeline_time,
                        metadata={
                            "evaluation_method": ("llm_judge_with_python_metrics"),
                            "evaluation_time_included": False,
                        },
                    )
                )

            failed_count = sum(
                result.generation_status == "generation_failed" for result in results
            )

            if failed_count == len(results):
                overall_status = "failed"
            elif failed_count > 0:
                overall_status = "partial_success"
            else:
                overall_status = "success"

            eval_model = (
                getattr(self.llm_service, "default_model", None)
                or getattr(self.llm_service, "model", None)
                or "unknown"
            )

            return ExperimentEvaluationResponse(
                status=overall_status,
                case_id=request.case_id,
                patient_id=request.patient_id,
                consultation_id=request.consultation_id,
                query=request.query,
                evaluated_strategy_count=len(results),
                results=results,
                metadata={
                    "strategy_count": len(results),
                    "reference_fact_count": len(
                        request.reference_dataset.reference_facts
                    ),
                    "relevant_record_count": len(request.relevant_record_ids),
                    "evaluation_model": eval_model,
                    "evaluation_method": (
                        "LLM-as-judge with Python metric calculation"
                    ),
                },
            )

        except LLMServiceError as exc:
            logger.exception("Evaluation LLM failed.")

            raise EvaluationServiceError(f"Answer evaluation failed: {exc}") from exc

        except Exception as exc:
            logger.exception("Evaluation failed.")

            raise EvaluationServiceError(f"Evaluation failed: {exc}") from exc

    # ========================================================
    # 1. RETRIEVAL EVALUATION
    # ========================================================

    def _evaluate_retrieval(
        self,
        strategy: EvaluationStrategy,
        retrieval,
        relevant_record_ids: list[str],
    ) -> RetrievalEvaluationResult:

        retrieved_ids = list(
            dict.fromkeys(record.record_id for record in retrieval.retrieved_records)
        )

        relevant_ids = list(dict.fromkeys(relevant_record_ids))

        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)

        retrieval_time = retrieval.metadata.retrieval_time_seconds

        retrieval_method = retrieval.metadata.retrieval_method

        # ----------------------------------------------------
        # CONSULTATION ONLY
        # ----------------------------------------------------

        if strategy == "consultation_only":
            return RetrievalEvaluationResult(
                retrieved_record_ids=[],
                relevant_record_ids=relevant_ids,
                relevant_record_ids_retrieved=[],
                missed_relevant_record_ids=relevant_ids,
                record_count=0,
                retrieval_time_seconds=retrieval_time,
                retrieval_method=None,
                metrics=RetrievalEvaluationMetrics(
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    mrr=None,
                    ndcg=None,
                ),
                metadata={
                    "status": "not_applicable",
                    "evaluation_type": "consultation_only",
                    "reason": (
                        "Historical retrieval is intentionally not "
                        "performed. Precision, recall and F1 are "
                        "operationally reported as zero. Ranking "
                        "metrics are not applicable."
                    ),
                },
            )

        # ----------------------------------------------------
        # COMMON SET-BASED METRICS
        # ----------------------------------------------------

        relevant_retrieved = [
            record_id for record_id in retrieved_ids if record_id in relevant_set
        ]

        missed_relevant = [
            record_id for record_id in relevant_ids if record_id not in retrieved_set
        ]

        retrieved_count = len(retrieved_ids)
        relevant_count = len(relevant_ids)
        hit_count = len(relevant_retrieved)

        precision = hit_count / retrieved_count if retrieved_count > 0 else 0.0

        recall = hit_count / relevant_count if relevant_count > 0 else 0.0

        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # ----------------------------------------------------
        # FULL HISTORY
        # ----------------------------------------------------

        if strategy == "full_history":
            return RetrievalEvaluationResult(
                retrieved_record_ids=retrieved_ids,
                relevant_record_ids=relevant_ids,
                relevant_record_ids_retrieved=relevant_retrieved,
                missed_relevant_record_ids=missed_relevant,
                record_count=retrieved_count,
                retrieval_time_seconds=retrieval_time,
                retrieval_method=retrieval_method,
                metrics=RetrievalEvaluationMetrics(
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score,
                    mrr=None,
                    ndcg=None,
                ),
                metadata={
                    "status": retrieval.status,
                    "evaluation_type": "full_history",
                    "relevant_record_count": relevant_count,
                    "retrieved_relevant_count": hit_count,
                    "reason": (
                        "Full-history retrieval is evaluated using "
                        "set-based precision, recall and F1. Ranking "
                        "metrics are not applicable."
                    ),
                },
            )

        # ----------------------------------------------------
        # SELECTIVE RETRIEVAL
        # ----------------------------------------------------

        if strategy == "selective_retrieval":
            mrr = self._calculate_mrr(
                retrieved_ids,
                relevant_set,
            )

            ndcg = self._calculate_ndcg(
                retrieved_ids,
                relevant_set,
            )

            return RetrievalEvaluationResult(
                retrieved_record_ids=retrieved_ids,
                relevant_record_ids=relevant_ids,
                relevant_record_ids_retrieved=relevant_retrieved,
                missed_relevant_record_ids=missed_relevant,
                record_count=retrieved_count,
                retrieval_time_seconds=retrieval_time,
                retrieval_method=retrieval_method,
                metrics=RetrievalEvaluationMetrics(
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score,
                    mrr=mrr,
                    ndcg=ndcg,
                ),
                metadata={
                    "status": retrieval.status,
                    "evaluation_type": "selective_retrieval",
                    "relevant_record_count": relevant_count,
                    "retrieved_relevant_count": hit_count,
                    "ranking_method": "binary_relevance",
                },
            )

        raise EvaluationServiceError(f"Unsupported evaluation strategy: {strategy}")

    # ========================================================
    # 2. MEAN RECIPROCAL RANK
    # ========================================================

    @staticmethod
    def _calculate_mrr(
        retrieved_ids: list[str],
        relevant_ids: set[str],
    ) -> float:

        for rank, record_id in enumerate(
            retrieved_ids,
            start=1,
        ):
            if record_id in relevant_ids:
                return 1.0 / rank

        return 0.0

    # ========================================================
    # 3. NORMALIZED DISCOUNTED CUMULATIVE GAIN
    # ========================================================

    @staticmethod
    def _calculate_ndcg(
        retrieved_ids: list[str],
        relevant_ids: set[str],
    ) -> float:

        if not relevant_ids:
            return 0.0

        dcg = 0.0

        for rank, record_id in enumerate(
            retrieved_ids,
            start=1,
        ):
            if record_id in relevant_ids:
                dcg += 1.0 / math.log2(rank + 1)

        ideal_count = min(
            len(relevant_ids),
            len(retrieved_ids),
        )

        if ideal_count == 0:
            return 0.0

        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))

        return dcg / idcg if idcg else 0.0

    # ========================================================
    # 4. DETERMINE FACT AVAILABILITY
    # ========================================================

    @staticmethod
    def _is_fact_available(
        fact,
        strategy: EvaluationStrategy,
        context,
        consultation_id: str,
    ) -> bool:

        if not context.context_available:
            return False

        if strategy not in fact.available_in:
            return False

        available_record_ids = set(context.context_record_ids)

        # Current consultation is part of every strategy.
        if context.consultation_id:
            available_record_ids.add(context.consultation_id)

        available_record_ids.add(consultation_id)

        # If no source IDs are annotated, rely on the
        # strategy availability annotation.
        if not fact.source_record_ids:
            return True

        # All annotated sources must be present in the
        # actual generation context.
        return set(fact.source_record_ids).issubset(available_record_ids)

    # ========================================================
    # 5. ANSWER EVALUATION
    # ========================================================

    def _invoke_evaluator_llm(
        self,
        *,
        strategy: EvaluationStrategy,
        evaluator_input: str,
        missing_keys_retry: set[str] | None = None,
        not_applicable_retry_ids: list[str] | None = None,
    ) -> tuple[str, dict[str, Any], set[str]]:
        prompt = evaluator_input
        corrections: list[str] = []

        if missing_keys_retry:
            missing_list = ", ".join(sorted(missing_keys_retry))
            corrections.append(
                f"Your previous response omitted required top-level JSON keys: "
                f"{missing_list}. Return ONLY ONE valid JSON object containing "
                f"ALL three keys: fact_assessments, unsupported_claims, and "
                f"answer_claims."
            )

        if not_applicable_retry_ids:
            ids_str = ", ".join(sorted(not_applicable_retry_ids))
            corrections.append(
                f"Facts with context_available=true CANNOT have status "
                f"'not_applicable'. Re-evaluate {ids_str} (which have "
                f"context_available=true) using one of: supported, "
                f"partially_supported, missing, or contradicted."
            )

        if corrections:
            joined = "\n- ".join(corrections)
            prompt = (
                f"{evaluator_input}\n\n"
                f"CRITICAL CORRECTION:\n- {joined}"
            )

        estimated_tokens = (len(self._get_evaluator_system_prompt()) + len(prompt)) // 3
        num_ctx = max(8192, min(32768, ((estimated_tokens + 4096) // 2048 + 1) * 2048))

        raw_response = self.llm_service.generate(
            system_prompt=self._get_evaluator_system_prompt(),
            user_prompt=prompt,
            temperature=0.0,
            json_mode=True,
            num_ctx=num_ctx,
            num_predict=4096,
        )

        logger.info(
            "[EVALUATION] Evaluator response received for %s | "
            "Response size: %d characters",
            strategy,
            len(raw_response) if raw_response else 0,
        )

        parsed_response = self._parse_json_response(raw_response)

        logger.info(
            "[EVALUATION] Parsed response keys: %s",
            list(parsed_response.keys()),
        )

        missing_keys = self.REQUIRED_RESPONSE_KEYS - set(parsed_response.keys())
        return raw_response, parsed_response, missing_keys

    def _evaluate_answer(
        self,
        strategy: EvaluationStrategy,
        query: str,
        generation,
        context,
        reference_facts,
        consultation_id: str,
    ) -> AnswerEvaluationResult:

        # Do not send failed or empty generations to the judge.
        if generation.status != "success" or not generation.answer:
            return AnswerEvaluationResult(
                metadata={
                    "status": "not_evaluated",
                    "reason": ("Generation did not produce a successful answer."),
                },
            )

        applicable_facts = []

        for fact in reference_facts:
            available = self._is_fact_available(
                fact=fact,
                strategy=strategy,
                context=context,
                consultation_id=consultation_id,
            )

            applicable_facts.append(
                {
                    "fact": fact,
                    "context_available": available,
                }
            )

        evaluator_input = self._build_evaluator_input(
            strategy=strategy,
            query=query,
            consultation_id=consultation_id,
            answer=generation.answer,
            context=context,
            applicable_facts=applicable_facts,
        )

        logger.info(
            "[EVALUATION] Strategy: %s | Prompt size: %d characters",
            strategy,
            len(evaluator_input),
        )

        logger.info(
            "[EVALUATION] Sending evaluator request for %s",
            strategy,
        )

        raw_response, parsed_response, missing_keys = self._invoke_evaluator_llm(
            strategy=strategy,
            evaluator_input=evaluator_input,
        )

        if missing_keys:
            logger.warning(
                "[EVALUATION] Evaluator response missing keys on first attempt: %s. "
                "Retrying once.",
                sorted(missing_keys),
            )
            raw_response, parsed_response, missing_keys = self._invoke_evaluator_llm(
                strategy=strategy,
                evaluator_input=evaluator_input,
                missing_keys_retry=missing_keys,
            )

        if missing_keys:
            logger.error(
                "[EVALUATION] Evaluator response missing keys: %s",
                sorted(missing_keys),
            )

            logger.error(
                "[EVALUATION] Raw evaluator response:\n%s",
                raw_response,
            )

            raise EvaluationServiceError(
                "Evaluator response is missing required keys: "
                + ", ".join(sorted(missing_keys))
            )

        try:
            return self._build_answer_result(
                parsed_response=parsed_response,
                applicable_facts=applicable_facts,
                strict_applicable_status=True,
            )
        except _ApplicableNotApplicableError as exc:
            logger.warning(
                "[EVALUATION] Evaluator marked available fact(s) as not_applicable: %s. "
                "Retrying once with corrective instruction.",
                exc.fact_ids,
            )
            _, retry_parsed, retry_missing = self._invoke_evaluator_llm(
                strategy=strategy,
                evaluator_input=evaluator_input,
                missing_keys_retry=retry_missing if (retry_missing := None) else None,
                not_applicable_retry_ids=exc.fact_ids,
            )
            if retry_missing:
                raise EvaluationServiceError(
                    "Evaluator retry response is missing required keys: "
                    + ", ".join(sorted(retry_missing))
                )

            return self._build_answer_result(
                parsed_response=retry_parsed,
                applicable_facts=applicable_facts,
                strict_applicable_status=False,
            )

    # ========================================================
    # 6. EVALUATOR SYSTEM PROMPT
    # ========================================================

    @staticmethod
    def _get_evaluator_system_prompt() -> str:

        return """
You are a strict evaluator of generated clinical consultation answers.

Your task is to evaluate the GENERATED ANSWER against:
1. The supplied reference facts.
2. The exact evidence supplied in the evaluation context.

You are NOT generating a clinical summary.
You are NOT extracting the patient's entire medical history.
You are NOT retrieving additional patient information.

STRICT RULES:

1. Evaluate every supplied reference fact exactly once.

2. Return the exact fact_id supplied for each reference fact.
   Never create, modify, or omit fact IDs.

3. Use only the supplied generation context as evidence.
   Reference facts are evaluation targets, NOT generation evidence.

4. The context_available field in each reference fact is
   calculated by Python. Copy it exactly.

5. If context_available is false:
   - status MUST be "not_applicable".
   - context_grounded MUST be null.
   - Do not classify the fact as missing.
   - Do not penalize the answer for omitting it.

6. If context_available is true:
   - status MUST be one of: supported, partially_supported, missing, contradicted.
   - status MUST NEVER be "not_applicable" when context_available is true.

7. Status definitions:
   - supported: The answer accurately represents the fact.
   - partially_supported: The answer represents the fact
     but omits an important detail or includes only part of it.
   - missing: The answer does not mention the applicable fact.
   - contradicted: The answer explicitly conflicts with the fact.

8. For context_grounded:
   - true: The answer's factual representation is supported
     by the supplied generation context.
   - false: The answer makes a claim that is not supported by,
     or conflicts with, the supplied generation context.
   - null: Use for missing and not_applicable assessments.

9. Do not confuse database existence with context availability.
   A historical record not included in the supplied context
   is unavailable to that generation strategy.

10. For negative constraints:
    - If the answer makes no prohibited claim, mark supported.
    - If the answer makes a prohibited claim, mark contradicted.
    - Do not penalize omission of a negative constraint.

11. Medication records do not establish current medication
    adherence. Preserve uncertainty when the consultation
    says the medication list is unconfirmed.

12. Historical conditions must not be represented as current
    acute conditions unless the context supports that.

13. Never invent diagnoses, allergies, reactions, medication
    changes, prescriptions, or treatment plans.

14. Do not extract unrelated historical records as reference
    facts. Only assess the reference facts supplied.

ANSWER CLAIM EXTRACTION:

15. Identify distinct factual claims actually stated in the
    generated answer.

16. Return answer_claims as a JSON array of STRINGS.
    Each item must be a plain string describing one factual claim.
    Do not return objects, dictionaries, or metadata.

17. Do not include facts that appear only in the generation
    context or reference facts but not in the generated answer.

18. unsupported_claims must contain only factual claims from
    answer_claims that are unsupported, invented, or contradictory.

19. Every unsupported claim must use this exact structure:
    {
      "claim": "The factual claim",
      "severity": "medium",
      "claim_type": "context_unsupported",
      "explanation": "Why the claim is unsupported"
    }

20. Allowed severity values:
    critical, high, medium, low.

21. Allowed claim_type values:
    factually_incorrect, context_unsupported,
    contradictory, invented.

22. Do not calculate evaluation metrics or percentages.

OUTPUT REQUIREMENTS:

Return ONE single JSON object as raw JSON text.
Do NOT use Markdown headings, bullet lists, prefixes like "F001:", or code fences.
Do NOT split fact_assessments, unsupported_claims, and answer_claims into separate sections.

It MUST contain all three top-level keys:
- fact_assessments
- unsupported_claims
- answer_claims

The fact_assessments array MUST contain one object
for every supplied reference fact:

{
  "fact_id": "F001",
  "status": "supported",
  "context_available": true,
  "context_grounded": true,
  "answer_evidence": "Evidence from the generated answer",
  "context_evidence": "Evidence from the supplied context",
  "explanation": "Brief explanation"
}

For a missing fact, answer_evidence and context_evidence
may be null and context_grounded must be null.

For an unavailable fact, status must be not_applicable
and context_grounded must be null.

The final JSON structure is:

{
  "fact_assessments": [],
  "unsupported_claims": [],
  "answer_claims": []
}

Do not return Markdown fences.
Do not return commentary.
Do not return additional keys.
"""

    # ========================================================
    # 7. BUILD EVALUATOR INPUT
    # ========================================================

    @staticmethod
    def _build_evaluator_input(
        strategy: EvaluationStrategy,
        query: str,
        consultation_id: str,
        answer: str,
        context,
        applicable_facts: list[dict[str, Any]],
    ) -> str:

        relevant_source_ids = set()

        has_fact_without_source_ids = False

        for item in applicable_facts:
            fact = item["fact"]

            if fact.source_record_ids:
                relevant_source_ids.update(fact.source_record_ids)
            elif item["context_available"]:
                # If a fact has no source annotations, Python
                # relies on strategy availability. In this case,
                # retain all historical context so the judge
                # has evidence to assess that fact.
                has_fact_without_source_ids = True

        if has_fact_without_source_ids:
            historical_records = [
                record.model_dump() for record in context.historical_records
            ]
        else:
            historical_records = [
                record.model_dump()
                for record in context.historical_records
                if record.record_id in relevant_source_ids
            ]

        context_data = {
            "consultation_id": context.consultation_id or consultation_id,
            "consultation_text": context.consultation_text,
            "current_measurements": [
                record.model_dump() for record in context.current_measurements
            ],
            "historical_records": historical_records,
        }

        facts_data = []

        for item in applicable_facts:
            fact = item["fact"]

            facts_data.append(
                {
                    "fact_id": fact.fact_id,
                    "description": fact.description,
                    "category": fact.category,
                    "priority": fact.priority,
                    "source_record_ids": fact.source_record_ids,
                    "context_available": item["context_available"],
                    "is_negative_constraint": (fact.is_negative_constraint),
                    "metadata": fact.metadata,
                }
            )

        return f"""
Evaluate this generated clinical answer.

STRATEGY:
{strategy}

CLINICAL QUESTION / QUERY:
{query}

EXACT GENERATION CONTEXT:
{json.dumps(context_data, indent=2, ensure_ascii=False)}

REFERENCE FACTS:
{json.dumps(facts_data, indent=2, ensure_ascii=False)}

GENERATED ANSWER:
{answer}

IMPORTANT:

- The context above contains only evidence from the actual
  generation context.
- Do not assume that records outside this context were available.
- Evaluate every supplied reference fact exactly once.
- Copy context_available exactly as provided.
- Unavailable facts (context_available=false) must be not_applicable.
- Available facts (context_available=true) must NEVER be not_applicable.
- Do not invent facts or source record IDs.
- Do not extract unrelated patient history.
- answer_claims must be an array of plain strings.
- unsupported_claims must be an array of objects matching
  the specified schema.
- Return a single top-level JSON object with all three required keys.
- Do not use Markdown headings or bullet points.
- Return only valid JSON.
"""

    # ========================================================
    # 8. PARSE EVALUATOR JSON
    # ========================================================

    @classmethod
    def _parse_json_response(cls, raw_response: str) -> dict[str, Any]:
        if not isinstance(raw_response, str) or not raw_response.strip():
            raise EvaluationServiceError("Evaluator returned an empty response.")

        cleaned = raw_response.strip()

        # Remove Markdown code fences.
        cleaned = (
            re.sub(
                r"```(?:json)?",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            .replace("```", "")
            .strip()
        )

        required_keys = cls.REQUIRED_RESPONSE_KEYS

        # 1. Fast path: response is already a single JSON object.
        try:
            direct = json.loads(cleaned)
            if isinstance(direct, dict) and required_keys.issubset(direct.keys()):
                return direct
        except json.JSONDecodeError:
            pass

        # 2. Fallback: model split output into section headings and bulleted items.
        section_map: dict[str, list[Any]] = {
            "fact_assessments": [],
            "unsupported_claims": [],
            "answer_claims": [],
        }
        current_section: str | None = None
        found_any_section = False

        for line in cleaned.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            header_match = re.match(
                r"^\*{0,2}(fact_assessments|unsupported_claims|answer_claims)\*{0,2}\s*:?\s*$",
                stripped,
                flags=re.IGNORECASE,
            )
            if header_match:
                current_section = header_match.group(1).lower()
                found_any_section = True
                continue

            if current_section and stripped.startswith(("-", "*")):
                item_text = re.sub(r"^[-*]\s*", "", stripped).strip()

                if item_text.lower() in {"none", "null", "n/a", "[]"}:
                    continue

                # Strip optional "F001: " prefix before inline JSON object.
                prefixed_json = re.match(
                    r"^[A-Za-z0-9_-]+\s*:\s*(\{.*\})\s*$", item_text
                )
                if prefixed_json:
                    item_text = prefixed_json.group(1)

                try:
                    parsed_item = json.loads(item_text)
                    if isinstance(parsed_item, list):
                        section_map[current_section].extend(parsed_item)
                    else:
                        section_map[current_section].append(parsed_item)
                except json.JSONDecodeError:
                    if current_section == "answer_claims":
                        cleaned_claim = item_text.strip('"').strip("'").strip()
                        if cleaned_claim:
                            section_map[current_section].append(cleaned_claim)

        if found_any_section:
            logger.warning(
                "Reconstructed evaluator JSON from section-formatted response."
            )
            return section_map

        # 3. Scan for top-level / embedded JSON objects.
        decoder = json.JSONDecoder()
        candidates: list[dict[str, Any]] = []

        for index, char in enumerate(cleaned):
            if char != "{":
                continue

            try:
                parsed, _ = decoder.raw_decode(cleaned[index:])

                if isinstance(parsed, dict):
                    candidates.append(parsed)

            except json.JSONDecodeError:
                continue

        if not candidates:
            logger.error(
                "No valid JSON object found. Raw response:\n%s",
                raw_response,
            )
            raise EvaluationServiceError("Could not parse evaluator JSON.")

        for candidate in candidates:
            if required_keys.issubset(candidate.keys()):
                return candidate

        best_candidate = max(
            candidates,
            key=lambda item: (
                len(required_keys.intersection(item.keys())),
                len(item),
            ),
        )

        logger.warning(
            "Evaluator JSON missing expected keys. Found: %s",
            list(best_candidate.keys()),
        )

        return best_candidate

    # ========================================================
    # 9. VALIDATE TOTAL CLAIM COUNT
    # ========================================================

    @staticmethod
    def _get_total_answer_claims(
        parsed_response: dict[str, Any],
        unsupported_claim_count: int,
    ) -> tuple[int, str]:
        """
        Count answer_claims in Python.

        Backward compatibility:
        If answer_claims is missing, accept a valid
        total_answer_claims integer from the evaluator.

        Never silently substitute zero.
        """

        answer_claims = parsed_response.get("answer_claims")

        if answer_claims is not None:
            if not isinstance(answer_claims, list):
                raise EvaluationServiceError("answer_claims must be a list.")

            if not all(
                isinstance(claim, str) and claim.strip() for claim in answer_claims
            ):
                raise EvaluationServiceError(
                    "Every answer_claims item must be a non-empty string."
                )

            total = len(answer_claims)
            source = "python_counted_answer_claims"

        else:
            raw_total = parsed_response.get("total_answer_claims")

            if isinstance(raw_total, bool):
                raise EvaluationServiceError("total_answer_claims cannot be a boolean.")

            if isinstance(raw_total, int):
                total = raw_total

            elif isinstance(raw_total, str) and raw_total.strip().isdigit():
                total = int(raw_total.strip())

            else:
                raise EvaluationServiceError(
                    "Evaluator returned neither a valid "
                    "answer_claims list nor a valid "
                    "total_answer_claims integer."
                )

            source = "llm_total_answer_claims_fallback"

        if total < 0:
            raise EvaluationServiceError("Total answer claims cannot be negative.")

        if total < unsupported_claim_count:
            raise EvaluationServiceError(
                "Unsupported claim count exceeds total answer claims. "
                "Check the evaluator's claim extraction."
            )

        return total, source

    # ========================================================
    # 10. VALIDATE ASSESSMENTS AND CALCULATE METRICS
    # ========================================================

    def _build_answer_result(
        self,
        parsed_response: dict[str, Any],
        applicable_facts: list[dict[str, Any]],
        strict_applicable_status: bool = True,
    ) -> AnswerEvaluationResult:

        raw_assessments = parsed_response.get("fact_assessments")

        if not isinstance(raw_assessments, list):
            raise EvaluationServiceError(
                "Evaluator response is missing fact_assessments."
            )

        expected_facts = {item["fact"].fact_id: item for item in applicable_facts}

        assessments_by_id = {}
        invalid_not_applicable_ids: list[str] = []

        # ----------------------------------------------------
        # VALIDATE EACH ASSESSMENT
        # ----------------------------------------------------

        for raw in raw_assessments:
            if not isinstance(raw, dict):
                raise EvaluationServiceError(
                    "Each fact assessment must be a JSON object."
                )

            fact_id = raw.get("fact_id")

            if not fact_id:
                raise EvaluationServiceError("Fact assessment is missing fact_id.")

            if fact_id not in expected_facts:
                raise EvaluationServiceError(f"Unknown fact ID: {fact_id}")

            if fact_id in assessments_by_id:
                raise EvaluationServiceError(f"Duplicate assessment: {fact_id}")

            expected_available = expected_facts[fact_id]["context_available"]

            # Python is the source of truth for availability.
            assessment_data = dict(raw)

            assessment_data["context_available"] = expected_available

            if not expected_available:
                # Unavailable facts cannot be evaluated.
                assessment_data["status"] = "not_applicable"
                assessment_data["context_grounded"] = None

            elif assessment_data.get("status") == "not_applicable":
                if strict_applicable_status:
                    invalid_not_applicable_ids.append(fact_id)
                    continue

                logger.warning(
                    "Coercing 'not_applicable' to 'missing' for available fact %s "
                    "after retry.",
                    fact_id,
                )
                assessment_data["status"] = "missing"
                assessment_data["context_grounded"] = None

            try:
                assessment = FactAssessment.model_validate(assessment_data)

            except Exception as exc:
                raise EvaluationServiceError(
                    f"Invalid assessment for {fact_id}: {exc}"
                ) from exc

            assessments_by_id[fact_id] = assessment

        if invalid_not_applicable_ids and strict_applicable_status:
            raise _ApplicableNotApplicableError(invalid_not_applicable_ids)

        # ----------------------------------------------------
        # CHECK FOR OMITTED FACTS
        # ----------------------------------------------------

        missing_ids = set(expected_facts.keys()) - set(assessments_by_id.keys())

        if missing_ids:
            raise EvaluationServiceError(
                "Evaluator omitted fact assessments: " + ", ".join(sorted(missing_ids))
            )

        assessments = [assessments_by_id[fact_id] for fact_id in expected_facts]

        # ----------------------------------------------------
        # FACT COUNTS
        # ----------------------------------------------------

        supported = sum(item.status == "supported" for item in assessments)

        partially_supported = sum(
            item.status == "partially_supported" for item in assessments
        )

        missing = sum(item.status == "missing" for item in assessments)

        contradicted = sum(item.status == "contradicted" for item in assessments)

        not_applicable = sum(item.status == "not_applicable" for item in assessments)

        applicable_count = supported + partially_supported + missing + contradicted

        # ----------------------------------------------------
        # COVERAGE
        # ----------------------------------------------------

        coverage_score = (
            (supported + 0.5 * partially_supported) / applicable_count * 100
            if applicable_count > 0
            else None
        )

        fully_supported_percentage = (
            supported / applicable_count * 100 if applicable_count > 0 else None
        )

        # ----------------------------------------------------
        # CRITICAL / HIGH PRIORITY FACT ERRORS
        # ----------------------------------------------------

        critical_fact_errors = 0

        for assessment in assessments:
            if assessment.status not in (
                "missing",
                "contradicted",
            ):
                continue

            fact = expected_facts[assessment.fact_id]["fact"]

            if fact.priority in ("critical", "high"):
                critical_fact_errors += 1

        # ----------------------------------------------------
        # CONTEXT GROUNDING
        # ----------------------------------------------------

        grounding_assessments = [
            item
            for item in assessments
            if (item.status != "not_applicable" and item.context_grounded is not None)
        ]

        context_grounding_score = (
            sum(item.context_grounded is True for item in grounding_assessments)
            / len(grounding_assessments)
            * 100
            if grounding_assessments
            else None
        )

        # ----------------------------------------------------
        # UNSUPPORTED CLAIMS
        # ----------------------------------------------------

        raw_claims = parsed_response.get(
            "unsupported_claims",
            [],
        )

        if not isinstance(raw_claims, list):
            raise EvaluationServiceError("unsupported_claims must be a list.")

        unsupported_claims = []

        for item in raw_claims:
            try:
                unsupported_claims.append(UnsupportedClaim.model_validate(item))

            except Exception as exc:
                raise EvaluationServiceError(
                    f"Invalid unsupported claim: {exc}"
                ) from exc

        # ----------------------------------------------------
        # TOTAL ANSWER CLAIMS
        # ----------------------------------------------------

        total_answer_claims, claim_count_source = self._get_total_answer_claims(
            parsed_response=parsed_response,
            unsupported_claim_count=len(unsupported_claims),
        )

        unsupported_claim_rate = (
            round(len(unsupported_claims) / total_answer_claims * 100, 2)
            if total_answer_claims > 0
            else None
        )

        # ----------------------------------------------------
        # FINAL ANSWER EVALUATION RESULT
        # ----------------------------------------------------

        return AnswerEvaluationResult(
            fact_assessments=assessments,
            unsupported_claims=unsupported_claims,
            total_answer_claims=total_answer_claims,
            applicable_fact_count=applicable_count,
            supported_count=supported,
            partially_supported_count=partially_supported,
            missing_count=missing,
            contradicted_count=contradicted,
            not_applicable_count=not_applicable,
            coverage_score=round(coverage_score, 2)
            if coverage_score is not None
            else None,
            fully_supported_percentage=round(fully_supported_percentage, 2)
            if fully_supported_percentage is not None
            else None,
            unsupported_claim_rate=unsupported_claim_rate,
            critical_fact_errors=critical_fact_errors,
            context_grounding_score=round(context_grounding_score, 2)
            if context_grounding_score is not None
            else None,
            metadata={
                "assessment_method": "llm_judge",
                "metric_calculation": "python",
                "reference_fact_count": len(assessments),
                "grounding_assessment_count": len(grounding_assessments),
                "claim_count_source": claim_count_source,
                "unsupported_claim_count": len(unsupported_claims),
            },
        )