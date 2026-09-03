"""PrefNugget Phase 3: nugget-based response grading via OpenRouter."""

from __future__ import annotations

from rag_framework.llm.client import OpenRouterLLM, get_llm_client

from prefnugget.models.judgment import GradeResult, RunScore
from prefnugget.models.nugget import NuggetQuestion, NuggetQuestionBank
from prefnugget.models.response import RagRunRecord
from prefnugget.models.topic import PrefNuggetTopic

_GRADE_SYSTEM = """Grade how well a passage answers a specific question.
Choose one grade:
5: highly relevant, complete, and accurate
4: mostly relevant and complete, minor gaps
3: partially relevant, noticeable gaps
2: limited relevance, significant gaps
1: minimally relevant
0: not relevant at all
Respond with JSON only: {"grade": 0-5, "reasoning": "...", "confidence": 0.0 to 1.0}"""


class NuggetGrader:
    """Grade responses against nugget questions using OpenRouter LLM.

    Mimics ``GradeNuggetAnswer``.

    Parameters
    ----------
    grading_mode : str
        ``response`` grades full response text; ``docs`` grades cited paragraphs.
    llm : OpenRouterLLM | None
        LLM client. Defaults to environment-configured OpenRouter client.
    """

    def __init__(
        self,
        grading_mode: str = "response",
        llm: OpenRouterLLM | None = None,
    ) -> None:
        self._grading_mode = grading_mode
        self._llm = llm or get_llm_client()

    @property
    def name(self) -> str:
        return f"prefnugget_llm_grader_{self._grading_mode}"

    def grade(
        self,
        topic: PrefNuggetTopic,
        run: RagRunRecord,
        bank: NuggetQuestionBank,
    ) -> RunScore:
        """Grade one run against all nugget questions for a topic.

        Parameters
        ----------
        topic : PrefNuggetTopic
            Topic context.
        run : RagRunRecord
            Response run to grade.
        bank : NuggetQuestionBank
            Nugget question bank.

        Returns
        -------
        RunScore
            Per-nugget grades and MAX aggregation.
        """
        if self._grading_mode == "docs":
            passages = run.cited_paragraphs() or [run.response_text()]
        else:
            passages = [run.response_text()]

        grades: list[GradeResult] = []
        for question in bank.questions:
            grade_value, reasoning, confidence = self._grade_question(question, passages)
            grades.append(
                GradeResult(
                    topic_id=topic.request_id,
                    run_id=run.metadata.run_id,
                    question_id=question.question_id,
                    question_text=question.text,
                    grade=grade_value,
                    reasoning=reasoning,
                    confidence=confidence,
                )
            )

        grade_values = [result.grade for result in grades]
        covered = [grade for grade in grade_values if grade >= 4]
        total = len(grade_values)
        return RunScore(
            run_id=run.metadata.run_id,
            topic_id=topic.request_id,
            max_grade=max(grade_values) if grade_values else 0,
            mean_grade=sum(grade_values) / total if total else 0.0,
            nugget_coverage=len(covered) / total if total else 0.0,
            covered_count=len(covered),
            total_nuggets=total,
            grades=grades,
        )

    def _grade_question(
        self,
        question: NuggetQuestion,
        passages: list[str],
    ) -> tuple[int, str, float]:
        best_grade = 0
        best_reasoning = ""
        best_confidence = 0.0
        for passage in passages:
            user = f"question: {question.text}\n\npassage:\n{passage}"
            payload = self._llm.complete_json(system=_GRADE_SYSTEM, user=user)
            grade = int(payload.get("grade", 0))
            reasoning = str(payload.get("reasoning", ""))
            confidence = float(payload.get("confidence", 0.5))
            if grade > best_grade:
                best_grade = grade
                best_reasoning = reasoning
                best_confidence = confidence
        return best_grade, best_reasoning, best_confidence
