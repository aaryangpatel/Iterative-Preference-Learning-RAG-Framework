"""PrefNugget Phase 2c: query-only nugget question generation via OpenRouter."""

from __future__ import annotations

from rag_framework.llm.client import OpenRouterLLM, get_llm_client

from prefnugget.judges.question_tracker import QuestionTracker
from prefnugget.models.nugget import NuggetQuestion, NuggetQuestionBank
from prefnugget.models.topic import PrefNuggetTopic

_QUERYONLY_SYSTEM = """For a query as title, problem statement, and user background, generate brief atomic
questions that target query-essential information which a good RAG response should answer well.
Avoid generic quality questions. Make questions self-contained.
Reuse given_exam_questions where possible and add new questions only if needed.
Respond with JSON only: {"questions": ["question1", "question2", ...], "confidence": 0.0 to 1.0}"""


class QueryOnlyNuggetExtractor:
    """Generate exam questions from topic fields using OpenRouter LLM.

    Mimics ``IterativeGenerateNuggetQuestionsReportRequest``.

    Parameters
    ----------
    max_questions : int
        Maximum questions per topic.
    llm : OpenRouterLLM | None
        LLM client. Defaults to environment-configured OpenRouter client.
    """

    def __init__(
        self,
        max_questions: int = 20,
        llm: OpenRouterLLM | None = None,
    ) -> None:
        self._max_questions = max_questions
        self._llm = llm or get_llm_client()

    @property
    def name(self) -> str:
        return "prefnugget_llm_queryonly_extractor"

    def extract(
        self,
        topic: PrefNuggetTopic,
        existing: NuggetQuestionBank | None = None,
    ) -> NuggetQuestionBank:
        """Generate parametric nugget questions from a topic.

        Parameters
        ----------
        topic : PrefNuggetTopic
            Topic with title, background, problem statement.
        existing : NuggetQuestionBank | None
            Prior questions to extend.

        Returns
        -------
        NuggetQuestionBank
            Generated atomic questions.
        """
        given = [question.text for question in (existing.questions if existing else [])]
        user = (
            f"query_title: {topic.title}\n"
            f"query_background: {topic.background}\n"
            f"query_problem: {topic.problem_statement}\n"
            f"given_exam_questions: {given}\n"
            f"max_questions: {self._max_questions}"
        )
        payload = self._llm.complete_json(system=_QUERYONLY_SYSTEM, user=user, max_tokens=1024, retries=6)
        raw_questions = payload.get("questions", [])
        confidence = float(payload.get("confidence", 0.8))

        start_index = len(given)
        questions: list[NuggetQuestion] = list(existing.questions) if existing else []
        for index, text in enumerate(raw_questions[: self._max_questions]):
            question_text = str(text).strip()
            if not question_text or question_text in given:
                continue
            questions.append(
                NuggetQuestion(
                    question_id=f"{topic.request_id}-q{start_index + index}",
                    text=question_text,
                    extraction_method="queryonly",
                    confidence=confidence,
                    metadata={"llm_model": self._llm.model},
                )
            )

        return NuggetQuestionBank(
            topic_id=topic.request_id,
            questions=questions[: self._max_questions],
            metadata={"extractor": self.name, "llm_model": self._llm.model},
        ).deduplicate()

    def extract_iterative(
        self,
        topic: PrefNuggetTopic,
        iterations: int = 3,
        tracker: QuestionTracker | None = None,
    ) -> NuggetQuestionBank:
        """Run multiple extraction rounds until question cap is reached.

        Parameters
        ----------
        topic : PrefNuggetTopic
            Topic to extract from.
        iterations : int
            Maximum extraction rounds.
        tracker : QuestionTracker | None
            Shared tracker. Created when omitted.

        Returns
        -------
        NuggetQuestionBank
            Deduplicated question bank.
        """
        tracker = tracker or QuestionTracker()
        bank: NuggetQuestionBank | None = None

        for _ in range(iterations):
            if tracker.is_done(topic.request_id):
                break
            bank = self.extract(topic, existing=bank)
            tracker.add_all(topic.request_id, [question.text for question in bank.questions])
            tracker.check_and_mark_done(topic.request_id, self._max_questions)

        return bank or NuggetQuestionBank(topic_id=topic.request_id)
