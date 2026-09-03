"""PrefNugget Phase 2a: contrastive nugget question extraction via OpenRouter."""

from __future__ import annotations

from rag_framework.llm.client import OpenRouterLLM, get_llm_client

from prefnugget.judges.question_tracker import QuestionTracker
from prefnugget.models.judgment import PreferenceResult
from prefnugget.models.nugget import NuggetQuestion, NuggetQuestionBank
from prefnugget.models.response import RagRunRecord
from prefnugget.models.topic import PrefNuggetTopic

_CONTRASTIVE_SYSTEM = """Compare Winner vs Loser RAG responses for a query. Focus on relevance, correctness, completeness.
From given_exam_questions, identify or generate questions the Winner addresses much better than the Loser.
Reuse questions where possible. New differentiating_questions must be brief, atomic questions about information
the Winner handles much better. Avoid generic quality questions. Make questions self-contained.
Respond with JSON only:
{"differentiating_questions": ["q1", "q2"], "reasoning": "...", "confidence": 0.0 to 1.0}"""


class ContrastiveNuggetExtractor:
    """Extract differentiating questions from winner/loser pairs via OpenRouter.

    Mimics ``IterativeExtractDifferentiatingNuggets`` with ``QuestionTracker``.

    Parameters
    ----------
    max_questions : int
        Maximum questions per topic (PrefNugget default 20).
    max_per_pair : int
        Max new questions per winner/loser pair (default 2).
    llm : OpenRouterLLM | None
        LLM client. Defaults to environment-configured OpenRouter client.
    """

    def __init__(
        self,
        max_questions: int = 20,
        max_per_pair: int = 2,
        llm: OpenRouterLLM | None = None,
    ) -> None:
        self._max_questions = max_questions
        self._max_per_pair = max_per_pair
        self._llm = llm or get_llm_client()

    @property
    def name(self) -> str:
        return "prefnugget_llm_contrastive_extractor"

    def extract(
        self,
        topic: PrefNuggetTopic,
        winner: RagRunRecord,
        loser: RagRunRecord,
        existing: NuggetQuestionBank | None = None,
    ) -> NuggetQuestionBank:
        """Extract differentiating questions from one winner/loser pair.

        Parameters
        ----------
        topic : PrefNuggetTopic
            Topic context.
        winner : RagRunRecord
            Preferred response.
        loser : RagRunRecord
            Non-preferred response.
        existing : NuggetQuestionBank | None
            Prior questions to reuse/extend.

        Returns
        -------
        NuggetQuestionBank
            Updated question bank.
        """
        bank = existing or NuggetQuestionBank(topic_id=topic.request_id)
        given = [question.text for question in bank.questions]
        user = (
            f"query_title: {topic.title}\n"
            f"query_background: {topic.background}\n\n"
            f"winner_passage:\n{winner.response_text()}\n\n"
            f"loser_passage:\n{loser.response_text()}\n\n"
            f"given_exam_questions: {given}\n"
            f"max_new_questions: {self._max_per_pair}"
        )
        payload = self._llm.complete_json(system=_CONTRASTIVE_SYSTEM, user=user)
        raw_questions = payload.get("differentiating_questions", [])
        confidence = float(payload.get("confidence", 0.8))

        new_questions: list[NuggetQuestion] = []
        for index, text in enumerate(raw_questions[: self._max_per_pair]):
            question_text = str(text).strip()
            if not question_text:
                continue
            new_questions.append(
                NuggetQuestion(
                    question_id=f"{topic.request_id}-contrastive-{len(bank.questions) + index}",
                    text=question_text,
                    extraction_method="contrastive",
                    source_run_id=winner.metadata.run_id,
                    confidence=confidence,
                    metadata={
                        "winner_run_id": winner.metadata.run_id,
                        "loser_run_id": loser.metadata.run_id,
                        "llm_model": self._llm.model,
                        "reasoning": payload.get("reasoning", ""),
                    },
                )
            )

        merged = bank.questions + new_questions
        return NuggetQuestionBank(
            topic_id=topic.request_id,
            questions=merged[: self._max_questions],
            metadata={"extractor": self.name, "llm_model": self._llm.model},
        ).deduplicate()

    def extract_iterative(
        self,
        topic: PrefNuggetTopic,
        pairs: list[tuple[RagRunRecord, RagRunRecord]],
        tracker: QuestionTracker | None = None,
    ) -> NuggetQuestionBank:
        """Iteratively extract questions across multiple winner/loser pairs.

        Parameters
        ----------
        topic : PrefNuggetTopic
            Topic context.
        pairs : list[tuple[RagRunRecord, RagRunRecord]]
            Ordered winner/loser pairs.
        tracker : QuestionTracker | None
            Shared question tracker. Created when omitted.

        Returns
        -------
        NuggetQuestionBank
            Deduplicated question bank up to ``max_questions``.
        """
        tracker = tracker or QuestionTracker()
        bank = NuggetQuestionBank(topic_id=topic.request_id)

        for winner, loser in pairs:
            if tracker.is_done(topic.request_id):
                break
            bank = self.extract(topic, winner, loser, existing=bank)
            tracker.add_all(topic.request_id, [question.text for question in bank.questions])
            tracker.check_and_mark_done(topic.request_id, self._max_questions)

        bank.metadata["tracker_counts"] = dict(tracker._counts.get(topic.request_id, {}))
        return bank

    def extract_from_preferences(
        self,
        topic: PrefNuggetTopic,
        preferences: list[PreferenceResult],
        run_lookup: dict[str, RagRunRecord],
    ) -> NuggetQuestionBank:
        """Build pairs from preference results and run iterative extraction.

        Parameters
        ----------
        topic : PrefNuggetTopic
            Topic context.
        preferences : list[PreferenceResult]
            Phase 1 preference judgments.
        run_lookup : dict[str, RagRunRecord]
            run_id -> run record.

        Returns
        -------
        NuggetQuestionBank
            Extracted nugget question bank.
        """
        pairs: list[tuple[RagRunRecord, RagRunRecord]] = []
        for preference in preferences:
            if preference.topic_id != topic.request_id:
                continue
            winner = run_lookup.get(preference.winner_run_id)
            loser = run_lookup.get(preference.loser_run_id)
            if winner is not None and loser is not None:
                pairs.append((winner, loser))
        return self.extract_iterative(topic, pairs)
