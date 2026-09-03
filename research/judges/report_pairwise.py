"""Pairwise LLM judge for full CRUCIBLE reports."""

from __future__ import annotations

from rag_framework.llm.client import OpenRouterLLM, get_llm_client

from prefnugget.models.judgment import PreferenceResult
from prefnugget.models.response import RagRunRecord
from prefnugget.models.topic import PrefNuggetTopic

_REPORT_PREF_SYSTEM = """You are a highly experienced TREC assessor comparing two cited RAG reports for the same query.
Select the report that answers the query better overall: more complete, accurate, well-cited, and useful.
Respond with JSON only:
{"better_report": 1 or 2 or 0, "confidence": 0.0 to 1.0, "reasoning": "brief explanation"}
Use 0 only when both reports are essentially equivalent."""


class ReportPairwiseJudge:
    """Compare two full reports (not single passages) via OpenRouter LLM.

    Parameters
    ----------
    llm : OpenRouterLLM | None
        LLM client. Defaults to environment-configured OpenRouter client.
    allow_ties : bool
        When True, ``better_report=0`` yields no winner.
    """

    def __init__(
        self,
        llm: OpenRouterLLM | None = None,
        allow_ties: bool = True,
    ) -> None:
        self._llm = llm or get_llm_client()
        self._allow_ties = allow_ties

    @property
    def name(self) -> str:
        return "research_report_pairwise_judge"

    def judge(
        self,
        topic: PrefNuggetTopic,
        report_a: RagRunRecord,
        report_b: RagRunRecord,
    ) -> PreferenceResult | None:
        """Judge which report is better for a topic.

        Parameters
        ----------
        topic : PrefNuggetTopic
            Query context.
        report_a : RagRunRecord
            First report (maps to ``better_report=1``).
        report_b : RagRunRecord
            Second report (maps to ``better_report=2``).

        Returns
        -------
        PreferenceResult | None
            Winner/loser preference, or None on tie.
        """
        user = (
            f"query_title: {topic.title}\n"
            f"query_background: {topic.background}\n"
            f"query_problem: {topic.problem_statement}\n\n"
            f"report_1 (run_id={report_a.metadata.run_id}):\n{report_a.response_text()}\n\n"
            f"report_2 (run_id={report_b.metadata.run_id}):\n{report_b.response_text()}"
        )
        payload = self._llm.complete_json(system=_REPORT_PREF_SYSTEM, user=user)
        better = int(payload.get("better_report", 0))
        confidence = float(payload.get("confidence", 0.5))

        if better == 0 and self._allow_ties:
            return None
        if better == 1:
            return PreferenceResult(
                topic_id=topic.request_id,
                winner_run_id=report_a.metadata.run_id,
                loser_run_id=report_b.metadata.run_id,
                confidence=confidence,
            )
        if better == 2:
            return PreferenceResult(
                topic_id=topic.request_id,
                winner_run_id=report_b.metadata.run_id,
                loser_run_id=report_a.metadata.run_id,
                confidence=confidence,
            )
        return None
