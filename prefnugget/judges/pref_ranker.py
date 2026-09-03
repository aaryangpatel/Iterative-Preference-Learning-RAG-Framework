"""PrefNugget Phase 1: pairwise preference ranking via OpenRouter LLM."""

from __future__ import annotations

from itertools import combinations
from math import gcd

from rag_framework.llm.client import OpenRouterLLM, get_llm_client

from prefnugget.models.judgment import PreferenceResult
from prefnugget.models.response import RagRunRecord
from prefnugget.models.topic import PrefNuggetTopic

_PREF_SYSTEM = """You are a highly experienced and accurate assessor for TREC.
Select the passage that answers the query better. Respond with JSON only:
{"better_passage": 1 or 2 or 0, "confidence": 0.0 to 1.0}
Use 0 when both passages are similar quality (tie)."""

_PREF_NO_TIES_SYSTEM = """You are a highly experienced and accurate assessor for TREC.
Select the passage that answers the query better. Respond with JSON only:
{"better_passage": 1 or 2, "confidence": 0.0 to 1.0}
If both passages are similar, select the simplest and clearest."""


class PrefRanker:
    """Pairwise preference ranker using OpenRouter (PrefJudgment mimic).

    Supports bidirectional judging, tie handling, and stratified sampling.

    Parameters
    ----------
    num_pivot : int
        Number of pivot runs for stratified sampling (default 1).
    num_others : int
        Number of other runs compared against each pivot (default 4).
    allow_ties : bool
        When True, ties (better_passage=0) are dropped from Borda counts.
    bidirectional : bool
        When True, judge each pair in both directions and average.
    llm : OpenRouterLLM | None
        LLM client. Defaults to environment-configured OpenRouter client.
    """

    def __init__(
        self,
        num_pivot: int = 1,
        num_others: int = 4,
        allow_ties: bool = True,
        bidirectional: bool = True,
        llm: OpenRouterLLM | None = None,
    ) -> None:
        self._num_pivot = num_pivot
        self._num_others = num_others
        self._allow_ties = allow_ties
        self._bidirectional = bidirectional
        self._llm = llm or get_llm_client()

    @property
    def name(self) -> str:
        return "prefnugget_llm_ranker"

    def rank_runs(
        self,
        topic: PrefNuggetTopic,
        runs: list[RagRunRecord],
    ) -> tuple[list[str], list[PreferenceResult]]:
        """Rank runs for one topic using Borda scores from LLM pairwise judgments.

        Parameters
        ----------
        topic : PrefNuggetTopic
            Topic context.
        runs : list[RagRunRecord]
            Candidate response runs for the same topic.

        Returns
        -------
        tuple[list[str], list[PreferenceResult]]
            Ranked run ids (best first) and pairwise preference records.
        """
        run_lookup = {run.metadata.run_id: run for run in runs}
        run_ids = [run.metadata.run_id for run in runs]
        preferences: list[PreferenceResult] = []

        for run_a_id, run_b_id in self._select_pairs(run_ids):
            winner_id, confidence = self._judge_pair_bidirectional(
                topic,
                run_lookup[run_a_id].response_text(),
                run_lookup[run_b_id].response_text(),
                run_a_id,
                run_b_id,
            )
            if winner_id is None:
                continue
            loser_id = run_b_id if winner_id == run_a_id else run_a_id
            preferences.append(
                PreferenceResult(
                    topic_id=topic.request_id,
                    winner_run_id=winner_id,
                    loser_run_id=loser_id,
                    confidence=confidence,
                )
            )

        borda: dict[str, int] = {run_id: 0 for run_id in run_ids}
        for preference in preferences:
            borda[preference.winner_run_id] += 1
            borda[preference.loser_run_id] -= 1

        ranked = sorted(run_ids, key=lambda run_id: borda[run_id], reverse=True)
        return ranked, preferences

    def _select_pairs(self, run_ids: list[str]) -> list[tuple[str, str]]:
        """Select pairwise comparisons using pivot + strided sampling."""
        if len(run_ids) <= 2:
            return list(combinations(run_ids, 2))

        pairs: list[tuple[str, str]] = []
        pivot_count = min(self._num_pivot, len(run_ids))
        stride = max(1, len(run_ids) // gcd(len(run_ids), self._num_others or 1))

        for pivot_index in range(pivot_count):
            pivot_id = run_ids[pivot_index]
            others_seen = 0
            offset = 0
            while others_seen < self._num_others and offset < len(run_ids):
                other_index = (pivot_index + 1 + offset * stride) % len(run_ids)
                offset += 1
                other_id = run_ids[other_index]
                if other_id == pivot_id:
                    continue
                pair = tuple(sorted((pivot_id, other_id)))
                if pair not in pairs:
                    pairs.append((pair[0], pair[1]))
                    others_seen += 1
        return pairs

    def _judge_pair_bidirectional(
        self,
        topic: PrefNuggetTopic,
        passage_a: str,
        passage_b: str,
        run_a_id: str,
        run_b_id: str,
    ) -> tuple[str | None, float]:
        winner, confidence = self._judge_pair(topic, passage_a, passage_b, run_a_id, run_b_id)
        if not self._bidirectional:
            return winner, confidence

        winner_flip, confidence_flip = self._judge_pair(
            topic, passage_b, passage_a, run_b_id, run_a_id
        )
        if winner is None and winner_flip is None:
            return None, 0.0
        if winner == run_a_id and winner_flip == run_b_id:
            return None, (confidence + confidence_flip) / 2.0
        if winner == run_b_id and winner_flip == run_a_id:
            return None, (confidence + confidence_flip) / 2.0
        if winner is not None:
            return winner, (confidence + confidence_flip) / 2.0
        return winner_flip, (confidence + confidence_flip) / 2.0

    def _judge_pair(
        self,
        topic: PrefNuggetTopic,
        passage_a: str,
        passage_b: str,
        run_a_id: str,
        run_b_id: str,
    ) -> tuple[str | None, float]:
        system = _PREF_SYSTEM if self._allow_ties else _PREF_NO_TIES_SYSTEM
        user = (
            f"query_title: {topic.title}\n"
            f"query_background: {topic.background}\n"
            f"query_problem: {topic.problem_statement}\n\n"
            f"passage_1:\n{passage_a}\n\n"
            f"passage_2:\n{passage_b}"
        )
        payload = self._llm.complete_json(system=system, user=user)
        better = int(payload.get("better_passage", 0))
        confidence = float(payload.get("confidence", 0.5))
        if better == 0 and self._allow_ties:
            return None, confidence
        if better == 1:
            return run_a_id, confidence
        if better == 2:
            return run_b_id, confidence
        return None, 0.0
