"""PrefNugget three-phase judge workflow (rank → extract → grade) via OpenRouter."""

from __future__ import annotations

from pathlib import Path

from rag_framework.llm.client import OpenRouterLLM, get_llm_client

from prefnugget.judges.extract_contrastive import ContrastiveNuggetExtractor
from prefnugget.judges.extract_queryonly import QueryOnlyNuggetExtractor
from prefnugget.judges.grader import NuggetGrader
from prefnugget.judges.pref_ranker import PrefRanker
from prefnugget.loaders import load_rag_runs, load_topics, load_truth_leaderboard
from prefnugget.models.judgment import PreferenceResult, RunScore
from prefnugget.models.nugget import NuggetQuestionBank
from prefnugget.models.response import RagRunRecord
from prefnugget.models.topic import PrefNuggetTopic
from prefnugget.pipeline.config import PrefNuggetWorkflowConfig, PrefNuggetWorkflowResult
from prefnugget.pipeline.output import write_config_snapshot, write_eval_leaderboard, write_nugget_banks_jsonl


class PrefNuggetWorkflow:
    """End-to-end PrefNugget judge workflow using OpenRouter LLM.

    Parameters
    ----------
    config : PrefNuggetWorkflowConfig | None
        Workflow variant settings.
    llm : OpenRouterLLM | None
        LLM client. Defaults to environment-configured OpenRouter client.
    """

    def __init__(
        self,
        config: PrefNuggetWorkflowConfig | None = None,
        llm: OpenRouterLLM | None = None,
    ) -> None:
        self.config = config or PrefNuggetWorkflowConfig()
        self._llm = llm or get_llm_client()
        grading_mode = self._resolve_grading_mode()
        self.ranker = PrefRanker(
            num_pivot=self.config.num_pivot,
            num_others=self.config.num_others,
            allow_ties=self.config.allow_ties,
            bidirectional=self.config.bidirectional,
            llm=self._llm,
        )
        self.contrastive_extractor = ContrastiveNuggetExtractor(
            max_questions=self.config.max_questions,
            llm=self._llm,
        )
        self.queryonly_extractor = QueryOnlyNuggetExtractor(
            max_questions=self.config.max_questions,
            llm=self._llm,
        )
        self.grader = NuggetGrader(grading_mode=grading_mode, llm=self._llm)

    def _resolve_grading_mode(self) -> str:
        if self.config.variant == "contrastive_docs":
            return "docs"
        return self.config.grading_mode

    def run_topic(
        self,
        topic: PrefNuggetTopic,
        runs: list[RagRunRecord],
        preferences: list[PreferenceResult] | None = None,
    ) -> tuple[NuggetQuestionBank, list[RunScore], list[PreferenceResult]]:
        """Execute rank → extract → grade for one topic.

        Parameters
        ----------
        topic : PrefNuggetTopic
            Topic to judge.
        runs : list[RagRunRecord]
            All runs for this topic.
        preferences : list[PreferenceResult] | None
            Optional precomputed preferences (skips Phase 1).

        Returns
        -------
        tuple[NuggetQuestionBank, list[RunScore], list[PreferenceResult]]
            Extracted nugget bank, per-run grades, and preferences.
        """
        run_lookup = {run.metadata.run_id: run for run in runs}
        if preferences is None:
            ranked_run_ids, preferences = self.ranker.rank_runs(topic, runs)
        else:
            topic_prefs = [pref for pref in preferences if pref.topic_id == topic.request_id]
            ranked_run_ids = self._rank_from_preferences(topic_prefs, list(run_lookup.keys()))

        if self.config.variant == "queryonly":
            bank = self.queryonly_extractor.extract_iterative(
                topic,
                iterations=self.config.queryonly_iterations,
            )
        else:
            topic_prefs = [pref for pref in preferences if pref.topic_id == topic.request_id]
            bank = self.contrastive_extractor.extract_from_preferences(
                topic, topic_prefs, run_lookup
            )

        scores = [
            self.grader.grade(topic, run_lookup[run_id], bank)
            for run_id in ranked_run_ids
            if run_id in run_lookup
        ]
        return bank, scores, preferences

    def run_dataset(
        self,
        topics_path: Path,
        runs_glob_dir: Path,
    ) -> PrefNuggetWorkflowResult:
        """Run workflow across a kiddie-style dataset directory.

        Parameters
        ----------
        topics_path : Path
            ``topics/kiddie-topics.jsonl``.
        runs_glob_dir : Path
            Directory containing ``run*.jsonl`` files.

        Returns
        -------
        PrefNuggetWorkflowResult
            Banks, scores, and preferences for all topics.
        """
        topics = {topic.request_id: topic for topic in load_topics(topics_path)}
        runs_by_topic: dict[str, list[RagRunRecord]] = {topic_id: [] for topic_id in topics}

        for run_path in sorted(runs_glob_dir.glob("run*.jsonl")):
            for record in load_rag_runs(run_path):
                topic_id = record.metadata.topic_id
                if topic_id in runs_by_topic:
                    runs_by_topic[topic_id].append(record)

        result = PrefNuggetWorkflowResult(config=self.config)
        for topic_id, topic in topics.items():
            topic_runs = runs_by_topic[topic_id]
            if topic_runs:
                bank, scores, preferences = self.run_topic(topic, topic_runs)
                result.banks[topic_id] = bank
                result.scores_by_topic[topic_id] = scores
                result.preferences.extend(preferences)
        return result

    def write_outputs(self, result: PrefNuggetWorkflowResult, out_dir: Path, variant: str) -> None:
        """Write eval leaderboard, nugget banks, and config snapshot.

        Parameters
        ----------
        result : PrefNuggetWorkflowResult
            Workflow result from ``run_dataset``.
        out_dir : Path
            Output directory.
        variant : str
            Variant label for filenames.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        write_eval_leaderboard(result.scores_by_topic, out_dir / f"{variant}.eval.txt")
        write_nugget_banks_jsonl(result.banks, out_dir / f"{variant}.nuggets.jsonl")
        write_config_snapshot(result.config, out_dir / f"{variant}.config.yml")

    def compare_to_truth(
        self,
        scores: list[RunScore],
        truth_path: Path,
    ) -> list[dict]:
        """Compare predicted run ordering to kiddie fake leaderboard.

        Parameters
        ----------
        scores : list[RunScore]
            Graded runs for one topic.
        truth_path : Path
            ``kiddie_fake.eval.ir_measures.txt``.

        Returns
        -------
        list[dict]
            Per-run predicted vs truth RELEVANCE values.
        """
        truth = load_truth_leaderboard(truth_path)
        comparisons: list[dict] = []
        for score in scores:
            truth_value = truth.get((score.run_id, score.topic_id))
            comparisons.append(
                {
                    "run_id": score.run_id,
                    "topic_id": score.topic_id,
                    "predicted_nugget_coverage": score.nugget_coverage,
                    "predicted_mean_grade": score.mean_grade,
                    "truth_relevance": truth_value,
                }
            )
        return comparisons

    @staticmethod
    def _rank_from_preferences(
        preferences: list[PreferenceResult],
        run_ids: list[str],
    ) -> list[str]:
        borda: dict[str, int] = {run_id: 0 for run_id in run_ids}
        for preference in preferences:
            if preference.winner_run_id in borda:
                borda[preference.winner_run_id] += 1
            if preference.loser_run_id in borda:
                borda[preference.loser_run_id] -= 1
        return sorted(run_ids, key=lambda run_id: borda[run_id], reverse=True)
