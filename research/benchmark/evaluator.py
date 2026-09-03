"""Numerical evaluation for benchmark runs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from crucible.eval.citation_eval import CitationEvaluator
from crucible.loaders import load_report_bundle
from crucible.models.report_output import CrucibleReportBundle
from prefnugget.judges.extract_queryonly import QueryOnlyNuggetExtractor
from prefnugget.judges.grader import NuggetGrader
from prefnugget.models.judgment import RunScore
from prefnugget.models.nugget import NuggetQuestionBank

from research.adapters import bundle_to_rag_run, query_to_topic
from research.benchmark.baselines import SYSTEM_IDS
from research.benchmark.config import BenchmarkConfig
from research.benchmark.datasets import RagtimeBenchmarkData, load_benchmark_data, topic_to_prefnugget_topic
from research.judges.report_pairwise import ReportPairwiseJudge


@dataclass
class TopicSystemScore:
    """Evaluation metrics for one system on one topic.

    Parameters
    ----------
    benchmark_id : str
        Benchmark id.
    system_id : str
        System id.
    topic_id : str
        Topic id.
    run_id : str
        Report run id.
    gold_mean_grade : float | None
        Mean 0–5 grade on gold nuggets.
    gold_max_grade : int | None
        MAX grade on gold nuggets.
    gold_nugget_coverage : float | None
        Fraction gold nuggets with grade ≥ 4.
    queryonly_mean_grade : float | None
        Mean grade on QueryOnly nugget bank.
    queryonly_nugget_coverage : float | None
        QueryOnly coverage.
    citation_coverage : float
        Fraction sentences with citations.
    citation_validity_rate : float
        Fraction sentences citing valid doc ids.
    mean_span_overlap : float
        Mean sentence–source overlap.
    ragtime_f1_proxy : float | None
        Harmonic mean of gold coverage and citation validity (AUTO-ARGUE F1 proxy).
    num_sentences : int
        Report sentence count.
    num_citations : int
        Total citation count.
    """

    benchmark_id: str
    system_id: str
    topic_id: str
    run_id: str
    gold_mean_grade: float | None = None
    gold_max_grade: int | None = None
    gold_nugget_coverage: float | None = None
    queryonly_mean_grade: float | None = None
    queryonly_nugget_coverage: float | None = None
    citation_coverage: float = 0.0
    citation_validity_rate: float = 0.0
    mean_span_overlap: float = 0.0
    ragtime_f1_proxy: float | None = None
    num_sentences: int = 0
    num_citations: int = 0


@dataclass
class PairwiseOutcome:
    """Pairwise preference between two systems on one topic."""

    topic_id: str
    system_a: str
    system_b: str
    winner_system: str | None
    confidence: float
    tie: bool


@dataclass
class BenchmarkEvaluation:
    """Full benchmark evaluation output."""

    benchmark_id: str
    topic_scores: list[TopicSystemScore] = field(default_factory=list)
    pairwise: list[PairwiseOutcome] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BenchmarkEvaluator:
    """Grade saved benchmark runs and compute aggregate metrics.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._data = load_benchmark_data(config)
        self._output_root = config.benchmark_output_dir()
        self._grader = NuggetGrader(grading_mode="response")
        max_qo = config.eval_max_queryonly_questions or 20
        self._queryonly = QueryOnlyNuggetExtractor(max_questions=max_qo)
        self._pairwise = ReportPairwiseJudge()

    @property
    def data(self) -> RagtimeBenchmarkData:
        return self._data

    def run_evaluate(self) -> BenchmarkEvaluation:
        """Evaluate all saved runs under ``output/benchmark/{id}/runs/``.

        Returns
        -------
        BenchmarkEvaluation
            Per-topic scores and pairwise outcomes.
        """
        if self.config.dry_run and not self._has_saved_runs():
            evaluation = BenchmarkEvaluation(
                benchmark_id=self.config.benchmark_id,
                metadata={"dry_run": True, "message": "Evaluation skipped in dry-run mode (no saved runs)."},
            )
            self._save_evaluation(evaluation)
            return evaluation

        topic_scores: list[TopicSystemScore] = []
        bundles_by_system_topic: dict[tuple[str, str], CrucibleReportBundle] = {}

        for system_id in self.config.systems:
            for topic in self._data.topics:
                bundle = self._load_run(system_id, topic.query.query_id)
                if bundle is None:
                    continue
                bundles_by_system_topic[(system_id, topic.query.query_id)] = bundle
                topic_scores.append(self._score_bundle(system_id, topic, bundle))

        pairwise = self._pairwise_matrix(bundles_by_system_topic)
        evaluation = BenchmarkEvaluation(
            benchmark_id=self.config.benchmark_id,
            topic_scores=topic_scores,
            pairwise=pairwise,
            metadata={
                "num_topics": len(self._data.topics),
                "systems": list(self.config.systems),
            },
        )
        self._save_evaluation(evaluation)
        return evaluation

    def _score_bundle(
        self,
        system_id: str,
        topic,
        bundle: CrucibleReportBundle,
    ) -> TopicSystemScore:
        valid_doc_ids = {document.doc_id for document in bundle.source_documents}
        citation_report = CitationEvaluator(valid_doc_ids=valid_doc_ids).evaluate(bundle.report)
        run = bundle_to_rag_run(bundle)

        gold_bank = self._data.gold_nuggets_by_topic.get(topic.query.query_id)
        if gold_bank is not None and self.config.eval_max_gold_questions is not None:
            gold_bank = self._truncate_bank(gold_bank, self.config.eval_max_gold_questions)
        gold_score = self._grade_run(topic, run, gold_bank) if gold_bank else None

        queryonly_bank = self._queryonly.extract(query_to_topic(topic.query))
        queryonly_score = self._grade_run(topic, run, queryonly_bank)

        gold_coverage = gold_score.nugget_coverage if gold_score else None
        citation_validity = citation_report.validity_rate
        f1_proxy = None
        if gold_coverage is not None:
            f1_proxy = _harmonic_mean(gold_coverage, citation_validity)

        return TopicSystemScore(
            benchmark_id=self.config.benchmark_id,
            system_id=system_id,
            topic_id=topic.query.query_id,
            run_id=bundle.report.run_id,
            gold_mean_grade=gold_score.mean_grade if gold_score else None,
            gold_max_grade=gold_score.max_grade if gold_score else None,
            gold_nugget_coverage=gold_coverage,
            queryonly_mean_grade=queryonly_score.mean_grade,
            queryonly_nugget_coverage=queryonly_score.nugget_coverage,
            citation_coverage=citation_report.coverage,
            citation_validity_rate=citation_validity,
            mean_span_overlap=citation_report.mean_span_overlap,
            ragtime_f1_proxy=f1_proxy,
            num_sentences=len(bundle.report.sentences),
            num_citations=sum(len(sentence.citations) for sentence in bundle.report.sentences),
        )

    def _truncate_bank(self, bank: NuggetQuestionBank, limit: int) -> NuggetQuestionBank:
        return NuggetQuestionBank(
            topic_id=bank.topic_id,
            questions=bank.questions[:limit],
        )

    def _grade_run(self, topic, run, bank: NuggetQuestionBank | None) -> RunScore | None:
        if bank is None or not bank.questions:
            return None
        pref_topic = topic_to_prefnugget_topic(topic)
        return self._grader.grade(pref_topic, run, bank)

    def _pairwise_matrix(
        self,
        bundles: dict[tuple[str, str], CrucibleReportBundle],
    ) -> list[PairwiseOutcome]:
        outcomes: list[PairwiseOutcome] = []
        primary = "preference_loop_full"
        if primary not in self.config.systems:
            return outcomes
        baselines = [system_id for system_id in self.config.systems if system_id != primary]
        for topic in self._data.topics:
            topic_id = topic.query.query_id
            primary_bundle = bundles.get((primary, topic_id))
            if primary_bundle is None:
                continue
            primary_run = bundle_to_rag_run(primary_bundle)
            pref_topic = topic_to_prefnugget_topic(topic)
            for baseline_id in baselines:
                baseline_bundle = bundles.get((baseline_id, topic_id))
                if baseline_bundle is None:
                    continue
                baseline_run = bundle_to_rag_run(baseline_bundle)
                preference = self._pairwise.judge(pref_topic, primary_run, baseline_run)
                if preference is None:
                    outcomes.append(
                        PairwiseOutcome(
                            topic_id=topic_id,
                            system_a=primary,
                            system_b=baseline_id,
                            winner_system=None,
                            confidence=0.0,
                            tie=True,
                        )
                    )
                else:
                    winner = primary if preference.winner_run_id == primary_run.metadata.run_id else baseline_id
                    outcomes.append(
                        PairwiseOutcome(
                            topic_id=topic_id,
                            system_a=primary,
                            system_b=baseline_id,
                            winner_system=winner,
                            confidence=preference.confidence,
                            tie=False,
                        )
                    )
        return outcomes

    def _load_run(self, system_id: str, topic_id: str) -> CrucibleReportBundle | None:
        path = self._output_root / "runs" / system_id / f"{topic_id}.json"
        if not path.exists():
            return None
        return load_report_bundle(path)

    def _has_saved_runs(self) -> bool:
        runs_root = self._output_root / "runs"
        if not runs_root.exists():
            return False
        return any(runs_root.glob("*/*.json"))

    def _save_evaluation(self, evaluation: BenchmarkEvaluation) -> Path:
        eval_dir = self._output_root / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        path = eval_dir / "scores.json"
        path.write_text(
            json.dumps(
                {
                    "benchmark_id": evaluation.benchmark_id,
                    "topic_scores": [score.__dict__ for score in evaluation.topic_scores],
                    "pairwise": [outcome.__dict__ for outcome in evaluation.pairwise],
                    "metadata": evaluation.metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path


def _harmonic_mean(a: float, b: float) -> float:
    if a + b == 0:
        return 0.0
    return 2 * a * b / (a + b)
