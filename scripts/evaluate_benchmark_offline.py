#!/usr/bin/env python3
"""Evaluate saved benchmark runs without OpenRouter (citation + lexical gold proxy)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crucible.eval.citation_eval import CitationEvaluator
from crucible.loaders import load_report_bundle
from prefnugget.models.nugget import NuggetQuestion, NuggetQuestionBank

from research.benchmark.baselines import SYSTEM_IDS
from research.benchmark.config import BenchmarkConfig
from research.benchmark.datasets import load_benchmark_data, load_gold_nuggets
from research.benchmark.evaluator import BenchmarkEvaluation, PairwiseOutcome, TopicSystemScore
from research.benchmark.poster_analysis import PosterAnalysis
from research.benchmark.reporting import BenchmarkReporter
from rag_framework.similarity import cosine_similarity, tfidf_vectors

ROOT = Path(__file__).resolve().parent.parent
PRIMARY_SYSTEM = "preference_loop_full"
GRADE_THRESHOLDS = [(0.45, 5), (0.35, 4), (0.25, 3), (0.15, 2), (0.08, 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline benchmark evaluation (no LLM)")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "benchmark_ragtime_pilot.yml",
    )
    parser.add_argument("--poster", action="store_true", help="Regenerate poster figures")
    return parser.parse_args()


def _grade_from_similarity(similarity: float) -> int:
    for threshold, grade in GRADE_THRESHOLDS:
        if similarity >= threshold:
            return grade
    return 0


def _lexical_nugget_scores(
    report_text: str,
    bank: NuggetQuestionBank | None,
) -> tuple[float | None, float | None, int | None]:
    if bank is None or not bank.questions:
        return None, None, None
    vectors = tfidf_vectors([report_text] + [question.text for question in bank.questions])
    report_vector = vectors[0]
    grades: list[int] = []
    for question_vector in vectors[1:]:
        similarity = cosine_similarity(report_vector, question_vector)
        grades.append(_grade_from_similarity(similarity))
    covered = sum(1 for grade in grades if grade >= 4) / len(grades)
    mean_grade = sum(grades) / len(grades)
    return mean_grade, covered, max(grades)


def _harmonic_mean(left: float, right: float) -> float:
    if left + right == 0:
        return 0.0
    return 2 * left * right / (left + right)


def _load_run(output_root: Path, system_id: str, topic_id: str):
    path = output_root / "runs" / system_id / f"{topic_id}.json"
    if not path.exists():
        return None
    return load_report_bundle(path)


def _pairwise_offline(
    bundles: dict[tuple[str, str], object],
    gold_nuggets: dict[str, NuggetQuestionBank],
    topic_ids: list[str],
    primary: str = PRIMARY_SYSTEM,
) -> list[PairwiseOutcome]:
    outcomes: list[PairwiseOutcome] = []
    baselines = [system_id for system_id in SYSTEM_IDS if system_id != primary and system_id in {key[0] for key in bundles}]
    for topic_id in topic_ids:
        primary_bundle = bundles.get((primary, topic_id))
        gold_bank = gold_nuggets.get(topic_id)
        if primary_bundle is None or gold_bank is None:
            continue
        primary_mean, primary_coverage, _ = _lexical_nugget_scores(primary_bundle.report_text(), gold_bank)
        primary_tuple = (primary_coverage or 0.0, primary_mean or 0.0, len(primary_bundle.report.sentences))
        for baseline_id in baselines:
            baseline_bundle = bundles.get((baseline_id, topic_id))
            if baseline_bundle is None:
                continue
            baseline_mean, baseline_coverage, _ = _lexical_nugget_scores(
                baseline_bundle.report_text(),
                gold_bank,
            )
            baseline_tuple = (baseline_coverage or 0.0, baseline_mean or 0.0, len(baseline_bundle.report.sentences))
            if primary_tuple == baseline_tuple:
                outcomes.append(
                    PairwiseOutcome(
                        topic_id=topic_id,
                        system_a=primary,
                        system_b=baseline_id,
                        winner_system=None,
                        confidence=0.5,
                        tie=True,
                    )
                )
                continue
            winner = primary if primary_tuple > baseline_tuple else baseline_id
            outcomes.append(
                PairwiseOutcome(
                    topic_id=topic_id,
                    system_a=primary,
                    system_b=baseline_id,
                    winner_system=winner,
                    confidence=0.7,
                    tie=False,
                )
            )
    return outcomes


def evaluate_offline(config: BenchmarkConfig) -> BenchmarkEvaluation:
    data = load_benchmark_data(config)
    gold_path = config.resolve_data_path(config.gold_nuggets_file)
    gold_nuggets = load_gold_nuggets(gold_path)
    output_root = config.benchmark_output_dir()

    topic_scores: list[TopicSystemScore] = []
    bundles: dict[tuple[str, str], object] = {}
    evaluated_topics: set[str] = set()

    for system_id in config.systems:
        for topic in data.topics:
            topic_id = topic.query.query_id
            bundle = _load_run(output_root, system_id, topic_id)
            if bundle is None:
                continue
            evaluated_topics.add(topic_id)
            bundles[(system_id, topic_id)] = bundle

            report_text = bundle.report_text()
            gold_bank = gold_nuggets.get(topic_id)
            gold_mean, gold_cov, gold_max = _lexical_nugget_scores(report_text, gold_bank)

            report_questions = [
                str(sentence.metadata.get("question", ""))
                for sentence in bundle.report.sentences
                if sentence.metadata.get("question")
            ]
            queryonly_bank = None
            if report_questions:
                queryonly_bank = NuggetQuestionBank(
                    topic_id=topic_id,
                    questions=[
                        NuggetQuestion(question_id=f"{topic_id}-qo-{index}", text=text)
                        for index, text in enumerate(report_questions)
                    ],
                )
            qo_mean, qo_cov, _ = _lexical_nugget_scores(report_text, queryonly_bank)

            valid_doc_ids = {document.doc_id for document in bundle.source_documents}
            citation_report = CitationEvaluator(valid_doc_ids=valid_doc_ids).evaluate(bundle.report)
            f1_proxy = _harmonic_mean(gold_cov or 0.0, citation_report.validity_rate) if gold_cov is not None else None

            topic_scores.append(
                TopicSystemScore(
                    benchmark_id=config.benchmark_id,
                    system_id=system_id,
                    topic_id=topic_id,
                    run_id=bundle.report.run_id,
                    gold_mean_grade=gold_mean,
                    gold_max_grade=gold_max,
                    gold_nugget_coverage=gold_cov,
                    queryonly_mean_grade=qo_mean,
                    queryonly_nugget_coverage=qo_cov,
                    citation_coverage=citation_report.coverage,
                    citation_validity_rate=citation_report.validity_rate,
                    mean_span_overlap=citation_report.mean_span_overlap,
                    ragtime_f1_proxy=f1_proxy,
                    num_sentences=len(bundle.report.sentences),
                    num_citations=sum(len(sentence.citations) for sentence in bundle.report.sentences),
                )
            )

    pairwise = _pairwise_offline(
        bundles,
        gold_nuggets,
        sorted(evaluated_topics, key=int),
        primary=PRIMARY_SYSTEM if PRIMARY_SYSTEM in config.systems else config.systems[0],
    )
    return BenchmarkEvaluation(
        benchmark_id=config.benchmark_id,
        topic_scores=topic_scores,
        pairwise=pairwise,
        metadata={
            "num_topics": len(evaluated_topics),
            "systems": list(config.systems),
            "grading_mode": "offline_lexical_proxy",
            "note": "Gold grades use TF-IDF cosine similarity proxy; add OpenRouter credits for LLM grading.",
        },
    )


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig.from_yaml(args.config)
    evaluation = evaluate_offline(config)
    output_root = config.benchmark_output_dir()
    eval_dir = output_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    scores_path = eval_dir / "scores.json"
    payload = {
        "benchmark_id": evaluation.benchmark_id,
        "topic_scores": [score.__dict__ for score in evaluation.topic_scores],
        "pairwise": [outcome.__dict__ for outcome in evaluation.pairwise],
        "metadata": evaluation.metadata,
    }
    scores_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    reporter = BenchmarkReporter(config)
    report_path = reporter.run_report(evaluation)
    print(f"Scored {len(evaluation.topic_scores)} system-topic pairs across {evaluation.metadata['num_topics']} topics")
    print(f"Mode: {evaluation.metadata['grading_mode']}")
    print(f"Scores: {scores_path}")
    print(f"Report: {report_path}")

    if args.poster:
        poster_path = PosterAnalysis(config).run(evaluation)
        print(f"Poster: {poster_path}")


if __name__ == "__main__":
    main()
