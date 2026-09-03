"""Poster-ready tables and summaries from benchmark evaluation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from research.benchmark.config import BenchmarkConfig
from research.benchmark.evaluator import BenchmarkEvaluation, BenchmarkEvaluator, TopicSystemScore


@dataclass
class SystemAggregate:
    """Macro-averaged metrics for one system."""

    system_id: str
    num_topics: int
    gold_mean_grade: float | None
    gold_nugget_coverage: float | None
    queryonly_nugget_coverage: float | None
    citation_validity_rate: float | None
    ragtime_f1_proxy: float | None
    pairwise_win_rate: float | None


class BenchmarkReporter:
    """Build summary tables from evaluation scores.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._output_root = config.benchmark_output_dir()

    def run_report(self, evaluation: BenchmarkEvaluation | None = None) -> Path:
        """Write Markdown and CSV summary tables.

        Parameters
        ----------
        evaluation : BenchmarkEvaluation | None
            Precomputed evaluation. Loaded from disk when omitted.

        Returns
        -------
        Path
            Path to ``summary_table.md``.
        """
        if evaluation is None:
            evaluation = self._load_evaluation()
        aggregates = aggregate_by_system(evaluation)
        md_path = self._output_root / "eval" / "summary_table.md"
        csv_path = self._output_root / "eval" / "summary_table.csv"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_render_markdown(self.config.benchmark_id, aggregates, evaluation), encoding="utf-8")
        _write_csv(csv_path, aggregates)
        return md_path

    def _load_evaluation(self) -> BenchmarkEvaluation:
        path = self._output_root / "eval" / "scores.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        topic_scores = [TopicSystemScore(**score) for score in payload.get("topic_scores", [])]
        from research.benchmark.evaluator import PairwiseOutcome

        pairwise = [PairwiseOutcome(**item) for item in payload.get("pairwise", [])]
        return BenchmarkEvaluation(
            benchmark_id=payload["benchmark_id"],
            topic_scores=topic_scores,
            pairwise=pairwise,
            metadata=payload.get("metadata", {}),
        )


def aggregate_by_system(evaluation: BenchmarkEvaluation) -> list[SystemAggregate]:
    """Compute macro averages per system."""
    by_system: dict[str, list[TopicSystemScore]] = defaultdict(list)
    for score in evaluation.topic_scores:
        by_system[score.system_id].append(score)

    win_counts: dict[str, list[bool]] = defaultdict(list)
    for outcome in evaluation.pairwise:
        if outcome.tie:
            continue
        win_counts[outcome.system_a].append(outcome.winner_system == outcome.system_a)

    aggregates: list[SystemAggregate] = []
    for system_id, scores in sorted(by_system.items()):
        aggregates.append(
            SystemAggregate(
                system_id=system_id,
                num_topics=len(scores),
                gold_mean_grade=_mean_optional([score.gold_mean_grade for score in scores]),
                gold_nugget_coverage=_mean_optional([score.gold_nugget_coverage for score in scores]),
                queryonly_nugget_coverage=_mean_optional([score.queryonly_nugget_coverage for score in scores]),
                citation_validity_rate=_mean_optional([score.citation_validity_rate for score in scores]),
                ragtime_f1_proxy=_mean_optional([score.ragtime_f1_proxy for score in scores]),
                pairwise_win_rate=_mean_optional([1.0 if won else 0.0 for won in win_counts.get(system_id, [])])
                if win_counts.get(system_id)
                else None,
            )
        )
    return aggregates


def _mean_optional(values: list[float | None]) -> float | None:
    kept = [value for value in values if value is not None]
    if not kept:
        return None
    return mean(kept)


def _render_markdown(benchmark_id: str, aggregates: list[SystemAggregate], evaluation: BenchmarkEvaluation) -> str:
    lines = [
        f"# Benchmark Summary: {benchmark_id}",
        "",
        f"Topics evaluated: {evaluation.metadata.get('num_topics', 'n/a')}",
        "",
        "| System | Topics | Gold mean grade | Gold coverage | QueryOnly coverage | Citation validity | F1 proxy | Pairwise win rate |",
        "|--------|--------|-----------------|---------------|------------------|-------------------|----------|-------------------|",
    ]
    for row in aggregates:
        lines.append(
            "| {system} | {topics} | {gold_mean} | {gold_cov} | {qo_cov} | {cite} | {f1} | {win} |".format(
                system=row.system_id,
                topics=row.num_topics,
                gold_mean=_fmt(row.gold_mean_grade),
                gold_cov=_fmt(row.gold_nugget_coverage),
                qo_cov=_fmt(row.queryonly_nugget_coverage),
                cite=_fmt(row.citation_validity_rate),
                f1=_fmt(row.ragtime_f1_proxy),
                win=_fmt(row.pairwise_win_rate),
            )
        )
    if evaluation.metadata.get("dry_run"):
        lines.extend(["", "_Dry-run mode: no live scores computed._"])
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, aggregates: list[SystemAggregate]) -> None:
    fieldnames = [
        "system_id",
        "num_topics",
        "gold_mean_grade",
        "gold_nugget_coverage",
        "queryonly_nugget_coverage",
        "citation_validity_rate",
        "ragtime_f1_proxy",
        "pairwise_win_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregates:
            writer.writerow(
                {
                    "system_id": row.system_id,
                    "num_topics": row.num_topics,
                    "gold_mean_grade": row.gold_mean_grade,
                    "gold_nugget_coverage": row.gold_nugget_coverage,
                    "queryonly_nugget_coverage": row.queryonly_nugget_coverage,
                    "citation_validity_rate": row.citation_validity_rate,
                    "ragtime_f1_proxy": row.ragtime_f1_proxy,
                    "pairwise_win_rate": row.pairwise_win_rate,
                }
            )


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"
