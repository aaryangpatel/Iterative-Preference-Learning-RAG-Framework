"""Orchestrate generate → evaluate → report benchmark phases."""

from __future__ import annotations

from dataclasses import dataclass

from research.benchmark.config import BenchmarkConfig
from research.benchmark.evaluator import BenchmarkEvaluation, BenchmarkEvaluator
from research.benchmark.poster_analysis import PosterAnalysis
from research.benchmark.reporting import BenchmarkReporter
from research.benchmark.runner import BenchmarkRunner


@dataclass
class BenchmarkPipelineResult:
    """Output from a full or partial benchmark pipeline run."""

    config: BenchmarkConfig
    evaluation: BenchmarkEvaluation | None = None
    report_path: str | None = None
    poster_path: str | None = None
    run_plan_written: bool = False


class BenchmarkPipeline:
    """Run benchmark phases according to ``BenchmarkConfig.phases``.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark settings.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def run(self) -> BenchmarkPipelineResult:
        """Execute configured phases without implicit live API calls when dry-run."""
        evaluation: BenchmarkEvaluation | None = None
        report_path = None
        poster_path = None
        run_plan_written = False

        if "generate" in self.config.phases:
            runner = BenchmarkRunner(self.config)
            runner.run_generate()
            run_plan_written = self.config.dry_run

        if "evaluate" in self.config.phases:
            evaluator = BenchmarkEvaluator(self.config)
            evaluation = evaluator.run_evaluate()

        if "report" in self.config.phases:
            reporter = BenchmarkReporter(self.config)
            path = reporter.run_report(evaluation)
            report_path = str(path)

        if "poster" in self.config.phases:
            if evaluation is None:
                evaluation = BenchmarkEvaluator(self.config).run_evaluate()
            poster = PosterAnalysis(self.config)
            path = poster.run(evaluation)
            poster_path = str(path)

        return BenchmarkPipelineResult(
            config=self.config,
            evaluation=evaluation,
            report_path=report_path,
            poster_path=poster_path,
            run_plan_written=run_plan_written,
        )
