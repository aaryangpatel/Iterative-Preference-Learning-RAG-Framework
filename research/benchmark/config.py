"""Benchmark configuration for TREC RAGTIME Tier B evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from research.config import ExperimentConfig

DEFAULT_SYSTEMS = [
    "preference_loop_full",
    "preference_loop_1round",
    "crucible_single",
    "crucible_dual_best",
    "vanilla_rag",
]


@dataclass
class BenchmarkConfig:
    """End-to-end benchmark settings for multi-system RAGTIME evaluation.

    Parameters
    ----------
    benchmark_id : str
        Benchmark run identifier.
    data_root : Path
        Root directory for RAGTIME metadata (topics, qrels, gold nuggets).
    topics_file : Path
        Relative path under ``data_root`` to RAGTIME topics JSONL.
    topic_filter : str
        ``short`` (limit 2000), ``assessed``, ``all``, or path to topic-id list file.
    assessed_topics_file : Path | None
        Optional JSON list of assessed topic ids (from nugget release).
    gold_nuggets_file : Path
        Normalized gold nugget bank JSONL path (relative to ``data_root``).
    systems : list[str]
        System ids to run (see ``research.benchmark.baselines.SYSTEM_IDS``).
    experiment : ExperimentConfig
        Shared preference-loop / CRUCIBLE generation settings.
    output_dir : Path
        Benchmark artifact root.
    dry_run : bool
        When True, validate config and write run plan without API/LLM calls.
    max_topics : int | None
        Optional cap on number of topics (dev subset).
    eval_max_gold_questions : int | None
        Cap gold nuggets graded per topic (reduces LLM cost).
    eval_max_queryonly_questions : int | None
        Cap QueryOnly nuggets extracted/graded per topic.
    phases : list[str]
        Subset of ``generate``, ``evaluate``, ``report``.
    """

    benchmark_id: str
    data_root: Path = field(default_factory=lambda: Path("data/benchmark/ragtime25"))
    topics_file: Path = field(default_factory=lambda: Path("topics/ragtime25_main_eng.jsonl"))
    topic_filter: str = "short"
    assessed_topics_file: Path | None = field(default_factory=lambda: Path("manifests/assessed_topics.json"))
    gold_nuggets_file: Path = field(default_factory=lambda: Path("gold_nuggets/gold_nuggets.jsonl"))
    systems: list[str] = field(default_factory=lambda: list(DEFAULT_SYSTEMS))
    experiment: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(experiment_id="ragtime_benchmark")
    )
    output_dir: Path = field(default_factory=lambda: Path("output/benchmark"))
    dry_run: bool = True
    max_topics: int | None = None
    eval_max_gold_questions: int | None = None
    eval_max_queryonly_questions: int | None = None
    phases: list[str] = field(default_factory=lambda: ["generate", "evaluate", "report"])

    @classmethod
    def from_yaml(cls, path: Path) -> BenchmarkConfig:
        """Load benchmark config from YAML."""
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BenchmarkConfig:
        """Build config from a dictionary."""
        data_root = Path(raw.get("data_root", "data/benchmark/ragtime25"))
        experiment_raw = raw.get("experiment", {})
        experiment_raw.setdefault("experiment_id", raw.get("benchmark_id", "ragtime_benchmark"))
        experiment = ExperimentConfig.from_dict(experiment_raw)

        assessed_raw = raw.get("assessed_topics_file")
        assessed_path = Path(assessed_raw) if assessed_raw else None

        return cls(
            benchmark_id=str(raw["benchmark_id"]),
            data_root=data_root,
            topics_file=Path(raw.get("topics_file", "topics/ragtime25_main_eng.jsonl")),
            topic_filter=str(raw.get("topic_filter", "short")),
            assessed_topics_file=assessed_path,
            gold_nuggets_file=Path(raw.get("gold_nuggets_file", "gold_nuggets/gold_nuggets.jsonl")),
            systems=list(raw.get("systems", DEFAULT_SYSTEMS)),
            experiment=experiment,
            output_dir=Path(raw.get("output_dir", "output/benchmark")),
            dry_run=bool(raw.get("dry_run", True)),
            max_topics=raw.get("max_topics"),
            eval_max_gold_questions=raw.get("eval_max_gold_questions"),
            eval_max_queryonly_questions=raw.get("eval_max_queryonly_questions"),
            phases=list(raw.get("phases", ["generate", "evaluate", "report"])),
        )

    def benchmark_output_dir(self) -> Path:
        """Return ``output_dir / benchmark_id``."""
        return self.output_dir / self.benchmark_id

    def resolve_data_path(self, relative: Path) -> Path:
        """Resolve a path relative to ``data_root``."""
        if relative.is_absolute():
            return relative
        return self.data_root / relative
