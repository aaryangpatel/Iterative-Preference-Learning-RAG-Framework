"""Configuration for the preference-learning research loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GenerationProfile:
    """CRUCIBLE report generation variant.

    Parameters
    ----------
    name : str
        Profile label (e.g. ``abstractive``, ``extractive``, ``improved``).
    run_id_suffix : str
        Suffix appended to run ids for this variant.
    match_threshold : float
        Minimum alignment confidence during extraction.
    use_request_context : bool
        Include title/background/problem in extraction prompts.
    prefer_shorter : bool
        Prefer shorter sentences during refinement.
    extractive : bool
        When True, use extracted document segments as report sentences.
    confidence_threshold : float
        Minimum sentence confidence during choose step.
    sentences_per_nugget : int
        Max sentences kept per nugget after choose.
    filter_citations : bool
        Drop sentences failing citation attestation.
    filter_nugget_coverage : bool
        Drop sentences failing nugget coverage filter.
    refine : bool
        Apply full refinement pipeline.
    """

    name: str
    run_id_suffix: str
    match_threshold: float = 0.3
    use_request_context: bool = True
    prefer_shorter: bool = False
    extractive: bool = False
    confidence_threshold: float = 0.3
    sentences_per_nugget: int = 1
    filter_citations: bool = True
    filter_nugget_coverage: bool = True
    refine: bool = True


DEFAULT_PROFILE_A = GenerationProfile(
    name="abstractive",
    run_id_suffix="a",
    match_threshold=0.25,
    prefer_shorter=True,
    extractive=False,
    confidence_threshold=0.25,
)

DEFAULT_PROFILE_B = GenerationProfile(
    name="extractive",
    run_id_suffix="b",
    match_threshold=0.35,
    prefer_shorter=False,
    extractive=True,
    confidence_threshold=0.35,
    sentences_per_nugget=2,
)

DEFAULT_IMPROVED_PROFILE = GenerationProfile(
    name="improved",
    run_id_suffix="improved",
    match_threshold=0.22,
    prefer_shorter=False,
    extractive=False,
    confidence_threshold=0.22,
    sentences_per_nugget=2,
    use_request_context=True,
)

DEFAULT_SYNTHESIS_PROFILE = GenerationProfile(
    name="synthesis",
    run_id_suffix="final",
    match_threshold=0.2,
    prefer_shorter=False,
    extractive=True,
    confidence_threshold=0.2,
    sentences_per_nugget=2,
    use_request_context=True,
)


@dataclass
class IterationConfig:
    """Multi-round iteration and convergence settings.

    Parameters
    ----------
    max_rounds : int
        Total rounds to run, including the initial dual-report round.
    min_improvement_rounds : int
        Minimum improvement rounds (after round 0) before convergence can stop early.
    enable_final_synthesis : bool
        When True, generate a synthesis report and pick the best candidate via tournament.
    min_new_nuggets_to_continue : int
        Minimum new contrastive nuggets required to keep iterating.
    stable_rounds_for_convergence : int
        Consecutive stable rounds (no champion change or no new nuggets) before stopping.
    """

    max_rounds: int = 5
    min_improvement_rounds: int = 2
    enable_final_synthesis: bool = True
    min_new_nuggets_to_continue: int = 1
    stable_rounds_for_convergence: int = 2


@dataclass
class DocumentSourceConfig:
    """Document retrieval settings.

    Parameters
    ----------
    provider : str
        ``jsonl`` (local corpus) or ``ragtime_api`` (TREC Search service).
    max_documents : int
        Documents fetched per query.
    max_docs_per_nugget : int
        Documents scanned per nugget during alignment.
    collection_path : Path | None
        Required when ``provider`` is ``jsonl``.
    """

    provider: str = "ragtime_api"
    max_documents: int = 8
    max_docs_per_nugget: int = 5
    collection_path: Path | None = None
    ragtime_pipeline: str = "ragtime1"
    ragtime_collection: str = "ragtime1"
    cache_dir: Path | None = None


@dataclass
class ExperimentConfig:
    """Configuration for the full preference-learning experiment.

    Parameters
    ----------
    experiment_id : str
        Unique experiment name for output paths.
    document_source : DocumentSourceConfig
        Where reports retrieve evidence from.
    generation_profiles : list[GenerationProfile]
        Two profiles used to generate the initial report pair.
    improved_profile : GenerationProfile
        Profile used for challenger reports in improvement rounds.
    iteration : IterationConfig
        Round limits and convergence thresholds.
    team_id : str
        CRUCIBLE team id stamped on reports.
    char_limit : int
        Report character limit after refinement.
    base_nuggets_count : int
        Auto-generated query nuggets shared across rounds.
    max_contrastive_nuggets : int
        Maximum PrefNugget contrastive questions to accumulate.
    output_dir : Path
        Root directory for saved artifacts.
    """

    experiment_id: str
    document_source: DocumentSourceConfig = field(default_factory=DocumentSourceConfig)
    generation_profiles: list[GenerationProfile] = field(
        default_factory=lambda: [DEFAULT_PROFILE_A, DEFAULT_PROFILE_B]
    )
    improved_profile: GenerationProfile = field(default_factory=lambda: DEFAULT_IMPROVED_PROFILE)
    iteration: IterationConfig = field(default_factory=IterationConfig)
    team_id: str = "research"
    char_limit: int = 2000
    base_nuggets_count: int = 8
    max_contrastive_nuggets: int = 10
    output_dir: Path = field(default_factory=lambda: Path("output/research"))

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        """Load experiment config from a YAML file."""
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        """Build config from a plain dictionary."""
        doc_raw = raw.get("document_source", {})
        profiles_raw = raw.get("generation_profiles", [])
        iteration_raw = raw.get("iteration", {})
        improved_raw = raw.get("improved_profile")

        profiles = [GenerationProfile(**profile) for profile in profiles_raw] if profiles_raw else [
            DEFAULT_PROFILE_A,
            DEFAULT_PROFILE_B,
        ]

        improved_profile = GenerationProfile(**improved_raw) if improved_raw else DEFAULT_IMPROVED_PROFILE

        document_source = DocumentSourceConfig(
            provider=doc_raw.get("provider", "ragtime_api"),
            max_documents=int(doc_raw.get("max_documents", 8)),
            max_docs_per_nugget=int(doc_raw.get("max_docs_per_nugget", 5)),
            collection_path=Path(doc_raw["collection_path"]) if doc_raw.get("collection_path") else None,
            ragtime_pipeline=str(doc_raw.get("ragtime_pipeline", "ragtime1")),
            ragtime_collection=str(doc_raw.get("ragtime_collection", "ragtime1")),
            cache_dir=Path(doc_raw["cache_dir"]) if doc_raw.get("cache_dir") else None,
        )

        iteration = IterationConfig(
            max_rounds=int(iteration_raw.get("max_rounds", 5)),
            min_improvement_rounds=int(iteration_raw.get("min_improvement_rounds", 2)),
            enable_final_synthesis=bool(iteration_raw.get("enable_final_synthesis", True)),
            min_new_nuggets_to_continue=int(iteration_raw.get("min_new_nuggets_to_continue", 1)),
            stable_rounds_for_convergence=int(iteration_raw.get("stable_rounds_for_convergence", 2)),
        )

        return cls(
            experiment_id=str(raw["experiment_id"]),
            document_source=document_source,
            generation_profiles=profiles,
            improved_profile=improved_profile,
            iteration=iteration,
            team_id=str(raw.get("team_id", "research")),
            char_limit=int(raw.get("char_limit", 2000)),
            base_nuggets_count=int(raw.get("base_nuggets_count", 8)),
            max_contrastive_nuggets=int(raw.get("max_contrastive_nuggets", 10)),
            output_dir=Path(raw.get("output_dir", "output/research")),
        )

    def experiment_output_dir(self) -> Path:
        """Return ``output_dir / experiment_id``."""
        return self.output_dir / self.experiment_id
