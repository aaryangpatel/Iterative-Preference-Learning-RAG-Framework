"""Preference-learning research loop (CRUCIBLE + PrefNugget)."""

from research.config import ExperimentConfig, GenerationProfile, IterationConfig
from research.loop import PreferenceLearningLoop
from research.models import ExperimentResult, RoundResult
from research.benchmark.config import BenchmarkConfig
from research.benchmark.pipeline import BenchmarkPipeline

__all__ = [
    "BenchmarkConfig",
    "BenchmarkPipeline",
    "ExperimentConfig",
    "ExperimentResult",
    "GenerationProfile",
    "IterationConfig",
    "PreferenceLearningLoop",
    "RoundResult",
]
