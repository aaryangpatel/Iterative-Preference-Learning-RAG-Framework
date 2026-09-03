"""PrefNugget workflow configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from prefnugget.models.judgment import PreferenceResult, RunScore
from prefnugget.models.nugget import NuggetQuestionBank


@dataclass
class PrefNuggetWorkflowConfig:
    """Workflow variant configuration.

    Parameters
    ----------
    variant : str
        ``queryonly``, ``contrastive``, or ``contrastive_docs``.
    grading_mode : str
        ``response`` or ``docs``.
    max_questions : int
        Questions per topic.
    num_pivot : int
        Pivot runs for stratified pairwise sampling.
    num_others : int
        Other runs compared per pivot.
    allow_ties : bool
        Allow tie judgments in preference phase.
    bidirectional : bool
        Judge each pair in both directions.
    queryonly_iterations : int
        Iterative rounds for query-only extraction.
    """

    variant: str = "queryonly"
    grading_mode: str = "response"
    max_questions: int = 20
    num_pivot: int = 1
    num_others: int = 4
    allow_ties: bool = True
    bidirectional: bool = True
    queryonly_iterations: int = 3


@dataclass
class PrefNuggetWorkflowResult:
    """Full workflow output for one variant run.

    Parameters
    ----------
    banks : dict[str, NuggetQuestionBank]
        topic_id -> extracted nugget bank.
    scores_by_topic : dict[str, list[RunScore]]
        topic_id -> graded runs.
    preferences : list[PreferenceResult]
        Phase 1 preference judgments.
    config : PrefNuggetWorkflowConfig
        Resolved workflow configuration.
    """

    banks: dict[str, NuggetQuestionBank] = field(default_factory=dict)
    scores_by_topic: dict[str, list[RunScore]] = field(default_factory=dict)
    preferences: list[PreferenceResult] = field(default_factory=list)
    config: PrefNuggetWorkflowConfig = field(default_factory=PrefNuggetWorkflowConfig)
