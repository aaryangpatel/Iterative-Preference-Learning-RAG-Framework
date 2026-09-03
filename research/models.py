"""Result models for multi-round preference-learning experiments."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crucible.models.nugget import CrucibleNuggetBank
from crucible.models.report_output import CrucibleReportBundle
from prefnugget.models.judgment import PreferenceResult
from prefnugget.models.nugget import NuggetQuestionBank


class RoundResult(BaseModel):
    """Outcome of one experiment round.

    Parameters
    ----------
    round_index : int
        Zero-based round number.
    candidate_bundles : list[CrucibleReportBundle]
        Reports compared this round (initial pair or champion vs challenger).
    champion : CrucibleReportBundle
        Best report after this round.
    runner_up : CrucibleReportBundle | None
        Second-best report after this round.
    challenger : CrucibleReportBundle | None
        Newly generated report in improvement rounds.
    preference : PreferenceResult | None
        Pairwise judgment for this round.
    merged_nugget_bank : CrucibleNuggetBank | None
        Base plus contrastive nuggets used for generation.
    contrastive_nugget_bank : NuggetQuestionBank | None
        Accumulated PrefNugget contrastive questions.
    new_nuggets_count : int
        Contrastive nuggets added during this round.
    converged : bool
        Whether the experiment stopped after this round.
    metadata : dict[str, Any]
        Round diagnostics (phase, stop reason, etc.).
    """

    round_index: int
    candidate_bundles: list[CrucibleReportBundle]
    champion: CrucibleReportBundle
    runner_up: CrucibleReportBundle | None = None
    challenger: CrucibleReportBundle | None = None
    preference: PreferenceResult | None = None
    merged_nugget_bank: CrucibleNuggetBank | None = None
    contrastive_nugget_bank: NuggetQuestionBank | None = None
    new_nuggets_count: int = 0
    converged: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentResult(BaseModel):
    """Full multi-round experiment output.

    Parameters
    ----------
    experiment_id : str
        Experiment identifier.
    query_id : str
        Topic / query id.
    rounds : list[RoundResult]
        Per-round results in order.
    final_champion : CrucibleReportBundle
        Best report after all rounds.
    total_rounds : int
        Number of rounds executed.
    converged : bool
        Whether the loop stopped before ``max_rounds``.
    metadata : dict[str, Any]
        Experiment-level diagnostics.
    """

    experiment_id: str
    query_id: str
    rounds: list[RoundResult]
    final_champion: CrucibleReportBundle
    total_rounds: int
    converged: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
