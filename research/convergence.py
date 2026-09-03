"""Convergence criteria for the preference-learning loop."""

from __future__ import annotations

from dataclasses import dataclass, field

from research.config import IterationConfig


@dataclass
class ConvergenceState:
    """Tracks stability signals across improvement rounds."""

    stable_rounds: int = 0
    rounds_without_new_nuggets: int = 0
    last_champion_run_id: str | None = None


class ConvergenceChecker:
    """Decide when to stop iterating report improvements."""

    def __init__(self, config: IterationConfig) -> None:
        self.config = config
        self.state = ConvergenceState()

    def should_stop(
        self,
        round_index: int,
        champion_run_id: str,
        new_nuggets_count: int,
    ) -> tuple[bool, str]:
        """Return whether the loop should stop after the given round."""
        if round_index + 1 >= self.config.max_rounds:
            return True, "max_rounds_reached"

        improvement_rounds_completed = round_index
        if improvement_rounds_completed < self.config.min_improvement_rounds:
            self.state.last_champion_run_id = champion_run_id
            return False, ""

        if new_nuggets_count < self.config.min_new_nuggets_to_continue:
            self.state.rounds_without_new_nuggets += 1
        else:
            self.state.rounds_without_new_nuggets = 0

        if self.state.rounds_without_new_nuggets >= self.config.stable_rounds_for_convergence:
            return True, "no_new_nuggets"

        if self.state.last_champion_run_id == champion_run_id:
            self.state.stable_rounds += 1
        else:
            self.state.stable_rounds = 0

        self.state.last_champion_run_id = champion_run_id

        if self.state.stable_rounds >= self.config.stable_rounds_for_convergence:
            return True, "champion_stable"

        return False, ""
