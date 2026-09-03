"""Persist multi-round preference-learning experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from crucible.loaders import save_report_bundle
from research.config import ExperimentConfig
from research.models import ExperimentResult, RoundResult


class ExperimentStorage:
    """Save per-round and experiment-level artifacts under ``config.output_dir``.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration with output paths.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._root = config.experiment_output_dir()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def save_round(self, round_result: RoundResult) -> Path:
        """Save one round's reports, nuggets, and summary.

        Parameters
        ----------
        round_result : RoundResult
            Completed round output.

        Returns
        -------
        Path
            Path to ``round_summary.json``.
        """
        round_dir = self._root / f"round_{round_result.round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        for index, bundle in enumerate(round_result.candidate_bundles):
            save_report_bundle(bundle, round_dir / f"candidate_{index}.json")

        save_report_bundle(round_result.champion, round_dir / "champion.json")
        if round_result.challenger is not None:
            save_report_bundle(round_result.challenger, round_dir / "challenger.json")

        if round_result.merged_nugget_bank is not None:
            (round_dir / "merged_nuggets.json").write_text(
                json.dumps(round_result.merged_nugget_bank.to_v3_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        if round_result.contrastive_nugget_bank is not None:
            contrastive_payload = {
                "query_id": round_result.contrastive_nugget_bank.topic_id,
                "questions": [
                    {
                        "question_id": question.question_id,
                        "text": question.text,
                        "extraction_method": question.extraction_method,
                        "confidence": question.confidence,
                    }
                    for question in round_result.contrastive_nugget_bank.questions
                ],
            }
            (round_dir / "contrastive_nuggets.json").write_text(
                json.dumps(contrastive_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        summary = {
            "round_index": round_result.round_index,
            "new_nuggets_count": round_result.new_nuggets_count,
            "converged": round_result.converged,
            "preference": round_result.preference.model_dump() if round_result.preference else None,
            "metadata": round_result.metadata,
            "champion_run_id": round_result.champion.report.run_id,
        }
        summary_path = round_dir / "round_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary_path

    def save_experiment(self, result: ExperimentResult) -> Path:
        """Save final champion and experiment summary.

        Parameters
        ----------
        result : ExperimentResult
            Full experiment output.

        Returns
        -------
        Path
            Path to ``experiment_summary.json``.
        """
        save_report_bundle(result.final_champion, self._root / "final_champion.json")

        summary = {
            "experiment_id": result.experiment_id,
            "query_id": result.query_id,
            "total_rounds": result.total_rounds,
            "converged": result.converged,
            "metadata": result.metadata,
            "rounds": [
                {
                    "round_index": round_result.round_index,
                    "champion_run_id": round_result.champion.report.run_id,
                    "new_nuggets_count": round_result.new_nuggets_count,
                    "converged": round_result.converged,
                }
                for round_result in result.rounds
            ],
        }
        summary_path = self._root / "experiment_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary_path
