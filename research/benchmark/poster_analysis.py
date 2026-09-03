"""Poster figures, statistical tests, and narrative summary for benchmark results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

from research.benchmark.config import BenchmarkConfig
from research.benchmark.evaluator import BenchmarkEvaluation, TopicSystemScore
from research.benchmark.reporting import BenchmarkReporter, SystemAggregate, aggregate_by_system

PRIMARY_SYSTEM = "preference_loop_full"

SYSTEM_ORDER = [
    "vanilla_rag",
    "crucible_single",
    "crucible_dual_best",
    "preference_loop_1round",
    PRIMARY_SYSTEM,
]

SYSTEM_LABELS = {
    "vanilla_rag": "Vanilla RAG",
    "crucible_single": "CRUCIBLE (single)",
    "crucible_dual_best": "CRUCIBLE (dual best)",
    "preference_loop_1round": "Pref. loop (1 round)",
    PRIMARY_SYSTEM: "Pref. loop (full)",
}

SYSTEM_COLORS = {
    "vanilla_rag": "#94a3b8",
    "crucible_single": "#0284c7",
    "crucible_dual_best": "#0369a1",
    "preference_loop_1round": "#7c3aed",
    PRIMARY_SYSTEM: "#5b21b6",
}

SCORE_METRICS: list[tuple[str, str, float | None]] = [
    ("gold_nugget_coverage", "Gold nugget coverage", 1.0),
    ("gold_mean_grade", "Gold mean grade (0–5)", 5.0),
    ("gold_max_grade", "Gold max grade (0–5)", 5.0),
    ("queryonly_nugget_coverage", "QueryOnly nugget coverage", 1.0),
    ("queryonly_mean_grade", "QueryOnly mean grade (0–5)", 5.0),
    ("citation_validity_rate", "Citation validity", 1.0),
    ("citation_coverage", "Citation coverage", 1.0),
    ("mean_span_overlap", "Mean span overlap", 1.0),
    ("ragtime_f1_proxy", "RAGTIME F1 proxy", 1.0),
]

COUNT_METRICS: list[tuple[str, str]] = [
    ("num_sentences", "Sentences per report"),
    ("num_citations", "Citations per report"),
]


@dataclass
class SignificanceResult:
    """Wilcoxon signed-rank test between two systems on one metric."""

    metric: str
    system_a: str
    system_b: str
    n_pairs: int
    mean_a: float
    mean_b: float
    statistic: float | None
    p_value: float | None
    significant_at_05: bool


def _apply_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#1e293b",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#cbd5e1",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _label_system(system_id: str) -> str:
    return SYSTEM_LABELS.get(system_id, system_id)


def _ordered_systems(system_ids: list[str]) -> list[str]:
    known = [system_id for system_id in SYSTEM_ORDER if system_id in system_ids]
    extras = sorted(system_id for system_id in system_ids if system_id not in SYSTEM_ORDER)
    return known + extras


def _adaptive_axis_max(values: pd.Series, nominal_max: float | None) -> float | None:
    """Zoom the y-axis when observed scores sit far below the nominal scale maximum."""
    if nominal_max is None:
        return None
    peak = float(values.max()) if not values.empty else 0.0
    if peak <= nominal_max * 0.4:
        return max(peak * 1.25, nominal_max * 0.1, 0.5)
    return nominal_max


def _normalize_overview_value(field: str, value: float) -> float:
    """Put grade metrics on a 0–1 axis for the aggregate overview panel."""
    if field in {"gold_mean_grade", "gold_max_grade", "queryonly_mean_grade"}:
        return value / 5.0
    return value


def _scores_dataframe(evaluation: BenchmarkEvaluation) -> pd.DataFrame:
    rows = [score.__dict__ for score in evaluation.topic_scores]
    frame = pd.DataFrame(rows)
    frame["topic_id"] = frame["topic_id"].astype(str)
    frame["topic_order"] = frame["topic_id"].astype(int)
    frame["system_label"] = frame["system_id"].map(_label_system)
    return frame.sort_values(["topic_order", "system_id"])


class PosterAnalysis:
    """Generate poster-ready assets from benchmark evaluation outputs."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._output_root = config.benchmark_output_dir()
        self._figures_dir = self._output_root / "poster" / "figures"
        self._figures_dir.mkdir(parents=True, exist_ok=True)

    def run(self, evaluation: BenchmarkEvaluation | None = None) -> Path:
        """Write figures, stats JSON, and poster summary markdown."""
        if evaluation is None:
            evaluation = BenchmarkReporter(self.config)._load_evaluation()
        _apply_plot_style()
        aggregates = aggregate_by_system(evaluation)
        significance = run_significance_tests(evaluation, primary_system=PRIMARY_SYSTEM)
        scores = _scores_dataframe(evaluation)

        self._plot_aggregate_overview(aggregates, evaluation.benchmark_id)
        for field, label, y_max in SCORE_METRICS:
            self._plot_single_metric_bars(scores, field, label, y_max, evaluation.benchmark_id)
            self._plot_per_topic_heatmap(scores, field, label, y_max)
            self._plot_per_topic_lines(scores, field, label, y_max)
        for field, label in COUNT_METRICS:
            self._plot_single_metric_bars(scores, field, label, None, evaluation.benchmark_id, integer_axis=True)
            self._plot_per_topic_lines(scores, field, label, None)

        self._plot_cumulative_means(scores, "ragtime_f1_proxy", "RAGTIME F1 proxy")
        self._plot_cumulative_means(scores, "gold_nugget_coverage", "Gold nugget coverage")
        self._plot_cumulative_means(scores, "gold_mean_grade", "Gold mean grade")
        self._plot_boxplots(scores)
        self._plot_pairwise_wins(aggregates)
        self._plot_pairwise_matrix(evaluation)
        self._plot_radar(aggregates)
        self._plot_system_rankings(scores)

        stats_path = self._output_root / "poster" / "significance.json"
        stats_path.write_text(
            json.dumps([_significance_to_json(result) for result in significance], indent=2),
            encoding="utf-8",
        )
        summary_path = self._output_root / "poster" / "poster_summary.md"
        summary_path.write_text(
            _render_poster_summary(self.config, aggregates, significance, evaluation),
            encoding="utf-8",
        )
        return summary_path

    def _save(self, figure: plt.Figure, name: str) -> None:
        figure.savefig(self._figures_dir / name, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)

    def _plot_aggregate_overview(self, aggregates: list[SystemAggregate], benchmark_id: str) -> None:
        rows: list[dict[str, object]] = []
        metric_specs = [
            ("gold_nugget_coverage", "Gold coverage"),
            ("gold_mean_grade", "Gold grade (÷5)"),
            ("queryonly_nugget_coverage", "QueryOnly coverage"),
            ("citation_validity_rate", "Citation validity"),
            ("ragtime_f1_proxy", "F1 proxy"),
        ]
        for aggregate in aggregates:
            for field, label in metric_specs:
                value = getattr(aggregate, field)
                if value is not None:
                    rows.append(
                        {
                            "system": _label_system(aggregate.system_id),
                            "system_id": aggregate.system_id,
                            "metric": label,
                            "value": _normalize_overview_value(field, float(value)),
                        }
                    )
        if not rows:
            return
        data = pd.DataFrame(rows)
        data["system"] = pd.Categorical(
            data["system"],
            categories=[_label_system(system_id) for system_id in _ordered_systems(data["system_id"].unique())],
            ordered=True,
        )
        figure, axis = plt.subplots(figsize=(11, 5.5))
        palette = [_system_color(system_id) for system_id in _ordered_systems(data["system_id"].unique())]
        sns.barplot(data=data, x="metric", y="value", hue="system", ax=axis, palette=palette, edgecolor="white", linewidth=0.6)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Score (normalized where applicable)")
        axis.set_xlabel("")
        axis.set_title(f"Aggregate system comparison — {benchmark_id}")
        axis.legend(title="System", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        figure.tight_layout()
        self._save(figure, "01_aggregate_overview.png")

    def _plot_single_metric_bars(
        self,
        scores: pd.DataFrame,
        field: str,
        label: str,
        y_max: float | None,
        benchmark_id: str,
        integer_axis: bool = False,
    ) -> None:
        grouped = (
            scores.groupby("system_id", as_index=False)[field]
            .mean()
            .rename(columns={field: "value"})
        )
        if grouped.empty or grouped["value"].isna().all():
            return
        grouped["system"] = grouped["system_id"].map(_label_system)
        order = [_label_system(system_id) for system_id in _ordered_systems(grouped["system_id"].tolist())]
        grouped["system"] = pd.Categorical(grouped["system"], categories=order, ordered=True)
        figure, axis = plt.subplots(figsize=(8, 4.5))
        palette = [_system_color(system_id) for system_id in _ordered_systems(grouped["system_id"].tolist())]
        bars = sns.barplot(
            data=grouped.sort_values("system"),
            x="system",
            y="value",
            ax=axis,
            palette=palette,
            edgecolor="white",
            linewidth=0.6,
        )
        axis.set_xlabel("")
        axis.set_ylabel(label)
        axis.set_title(f"Mean {label} across topics — {benchmark_id}")
        axis.tick_params(axis="x", rotation=25)
        if y_max is not None:
            axis.set_ylim(0, y_max * 1.08)
        if integer_axis:
            axis.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        for container in bars.containers:
            axis.bar_label(container, fmt="%.2f", padding=2, fontsize=8)
        figure.tight_layout()
        slug = field.replace("_", "-")
        self._save(figure, f"02_bar_{slug}.png")

    def _plot_per_topic_heatmap(
        self,
        scores: pd.DataFrame,
        field: str,
        label: str,
        y_max: float | None,
    ) -> None:
        subset = scores.dropna(subset=[field])
        if subset.empty:
            return
        pivot = subset.pivot(index="topic_id", columns="system_id", values=field)
        pivot = pivot[_ordered_systems(pivot.columns.tolist())]
        pivot.columns = [_label_system(column) for column in pivot.columns]
        heatmap_max = _adaptive_axis_max(pivot.stack(), y_max)
        figure, axis = plt.subplots(figsize=(10, max(3.5, 0.45 * len(pivot) + 1.5)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=heatmap_max,
            ax=axis,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": label},
        )
        axis.set_title(f"Per-topic {label}")
        axis.set_xlabel("System")
        axis.set_ylabel("Topic ID")
        figure.tight_layout()
        slug = field.replace("_", "-")
        self._save(figure, f"03_heatmap_{slug}.png")

    def _plot_per_topic_lines(
        self,
        scores: pd.DataFrame,
        field: str,
        label: str,
        y_max: float | None,
    ) -> None:
        subset = scores.dropna(subset=[field])
        if subset.empty:
            return
        figure, axis = plt.subplots(figsize=(9, 4.8))
        for system_id in _ordered_systems(subset["system_id"].unique().tolist()):
            part = subset[subset["system_id"] == system_id].sort_values("topic_order")
            axis.plot(
                part["topic_id"],
                part[field],
                marker="o",
                linewidth=2,
                markersize=6,
                label=_label_system(system_id),
                color=_system_color(system_id),
            )
        axis.set_xlabel("Topic ID")
        axis.set_ylabel(label)
        axis.set_title(f"Per-topic {label}")
        line_max = _adaptive_axis_max(subset[field], y_max)
        if line_max is not None:
            axis.set_ylim(0, line_max * 1.08)
        axis.legend(title="System", fontsize=8, loc="best")
        axis.grid(True, axis="y", alpha=0.35)
        figure.tight_layout()
        slug = field.replace("_", "-")
        self._save(figure, f"04_lines_{slug}.png")

    def _plot_cumulative_means(self, scores: pd.DataFrame, field: str, label: str) -> None:
        subset = scores.dropna(subset=[field])
        if subset.empty:
            return
        topics = sorted(subset["topic_id"].unique(), key=int)
        figure, axis = plt.subplots(figsize=(9, 5))
        for system_id in _ordered_systems(subset["system_id"].unique().tolist()):
            cumulative: list[float] = []
            topic_labels: list[str] = []
            values: list[float] = []
            for topic_id in topics:
                row = subset[(subset["system_id"] == system_id) & (subset["topic_id"] == topic_id)]
                if row.empty:
                    continue
                values.append(float(row[field].iloc[0]))
                cumulative.append(mean(values))
                topic_labels.append(f"≤{topic_id}")
            if not cumulative:
                continue
            axis.plot(
                range(1, len(cumulative) + 1),
                cumulative,
                marker="o",
                linewidth=2.2,
                markersize=7,
                label=_label_system(system_id),
                color=_system_color(system_id),
            )
        axis.set_xticks(range(1, len(topics) + 1))
        axis.set_xticklabels([f"≤{topic_id}" for topic_id in topics])
        axis.set_xlabel("Topics accumulated (in run order)")
        axis.set_ylabel(f"Cumulative mean {label.lower()}")
        axis.set_title(f"Cumulative performance — {label}")
        axis.legend(title="System", fontsize=8, loc="best")
        axis.grid(True, axis="y", alpha=0.35)
        figure.tight_layout()
        slug = field.replace("_", "-")
        self._save(figure, f"05_cumulative_{slug}.png")

    def _plot_boxplots(self, scores: pd.DataFrame) -> None:
        key_metrics = [
            ("ragtime_f1_proxy", "RAGTIME F1 proxy"),
            ("gold_nugget_coverage", "Gold nugget coverage"),
            ("gold_mean_grade", "Gold mean grade"),
            ("queryonly_nugget_coverage", "QueryOnly coverage"),
        ]
        figure, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes_list = axes.flatten()
        for axis, (field, label) in zip(axes_list, key_metrics):
            subset = scores.dropna(subset=[field]).copy()
            if subset.empty:
                axis.set_visible(False)
                continue
            order = [_label_system(system_id) for system_id in _ordered_systems(subset["system_id"].unique().tolist())]
            subset["system"] = pd.Categorical(subset["system_id"].map(_label_system), categories=order, ordered=True)
            palette = [_system_color(system_id) for system_id in _ordered_systems(subset["system_id"].unique().tolist())]
            sns.boxplot(data=subset, x="system", y=field, ax=axis, palette=palette, linewidth=1.0)
            sns.stripplot(
                data=subset,
                x="system",
                y=field,
                ax=axis,
                color="#0f172a",
                alpha=0.55,
                size=4,
                jitter=0.15,
            )
            axis.set_title(label)
            axis.set_xlabel("")
            axis.tick_params(axis="x", rotation=20)
        figure.suptitle("Score distributions across topics", fontsize=13, fontweight="bold", y=1.02)
        figure.tight_layout()
        self._save(figure, "06_boxplots_key_metrics.png")

    def _plot_pairwise_wins(self, aggregates: list[SystemAggregate]) -> None:
        rows = [
            {
                "system": _label_system(row.system_id),
                "system_id": row.system_id,
                "pairwise_win_rate": row.pairwise_win_rate,
            }
            for row in aggregates
            if row.pairwise_win_rate is not None
        ]
        if not rows:
            return
        data = pd.DataFrame(rows)
        order = [_label_system(system_id) for system_id in _ordered_systems(data["system_id"].tolist())]
        data["system"] = pd.Categorical(data["system"], categories=order, ordered=True)
        figure, axis = plt.subplots(figsize=(8, 4.8))
        palette = [_system_color(system_id) for system_id in _ordered_systems(data["system_id"].tolist())]
        bars = sns.barplot(
            data=data.sort_values("system"),
            x="system",
            y="pairwise_win_rate",
            ax=axis,
            palette=palette,
            edgecolor="white",
        )
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Win rate vs. baselines")
        axis.set_xlabel("")
        axis.set_title(f"Pairwise win rate — {PRIMARY_SYSTEM}")
        axis.tick_params(axis="x", rotation=20)
        for container in bars.containers:
            axis.bar_label(container, fmt="%.2f", padding=2, fontsize=8)
        figure.tight_layout()
        self._save(figure, "07_pairwise_win_rate.png")

    def _plot_pairwise_matrix(self, evaluation: BenchmarkEvaluation) -> None:
        if not evaluation.pairwise:
            return
        baselines = sorted({outcome.system_b for outcome in evaluation.pairwise if outcome.system_a == PRIMARY_SYSTEM})
        if not baselines:
            return
        rows: list[dict[str, object]] = []
        for baseline in baselines:
            outcomes = [
                outcome
                for outcome in evaluation.pairwise
                if outcome.system_a == PRIMARY_SYSTEM and outcome.system_b == baseline
            ]
            wins = sum(1 for outcome in outcomes if outcome.winner_system == PRIMARY_SYSTEM and not outcome.tie)
            rows.append(
                {
                    "baseline": _label_system(baseline),
                    "win_rate": wins / len(outcomes) if outcomes else 0.0,
                    "n": len(outcomes),
                }
            )
        data = pd.DataFrame(rows)
        figure, axis = plt.subplots(figsize=(7, max(3, 0.55 * len(data) + 1.5)))
        sns.barplot(
            data=data,
            y="baseline",
            x="win_rate",
            ax=axis,
            color=_system_color(PRIMARY_SYSTEM),
            edgecolor="white",
        )
        axis.set_xlim(0, 1.05)
        axis.set_xlabel("Win rate")
        axis.set_ylabel("")
        axis.set_title(f"{_label_system(PRIMARY_SYSTEM)} vs. each baseline")
        for index, row in data.iterrows():
            axis.text(row["win_rate"] + 0.02, index, f"{row['win_rate']:.0%} (n={int(row['n'])})", va="center", fontsize=9)
        figure.tight_layout()
        self._save(figure, "08_pairwise_vs_baselines.png")

    def _plot_radar(self, aggregates: list[SystemAggregate]) -> None:
        fields = [
            ("gold_nugget_coverage", "Gold cov."),
            ("gold_mean_grade", "Gold grade"),
            ("queryonly_nugget_coverage", "QO cov."),
            ("citation_validity_rate", "Cite valid."),
            ("ragtime_f1_proxy", "F1 proxy"),
        ]
        labels = [label for _, label in fields]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        figure, axis = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
        for aggregate in aggregates:
            values = []
            for field, _label in fields:
                raw = getattr(aggregate, field)
                if raw is None:
                    values.append(0.0)
                elif field.endswith("_grade"):
                    values.append(raw / 5.0)
                else:
                    values.append(float(raw))
            values += values[:1]
            axis.plot(angles, values, linewidth=2, label=_label_system(aggregate.system_id), color=_system_color(aggregate.system_id))
            axis.fill(angles, values, alpha=0.08, color=_system_color(aggregate.system_id))
        axis.set_xticks(angles[:-1])
        axis.set_xticklabels(labels)
        axis.set_ylim(0, 1.05)
        axis.set_title("Normalized multi-metric profile", pad=20)
        axis.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)
        figure.tight_layout()
        self._save(figure, "09_radar_profiles.png")

    def _plot_system_rankings(self, scores: pd.DataFrame) -> None:
        if scores.empty:
            return
        metrics = ["ragtime_f1_proxy", "gold_nugget_coverage", "gold_mean_grade", "queryonly_nugget_coverage"]
        rank_rows: list[dict[str, object]] = []
        for topic_id in sorted(scores["topic_id"].unique(), key=int):
            part = scores[scores["topic_id"] == topic_id]
            for field in metrics:
                ordered = part.dropna(subset=[field]).sort_values(field, ascending=False)
                for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
                    rank_rows.append(
                        {
                            "topic_id": topic_id,
                            "metric": field.replace("_", " "),
                            "system": _label_system(row["system_id"]),
                            "system_id": row["system_id"],
                            "rank": rank,
                        }
                    )
        if not rank_rows:
            return
        data = pd.DataFrame(rank_rows)
        summary = data.groupby(["system_id", "metric"], as_index=False)["rank"].mean()
        summary["system"] = summary["system_id"].map(_label_system)
        figure, axis = plt.subplots(figsize=(10, 5))
        order = [_label_system(system_id) for system_id in _ordered_systems(summary["system_id"].unique().tolist())]
        summary["system"] = pd.Categorical(summary["system"], categories=order, ordered=True)
        palette = [_system_color(system_id) for system_id in _ordered_systems(summary["system_id"].unique().tolist())]
        sns.barplot(data=summary, x="metric", y="rank", hue="system", ax=axis, palette=palette)
        axis.set_ylabel("Mean rank (lower is better)")
        axis.set_xlabel("")
        axis.set_title("Average rank by metric across topics")
        axis.invert_yaxis()
        axis.legend(title="System", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        figure.tight_layout()
        self._save(figure, "10_mean_rank_by_metric.png")


def _significance_to_json(result: SignificanceResult) -> dict[str, object]:
    payload = result.__dict__.copy()
    for key, value in payload.items():
        if isinstance(value, (np.bool_, bool)):
            payload[key] = bool(value)
        elif isinstance(value, (np.floating, float)):
            payload[key] = float(value) if value is not None else None
        elif isinstance(value, (np.integer, int)):
            payload[key] = int(value) if value is not None else None
    return payload


def _system_color(system_id: str) -> str:
    return SYSTEM_COLORS.get(system_id, "#475569")


def run_significance_tests(
    evaluation: BenchmarkEvaluation,
    primary_system: str = PRIMARY_SYSTEM,
    alpha: float = 0.05,
) -> list[SignificanceResult]:
    """Wilcoxon signed-rank tests: primary system vs each baseline per metric."""
    metrics = [
        ("gold_nugget_coverage", "Gold coverage"),
        ("gold_mean_grade", "Gold mean grade"),
        ("queryonly_nugget_coverage", "QueryOnly coverage"),
        ("citation_validity_rate", "Citation validity"),
        ("ragtime_f1_proxy", "F1 proxy"),
    ]
    by_system_topic: dict[str, dict[str, TopicSystemScore]] = {}
    for score in evaluation.topic_scores:
        by_system_topic.setdefault(score.system_id, {})[score.topic_id] = score

    if primary_system not in by_system_topic:
        return []

    results: list[SignificanceResult] = []
    primary_topics = by_system_topic[primary_system]
    for system_id, topic_scores in sorted(by_system_topic.items()):
        if system_id == primary_system:
            continue
        shared_topics = sorted(set(primary_topics) & set(topic_scores), key=int)
        for field, label in metrics:
            pairs_a: list[float] = []
            pairs_b: list[float] = []
            for topic_id in shared_topics:
                value_a = getattr(primary_topics[topic_id], field)
                value_b = getattr(topic_scores[topic_id], field)
                if value_a is not None and value_b is not None:
                    pairs_a.append(value_a)
                    pairs_b.append(value_b)
            if len(pairs_a) < 2:
                continue
            if all(left == right for left, right in zip(pairs_a, pairs_b)):
                statistic, p_value = None, 1.0
            else:
                statistic, p_value = wilcoxon(pairs_a, pairs_b, alternative="two-sided")
            results.append(
                SignificanceResult(
                    metric=label,
                    system_a=primary_system,
                    system_b=system_id,
                    n_pairs=len(pairs_a),
                    mean_a=mean(pairs_a),
                    mean_b=mean(pairs_b),
                    statistic=float(statistic) if statistic is not None else None,
                    p_value=float(p_value),
                    significant_at_05=p_value < alpha,
                )
            )
    return results


def _render_poster_summary(
    config: BenchmarkConfig,
    aggregates: list[SystemAggregate],
    significance: list[SignificanceResult],
    evaluation: BenchmarkEvaluation,
) -> str:
    primary_row = next((row for row in aggregates if row.system_id == PRIMARY_SYSTEM), None)
    figure_list = sorted(path.name for path in (config.benchmark_output_dir() / "poster" / "figures").glob("*.png"))
    lines = [
        f"# Poster Summary — {config.benchmark_id}",
        "",
        "## Research question",
        "",
        "Does iterative preference learning (PrefNugget contrastive nuggets + CRUCIBLE report generation)",
        "outperform single-pass RAG baselines on TREC RAGTIME Tier B metrics?",
        "",
        "## Setup",
        "",
        f"- Topics: {evaluation.metadata.get('num_topics', 'n/a')} ({config.topic_filter})",
        f"- Systems: {', '.join(config.systems)}",
        f"- Retrieval: TREC RAGTIME Search API (`eng` pipeline, `ragtime1` collection)",
        "",
        "## Main results (macro average)",
        "",
    ]
    if primary_row:
        lines.extend(
            [
                f"**{PRIMARY_SYSTEM}** (proposed system):",
                f"- Gold nugget coverage: {_fmt(primary_row.gold_nugget_coverage)}",
                f"- Gold mean grade: {_fmt(primary_row.gold_mean_grade)}",
                f"- Citation validity: {_fmt(primary_row.citation_validity_rate)}",
                f"- RAGTIME F1 proxy: {_fmt(primary_row.ragtime_f1_proxy)}",
                f"- Pairwise win rate: {_fmt(primary_row.pairwise_win_rate)}",
                "",
            ]
        )
    lines.extend(
        [
            "## All systems",
            "",
            "| System | Gold cov. | Gold grade | QO cov. | Cite valid. | F1 proxy | Win rate |",
            "|--------|-----------|------------|---------|-------------|----------|----------|",
        ]
    )
    for row in aggregates:
        lines.append(
            f"| {row.system_id} | {_fmt(row.gold_nugget_coverage)} | {_fmt(row.gold_mean_grade)} | "
            f"{_fmt(row.queryonly_nugget_coverage)} | {_fmt(row.citation_validity_rate)} | "
            f"{_fmt(row.ragtime_f1_proxy)} | {_fmt(row.pairwise_win_rate)} |"
        )
    lines.extend(["", "## Statistical significance (Wilcoxon vs. preference loop)", ""])
    if not significance:
        lines.append("_No paired significance tests (need ≥2 shared topics)._")
    else:
        lines.extend(["| Metric | Baseline | n | Mean Δ | p-value | Sig. |", "|--------|----------|---|--------|---------|------|"])
        for result in significance:
            delta = result.mean_a - result.mean_b
            sig = "yes" if result.significant_at_05 else "—"
            lines.append(
                f"| {result.metric} | {result.system_b} | {result.n_pairs} | {delta:+.3f} | "
                f"{result.p_value:.4f} | {sig} |"
            )
    lines.extend(["", "## Figures", ""])
    for name in figure_list:
        lines.append(f"- `poster/figures/{name}`")
    lines.extend(["", "## Takeaway", "", _build_takeaway(primary_row, aggregates, significance)])
    return "\n".join(lines) + "\n"


def _build_takeaway(
    primary_row: SystemAggregate | None,
    aggregates: list[SystemAggregate],
    significance: list[SignificanceResult],
) -> str:
    if primary_row is None or primary_row.ragtime_f1_proxy is None:
        return "Run live benchmark to populate scores before finalizing poster text."
    best = max(aggregates, key=lambda row: row.ragtime_f1_proxy or 0.0)
    sig_wins = [result for result in significance if result.significant_at_05 and result.mean_a > result.mean_b]
    if best.system_id == primary_row.system_id:
        if sig_wins:
            return (
                "The preference-learning loop achieves the highest RAGTIME F1 proxy and significantly "
                "outperforms baselines on paired topics."
            )
        return (
            "The preference-learning loop leads on aggregate F1 proxy; add more topics to test "
            "whether the cumulative trend separates from CRUCIBLE baselines."
        )
    return (
        f"Best aggregate F1 proxy: **{best.system_id}** ({best.ragtime_f1_proxy:.3f}). "
        f"Monitor cumulative-mean plots as more topics are added."
    )


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"
