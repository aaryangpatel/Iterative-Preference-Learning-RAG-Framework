"""PrefNugget judgment result models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PreferenceResult(BaseModel):
    """Pairwise preference judgment (PrefJudgment phase).

    Parameters
    ----------
    topic_id : str
        Topic id.
    winner_run_id : str
        Preferred response run.
    loser_run_id : str
        Non-preferred response run.
    confidence : float
        Judge confidence 0-1.
    """

    topic_id: str
    winner_run_id: str
    loser_run_id: str
    confidence: float = 1.0


class GradeResult(BaseModel):
    """Nugget grading result (GradeNuggetAnswer phase).

    Parameters
    ----------
    topic_id : str
        Topic id.
    run_id : str
        Graded response run.
    question_id : str
        Nugget question id.
    question_text : str
        Question text.
    grade : int
        Integer grade 0-5.
    reasoning : str
        Brief grade explanation.
    confidence : float
        Judge confidence 0-1.
    """

    topic_id: str
    run_id: str
    question_id: str
    question_text: str
    grade: int
    reasoning: str = ""
    confidence: float = 1.0


class RunScore(BaseModel):
    """Aggregated run score for one topic.

    Parameters
    ----------
    run_id : str
        Response run id.
    topic_id : str
        Topic id.
    max_grade : int
        MAX over nugget grades (PrefNugget aggregation).
    mean_grade : float
        Mean grade across nuggets.
    nugget_coverage : float
        Fraction of nuggets with grade >= 4.
    covered_count : int
        Number of nuggets with grade >= 4.
    total_nuggets : int
        Total nuggets graded.
    grades : list[GradeResult]
        Per-nugget grades.
    """

    run_id: str
    topic_id: str
    max_grade: int
    mean_grade: float
    nugget_coverage: float = 0.0
    covered_count: int = 0
    total_nuggets: int = 0
    grades: list[GradeResult] = Field(default_factory=list)
