"""CRUCIBLE-style nugget alignment results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NuggetAlignmentResult(BaseModel):
    """Alignment of one nugget to a text segment (CRUCIBLE ``align.py`` output).

    Parameters
    ----------
    nugget_text : str
        Nugget question or factual statement.
    nugget_id : str | None
        Nugget identifier.
    query_id : str | None
        Topic id.
    context : str | None
        Surrounding context (sentence or paragraph).
    sentence : str | None
        Matched sentence text.
    source_document : str | None
        Full source document text.
    extracted_text_segment : str | None
        Supporting passage span.
    is_match : bool | None
        Whether nugget is supported by the segment.
    match_score : float | None
        Continuous alignment score.
    answerability : int | None
        0-5 answerability grade.
    confidence : float | None
        Model confidence.
    summary : str | None
        Abstractive summary sentence for report assembly.
    answer : str | None
        Gold answer text used during extraction.
    reasoning : str | None
        Model reasoning trace.
    metadata : dict[str, Any]
        Prompt name and extra fields.
    """

    nugget_text: str
    nugget_id: str | None = None
    query_id: str | None = None
    context: str | None = None
    sentence: str | None = None
    source_document: str | None = None
    extracted_text_segment: str | None = None
    is_match: bool | None = None
    match_score: float | None = None
    answerability: int | None = None
    confidence: float | None = None
    summary: str | None = None
    answer: str | None = None
    reasoning: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NuggetPairAlignment(BaseModel):
    """Semantic alignment between nuggets from two CRUCIBLE reports.

    Parameters
    ----------
    nugget_a_id : str
        Nugget from report A.
    nugget_b_id : str
        Nugget from report B.
    similarity : float
        Semantic similarity score.
    alignment_type : str
        equivalent | partial | related | unrelated.
    metadata : dict[str, Any]
        Thresholds and method metadata.
    """

    nugget_a_id: str
    nugget_b_id: str
    similarity: float
    alignment_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CrossReportAlignment(BaseModel):
    """Full cross-report nugget alignment for one query.

    Parameters
    ----------
    query_id : str
        Shared topic id.
    bank_a_id : str
        Source bank A identifier.
    bank_b_id : str
        Source bank B identifier.
    pairs : list[NuggetPairAlignment]
        Aligned nugget pairs.
    unmatched_a : list[str]
        Nugget ids in A with no partner above threshold.
    unmatched_b : list[str]
        Nugget ids in B with no partner above threshold.
    metadata : dict[str, Any]
        Algorithm configuration.
    """

    query_id: str
    bank_a_id: str
    bank_b_id: str
    pairs: list[NuggetPairAlignment] = Field(default_factory=list)
    unmatched_a: list[str] = Field(default_factory=list)
    unmatched_b: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
