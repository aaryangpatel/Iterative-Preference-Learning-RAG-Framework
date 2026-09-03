"""Automatic citation quality evaluation for CRUCIBLE reports."""

from __future__ import annotations

from pydantic import BaseModel, Field

from crucible.models.citation import CitedReport


class SentenceCitationScore(BaseModel):
    """Per-sentence citation quality metrics.

    Parameters
    ----------
    sentence_id : str
        Sentence identifier.
    has_citation : bool
        Whether at least one citation exists.
    valid_doc_ids : bool
        Whether all cited ids are in the retrieval pool.
    span_overlap : float
        Token overlap between sentence and cited span text.
    """

    sentence_id: str
    has_citation: bool
    valid_doc_ids: bool
    span_overlap: float


class CitationQualityReport(BaseModel):
    """Aggregate citation evaluation for one report.

    Parameters
    ----------
    report_id : str
        Evaluated report id.
    coverage : float
        Fraction of sentences with citations.
    validity_rate : float
        Fraction of sentences with only valid doc ids.
    mean_span_overlap : float
        Mean token overlap between sentence and cited spans.
    sentence_scores : list[SentenceCitationScore]
        Per-sentence breakdown.
    """

    report_id: str
    coverage: float
    validity_rate: float
    mean_span_overlap: float
    sentence_scores: list[SentenceCitationScore] = Field(default_factory=list)


class CitationEvaluator:
    """Evaluate citation quality for CRUCIBLE cited reports.

    Parameters
    ----------
    valid_doc_ids : set[str] | None
        Retrieval pool used to check citation validity.
    """

    def __init__(self, valid_doc_ids: set[str] | None = None) -> None:
        self._valid_doc_ids = valid_doc_ids

    def evaluate(self, report: CitedReport) -> CitationQualityReport:
        """Compute citation quality metrics.

        Parameters
        ----------
        report : CitedReport
            Report to evaluate.

        Returns
        -------
        CitationQualityReport
            Aggregate and per-sentence scores.
        """
        sentence_scores: list[SentenceCitationScore] = []
        overlap_values: list[float] = []
        valid_count = 0

        for sentence in report.sentences:
            has_citation = bool(sentence.citations)
            valid = True
            if self._valid_doc_ids is not None:
                valid = sentence.doc_ids().issubset(self._valid_doc_ids)
            if valid:
                valid_count += 1

            overlap = self._sentence_span_overlap(sentence.text, sentence.citations)
            overlap_values.append(overlap)
            sentence_scores.append(
                SentenceCitationScore(
                    sentence_id=sentence.sentence_id,
                    has_citation=has_citation,
                    valid_doc_ids=valid,
                    span_overlap=overlap,
                )
            )

        sentence_count = len(report.sentences) or 1
        return CitationQualityReport(
            report_id=report.report_id,
            coverage=report.citation_coverage(),
            validity_rate=valid_count / sentence_count,
            mean_span_overlap=sum(overlap_values) / sentence_count,
            sentence_scores=sentence_scores,
        )

    def _sentence_span_overlap(self, sentence_text: str, citations: list) -> float:
        if not citations:
            return 0.0
        sentence_tokens = set(sentence_text.lower().split())
        overlaps: list[float] = []
        for citation in citations:
            span_tokens = set(citation.span.text.lower().split())
            if not span_tokens:
                overlaps.append(0.0)
                continue
            overlaps.append(len(sentence_tokens & span_tokens) / len(sentence_tokens | span_tokens))
        return max(overlaps)
