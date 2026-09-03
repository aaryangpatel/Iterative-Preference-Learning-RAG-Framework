"""Post-process CRUCIBLE reports: filter, choose, and chop."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from rag_framework.llm.client import OpenRouterLLM, get_llm_client

from crucible.models.citation import CitedReport, CitedSentence
from crucible.models.nugget import CrucibleNuggetBank

_ATTEST_SYSTEM = """Does the cited document passage support the claim in the sentence?
Respond with JSON only: {"supported": true/false, "confidence": 0.0 to 1.0}"""

_COVERAGE_SYSTEM = """Does the sentence cover the nugget question with the given gold answers?
Respond with JSON only: {"covers": true/false, "confidence": 0.0 to 1.0}"""


@dataclass
class RefinementConfig:
    """Report refinement settings mimicking CRUCIBLE choose/chop/filter.

    Parameters
    ----------
    confidence_threshold : float
        Minimum sentence confidence to keep during choose.
    sentences_per_nugget : int
        Max sentences retained per nugget question.
    prefer_shorter : bool
        When True, prefer shorter sentences among candidates.
    char_limit : int | None
        Maximum report length in Unicode-normalized characters.
    word_limit : int | None
        Maximum report length in words.
    filter_citations : bool
        Drop sentences whose citations do not attest the claim.
    filter_nugget_coverage : bool
        Drop sentences that do not cover their nugget.
    avoid_similar : bool
        Skip duplicate sentence fingerprints.
    """

    confidence_threshold: float = 0.3
    sentences_per_nugget: int = 1
    prefer_shorter: bool = False
    char_limit: int | None = 2000
    word_limit: int | None = None
    filter_citations: bool = True
    filter_nugget_coverage: bool = True
    avoid_similar: bool = True


class ReportRefinement:
    """Filter, choose, and chop CRUCIBLE reports via LLM attestation prompts.

    Mimics ``filter_argue_*``, ``choose_sentences``, and ``chop_sentences``.

    Parameters
    ----------
    config : RefinementConfig | None
        Refinement settings.
    llm : OpenRouterLLM | None
        LLM client for attestation checks.
    """

    def __init__(
        self,
        config: RefinementConfig | None = None,
        llm: OpenRouterLLM | None = None,
    ) -> None:
        self.config = config or RefinementConfig()
        self._llm = llm or get_llm_client()

    def refine(
        self,
        report: CitedReport,
        bank: CrucibleNuggetBank,
    ) -> CitedReport:
        """Apply filter, choose, and chop stages in order.

        Parameters
        ----------
        report : CitedReport
            Raw generated report.
        bank : CrucibleNuggetBank
            Nugget bank for coverage filtering.

        Returns
        -------
        CitedReport
            Refined report.
        """
        sentences = list(report.sentences)
        if self.config.filter_citations:
            sentences = self.filter_citation_support(sentences)
        if self.config.filter_nugget_coverage:
            sentences = self.filter_nugget_coverage(sentences, bank)
        sentences = self.choose_sentences(sentences)
        sentences = self.chop_sentences(sentences)
        refined = report.model_copy(update={"sentences": sentences})
        refined.autofill_references()
        refined.metadata["refinement"] = {
            "confidence_threshold": self.config.confidence_threshold,
            "char_limit": self.config.char_limit,
            "word_limit": self.config.word_limit,
        }
        return refined

    def filter_citation_support(self, sentences: list[CitedSentence]) -> list[CitedSentence]:
        """Keep sentences whose primary citation attests the claim."""
        kept: list[CitedSentence] = []
        for sentence in sentences:
            citation = sentence.primary_citation()
            if citation is None:
                continue
            user = f"sentence: {sentence.text}\n\ndocument:\n{citation.span.text}"
            payload = self._llm.complete_json(system=_ATTEST_SYSTEM, user=user)
            if bool(payload.get("supported", False)):
                kept.append(sentence)
        return kept

    def filter_nugget_coverage(
        self,
        sentences: list[CitedSentence],
        bank: CrucibleNuggetBank,
    ) -> list[CitedSentence]:
        """Keep sentences that cover their associated nugget question."""
        nugget_lookup = {nugget.alignment_text(): nugget for nugget in bank.nuggets}
        kept: list[CitedSentence] = []
        for sentence in sentences:
            question = sentence.metadata.get("question", "")
            nugget = nugget_lookup.get(question)
            if nugget is None:
                kept.append(sentence)
                continue
            answers = [answer.answer for answer in nugget.answers.values()] or [nugget.text]
            user = (
                f"question: {question}\n"
                f"gold_answers: {', '.join(answers)}\n\n"
                f"sentence: {sentence.text}"
            )
            payload = self._llm.complete_json(system=_COVERAGE_SYSTEM, user=user)
            if bool(payload.get("covers", False)):
                kept.append(sentence)
        return kept

    def choose_sentences(self, sentences: list[CitedSentence]) -> list[CitedSentence]:
        """Select up to ``sentences_per_nugget`` per nugget question."""
        by_question: dict[str, list[CitedSentence]] = {}
        for sentence in sentences:
            question = sentence.metadata.get("question", sentence.sentence_id)
            by_question.setdefault(question, []).append(sentence)

        chosen: list[CitedSentence] = []
        seen_fingerprints: set[str] = set()
        for candidates in by_question.values():
            above = [
                sentence
                for sentence in candidates
                if float(sentence.metadata.get("confidence", 0.0)) >= self.config.confidence_threshold
            ]
            if self.config.prefer_shorter:
                ranked = sorted(above, key=lambda sentence: len(sentence.text))
            else:
                ranked = sorted(
                    above,
                    key=lambda sentence: float(sentence.metadata.get("confidence", 0.0)),
                    reverse=True,
                )
            for sentence in ranked[: self.config.sentences_per_nugget]:
                fingerprint = self._fingerprint(sentence.text)
                if self.config.avoid_similar and fingerprint in seen_fingerprints:
                    continue
                chosen.append(sentence)
                seen_fingerprints.add(fingerprint)
        return chosen

    def chop_sentences(self, sentences: list[CitedSentence]) -> list[CitedSentence]:
        """Trim report to char/word limits by dropping lowest-confidence sentences."""
        trimmed = list(sentences)
        while self._too_long(trimmed) and trimmed:
            lowest = min(
                trimmed,
                key=lambda sentence: float(sentence.metadata.get("confidence", 0.0)),
            )
            trimmed.remove(lowest)
        return trimmed

    def _too_long(self, sentences: list[CitedSentence]) -> bool:
        if self.config.word_limit is not None:
            return self._word_length(sentences) > self.config.word_limit
        if self.config.char_limit is not None:
            return self._char_length(sentences) > self.config.char_limit
        return False

    @staticmethod
    def _char_length(sentences: list[CitedSentence]) -> int:
        return sum(len(unicodedata.normalize("NFKC", sentence.text)) for sentence in sentences)

    @staticmethod
    def _word_length(sentences: list[CitedSentence]) -> int:
        lengths = []
        for sentence in sentences:
            normalized = unicodedata.normalize("NFKC", sentence.text)
            lengths.append(len(normalized.split()))
            lengths.append(len(re.findall(r"\w+", normalized)))
        return max(lengths) if lengths else 0

    @staticmethod
    def _fingerprint(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text.lower())
        return " ".join(normalized.split())
