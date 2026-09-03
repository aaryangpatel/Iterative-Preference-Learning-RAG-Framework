"""Align CRUCIBLE nuggets to document text segments via OpenRouter."""

from __future__ import annotations

from rag_framework.llm.client import OpenRouterLLM, get_llm_client

from crucible.models.alignment import NuggetAlignmentResult
from crucible.models.nugget import CrucibleNugget

_ALIGN_SYSTEM = """Can the question be answered based on the available context? Choose one:
5: highly relevant, complete, and accurate
4: mostly relevant and complete
3: partially relevant
2: limited relevance
1: minimally relevant
0: not relevant at all
Also extract the shortest supporting text segment from the context.
Respond with JSON only:
{"answerability": 0-5, "extracted_text_segment": "...", "is_match": true/false, "confidence": 0.0 to 1.0}"""


class CrucibleTextAligner:
    """Align nuggets to document text using OpenRouter LLM (answerability prompt).

    Mimics CRUCIBLE ``NuggetAnswerability`` / argue_eval alignment.

    Parameters
    ----------
    match_threshold : int
        Minimum answerability score for ``is_match=True``.
    llm : OpenRouterLLM | None
        LLM client. Defaults to environment-configured OpenRouter client.
    """

    def __init__(
        self,
        match_threshold: int = 2,
        llm: OpenRouterLLM | None = None,
    ) -> None:
        self._match_threshold = match_threshold
        self._llm = llm or get_llm_client()

    @property
    def name(self) -> str:
        return "crucible_llm_text_aligner"

    def align_nugget_to_document(
        self,
        nugget: CrucibleNugget,
        document_text: str,
        query_id: str | None = None,
    ) -> NuggetAlignmentResult:
        """Align one nugget to a full document text.

        Parameters
        ----------
        nugget : CrucibleNugget
            Nugget to align.
        document_text : str
            Full document body.
        query_id : str | None
            Optional topic id.

        Returns
        -------
        NuggetAlignmentResult
            Alignment result with extracted segment and scores.
        """
        user = (
            f"question: {nugget.alignment_text()}\n\n"
            f"context:\n{document_text}"
        )
        payload = self._llm.complete_json(system=_ALIGN_SYSTEM, user=user)
        answerability = int(payload.get("answerability", 0))
        segment = str(payload.get("extracted_text_segment", "")).strip()
        is_match = bool(payload.get("is_match", answerability >= self._match_threshold))
        confidence = float(payload.get("confidence", answerability / 5.0))

        return NuggetAlignmentResult(
            nugget_text=nugget.alignment_text(),
            nugget_id=nugget.nugget_id,
            query_id=query_id,
            source_document=document_text,
            extracted_text_segment=segment,
            sentence=segment,
            is_match=is_match,
            match_score=answerability / 5.0,
            answerability=answerability,
            confidence=confidence,
            metadata={"aligner": self.name, "llm_model": self._llm.model},
        )
