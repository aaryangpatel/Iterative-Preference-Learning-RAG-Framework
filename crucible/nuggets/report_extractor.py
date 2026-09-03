"""Extract atomic factual nuggets from CRUCIBLE cited reports via OpenRouter."""

from __future__ import annotations

from rag_framework.llm.client import OpenRouterLLM, get_llm_client

from crucible.models.citation import CitedReport
from crucible.models.nugget import CrucibleNugget, CrucibleNuggetBank, NuggetAnswer

_EXTRACT_SYSTEM = """Decompose a cited report sentence into atomic factual statements.
Each statement must be self-contained and supported only by information in the sentence.
Do not add facts not present in the sentence.
Respond with JSON only:
{"statements": [{"text": "...", "question": "brief question form"}], "confidence": 0.0 to 1.0}"""


class ReportNuggetExtractor:
    """Decompose cited report sentences into atomic factual nuggets via OpenRouter.

    Each nugget inherits parent sentence citations.

    Parameters
    ----------
    llm : OpenRouterLLM | None
        LLM client. Defaults to environment-configured OpenRouter client.
    """

    def __init__(self, llm: OpenRouterLLM | None = None) -> None:
        self._llm = llm or get_llm_client()

    @property
    def name(self) -> str:
        return "crucible_llm_extractor"

    def extract(self, report: CitedReport) -> CrucibleNuggetBank:
        """Extract a nugget bank from one cited report.

        Parameters
        ----------
        report : CitedReport
            Input report with per-sentence citations.

        Returns
        -------
        CrucibleNuggetBank
            Atomic factual nuggets with inherited citations.
        """
        nuggets: list[CrucibleNugget] = []
        for sentence in report.sentences:
            user = f"sentence: {sentence.text}"
            payload = self._llm.complete_json(system=_EXTRACT_SYSTEM, user=user)
            statements = payload.get("statements", [])
            confidence = float(payload.get("confidence", 0.8))

            for clause_index, entry in enumerate(statements):
                text = str(entry.get("text", "")).strip()
                if not text:
                    continue
                question = str(entry.get("question", f"What is true about: {text}?")).strip()
                nugget_id = f"{sentence.sentence_id}-n{clause_index}"
                nuggets.append(
                    CrucibleNugget(
                        nugget_id=nugget_id,
                        text=text,
                        question=question,
                        citations=list(sentence.citations),
                        source_sentence_id=sentence.sentence_id,
                        source_sentence_text=sentence.text,
                        answers={
                            text: NuggetAnswer(
                                answer=text,
                                references=[
                                    {"doc_id": citation.span.doc_id}
                                    for citation in sentence.citations
                                ],
                            )
                        },
                        confidence=confidence,
                        metadata={"extractor": self.name, "llm_model": self._llm.model},
                    )
                )

        return CrucibleNuggetBank(
            query_id=report.topic_id,
            title_query=report.metadata.get("title"),
            nuggets=nuggets,
            metadata={"report_id": report.report_id, "extractor": self.name, "llm_model": self._llm.model},
        )
