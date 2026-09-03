"""Semantic alignment of nuggets across two CRUCIBLE reports via OpenRouter."""

from __future__ import annotations

from dataclasses import dataclass

from rag_framework.llm.client import OpenRouterLLM, get_llm_client

from crucible.models.alignment import CrossReportAlignment, NuggetPairAlignment
from crucible.models.nugget import CrucibleNuggetBank

_PAIR_SYSTEM = """Determine if two atomic factual statements express the same or overlapping information.
Respond with JSON only:
{"similarity": 0.0 to 1.0, "alignment_type": "equivalent|partial|related|unrelated", "reasoning": "..."}"""


@dataclass
class CrucibleAlignerConfig:
    """Thresholds for CRUCIBLE cross-report nugget alignment.

    Parameters
    ----------
    partial_threshold : float
        Minimum similarity to emit a pair.
    """

    partial_threshold: float = 0.35


class CrucibleReportNuggetAligner:
    """Align nugget banks from two reports via OpenRouter LLM pairwise comparison.

    Parameters
    ----------
    config : CrucibleAlignerConfig | None
        Alignment thresholds.
    llm : OpenRouterLLM | None
        LLM client. Defaults to environment-configured OpenRouter client.
    """

    def __init__(
        self,
        config: CrucibleAlignerConfig | None = None,
        llm: OpenRouterLLM | None = None,
    ) -> None:
        self.config = config or CrucibleAlignerConfig()
        self._llm = llm or get_llm_client()

    @property
    def name(self) -> str:
        return "crucible_llm_report_aligner"

    def align(self, bank_a: CrucibleNuggetBank, bank_b: CrucibleNuggetBank) -> CrossReportAlignment:
        """Align two nugget banks for the same query.

        Parameters
        ----------
        bank_a : CrucibleNuggetBank
            Nuggets from report A.
        bank_b : CrucibleNuggetBank
            Nuggets from report B.

        Returns
        -------
        CrossReportAlignment
            Matched pairs and unmatched ids.
        """
        if bank_a.query_id != bank_b.query_id:
            raise ValueError(
                f"Query ids must match for alignment: {bank_a.query_id} vs {bank_b.query_id}"
            )

        pairs: list[NuggetPairAlignment] = []
        matched_a: set[str] = set()
        matched_b: set[str] = set()

        for nugget_a in bank_a.nuggets:
            best_score = 0.0
            best_b = None
            best_type = "unrelated"
            for nugget_b in bank_b.nuggets:
                if nugget_b.nugget_id in matched_b:
                    continue
                user = (
                    f"statement_a: {nugget_a.alignment_text()}\n"
                    f"statement_b: {nugget_b.alignment_text()}"
                )
                payload = self._llm.complete_json(system=_PAIR_SYSTEM, user=user)
                score = float(payload.get("similarity", 0.0))
                alignment_type = str(payload.get("alignment_type", "unrelated"))
                if score > best_score:
                    best_score = score
                    best_b = nugget_b
                    best_type = alignment_type

            if best_b is not None and best_score >= self.config.partial_threshold:
                pairs.append(
                    NuggetPairAlignment(
                        nugget_a_id=nugget_a.nugget_id,
                        nugget_b_id=best_b.nugget_id,
                        similarity=best_score,
                        alignment_type=best_type,
                        metadata={"method": self.name, "llm_model": self._llm.model},
                    )
                )
                matched_a.add(nugget_a.nugget_id)
                matched_b.add(best_b.nugget_id)

        unmatched_a = [nugget.nugget_id for nugget in bank_a.nuggets if nugget.nugget_id not in matched_a]
        unmatched_b = [nugget.nugget_id for nugget in bank_b.nuggets if nugget.nugget_id not in matched_b]

        return CrossReportAlignment(
            query_id=bank_a.query_id,
            bank_a_id=str(bank_a.metadata.get("report_id", "bank_a")),
            bank_b_id=str(bank_b.metadata.get("report_id", "bank_b")),
            pairs=pairs,
            unmatched_a=unmatched_a,
            unmatched_b=unmatched_b,
            metadata={"aligner": self.name, "llm_model": self._llm.model},
        )
