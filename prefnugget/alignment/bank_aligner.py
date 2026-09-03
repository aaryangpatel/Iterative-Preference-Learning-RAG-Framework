"""Semantic alignment of PrefNugget question banks via OpenRouter."""

from __future__ import annotations

from dataclasses import dataclass

from rag_framework.llm.client import OpenRouterLLM, get_llm_client

from prefnugget.models.nugget import NuggetQuestionBank

_PAIR_SYSTEM = """Determine if two exam questions target the same or overlapping information.
Respond with JSON only:
{"similarity": 0.0 to 1.0, "alignment_type": "equivalent|partial|unrelated", "reasoning": "..."}"""


@dataclass
class PrefNuggetAlignerConfig:
    """Thresholds for PrefNugget bank alignment.

    Parameters
    ----------
    partial_threshold : float
        Minimum similarity to emit a pair.
    equivalent_threshold : float
        Threshold for equivalent alignment type.
    """

    partial_threshold: float = 0.35
    equivalent_threshold: float = 0.60


class PrefNuggetBankAligner:
    """Align nugget question banks using OpenRouter LLM pairwise comparison.

    Parameters
    ----------
    config : PrefNuggetAlignerConfig | None
        Alignment thresholds.
    llm : OpenRouterLLM | None
        LLM client. Defaults to environment-configured OpenRouter client.
    """

    def __init__(
        self,
        config: PrefNuggetAlignerConfig | None = None,
        llm: OpenRouterLLM | None = None,
    ) -> None:
        self.config = config or PrefNuggetAlignerConfig()
        self._llm = llm or get_llm_client()

    @property
    def name(self) -> str:
        return "prefnugget_llm_bank_aligner"

    def align(self, bank_a: NuggetQuestionBank, bank_b: NuggetQuestionBank) -> dict:
        """Align two question banks for the same topic.

        Parameters
        ----------
        bank_a : NuggetQuestionBank
            Questions from extractor A.
        bank_b : NuggetQuestionBank
            Questions from extractor B.

        Returns
        -------
        dict
            Alignment pairs, unmatched ids, and metadata.
        """
        if bank_a.topic_id != bank_b.topic_id:
            raise ValueError(f"Topic mismatch: {bank_a.topic_id} vs {bank_b.topic_id}")

        pairs: list[dict] = []
        matched_a: set[str] = set()
        matched_b: set[str] = set()

        for question_a in bank_a.questions:
            best_score = 0.0
            best_b = None
            best_type = "unrelated"
            for question_b in bank_b.questions:
                if question_b.question_id in matched_b:
                    continue
                user = f"question_a: {question_a.text}\nquestion_b: {question_b.text}"
                payload = self._llm.complete_json(system=_PAIR_SYSTEM, user=user)
                score = float(payload.get("similarity", 0.0))
                alignment_type = str(payload.get("alignment_type", "unrelated"))
                if score > best_score:
                    best_score = score
                    best_b = question_b
                    best_type = alignment_type

            if best_b is not None and best_score >= self.config.partial_threshold:
                pairs.append(
                    {
                        "question_a_id": question_a.question_id,
                        "question_b_id": best_b.question_id,
                        "similarity": best_score,
                        "alignment_type": (
                            "equivalent"
                            if best_score >= self.config.equivalent_threshold
                            else best_type
                        ),
                    }
                )
                matched_a.add(question_a.question_id)
                matched_b.add(best_b.question_id)

        return {
            "topic_id": bank_a.topic_id,
            "pairs": pairs,
            "unmatched_a": [
                question.question_id
                for question in bank_a.questions
                if question.question_id not in matched_a
            ],
            "unmatched_b": [
                question.question_id
                for question in bank_b.questions
                if question.question_id not in matched_b
            ],
            "metadata": {"aligner": self.name, "llm_model": self._llm.model},
        }
