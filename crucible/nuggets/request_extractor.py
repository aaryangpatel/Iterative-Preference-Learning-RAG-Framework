"""Generate CRUCIBLE v3 nugget banks from RAGTIME report requests."""

from __future__ import annotations

from rag_framework.llm.client import OpenRouterLLM, get_llm_client

from crucible.models.nugget import CrucibleNugget, CrucibleNuggetBank, NuggetAnswer

_AUTO_NUGGET_SYSTEM = """Create evaluation nuggets for a report request.
Each nugget is one short factual question about a single specific fact relevant to the request.
Provide 3-5 short paraphrases of the SAME fact as gold answers (not different facts).

Do not ask for lists or multiple items in one question.
Respond with JSON only:
{"nuggets": [{"question": "...", "answers": ["paraphrase1", "paraphrase2", "paraphrase3"]}],
 "confidence": 0.0 to 1.0}"""


class RequestNuggetExtractor:
    """Generate nugget banks from RAGTIME requests via OpenRouter.

    Mimics ``AutoNuggetsGpt`` / ``AutoNuggetsLlama`` (simplified for free-tier LLM).

    Parameters
    ----------
    max_nuggets : int
        Maximum nuggets to generate per request.
    llm : OpenRouterLLM | None
        LLM client. Defaults to environment-configured OpenRouter client.
    """

    def __init__(
        self,
        max_nuggets: int = 20,
        llm: OpenRouterLLM | None = None,
    ) -> None:
        self._max_nuggets = max_nuggets
        self._llm = llm or get_llm_client()

    @property
    def name(self) -> str:
        return "crucible_llm_request_nugget_extractor"

    def extract(self, request: dict) -> CrucibleNuggetBank:
        """Generate a v3 nugget bank from one RAGTIME request record.

        Parameters
        ----------
        request : dict
            Request with ``request_id``, ``title``, ``background``, ``problem_statement``.

        Returns
        -------
        CrucibleNuggetBank
            Generated nugget bank.
        """
        query_id = str(request.get("request_id", request.get("topic_id", "0")))
        user = (
            f"request_id: {query_id}\n"
            f"title: {request.get('title', '')}\n"
            f"background: {request.get('background', '')}\n"
            f"problem_statement: {request.get('problem_statement', '')}\n"
            f"max_nuggets: {self._max_nuggets}"
        )
        payload = self._llm.complete_json(system=_AUTO_NUGGET_SYSTEM, user=user)
        confidence = float(payload.get("confidence", 0.8))
        nuggets: list[CrucibleNugget] = []

        for index, entry in enumerate(payload.get("nuggets", [])[: self._max_nuggets]):
            question = str(entry.get("question", "")).strip()
            if not question:
                continue
            answer_texts = [str(answer).strip() for answer in entry.get("answers", []) if str(answer).strip()]
            if not answer_texts:
                answer_texts = [question]
            answers = {
                answer: NuggetAnswer(answer=answer, references=[])
                for answer in answer_texts
            }
            nuggets.append(
                CrucibleNugget(
                    nugget_id=CrucibleNugget.make_question_id(question),
                    text=question,
                    question=question,
                    answers=answers,
                    confidence=confidence,
                    metadata={"extractor": self.name, "llm_model": self._llm.model},
                )
            )

        return CrucibleNuggetBank(
            query_id=query_id,
            title_query=request.get("title"),
            nuggets=nuggets,
            metadata={"extractor": self.name, "llm_model": self._llm.model},
        )
