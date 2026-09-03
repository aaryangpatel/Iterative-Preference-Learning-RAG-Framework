"""PrefNugget nugget question models (atomic questions, not declarative facts)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class NuggetQuestion(BaseModel):
    """Atomic exam question extracted by a PrefNugget judge.

    PrefNugget nuggets are brief self-contained QUESTIONS used to grade
    responses. This is intentionally separate from CRUCIBLE declarative facts.

    Parameters
    ----------
    question_id : str
        Stable question identifier.
    text : str
        Atomic question text.
    extraction_method : str
        contrastive | grounded | queryonly.
    source_run_id : str | None
        Run that produced this question (if grounded/contrastive).
    confidence : float
        Extractor confidence 0-1.
    metadata : dict[str, Any]
        Prompt and iteration metadata.
    """

    question_id: str
    text: str
    extraction_method: Literal["contrastive", "grounded", "queryonly"] = "queryonly"
    source_run_id: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def alignment_text(self) -> str:
        """Text used for semantic bank alignment."""
        return self.text


class NuggetQuestionBank(BaseModel):
    """Bank of exam questions for one topic (PrefNugget ``*.nuggets.jsonl``).

    Parameters
    ----------
    topic_id : str
        Topic / request id.
    questions : list[NuggetQuestion]
        Up to 20 atomic questions per topic.
    metadata : dict[str, Any]
        Judge variant and workflow metadata.
    """

    topic_id: str
    questions: list[NuggetQuestion] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def deduplicate(self, similarity_threshold: float = 0.85) -> NuggetQuestionBank:
        """Remove near-duplicate questions by token Jaccard similarity.

        Parameters
        ----------
        similarity_threshold : float
            Jaccard threshold for merging duplicates.

        Returns
        -------
        NuggetQuestionBank
            Deduplicated bank.
        """
        from rag_framework.similarity import tokenize

        kept: list[NuggetQuestion] = []
        kept_tokens: list[set[str]] = []
        for question in self.questions:
            tokens = set(tokenize(question.text))
            duplicate = False
            for existing_tokens in kept_tokens:
                jaccard = len(tokens & existing_tokens) / len(tokens | existing_tokens)
                if jaccard >= similarity_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(question)
                kept_tokens.append(tokens)
        return self.model_copy(update={"questions": kept})
