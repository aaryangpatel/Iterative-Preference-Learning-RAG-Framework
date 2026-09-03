"""PrefNugget topic / report-request models (kiddie format)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PrefNuggetTopic(BaseModel):
    """Auto-Judge topic in kiddie / RAGTIME request style.

    Parameters
    ----------
    request_id : str
        Topic identifier (``leaf``, ``cloud``, etc. in kiddie).
    title : str
        Short query title.
    problem_statement : str
        Problem to address.
    background : str
        User background context.
    metadata : dict[str, Any]
        Extra topic fields.
    """

    request_id: str
    title: str
    problem_statement: str
    background: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def grading_context(self) -> str:
        """Concatenated context for judge prompts."""
        return f"{self.title} {self.background} {self.problem_statement}".strip()
