"""Track unique nugget questions per topic during iterative extraction."""

from __future__ import annotations

import collections


class QuestionTracker:
    """Track unique questions and occurrence counts per topic.

    Mimics ``QuestionTracker`` from prefnugget-starterkit ``nugget_judge_base.py``.

    Parameters
    ----------
    None
        Instantiate empty; questions are added via ``add`` / ``add_all``.
    """

    def __init__(self) -> None:
        self._counts: dict[str, dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        self._topics_done: set[str] = set()

    def add(self, query_id: str, question: str, count: int = 1) -> None:
        """Add one question, incrementing its count."""
        self._counts[query_id][question] += count

    def add_all(self, query_id: str, questions: list[str], count: int = 1) -> None:
        """Add multiple questions."""
        for question in questions:
            self.add(query_id, question, count=count)

    def questions(self, query_id: str) -> list[str]:
        """Return unique questions for a topic."""
        return list(self._counts[query_id].keys())

    def num_questions(self, query_id: str) -> int:
        """Return number of unique questions for a topic."""
        return len(self._counts[query_id])

    def is_done(self, query_id: str) -> bool:
        """Return True if topic has reached the question cap."""
        return query_id in self._topics_done

    def mark_done(self, query_id: str) -> None:
        """Mark topic as done collecting questions."""
        self._topics_done.add(query_id)

    def check_and_mark_done(self, query_id: str, stop_at_count: int) -> bool:
        """Mark done when unique question count exceeds threshold."""
        if self.num_questions(query_id) >= stop_at_count:
            self._topics_done.add(query_id)
            return True
        return False
