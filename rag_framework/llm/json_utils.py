"""JSON parsing helpers for LLM responses."""

from __future__ import annotations

import json
import re


def parse_json_from_text(text: str) -> dict | list:
    """Extract and parse JSON object or array from LLM output.

    Parameters
    ----------
    text : str
        Raw LLM response text.

    Returns
    -------
    dict | list
        Parsed JSON value.

    Raises
    ------
    ValueError
        If no valid JSON is found.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char not in "{[":
            continue
        value, _end = decoder.raw_decode(stripped[index:])
        return value

    raise ValueError(f"No JSON object found in LLM response: {text[:200]}")
