"""OpenRouter LLM client (OpenAI-compatible API)."""

from __future__ import annotations

import time
from functools import lru_cache

from openai import APIStatusError, OpenAI, RateLimitError

from rag_framework.llm.config import LLMConfig, get_llm_config
from rag_framework.llm.json_utils import parse_json_from_text


class OpenRouterLLM:
    """Chat-completions client for OpenRouter.

    Parameters
    ----------
    config : LLMConfig | None
        API configuration. Loads from environment when omitted.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or get_llm_config()
        extra_headers: dict[str, str] = {}
        if self.config.site_url:
            extra_headers["HTTP-Referer"] = self.config.site_url
        if self.config.site_name:
            extra_headers["X-Title"] = self.config.site_name

        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            default_headers=extra_headers or None,
        )
        self._last_request_at: float = 0.0

    @property
    def model(self) -> str:
        """Configured model slug."""
        return self.config.model

    def _wait_for_rate_limit(self) -> None:
        delay = self.config.min_request_delay_seconds
        if delay <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < delay:
            time.sleep(delay - elapsed)

    def _rate_limit_backoff(self, attempt: int) -> None:
        time.sleep(min(60.0, 5.0 * (2**attempt)))

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        api_retries: int | None = None,
    ) -> str:
        """Send a chat completion request.

        Parameters
        ----------
        system : str
            System prompt.
        user : str
            User prompt.
        temperature : float
            Sampling temperature.
        max_tokens : int
            Maximum completion tokens.
        api_retries : int | None
            Retries on rate limit errors (defaults to config max).

        Returns
        -------
        str
            Assistant message content.
        """
        retries = api_retries if api_retries is not None else self.config.max_rate_limit_retries
        token_budget = max_tokens if max_tokens is not None else self.config.default_max_tokens
        last_error: Exception | None = None
        for attempt in range(retries):
            self._wait_for_rate_limit()
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=token_budget,
                )
                message = response.choices[0].message.content
                if not message:
                    reasoning = getattr(response.choices[0].message, "reasoning", None)
                    if reasoning:
                        message = reasoning
                if not (message or "").strip():
                    if attempt + 1 < retries:
                        self._rate_limit_backoff(attempt)
                        continue
                    raise ValueError("Empty LLM response")
                self._last_request_at = time.monotonic()
                return message or ""
            except RateLimitError as error:
                last_error = error
                self._rate_limit_backoff(attempt)
            except APIStatusError as error:
                if error.status_code == 429:
                    last_error = error
                    self._rate_limit_backoff(attempt)
                    continue
                if error.status_code == 402 and token_budget > 256:
                    token_budget = max(256, token_budget // 2)
                    last_error = error
                    continue
                raise
        if last_error is not None:
            raise last_error
        return ""

    def complete_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        retries: int = 8,
    ) -> dict | list:
        """Chat and parse a JSON object or array from the response.

        Parameters
        ----------
        system : str
            System prompt (should instruct JSON-only output).
        user : str
            User prompt.
        temperature : float
            Sampling temperature.
        max_tokens : int
            Maximum completion tokens.
        retries : int
            Parse attempts before raising.

        Returns
        -------
        dict | list
            Parsed JSON value.
        """
        last_error: ValueError | None = None
        for attempt in range(retries):
            prompt_user = user
            if attempt > 0:
                prompt_user = (
                    f"{user}\n\n"
                    "IMPORTANT: Respond with valid JSON only. No prose, labels, or markdown."
                )
            text = self.chat(
                system=system,
                user=prompt_user,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if not text.strip():
                last_error = ValueError("Empty LLM response")
                continue
            try:
                return parse_json_from_text(text)
            except ValueError as error:
                last_error = error
        if last_error is not None:
            raise last_error
        return {}


@lru_cache(maxsize=1)
def get_llm_client() -> OpenRouterLLM:
    """Return a shared OpenRouter client configured from environment."""
    return OpenRouterLLM()


def clear_llm_client_cache() -> None:
    """Clear cached shared LLM client."""
    get_llm_client.cache_clear()
