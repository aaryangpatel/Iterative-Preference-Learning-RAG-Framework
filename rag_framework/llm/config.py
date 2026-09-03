"""OpenRouter LLM configuration loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class LLMConfig(BaseModel):
    """OpenRouter API configuration.

    Parameters
    ----------
    api_key : str
        OpenRouter API key (``OPENROUTER_API_KEY``).
    base_url : str
        API base URL (``OPENROUTER_BASE_URL``).
    model : str
        Model slug (``OPENROUTER_MODEL``).
    site_url : str | None
        Optional HTTP-Referer for OpenRouter rankings.
    site_name : str | None
        Optional X-Title for OpenRouter rankings.
    """

    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "openrouter/free"
    site_url: str | None = None
    site_name: str | None = None
    min_request_delay_seconds: float = 0.0
    max_rate_limit_retries: int = 8
    default_max_tokens: int = 1024

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Load configuration from environment variables.

        Returns
        -------
        LLMConfig
            Parsed config.

        Raises
        ------
        ValueError
            If ``OPENROUTER_API_KEY`` is missing.
        """
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return cls(
            api_key=api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip(),
            model=os.getenv("OPENROUTER_MODEL", "openrouter/free").strip(),
            site_url=os.getenv("OPENROUTER_SITE_URL"),
            site_name=os.getenv("OPENROUTER_SITE_NAME"),
            min_request_delay_seconds=float(os.getenv("OPENROUTER_MIN_DELAY_SECONDS", "2.5")),
            max_rate_limit_retries=int(os.getenv("OPENROUTER_MAX_RETRIES", "8")),
            default_max_tokens=int(os.getenv("OPENROUTER_MAX_TOKENS", "1024")),
        )


@lru_cache(maxsize=1)
def get_llm_config() -> LLMConfig:
    """Return cached LLM config from environment."""
    return LLMConfig.from_env()


def clear_llm_config_cache() -> None:
    """Clear cached config (useful after env changes in tests)."""
    get_llm_config.cache_clear()
