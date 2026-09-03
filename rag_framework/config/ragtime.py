"""RAGTIME search API configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class RagtimeApiConfig:
    """Credentials and defaults for the TREC RAGTIME search service.

    Parameters
    ----------
    api_url : str
        Base URL assigned after TREC registration (no trailing slash).
    bearer_token : str
        Authorization bearer token from TREC.
    pipeline : str
        RoutIR pipeline alias (``ragtime1`` for 2025 languages).
    collection : str
        Collection alias for ``/content`` document fetch.
    dry_run : bool
        When True, HTTP calls are skipped (benchmark wiring / tests).
    timeout_seconds : float
        HTTP request timeout.
    """

    api_url: str
    bearer_token: str
    pipeline: str = "ragtime1"
    collection: str = "ragtime1"
    dry_run: bool = False
    timeout_seconds: float = 120.0
    user_agent: str = "rag-framework/0.1.0 (TREC-RAGTIME-benchmark)"

    @property
    def is_configured(self) -> bool:
        """Return True when URL and token are non-empty."""
        return bool(self.api_url.strip() and self.bearer_token.strip())

    @property
    def search_endpoint(self) -> str:
        """Full URL for the search ``/pipeline`` endpoint."""
        return f"{self.api_url.rstrip('/')}/pipeline"

    @property
    def content_endpoint(self) -> str:
        """Full URL for the document ``/content`` endpoint."""
        return f"{self.api_url.rstrip('/')}/content"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_ragtime_api_config() -> RagtimeApiConfig:
    """Load RAGTIME API settings from environment.

    Environment variables
    ---------------------
    RAGTIME_API_URL : str
        Base endpoint URL from TREC registration.
    RAGTIME_BEARER_TOKEN : str
        Bearer token from TREC registration.
    RAGTIME_PIPELINE : str
        Search pipeline alias (default ``ragtime1``).
    RAGTIME_COLLECTION : str
        Content collection alias (default ``ragtime1``).
    RAGTIME_DRY_RUN : bool
        Skip HTTP when ``true`` (also respects ``BENCHMARK_DRY_RUN``).
    RAGTIME_TIMEOUT_SECONDS : float
        Request timeout (default 60).
    RAGTIME_USER_AGENT : str
        HTTP User-Agent (Cloudflare may block default Python urllib).

    Returns
    -------
    RagtimeApiConfig
        Parsed configuration (may be unconfigured when dry-run testing).
    """
    dry_run = _env_bool("RAGTIME_DRY_RUN") or _env_bool("BENCHMARK_DRY_RUN")
    timeout_raw = os.environ.get("RAGTIME_TIMEOUT_SECONDS", "120")
    return RagtimeApiConfig(
        api_url=os.environ.get("RAGTIME_API_URL", "").strip(),
        bearer_token=os.environ.get("RAGTIME_BEARER_TOKEN", "").strip(),
        pipeline=os.environ.get("RAGTIME_PIPELINE", "ragtime1").strip(),
        collection=os.environ.get("RAGTIME_COLLECTION", "ragtime1").strip(),
        dry_run=dry_run,
        timeout_seconds=float(timeout_raw),
        user_agent=os.environ.get(
            "RAGTIME_USER_AGENT",
            "rag-framework/0.1.0 (TREC-RAGTIME-benchmark)",
        ).strip(),
    )


def clear_ragtime_api_config_cache() -> None:
    """Clear cached RAGTIME config (for tests)."""
    get_ragtime_api_config.cache_clear()
