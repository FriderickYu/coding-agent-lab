"""Optional, bounded GitHub code search: one GET endpoint, public results, small records."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

API_URL = "https://api.github.com/search/code"
GITHUB_WEB_PREFIX = "https://github.com/"
MAX_QUERY_CHARS = 180
MAX_REQUESTS = 3
MAX_RESULTS = 3
MAX_SNIPPET_CHARS = 500
MAX_RESPONSE_BYTES = 1_000_000


class ResearchError(ValueError):
    """Raised for a rejected query, a used-up budget, or a failed search."""


class ResearchConfigError(ResearchError):
    """Raised when research is requested but GITHUB_TOKEN is not set."""


def read_token(environ: Mapping[str, str]) -> str:
    """Read GITHUB_TOKEN from the given environment at composition time."""
    token = (environ.get("GITHUB_TOKEN") or "").strip()
    if not token:
        raise ResearchConfigError(
            "--research needs a GITHUB_TOKEN environment variable (a short-lived, read-only token)."
        )
    return token


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so a request can never be steered to another URL."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def _default_opener() -> Callable[..., Any]:
    """An opener without redirect handling."""
    return urllib.request.build_opener(_NoRedirect()).open


class GitHubCodeSearch:
    """Implements ResearchPort with GitHub's code-search API and per-repair budgets."""

    def __init__(
        self,
        token: str,
        opener: Callable[..., Any] | None = None,
        max_requests: int = MAX_REQUESTS,
        max_results: int = MAX_RESULTS,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not isinstance(token, str) or not token.strip():
            raise ResearchConfigError("A GitHub token is required.")
        self._token = token.strip()
        self._open = opener or _default_opener()
        self.max_requests = max_requests
        self.max_results = max_results
        self.timeout_seconds = timeout_seconds
        self.requests_made = 0
        self.results_returned = 0

    def __repr__(self) -> str:
        return f"GitHubCodeSearch(requests_made={self.requests_made}, results_returned={self.results_returned})"

    def search_github(self, query: str) -> dict[str, Any]:
        """Search public code once and return query, retrieval time, and up to the remaining sources."""
        query = self._validate(query)
        if self.requests_made >= self.max_requests:
            raise ResearchError(f"Research request budget used: at most {self.max_requests} searches per repair.")
        remaining = self.max_results - self.results_returned
        if remaining <= 0:
            raise ResearchError(f"Research result budget used: at most {self.max_results} results per repair.")

        self.requests_made += 1
        payload = self._get(query, per_page=remaining)
        sources = self._public_sources(payload, limit=remaining)
        self.results_returned += len(sources)
        return {
            "query": query,
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sources": sources,
        }

    @staticmethod
    def _validate(query: str) -> str:
        """Accept one printable line of at most MAX_QUERY_CHARS characters."""
        if not isinstance(query, str) or not query.strip():
            raise ResearchError("Search query must not be empty.")
        if "\n" in query or "\r" in query:
            raise ResearchError("Search query must be a single line.")
        query = query.strip()
        if not query.isprintable():
            raise ResearchError("Search query must contain only printable characters.")
        if len(query) > MAX_QUERY_CHARS:
            raise ResearchError(f"Search query must be at most {MAX_QUERY_CHARS} characters.")
        return query

    def _get(self, query: str, per_page: int) -> dict[str, Any]:
        """Send one GET to the code-search endpoint and decode its JSON body."""
        url = f"{API_URL}?{urllib.parse.urlencode({'q': query, 'per_page': per_page})}"
        request = urllib.request.Request(
            url,
            method="GET",
            headers={
                "Accept": "application/vnd.github.text-match+json",
                "Authorization": f"Bearer {self._token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "coding-agent-lab-research",
            },
        )
        try:
            with self._open(request, timeout=self.timeout_seconds) as response:
                body = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:  # Report the status only: never headers or body.
            raise ResearchError(f"GitHub search failed with HTTP {exc.code}.") from None
        except (urllib.error.URLError, OSError) as exc:
            raise ResearchError(f"GitHub search could not connect: {type(exc).__name__}.") from None
        if len(body) > MAX_RESPONSE_BYTES:
            raise ResearchError("GitHub search response was too large.")
        try:
            payload = json.loads(body)
        except ValueError:
            raise ResearchError("GitHub search returned invalid JSON.") from None
        if not isinstance(payload, dict):
            raise ResearchError("GitHub search returned an unexpected response.")
        return payload

    @staticmethod
    def _public_sources(payload: dict[str, Any], limit: int) -> list[dict[str, str]]:
        """Keep only clearly public results, reduced to repository, path, URL, and a short snippet."""
        sources: list[dict[str, str]] = []
        for item in payload.get("items") or []:
            if len(sources) >= limit:
                break
            if not isinstance(item, dict):
                continue
            repository = item.get("repository") if isinstance(item.get("repository"), dict) else {}
            if repository.get("private") is not False or repository.get("visibility", "public") != "public":
                continue  # Private, or not explicitly public: discard.
            full_name, path, url = repository.get("full_name"), item.get("path"), item.get("html_url")
            if not all(isinstance(value, str) and value for value in (full_name, path, url)):
                continue
            if not url.startswith(GITHUB_WEB_PREFIX):
                continue
            fragments = [
                match["fragment"]
                for match in item.get("text_matches") or []
                if isinstance(match, dict) and isinstance(match.get("fragment"), str)
            ]
            sources.append(
                {
                    "repository": full_name,
                    "path": path,
                    "url": url,
                    "snippet": "\n...\n".join(fragments)[:MAX_SNIPPET_CHARS],
                }
            )
        return sources
