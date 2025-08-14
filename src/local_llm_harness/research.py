"""Bounded SearXNG search and untrusted result-page collection."""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Sequence
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict

from local_llm_harness.config import SearxNGSettings


class ResearchError(RuntimeError):
    """Mandatory external research could not be completed safely."""


class WebSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    title: str
    url: str
    snippet: str = ""
    content: str = ""
    fetch_error: str | None = None


class SearxNGClient:
    def __init__(
        self,
        settings: SearxNGSettings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self._client = client or httpx.AsyncClient(
            timeout=settings.timeout_seconds,
            follow_redirects=False,
        )
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def search_and_fetch(self, queries: Sequence[str]) -> list[WebSource]:
        if not queries:
            raise ResearchError("at least one research query is required")
        sources: list[WebSource] = []
        seen_urls: set[str] = set()
        for query in queries:
            results = await self._search(query)
            for raw in results:
                url = raw["url"]
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                content = ""
                fetch_error = None
                if (
                    len([source for source in sources if source.query == query])
                    < self.settings.fetch_result_limit
                ):
                    try:
                        content = await self._fetch_text(url)
                    except ResearchError as exc:
                        fetch_error = str(exc)
                sources.append(
                    WebSource(
                        query=query,
                        title=raw["title"],
                        url=url,
                        snippet=raw.get("content", ""),
                        content=content,
                        fetch_error=fetch_error,
                    )
                )
        if not sources:
            raise ResearchError("SearXNG returned no safe research results")
        return sources

    async def _search(self, query: str) -> list[dict[str, str]]:
        endpoint = f"{self.settings.base_url.rstrip('/')}/search"
        try:
            response = await self._client.get(
                endpoint,
                params={"q": query, "format": "json", "safesearch": 1},
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ResearchError(f"SearXNG search failed for {query!r}: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ResearchError(f"SearXNG returned malformed JSON for {query!r}")

        results: list[dict[str, str]] = []
        for item in payload["results"][: self.settings.result_limit]:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            url = item.get("url")
            content = item.get("content", "")
            if not isinstance(title, str) or not isinstance(url, str):
                continue
            if not _is_safe_result_url(url):
                continue
            results.append(
                {
                    "title": title,
                    "url": url,
                    "content": content if isinstance(content, str) else "",
                }
            )
        return results

    async def _fetch_text(self, url: str) -> str:
        if not _is_safe_result_url(url):
            raise ResearchError(f"unsafe research result URL: {url}")
        try:
            async with self._client.stream("GET", url, follow_redirects=False) as response:
                if response.is_redirect:
                    raise ResearchError("research result redirects are not followed")
                response.raise_for_status()
                if not _is_safe_result_url(str(response.url)):
                    raise ResearchError("research result redirected to an unsafe URL")
                content_type = response.headers.get("content-type", "").lower()
                if not (
                    content_type.startswith("text/")
                    or "application/json" in content_type
                    or "application/xhtml+xml" in content_type
                ):
                    raise ResearchError(
                        f"unsupported research content type: {content_type or 'unknown'}"
                    )
                collected = bytearray()
                async for chunk in response.aiter_bytes():
                    remaining = self.settings.max_fetch_bytes - len(collected)
                    if remaining <= 0:
                        break
                    collected.extend(chunk[:remaining])
                    if len(collected) >= self.settings.max_fetch_bytes:
                        break
        except httpx.HTTPError as exc:
            raise ResearchError(f"research result fetch failed: {exc}") from exc
        return bytes(collected).decode(response.encoding or "utf-8", errors="replace")


def _is_safe_result_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return False
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return True
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def render_untrusted_sources(sources: Sequence[WebSource]) -> str:
    """Serialize web material as explicitly delimited untrusted data."""

    blocks = []
    for index, source in enumerate(sources, start=1):
        payload = json.dumps(source.model_dump(mode="json"), indent=2)
        blocks.append(
            f'<untrusted-web-content source="{index}">\n{payload}\n</untrusted-web-content>'
        )
    return "\n\n".join(blocks)
