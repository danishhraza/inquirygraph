import ipaddress
import re
import socket
import time
import uuid
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from ddgs import DDGS

from inquirygraph.config.settings import settings


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class FetchedSource:
    source_id: str
    title: str
    url: str
    text: str


def web_search(query: str, max_results: int | None = None) -> list[SearchResult]:
    limit = max_results or settings.max_web_results

    if settings.tavily_api_key:
        return _tavily_search(query, limit)
    return _duckduckgo_search(query, limit)


def _duckduckgo_search(query: str, limit: int) -> list[SearchResult]:
    for attempt in range(2):
        try:
            with DDGS(timeout=10) as ddgs:
                hits = list(ddgs.text(query, max_results=limit))
            return [
                SearchResult(title=h.get("title", ""), url=h.get("href", ""), snippet=h.get("body", ""))
                for h in hits
                if h.get("href")
            ]
        except Exception:
            # DuckDuckGo rate-limits or returns transient errors; retry once,
            # then degrade to no results rather than crashing the pipeline.
            if attempt == 0:
                time.sleep(2)
    return []


def _tavily_search(query: str, limit: int) -> list[SearchResult]:
    # Retry once on transient failures (DNS blips, dropped connections, 429/5xx);
    # a second failure propagates so the caller can record it against the task.
    for attempt in range(2):
        try:
            response = httpx.post(
                "https://api.tavily.com/search",
                json={"api_key": settings.tavily_api_key, "query": query, "max_results": limit},
                timeout=30.0,
            )
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            break
        except (httpx.TransportError, httpx.HTTPStatusError):
            if attempt == 1:
                raise
            time.sleep(2)
    response.raise_for_status()
    data = response.json()
    return [
        SearchResult(title=r.get("title", ""), url=r.get("url", ""), snippet=r.get("content", ""))
        for r in data.get("results", [])
        if r.get("url")
    ]


def _is_private_host(host: str) -> bool:
    """Block requests to loopback, link-local, and private network hosts (SSRF guard)."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # Host is a domain name — resolve it and check every address.
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror:
            return True
        ips = {info[4][0] for info in infos}
        return any(ipaddress.ip_address(ip).is_private for ip in ips)

    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
    )


def fetch_url(url: str, title: str = "") -> FetchedSource | None:
    host = urlparse(url).hostname or ""
    if _is_private_host(host):
        return None
    try:
        response = httpx.get(
            url,
            timeout=httpx.Timeout(8.0, connect=3.0),
            follow_redirects=True,
            headers={"User-Agent": "InquiryGraph/0.1 (research bot)"},
        )
        response.raise_for_status()
        text = _extract_text(response.text)
        if len(text) < 200:
            return None
        return FetchedSource(
            source_id=str(uuid.uuid4()),
            title=title or url,
            url=url,
            text=text[:12000],
        )
    except Exception:
        return None


def _extract_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    html = html.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", html).strip()
