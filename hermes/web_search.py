"""
Web search tool for Hermes Edge agent.
Bypasses Google/blocks via DuckDuckGo (no API key) + proxy rotation + UA rotation.
"""

import json
import logging
import random
import urllib.request
import urllib.parse
import urllib.error
from dataclasses import dataclass

log = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.6478.122 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.6478.122 Mobile Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.6478.122 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.6478.122 Safari/537.36",
]

PROXIES = [
    None,  # direct connection first
]


class WebSearchError(Exception):
    pass


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str


def search_duckduckgo(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search via DuckDuckGo Lite API. No API key needed, no rate limit headers."""
    url = "https://lite.duckduckgo.com/lite/"
    data = urllib.parse.urlencode({"q": query}).encode()
    ua = random.choice(USER_AGENTS)
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("User-Agent", ua)
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Accept", "text/html,application/xhtml+xml")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise WebSearchError(f"DuckDuckGo blocked: {exc}") from exc

    return _parse_ddg_html(html, max_results)


def _parse_ddg_html(html: str, max_results: int) -> list[SearchResult]:
    """Parse DuckDuckGo Lite HTML results (fast, no external deps)."""
    import re

    results = []
    # DDG Lite uses mixed quotes: <a ... class='result-link'>TITLE</a>
    # href can have single or double quotes
    link_pattern = re.compile(
        r"<a[^>]+href=([\"'])([^\"']+)\1[^>]*class='result-link'[^>]*>([^<]+)</a>"
    )
    snippet_pattern = re.compile(r"<td class='result-snippet'>\s*(.*?)\s*</td>", re.DOTALL)

    links = link_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for i, (_, url, title) in enumerate(links[:max_results]):
        snippet = snippets[i] if i < len(snippets) else ""
        results.append(SearchResult(
            title=title.strip(),
            snippet=snippet.strip(),
            url=url,
        ))

    return results


def search_html_scrape(query: str, max_results: int = 5) -> list[SearchResult]:
    """Fallback: scrape HTML search from DuckDuckGo HTML endpoint."""
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    ua = random.choice(USER_AGENTS)
    req = urllib.request.Request(url)
    req.add_header("User-Agent", ua)
    req.add_header("Accept", "text/html")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise WebSearchError(f"HTML search blocked: {exc}") from exc

    import re
    results = []
    for match in re.finditer(
        r'<a[^>]+href=(["\'])([^"\']+)\1[^>]*>([^<]+)</a>',
        html,
    ):
        if len(results) >= max_results:
            break
        _, url, title = match.groups()
        results.append(SearchResult(title=title.strip(), snippet="", url=url))

    return results


def web_search(query: str, max_results: int = 5) -> str:
    """Tool-callable web search. Tries DuckDuckGo Lite first, falls back to HTML scrape."""
    try:
        results = search_duckduckgo(query, max_results)
    except WebSearchError:
        try:
            results = search_html_scrape(query, max_results)
        except WebSearchError as exc:
            return json.dumps({"error": f"Web search unavailable: {exc}", "query": query})

    if not results:
        return json.dumps({"query": query, "results": [], "message": "No results found"})

    return json.dumps(
        {
            "query": query,
            "results": [
                {
                    "title": _strip_html(r.title),
                    "snippet": _strip_html(r.snippet),
                    "url": r.url,
                }
                for r in results
            ],
        },
        indent=2,
    )


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities."""
    import re
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&#x27;", "'").replace("&quot;", '"')
    return text.strip()
