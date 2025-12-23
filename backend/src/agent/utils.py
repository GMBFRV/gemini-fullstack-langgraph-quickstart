from typing import Any, Dict, List, Optional
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage

import time
import random
import requests
from urllib.parse import urlparse


def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def _safe_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.replace("www.", "") if netloc else "source"
    except Exception:
        return "source"


def searxng_search(
    query: str,
    *,
    instances: List[str],
    max_results: int = 8,
    timeout: float = 12.0,
    lang: str = "en",
) -> List[Dict[str, str]]:
    """
    Perform a web search via SearXNG JSON API (no API key).

    Returns a list of dicts:
      { "title": str, "url": str, "snippet": str }
    """
    # Simple shuffle for load distribution
    instance_list = instances[:]
    random.shuffle(instance_list)

    last_error: Optional[Exception] = None

    for base in instance_list:
        base = base.rstrip("/")
        endpoint = f"{base}/search"

        try:
            r = requests.get(
                endpoint,
                params={
                    "q": query,
                    "format": "json",
                    "language": lang,
                    "safesearch": 1,
                },
                timeout=timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; research-agent/1.0)",
                    "Accept": "application/json",
                },
            )
            r.raise_for_status()
            data = r.json()

            results = []
            for item in data.get("results", [])[: max_results * 2]:
                url = item.get("url") or ""
                title = item.get("title") or ""
                snippet = item.get("content") or item.get("snippet") or ""
                if not url or not title:
                    continue

                results.append(
                    {
                        "title": title.strip(),
                        "url": url.strip(),
                        "snippet": snippet.strip(),
                    }
                )

                if len(results) >= max_results:
                    break

            if results:
                return results

        except Exception as e:
            last_error = e
            # small backoff before trying next instance
            time.sleep(0.2)

    # If everything failed – return empty list
    return []


def shorten_sources(sources: List[Dict[str, str]], id: int) -> List[Dict[str, str]]:
    """
    Convert list of {title,url,snippet} to a list of source dicts with short_url
    compatible with the rest of the pipeline.

    Output item format (compatible with finalize_answer):
      { label, short_url, value, title, snippet }
    """
    prefix = "https://vertexaisearch.cloud.google.com/id/"
    seen: Dict[str, str] = {}

    out: List[Dict[str, str]] = []
    for idx, src in enumerate(sources):
        url = src["url"]
        if url not in seen:
            seen[url] = f"{prefix}{id}-{len(seen)}"

        short_url = seen[url]
        title = src.get("title", "").strip()
        snippet = src.get("snippet", "").strip()

        # Label: prefer domain, fallback to title
        label = _safe_domain(url)
        if not label or label == "source":
            label = title[:40] if title else "source"

        out.append(
            {
                "label": label,
                "short_url": short_url,
                "value": url,
                "title": title,
                "snippet": snippet,
            }
        )
    return out


def format_search_results_for_prompt(short_sources: List[Dict[str, str]]) -> str:
    """
    Creates a compact text block for the LLM, giving it:
    - citation labels [S1], [S2] ...
    - title/snippet
    - short_url that will later be replaced with real URL
    """
    lines: List[str] = []
    for i, s in enumerate(short_sources, start=1):
        title = s.get("title", "").strip()
        snippet = s.get("snippet", "").strip()
        short_url = s.get("short_url", "").strip()

        lines.append(f"S{i}: {title}")
        if snippet:
            lines.append(f"Snippet: {snippet}")
        lines.append(f"Use this citation exactly: [S{i}]({short_url})")
        lines.append("")  # spacing
    return "\n".join(lines).strip()
