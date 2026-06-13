"""
tavily_search.py — Clean Tavily wrapper for live government source search.

Called only when FAISS confidence < 0.65.
Returns clean text ready for LLM consumption.
"""

import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def search_welfare_schemes(query: str, max_results: int = 4) -> str:
    """
    Search for welfare scheme information from official sources.
    Returns merged text ready for LLM prompt injection.
    Prioritises .gov.in domains automatically via search depth.
    """
    try:
        results = _client.search(
            query=f"{query} India government scheme site:gov.in OR site:myscheme.gov.in",
            max_results=max_results,
            search_depth="advanced",
        )

        chunks = []
        for item in results.get("results", []):
            content = item.get("content", "").strip()
            url     = item.get("url", "")
            if content:
                chunks.append(f"[Source: {url}]\n{content[:800]}")

        if not chunks:
            # Retry without site restriction
            results = _client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
            )
            for item in results.get("results", []):
                content = item.get("content", "").strip()
                url     = item.get("url", "")
                if content:
                    chunks.append(f"[Source: {url}]\n{content[:800]}")

        return "\n\n".join(chunks) if chunks else ""

    except Exception as e:
        print(f"[Tavily ERROR] {e}")
        return ""


def is_available() -> bool:
    """Check if Tavily API key is configured."""
    return bool(os.getenv("TAVILY_API_KEY"))