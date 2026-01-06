"""
Web search tool using DuckDuckGo.

Provides free web search without API keys.
"""

from typing import List, Optional
from duckduckgo_search import DDGS

from ..config import settings


def search_duckduckgo(
    query: str,
    max_results: Optional[int] = None,
    region: str = "wt-wt",
    safesearch: str = "moderate",
) -> List[dict]:
    """
    Search DuckDuckGo and return results.

    Args:
        query: Search query string
        max_results: Maximum number of results (default: from settings)
        region: Region code (default: "wt-wt" for worldwide)
        safesearch: Safety level ("off", "moderate", "strict")

    Returns:
        List of search result dictionaries with keys:
        - title: Result title
        - href: Result URL
        - body: Result snippet/description

    Raises:
        Exception: If search fails

    Example:
        >>> results = search_duckduckgo("artificial intelligence", max_results=5)
        >>> for result in results:
        ...     print(f"{result['title']}: {result['href']}")
    """
    max_results = max_results or settings.max_search_results_per_query

    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(query, region=region, safesearch=safesearch, max_results=max_results)
            )
            return results

    except Exception as e:
        raise Exception(f"DuckDuckGo search failed for query '{query}': {str(e)}")


def search_multiple_queries(
    queries: List[str], max_results_per_query: Optional[int] = None, deduplicate: bool = True
) -> List[str]:
    """
    Search multiple queries and return unique URLs.

    Args:
        queries: List of search queries
        max_results_per_query: Max results per query (default: from settings)
        deduplicate: Remove duplicate URLs (default: True)

    Returns:
        List of unique URLs from all queries

    Example:
        >>> queries = ["AI ethics", "artificial intelligence safety"]
        >>> urls = search_multiple_queries(queries, max_results_per_query=5)
        >>> print(f"Found {len(urls)} unique URLs")
    """
    all_urls = []

    for query in queries:
        try:
            results = search_duckduckgo(query, max_results=max_results_per_query)
            urls = [result["href"] for result in results]
            all_urls.extend(urls)

        except Exception as e:
            # Log error but continue with other queries
            print(f"Warning: Search failed for query '{query}': {e}")
            continue

    # Deduplicate while preserving order
    if deduplicate:
        seen = set()
        unique_urls = []
        for url in all_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        return unique_urls

    return all_urls


def extract_urls_from_results(results: List[dict]) -> List[str]:
    """
    Extract URLs from search results.

    Args:
        results: List of search result dictionaries

    Returns:
        List of URLs

    Example:
        >>> results = search_duckduckgo("python")
        >>> urls = extract_urls_from_results(results)
    """
    return [result["href"] for result in results]


def filter_urls_by_domain(
    urls: List[str],
    allowed_domains: Optional[List[str]] = None,
    blocked_domains: Optional[List[str]] = None,
) -> List[str]:
    """
    Filter URLs by domain whitelist/blacklist.

    Args:
        urls: List of URLs to filter
        allowed_domains: Only include these domains (if specified)
        blocked_domains: Exclude these domains (if specified)

    Returns:
        Filtered list of URLs

    Example:
        >>> urls = ["https://github.com/...", "https://reddit.com/..."]
        >>> filtered = filter_urls_by_domain(urls, allowed_domains=["github.com"])
        >>> # Returns only github.com URLs
    """
    from urllib.parse import urlparse

    filtered = []

    for url in urls:
        try:
            domain = urlparse(url).netloc

            # Check allowed domains
            if allowed_domains:
                if not any(allowed in domain for allowed in allowed_domains):
                    continue

            # Check blocked domains
            if blocked_domains:
                if any(blocked in domain for blocked in blocked_domains):
                    continue

            filtered.append(url)

        except Exception:
            # Skip malformed URLs
            continue

    return filtered


class SearchResult:
    """
    Wrapper for search results with utility methods.
    """

    def __init__(self, results: List[dict]):
        """
        Initialize with DuckDuckGo search results.

        Args:
            results: List of search result dictionaries
        """
        self.results = results

    @property
    def urls(self) -> List[str]:
        """Get all URLs from results"""
        return extract_urls_from_results(self.results)

    @property
    def titles(self) -> List[str]:
        """Get all titles from results"""
        return [r["title"] for r in self.results]

    @property
    def snippets(self) -> List[str]:
        """Get all snippets/descriptions from results"""
        return [r.get("body", "") for r in self.results]

    def filter_by_domain(
        self, allowed: Optional[List[str]] = None, blocked: Optional[List[str]] = None
    ) -> "SearchResult":
        """
        Filter results by domain.

        Args:
            allowed: Allowed domains
            blocked: Blocked domains

        Returns:
            New SearchResult with filtered results
        """
        filtered_urls = filter_urls_by_domain(self.urls, allowed, blocked)
        filtered_results = [r for r in self.results if r["href"] in filtered_urls]
        return SearchResult(filtered_results)

    def limit(self, n: int) -> "SearchResult":
        """
        Limit to first n results.

        Args:
            n: Maximum number of results

        Returns:
            New SearchResult with limited results
        """
        return SearchResult(self.results[:n])

    def __len__(self) -> int:
        """Number of results"""
        return len(self.results)

    def __iter__(self):
        """Iterate over results"""
        return iter(self.results)

    def __repr__(self) -> str:
        """String representation"""
        return f"SearchResult({len(self.results)} results)"


# Convenience function for common use case
def search_and_get_urls(
    query: str,
    max_results: int = 5,
    allowed_domains: Optional[List[str]] = None,
    blocked_domains: Optional[List[str]] = None,
) -> List[str]:
    """
    Search and get filtered URLs in one call.

    Args:
        query: Search query
        max_results: Maximum results
        allowed_domains: Whitelist domains
        blocked_domains: Blacklist domains

    Returns:
        List of filtered URLs

    Example:
        >>> urls = search_and_get_urls(
        ...     "machine learning",
        ...     max_results=10,
        ...     blocked_domains=["reddit.com", "quora.com"]
        ... )
    """
    results = search_duckduckgo(query, max_results=max_results)
    urls = extract_urls_from_results(results)

    if allowed_domains or blocked_domains:
        urls = filter_urls_by_domain(urls, allowed_domains, blocked_domains)

    return urls
