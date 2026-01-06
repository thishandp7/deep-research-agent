"""
Mock implementation of DuckDuckGo search.

Simulates DDGS without making real API calls.
"""

from typing import List, Optional, Iterator


# ============================================================================
# Predefined Search Result Datasets
# ============================================================================

SEARCH_RESULTS_EMPTY: List[dict] = []


SEARCH_RESULTS_SINGLE: List[dict] = [
    {
        "title": "Artificial Intelligence - Wikipedia",
        "href": "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "body": "Artificial intelligence (AI) is intelligence demonstrated by machines..."
    }
]


SEARCH_RESULTS_DEFAULT: List[dict] = [
    {
        "title": "Artificial Intelligence - Wikipedia",
        "href": "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "body": "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans."
    },
    {
        "title": "What is Artificial Intelligence (AI)? | IBM",
        "href": "https://www.ibm.com/topics/artificial-intelligence",
        "body": "Artificial intelligence leverages computers and machines to mimic the problem-solving and decision-making capabilities of the human mind."
    },
    {
        "title": "AI Ethics: Stanford Research",
        "href": "https://stanford.edu/ai-ethics",
        "body": "Research on ethical considerations in artificial intelligence development and deployment."
    },
    {
        "title": "MIT - Artificial Intelligence",
        "href": "https://news.mit.edu/topic/artificial-intelligence2",
        "body": "Latest news and research on artificial intelligence from MIT."
    },
    {
        "title": "Introduction to AI | Google",
        "href": "https://ai.google/education/",
        "body": "Learn about artificial intelligence fundamentals with Google's educational resources."
    }
]


SEARCH_RESULTS_MANY: List[dict] = SEARCH_RESULTS_DEFAULT + [
    {
        "title": f"AI Article {i}",
        "href": f"https://example{i}.com/ai-article",
        "body": f"Article {i} about artificial intelligence and machine learning."
    }
    for i in range(6, 25)  # Total 24 results
]


SEARCH_RESULTS_DUPLICATE_URLS: List[dict] = [
    {
        "title": "AI Article 1",
        "href": "https://example.com/ai",
        "body": "First article about AI"
    },
    {
        "title": "AI Article 2",
        "href": "https://example.com/ai",  # Duplicate URL
        "body": "Second article with same URL"
    },
    {
        "title": "AI Article 3",
        "href": "https://other.com/ai",
        "body": "Different URL"
    }
]


SEARCH_RESULTS_BY_QUERY: dict[str, List[dict]] = {
    "artificial intelligence": SEARCH_RESULTS_DEFAULT,
    "machine learning": [
        {
            "title": "Machine Learning | Stanford",
            "href": "https://stanford.edu/ml",
            "body": "Introduction to machine learning concepts."
        },
        {
            "title": "Deep Learning Basics",
            "href": "https://deeplearning.ai/basics",
            "body": "Fundamentals of deep learning and neural networks."
        }
    ],
    "python programming": [
        {
            "title": "Python.org",
            "href": "https://python.org",
            "body": "Official Python programming language website."
        },
        {
            "title": "Learn Python - Codecademy",
            "href": "https://codecademy.com/learn/python",
            "body": "Interactive Python programming tutorial."
        }
    ]
}


# ============================================================================
# Mock DDGS Implementation
# ============================================================================

class MockDDGS:
    """
    Mock DuckDuckGo Search implementation.

    Simulates the DDGS class from duckduckgo_search without making real API calls.
    """

    def __init__(self):
        """Initialize mock DDGS."""
        self.call_count = 0
        self.last_query = None
        self.last_kwargs = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass

    def text(
        self,
        query: str,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        max_results: Optional[int] = None
    ) -> Iterator[dict]:
        """
        Mock text search.

        Args:
            query: Search query
            region: Region code
            safesearch: Safety level
            max_results: Maximum results to return

        Returns:
            Iterator of search result dictionaries
        """
        # Track call for assertions
        self.call_count += 1
        self.last_query = query
        self.last_kwargs = {
            "region": region,
            "safesearch": safesearch,
            "max_results": max_results
        }

        # Get results based on query
        results = self._get_results_for_query(query)

        # Apply max_results limit
        if max_results is not None:
            results = results[:max_results]

        return iter(results)

    def _get_results_for_query(self, query: str) -> List[dict]:
        """
        Get results for a specific query.

        Args:
            query: Search query

        Returns:
            List of search results
        """
        # Normalize query
        query_lower = query.lower().strip()

        # Check for exact matches
        if query_lower in SEARCH_RESULTS_BY_QUERY:
            return SEARCH_RESULTS_BY_QUERY[query_lower].copy()

        # Check for partial matches
        for key in SEARCH_RESULTS_BY_QUERY:
            if key in query_lower or query_lower in key:
                return SEARCH_RESULTS_BY_QUERY[key].copy()

        # Default to standard results
        return SEARCH_RESULTS_DEFAULT.copy()


class EmptyMockDDGS(MockDDGS):
    """Mock DDGS that always returns empty results."""

    def text(self, *args, **kwargs) -> Iterator[dict]:
        """Return empty results."""
        self.call_count += 1
        self.last_query = args[0] if args else None
        self.last_kwargs = kwargs
        return iter([])


class FailingMockDDGS(MockDDGS):
    """Mock DDGS that always fails."""

    def text(self, *args, **kwargs) -> Iterator[dict]:
        """Raise an exception."""
        self.call_count += 1
        raise Exception("DuckDuckGo search failed")


# ============================================================================
# Factory Functions
# ============================================================================

def create_mock_ddgs_with_results(results: List[dict]) -> type[MockDDGS]:
    """
    Create a MockDDGS class that returns specific results.

    Args:
        results: List of search result dictionaries

    Returns:
        MockDDGS subclass

    Example:
        >>> custom_results = [{"title": "Test", "href": "...", "body": "..."}]
        >>> MockClass = create_mock_ddgs_with_results(custom_results)
        >>> with MockClass() as ddgs:
        ...     results = list(ddgs.text("test query"))
    """
    class CustomMockDDGS(MockDDGS):
        def _get_results_for_query(self, query: str) -> List[dict]:
            return results.copy()

    return CustomMockDDGS


def create_mock_ddgs_with_query_map(query_map: dict[str, List[dict]]) -> type[MockDDGS]:
    """
    Create a MockDDGS class with custom query-to-results mapping.

    Args:
        query_map: Dictionary mapping queries to results

    Returns:
        MockDDGS subclass

    Example:
        >>> query_map = {
        ...     "ai": [{"title": "AI", "href": "...", "body": "..."}],
        ...     "python": [{"title": "Python", "href": "...", "body": "..."}]
        ... }
        >>> MockClass = create_mock_ddgs_with_query_map(query_map)
    """
    class CustomMockDDGS(MockDDGS):
        def _get_results_for_query(self, query: str) -> List[dict]:
            query_lower = query.lower().strip()

            # Check exact match
            if query_lower in query_map:
                return query_map[query_lower].copy()

            # Check partial match
            for key in query_map:
                if key in query_lower or query_lower in key:
                    return query_map[key].copy()

            # Default to empty
            return []

    return CustomMockDDGS
