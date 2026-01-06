"""
Unit tests for search module.

Tests DuckDuckGo search functionality without making real API calls.
"""

import pytest

from research_assistant.tools.search import (
    search_duckduckgo,
    search_multiple_queries,
    extract_urls_from_results,
    filter_urls_by_domain,
    SearchResult,
    search_and_get_urls,
)
from tests.mocks.mock_ddgs import (
    SEARCH_RESULTS_MANY,
    SEARCH_RESULTS_DUPLICATE_URLS,
    create_mock_ddgs_with_results,
)
from tests.utils.factories import SearchResultFactory
from tests.utils.assertions import (
    assert_search_results_valid,
    assert_url_list_unique,
    assert_urls_from_domain,
)


# ============================================================================
# TestSearchDuckDuckGo - Basic search functionality
# ============================================================================


class TestSearchDuckDuckGo:
    """Test basic DuckDuckGo search functionality."""

    def test_basic_search_returns_results(self, mock_ddgs):
        """Test that basic search returns results."""
        results = search_duckduckgo("artificial intelligence")

        assert isinstance(results, list)
        assert len(results) > 0
        assert_search_results_valid(results)

    def test_search_max_results_limit(self, mock_ddgs):
        """Test that max_results parameter limits results."""
        max_results = 3
        results = search_duckduckgo("test query", max_results=max_results)

        assert len(results) <= max_results

    def test_search_empty_results(self, mock_ddgs_empty):
        """Test handling of empty search results."""
        results = search_duckduckgo("nonexistent query")

        assert isinstance(results, list)
        assert len(results) == 0

    def test_search_custom_region(self, mock_ddgs):
        """Test search with custom region parameter."""
        results = search_duckduckgo("test", region="us-en")

        assert isinstance(results, list)
        # Mock should still return results regardless of region

    @pytest.mark.parametrize("safesearch_level", ["off", "moderate", "strict"])
    def test_search_safesearch_levels(self, mock_ddgs, safesearch_level):
        """Test search with different safesearch levels."""
        results = search_duckduckgo("test", safesearch=safesearch_level)

        assert isinstance(results, list)
        # All safesearch levels should work with mock

    def test_search_preserves_result_structure(self, mock_ddgs):
        """Test that search results have required keys."""
        results = search_duckduckgo("test")

        for result in results:
            assert "title" in result
            assert "href" in result
            assert "body" in result

    def test_search_with_many_results(self, monkeypatch):
        """Test search that returns many results."""
        # Use custom mock with many results
        MockDDGS = create_mock_ddgs_with_results(SEARCH_RESULTS_MANY)
        monkeypatch.setattr("research_assistant.tools.search.DDGS", MockDDGS)

        results = search_duckduckgo("test", max_results=20)

        assert len(results) == 20

    def test_search_failure_raises_exception(self, monkeypatch):
        """Test that search failures raise appropriate exceptions."""
        from tests.mocks.mock_ddgs import FailingMockDDGS

        monkeypatch.setattr("research_assistant.tools.search.DDGS", FailingMockDDGS)

        with pytest.raises(Exception, match="DuckDuckGo search failed"):
            search_duckduckgo("test query")


# ============================================================================
# TestSearchMultipleQueries - Multiple query searches
# ============================================================================


class TestSearchMultipleQueries:
    """Test searching multiple queries."""

    def test_multiple_queries_searched(self, mock_ddgs):
        """Test that all queries are searched."""
        queries = ["AI ethics", "machine learning", "deep learning"]
        urls = search_multiple_queries(queries, max_results_per_query=5)

        assert isinstance(urls, list)
        assert len(urls) > 0

    def test_url_deduplication_default(self, monkeypatch):
        """Test that URLs are deduplicated by default."""
        # Create mock with duplicate URLs
        MockDDGS = create_mock_ddgs_with_results(SEARCH_RESULTS_DUPLICATE_URLS)
        monkeypatch.setattr("research_assistant.tools.search.DDGS", MockDDGS)

        queries = ["test1", "test2"]
        urls = search_multiple_queries(queries, deduplicate=True)

        assert_url_list_unique(urls)

    def test_no_deduplication_option(self, monkeypatch):
        """Test disabling deduplication."""
        # Create mock with duplicate URLs
        MockDDGS = create_mock_ddgs_with_results(SEARCH_RESULTS_DUPLICATE_URLS)
        monkeypatch.setattr("research_assistant.tools.search.DDGS", MockDDGS)

        queries = ["test1", "test2"]
        urls = search_multiple_queries(queries, deduplicate=False)

        # Should have duplicates
        assert len(urls) > len(set(urls))

    def test_error_handling_continues(self, monkeypatch, capsys):
        """Test that errors in one query don't stop others."""
        call_count = {"count": 0}

        # Create mock that fails on first call, succeeds after
        from tests.mocks.mock_ddgs import MockDDGS

        class PartiallyFailingDDGS(MockDDGS):
            def text(self, *args, **kwargs):
                call_count["count"] += 1
                if call_count["count"] == 1:
                    raise Exception("First query failed")
                return super().text(*args, **kwargs)

        monkeypatch.setattr("research_assistant.tools.search.DDGS", PartiallyFailingDDGS)

        queries = ["failing query", "working query"]
        urls = search_multiple_queries(queries)

        # Should still get results from second query
        assert len(urls) > 0

        # Should print warning
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "failed" in captured.out.lower()


# ============================================================================
# TestFilterUrlsByDomain - Domain filtering
# ============================================================================


class TestFilterUrlsByDomain:
    """Test URL filtering by domain."""

    def test_allowed_domains_whitelist(self):
        """Test filtering with allowed domains."""
        urls = [
            "https://github.com/repo1",
            "https://stackoverflow.com/question",
            "https://reddit.com/post",
        ]

        filtered = filter_urls_by_domain(urls, allowed_domains=["github.com"])

        assert len(filtered) == 1
        assert_urls_from_domain(filtered, "github.com")

    def test_blocked_domains_blacklist(self):
        """Test filtering with blocked domains."""
        urls = [
            "https://github.com/repo1",
            "https://reddit.com/post",
            "https://twitter.com/status",
        ]

        filtered = filter_urls_by_domain(urls, blocked_domains=["reddit.com", "twitter.com"])

        assert len(filtered) == 1
        assert "github.com" in filtered[0]

    def test_both_allowed_and_blocked(self):
        """Test filtering with both allowed and blocked domains."""
        urls = [
            "https://github.com/repo1",
            "https://gitlab.com/repo2",
            "https://bitbucket.com/repo3",
            "https://reddit.com/post",
        ]

        # Allow git hosting, but block gitlab specifically
        filtered = filter_urls_by_domain(
            urls,
            allowed_domains=["github.com", "gitlab.com", "bitbucket.com"],
            blocked_domains=["gitlab.com"],
        )

        assert len(filtered) == 2
        assert all("github.com" in url or "bitbucket.com" in url for url in filtered)

    def test_malformed_urls_skipped(self):
        """Test that malformed URLs are gracefully skipped."""
        urls = [
            "https://github.com/repo",
            "not a url",
            "ftp://invalid.com",
            "https://stackoverflow.com/question",
        ]

        filtered = filter_urls_by_domain(urls, allowed_domains=["github.com", "stackoverflow.com"])

        # Should only get valid URLs
        assert len(filtered) == 2


# ============================================================================
# TestSearchResult - SearchResult wrapper class
# ============================================================================


class TestSearchResult:
    """Test SearchResult wrapper class."""

    def test_initialization(self):
        """Test SearchResult initialization."""
        results = SearchResultFactory.create_batch(5)
        search_result = SearchResult(results)

        assert len(search_result) == 5
        assert search_result.results == results

    def test_urls_property(self):
        """Test extracting URLs from results."""
        results = SearchResultFactory.create_batch(3)
        search_result = SearchResult(results)

        urls = search_result.urls

        assert isinstance(urls, list)
        assert len(urls) == 3
        assert all(isinstance(url, str) for url in urls)

    def test_titles_property(self):
        """Test extracting titles from results."""
        results = SearchResultFactory.create_batch(3)
        search_result = SearchResult(results)

        titles = search_result.titles

        assert isinstance(titles, list)
        assert len(titles) == 3
        assert all(isinstance(title, str) for title in titles)

    def test_snippets_property(self):
        """Test extracting snippets/bodies from results."""
        results = SearchResultFactory.create_batch(3)
        search_result = SearchResult(results)

        snippets = search_result.snippets

        assert isinstance(snippets, list)
        assert len(snippets) == 3

    def test_filter_by_domain(self):
        """Test filtering SearchResult by domain."""
        results = SearchResultFactory.create_from_domain("github.com", 3)
        results += SearchResultFactory.create_from_domain("stackoverflow.com", 2)

        search_result = SearchResult(results)
        filtered = search_result.filter_by_domain(allowed=["github.com"])

        assert isinstance(filtered, SearchResult)
        assert len(filtered) == 3

    def test_limit_method(self):
        """Test limiting SearchResult to n results."""
        results = SearchResultFactory.create_batch(10)
        search_result = SearchResult(results)

        limited = search_result.limit(3)

        assert isinstance(limited, SearchResult)
        assert len(limited) == 3

    def test_iteration_support(self):
        """Test that SearchResult is iterable."""
        results = SearchResultFactory.create_batch(3)
        search_result = SearchResult(results)

        count = 0
        for result in search_result:
            count += 1
            assert isinstance(result, dict)

        assert count == 3


# ============================================================================
# TestUtilityFunctions - Helper functions
# ============================================================================


class TestUtilityFunctions:
    """Test utility functions."""

    def test_extract_urls_from_results(self):
        """Test extracting URLs from search results."""
        results = SearchResultFactory.create_batch(5)
        urls = extract_urls_from_results(results)

        assert isinstance(urls, list)
        assert len(urls) == 5
        assert all(isinstance(url, str) for url in urls)


# ============================================================================
# TestSearchAndGetUrls - Convenience function
# ============================================================================


class TestSearchAndGetUrls:
    """Test search_and_get_urls convenience function."""

    def test_search_and_get_urls_basic(self, mock_ddgs):
        """Test basic search and URL extraction."""
        urls = search_and_get_urls("test query", max_results=5)

        assert isinstance(urls, list)
        assert len(urls) > 0
        assert all(isinstance(url, str) for url in urls)

    def test_search_and_get_urls_with_filters(self, mock_ddgs):
        """Test search with domain filters."""
        urls = search_and_get_urls("test query", max_results=10, blocked_domains=["reddit.com"])

        assert isinstance(urls, list)
        assert not any("reddit.com" in url for url in urls)


# ============================================================================
# Parametrized Tests
# ============================================================================


@pytest.mark.parametrize(
    "query,expected_min_results",
    [
        ("artificial intelligence", 1),
        ("machine learning", 1),
        ("python programming", 1),
    ],
)
def test_search_queries_parametrized(mock_ddgs, query, expected_min_results):
    """Test various search queries return results."""
    results = search_duckduckgo(query, max_results=10)

    assert len(results) >= expected_min_results
    assert_search_results_valid(results)


@pytest.mark.parametrize("max_results", [1, 5, 10, 20])
def test_max_results_limits_parametrized(mock_ddgs, max_results):
    """Test that max_results properly limits results."""
    results = search_duckduckgo("test", max_results=max_results)

    assert len(results) <= max_results
