"""
Custom assertion functions for tests.

Provides semantic assertions specific to the research assistant domain.
"""

from datetime import datetime
from typing import List, Optional
from research_assistant.models.source import Source


# ============================================================================
# Source Assertions
# ============================================================================

def assert_valid_source(source: Source, min_content_length: int = 50):
    """
    Assert that a Source object is valid.

    Args:
        source: Source to validate
        min_content_length: Minimum content length

    Raises:
        AssertionError: If source is invalid

    Example:
        >>> from tests.utils.factories import SourceFactory
        >>> source = SourceFactory.create()
        >>> assert_valid_source(source)
    """
    assert isinstance(source, Source), "Must be a Source instance"
    assert source.url, "URL must not be empty"
    assert source.title, "Title must not be empty"
    assert source.content, "Content must not be empty"
    assert len(source.content) >= min_content_length, \
        f"Content too short: {len(source.content)} < {min_content_length}"
    assert 0 <= source.trustworthiness_score <= 100, \
        f"Score out of range: {source.trustworthiness_score}"


def assert_source_trustworthy(source: Source, threshold: float = 85.0):
    """
    Assert that a Source meets trustworthiness threshold.

    Args:
        source: Source to check
        threshold: Minimum trustworthiness score

    Raises:
        AssertionError: If source is not trustworthy

    Example:
        >>> from tests.utils.factories import SourceFactory
        >>> source = SourceFactory.create_trustworthy()
        >>> assert_source_trustworthy(source)
    """
    assert source.is_trustworthy(threshold), \
        f"Source not trustworthy: {source.trustworthiness_score} < {threshold}"


def assert_source_untrustworthy(source: Source, threshold: float = 85.0):
    """
    Assert that a Source does NOT meet trustworthiness threshold.

    Args:
        source: Source to check
        threshold: Trustworthiness threshold

    Raises:
        AssertionError: If source is trustworthy

    Example:
        >>> from tests.utils.factories import SourceFactory
        >>> source = SourceFactory.create_untrustworthy()
        >>> assert_source_untrustworthy(source)
    """
    assert not source.is_trustworthy(threshold), \
        f"Source is trustworthy: {source.trustworthiness_score} >= {threshold}"


def assert_sources_equal(source1: Source, source2: Source):
    """
    Assert that two sources are equal.

    Compares URL, title, content, and trustworthiness score.

    Args:
        source1: First source
        source2: Second source

    Raises:
        AssertionError: If sources are not equal

    Example:
        >>> from tests.utils.factories import SourceFactory
        >>> s1 = SourceFactory.create(url="https://test.com")
        >>> s2 = SourceFactory.create(url="https://test.com")
        >>> assert_sources_equal(s1, s2)
    """
    assert source1.url == source2.url, "URLs don't match"
    assert source1.title == source2.title, "Titles don't match"
    assert source1.content == source2.content, "Content doesn't match"
    assert source1.trustworthiness_score == source2.trustworthiness_score, \
        "Scores don't match"


def assert_all_trustworthy(sources: List[Source], threshold: float = 85.0):
    """
    Assert that all sources meet trustworthiness threshold.

    Args:
        sources: List of sources to check
        threshold: Minimum trustworthiness score

    Raises:
        AssertionError: If any source is not trustworthy

    Example:
        >>> from tests.utils.factories import SourceFactory
        >>> sources = [SourceFactory.create_trustworthy() for _ in range(3)]
        >>> assert_all_trustworthy(sources)
    """
    untrustworthy = [s for s in sources if not s.is_trustworthy(threshold)]

    assert not untrustworthy, \
        f"{len(untrustworthy)} sources below threshold: {[s.url for s in untrustworthy]}"


def assert_source_count(sources: List[Source], expected: int):
    """
    Assert number of sources matches expected count.

    Args:
        sources: List of sources
        expected: Expected count

    Raises:
        AssertionError: If count doesn't match

    Example:
        >>> from tests.utils.factories import SourceFactory
        >>> sources = SourceFactory.create_batch(5)
        >>> assert_source_count(sources, 5)
    """
    actual = len(sources)
    assert actual == expected, f"Expected {expected} sources, got {actual}"


def assert_source_from_domain(source: Source, domain: str):
    """
    Assert that source is from a specific domain.

    Args:
        source: Source to check
        domain: Expected domain

    Raises:
        AssertionError: If domain doesn't match

    Example:
        >>> from tests.utils.factories import SourceFactory
        >>> source = SourceFactory.create(url="https://example.com/article")
        >>> assert_source_from_domain(source, "example.com")
    """
    actual_domain = source.get_domain()
    assert domain in actual_domain, \
        f"Domain mismatch: expected '{domain}' in '{actual_domain}'"


# ============================================================================
# URL Assertions
# ============================================================================

def assert_url_list_unique(urls: List[str]):
    """
    Assert that URL list contains no duplicates.

    Args:
        urls: List of URLs

    Raises:
        AssertionError: If duplicates found

    Example:
        >>> urls = ["https://a.com", "https://b.com", "https://c.com"]
        >>> assert_url_list_unique(urls)
    """
    unique_urls = set(urls)
    assert len(unique_urls) == len(urls), \
        f"Duplicate URLs found: {len(urls)} total, {len(unique_urls)} unique"


def assert_url_valid(url: str):
    """
    Assert that URL is valid.

    Args:
        url: URL to validate

    Raises:
        AssertionError: If URL is invalid

    Example:
        >>> assert_url_valid("https://example.com")
    """
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
        assert parsed.scheme in ("http", "https"), f"Invalid scheme: {parsed.scheme}"
        assert parsed.netloc, "Missing netloc (domain)"
    except Exception as e:
        raise AssertionError(f"Invalid URL '{url}': {e}")


def assert_urls_from_domain(urls: List[str], domain: str):
    """
    Assert that all URLs are from a specific domain.

    Args:
        urls: List of URLs
        domain: Expected domain

    Raises:
        AssertionError: If any URL is not from domain

    Example:
        >>> urls = ["https://example.com/1", "https://example.com/2"]
        >>> assert_urls_from_domain(urls, "example.com")
    """
    from urllib.parse import urlparse

    for url in urls:
        parsed = urlparse(url)
        assert domain in parsed.netloc, \
            f"URL '{url}' not from domain '{domain}' (got: {parsed.netloc})"


# ============================================================================
# Search Result Assertions
# ============================================================================

def assert_valid_search_result(result: dict):
    """
    Assert that search result dictionary is valid.

    Args:
        result: Search result dict

    Raises:
        AssertionError: If result is invalid

    Example:
        >>> result = {"title": "Test", "href": "https://test.com", "body": "..."}
        >>> assert_valid_search_result(result)
    """
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "title" in result, "Result missing 'title' key"
    assert "href" in result, "Result missing 'href' key"
    assert "body" in result, "Result missing 'body' key"

    assert result["title"], "Title must not be empty"
    assert result["href"], "Href must not be empty"

    # Validate href is a valid URL
    assert_url_valid(result["href"])


def assert_search_results_valid(results: List[dict]):
    """
    Assert that all search results are valid.

    Args:
        results: List of search results

    Raises:
        AssertionError: If any result is invalid

    Example:
        >>> from tests.utils.factories import SearchResultFactory
        >>> results = SearchResultFactory.create_batch(5)
        >>> assert_search_results_valid(results)
    """
    for i, result in enumerate(results):
        try:
            assert_valid_search_result(result)
        except AssertionError as e:
            raise AssertionError(f"Result {i} invalid: {e}")


def assert_search_result_count(results: List[dict], expected: int):
    """
    Assert number of search results matches expected.

    Args:
        results: List of search results
        expected: Expected count

    Raises:
        AssertionError: If count doesn't match

    Example:
        >>> from tests.utils.factories import SearchResultFactory
        >>> results = SearchResultFactory.create_batch(5)
        >>> assert_search_result_count(results, 5)
    """
    actual = len(results)
    assert actual == expected, f"Expected {expected} results, got {actual}"


# ============================================================================
# Vector Store Assertions
# ============================================================================

def assert_query_results_valid(results: dict):
    """
    Assert that vector store query results are valid.

    Args:
        results: Query results dictionary

    Raises:
        AssertionError: If results are invalid

    Example:
        >>> results = {
        ...     "ids": [["id1", "id2"]],
        ...     "documents": [["doc1", "doc2"]],
        ...     "metadatas": [[{}, {}]],
        ...     "distances": [[0.1, 0.2]]
        ... }
        >>> assert_query_results_valid(results)
    """
    assert isinstance(results, dict), "Results must be a dictionary"

    required_keys = ["ids", "documents", "metadatas", "distances"]
    for key in required_keys:
        assert key in results, f"Results missing '{key}' key"

    # Check structure (list of lists)
    for key in required_keys:
        assert isinstance(results[key], list), f"'{key}' must be a list"
        if results[key]:
            assert isinstance(results[key][0], list), \
                f"'{key}' must be a list of lists"


def assert_collection_count(count: int, expected: int):
    """
    Assert that collection count matches expected.

    Args:
        count: Actual count
        expected: Expected count

    Raises:
        AssertionError: If counts don't match

    Example:
        >>> assert_collection_count(5, 5)
    """
    assert count == expected, f"Expected {expected} documents, got {count}"


# ============================================================================
# Content Assertions
# ============================================================================

def assert_content_length(content: str, min_length: int, max_length: Optional[int] = None):
    """
    Assert that content length is within bounds.

    Args:
        content: Content string
        min_length: Minimum length
        max_length: Maximum length (optional)

    Raises:
        AssertionError: If length is out of bounds

    Example:
        >>> assert_content_length("Test content", min_length=5, max_length=100)
    """
    actual = len(content)

    assert actual >= min_length, \
        f"Content too short: {actual} < {min_length}"

    if max_length is not None:
        assert actual <= max_length, \
            f"Content too long: {actual} > {max_length}"


def assert_contains_keywords(text: str, keywords: List[str], case_sensitive: bool = False):
    """
    Assert that text contains all keywords.

    Args:
        text: Text to search
        keywords: Keywords to find
        case_sensitive: Whether search is case-sensitive

    Raises:
        AssertionError: If any keyword is missing

    Example:
        >>> assert_contains_keywords("artificial intelligence", ["artificial", "intelligence"])
    """
    search_text = text if case_sensitive else text.lower()

    missing = []
    for keyword in keywords:
        search_keyword = keyword if case_sensitive else keyword.lower()
        if search_keyword not in search_text:
            missing.append(keyword)

    assert not missing, f"Missing keywords: {missing}"


# ============================================================================
# Metadata Assertions
# ============================================================================

def assert_metadata_has_keys(metadata: dict, required_keys: List[str]):
    """
    Assert that metadata contains required keys.

    Args:
        metadata: Metadata dictionary
        required_keys: Required keys

    Raises:
        AssertionError: If any key is missing

    Example:
        >>> metadata = {"domain": "example.com", "word_count": 100}
        >>> assert_metadata_has_keys(metadata, ["domain", "word_count"])
    """
    missing = [key for key in required_keys if key not in metadata]
    assert not missing, f"Metadata missing keys: {missing}"


def assert_timestamp_recent(timestamp: Optional[datetime], max_age_seconds: int = 3600):
    """
    Assert that timestamp is recent.

    Args:
        timestamp: Timestamp to check
        max_age_seconds: Maximum age in seconds

    Raises:
        AssertionError: If timestamp is too old or None

    Example:
        >>> from datetime import datetime
        >>> assert_timestamp_recent(datetime.now(), max_age_seconds=60)
    """
    from datetime import datetime, timedelta

    assert timestamp is not None, "Timestamp is None"

    age = datetime.now() - timestamp
    max_age = timedelta(seconds=max_age_seconds)

    assert age <= max_age, \
        f"Timestamp too old: {age.total_seconds()}s > {max_age_seconds}s"
