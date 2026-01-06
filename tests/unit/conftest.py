"""
Unit test specific fixtures.

These fixtures are available to all unit tests.
"""

import pytest
from datetime import datetime
from pathlib import Path


# ============================================================================
# Data Fixtures
# ============================================================================


@pytest.fixture
def sample_source():
    """
    Create a sample Source object for testing.

    Returns:
        Source instance with typical data
    """
    from research_assistant.models.source import Source

    return Source(
        url="https://example.com/article",
        title="Sample Article Title",
        content="This is sample article content for testing purposes. " * 10,
        trustworthiness_score=87.5,
        metadata={
            "domain": "example.com",
            "authors": ["Test Author"],
            "word_count": 100,
        },
        scraped_at=datetime(2024, 1, 15, 12, 0, 0),
    )


@pytest.fixture
def sample_sources():
    """
    Create multiple Source objects for testing.

    Returns:
        List of Source instances
    """
    from research_assistant.models.source import Source

    sources = []
    for i in range(5):
        source = Source(
            url=f"https://example{i}.com/article",
            title=f"Article {i}",
            content=f"Content for article {i}. " * 20,
            trustworthiness_score=80.0 + i * 2,
            metadata={
                "domain": f"example{i}.com",
                "word_count": 100 + i * 10,
            },
            scraped_at=datetime(2024, 1, 15, 12, i, 0),
        )
        sources.append(source)

    return sources


@pytest.fixture
def trustworthy_source():
    """
    Create a trustworthy Source (score >= 85).

    Returns:
        Source instance with high trustworthiness score
    """
    from research_assistant.models.source import Source

    return Source(
        url="https://stanford.edu/research/ai",
        title="AI Research Paper",
        content="Well-researched content with citations. " * 30,
        trustworthiness_score=95.0,
        metadata={
            "domain": "stanford.edu",
            "authors": ["Dr. Jane Smith", "Dr. John Doe"],
            "word_count": 300,
        },
    )


@pytest.fixture
def untrustworthy_source():
    """
    Create an untrustworthy Source (score < 85).

    Returns:
        Source instance with low trustworthiness score
    """
    from research_assistant.models.source import Source

    return Source(
        url="https://random-blog.com/opinion",
        title="My Hot Take",
        content="Just my opinion, not backed by evidence. " * 10,
        trustworthiness_score=45.0,
        metadata={
            "domain": "random-blog.com",
            "word_count": 100,
        },
    )


# ============================================================================
# Search Module Fixtures
# ============================================================================


@pytest.fixture
def mock_ddgs(monkeypatch):
    """
    Mock DuckDuckGo search with default results.

    Returns:
        Mock DDGS instance
    """
    from tests.mocks.mock_ddgs import MockDDGS

    monkeypatch.setattr("research_assistant.tools.search.DDGS", MockDDGS)
    return MockDDGS


@pytest.fixture
def mock_ddgs_empty(monkeypatch):
    """
    Mock DuckDuckGo search with no results.

    Returns:
        Mock DDGS instance that returns empty results
    """
    from tests.mocks.mock_ddgs import MockDDGS, SEARCH_RESULTS_EMPTY

    class EmptyDDGS(MockDDGS):
        def text(self, *args, **kwargs):
            return iter(SEARCH_RESULTS_EMPTY)

    monkeypatch.setattr("research_assistant.tools.search.DDGS", EmptyDDGS)
    return EmptyDDGS


@pytest.fixture
def mock_ddgs_with_results(monkeypatch, request):
    """
    Mock DuckDuckGo search with custom results.

    Use with @pytest.mark.parametrize to provide custom results.

    Example:
        @pytest.mark.parametrize("mock_ddgs_with_results", [custom_results], indirect=True)
        def test_search(mock_ddgs_with_results):
            ...
    """
    from tests.mocks.mock_ddgs import MockDDGS

    results = request.param if hasattr(request, "param") else []

    class CustomDDGS(MockDDGS):
        def text(self, *args, **kwargs):
            return iter(results)

    monkeypatch.setattr("research_assistant.tools.search.DDGS", CustomDDGS)
    return CustomDDGS


# ============================================================================
# Scraper Module Fixtures
# ============================================================================


@pytest.fixture
def mock_requests_get_success(monkeypatch):
    """
    Mock successful HTTP request.

    Returns:
        Mock requests.get function
    """
    from tests.mocks.mock_requests import create_mock_requests_get, HTML_ARTICLE_GOOD

    mock_get = create_mock_requests_get(HTML_ARTICLE_GOOD, status_code=200)
    monkeypatch.setattr("research_assistant.tools.scraper.requests.get", mock_get)
    return mock_get


@pytest.fixture
def mock_requests_get_minimal(monkeypatch):
    """
    Mock HTTP request returning minimal content.

    Returns:
        Mock requests.get function
    """
    from tests.mocks.mock_requests import create_mock_requests_get, HTML_ARTICLE_MINIMAL

    mock_get = create_mock_requests_get(HTML_ARTICLE_MINIMAL, status_code=200)
    monkeypatch.setattr("research_assistant.tools.scraper.requests.get", mock_get)
    return mock_get


@pytest.fixture
def mock_requests_get_failure(monkeypatch):
    """
    Mock HTTP request that fails.

    Returns:
        Mock requests.get function that raises exception
    """
    from tests.mocks.mock_requests import create_failing_requests_get

    mock_get = create_failing_requests_get()
    monkeypatch.setattr("research_assistant.tools.scraper.requests.get", mock_get)
    return mock_get


@pytest.fixture
def mock_newspaper_article(monkeypatch):
    """
    Mock newspaper3k Article class.

    Returns:
        Mock Article class
    """
    from tests.mocks.mock_newspaper import MockArticle

    monkeypatch.setattr("research_assistant.tools.scraper.Article", MockArticle)
    return MockArticle


@pytest.fixture
def mock_newspaper_failing(monkeypatch):
    """
    Mock newspaper3k Article that always fails.

    Returns:
        Mock Article class that raises exceptions
    """
    from tests.mocks.mock_newspaper import FailingMockArticle

    monkeypatch.setattr("research_assistant.tools.scraper.Article", FailingMockArticle)
    return FailingMockArticle


@pytest.fixture
def mock_scraper_full(mock_newspaper_article, mock_requests_get_success):
    """
    Complete scraper mocking with both newspaper and requests.

    Combines multiple fixtures for full scraping functionality.

    Returns:
        Tuple of (MockArticle, mock_get)
    """
    return (mock_newspaper_article, mock_requests_get_success)


# ============================================================================
# Vector Store Module Fixtures
# ============================================================================


@pytest.fixture
def mock_chroma_client(monkeypatch, temp_data_dir: Path):
    """
    Mock ChromaDB client.

    Returns:
        Mock ChromaDB client
    """
    from tests.mocks.mock_chroma import MockChromaClient

    client = MockChromaClient()
    monkeypatch.setattr(
        "research_assistant.tools.vector_store.chromadb.Client", lambda *args, **kwargs: client
    )
    return client


@pytest.fixture
def mock_sentence_transformer(monkeypatch):
    """
    Mock SentenceTransformer.

    Returns:
        Mock SentenceTransformer class
    """
    from tests.mocks.mock_embedder import MockSentenceTransformer

    monkeypatch.setattr(
        "research_assistant.tools.vector_store.SentenceTransformer", MockSentenceTransformer
    )
    return MockSentenceTransformer


@pytest.fixture
def mock_vector_store_full(mock_chroma_client, mock_sentence_transformer, temp_data_dir: Path):
    """
    Complete vector store mocking.

    Combines ChromaDB and SentenceTransformer mocks.

    Returns:
        Tuple of (mock_client, mock_embedder)
    """
    return (mock_chroma_client, mock_sentence_transformer)


# ============================================================================
# HTML Fixtures
# ============================================================================


@pytest.fixture
def html_article_good(html_samples_dir: Path) -> str:
    """
    Load good HTML article sample.

    Returns:
        HTML content as string
    """
    file_path = html_samples_dir / "article_good.html"
    if file_path.exists():
        return file_path.read_text()
    # Fallback if file doesn't exist yet
    from tests.mocks.mock_requests import HTML_ARTICLE_GOOD

    return HTML_ARTICLE_GOOD


@pytest.fixture
def html_article_minimal(html_samples_dir: Path) -> str:
    """
    Load minimal HTML article sample.

    Returns:
        HTML content as string
    """
    file_path = html_samples_dir / "article_minimal.html"
    if file_path.exists():
        return file_path.read_text()
    # Fallback if file doesn't exist yet
    from tests.mocks.mock_requests import HTML_ARTICLE_MINIMAL

    return HTML_ARTICLE_MINIMAL


@pytest.fixture
def html_no_content(html_samples_dir: Path) -> str:
    """
    Load HTML with no content sample.

    Returns:
        HTML content as string
    """
    file_path = html_samples_dir / "article_no_content.html"
    if file_path.exists():
        return file_path.read_text()
    # Fallback if file doesn't exist yet
    from tests.mocks.mock_requests import HTML_NO_CONTENT

    return HTML_NO_CONTENT
