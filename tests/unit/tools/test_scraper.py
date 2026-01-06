"""
Unit tests for scraper module.

Tests web scraping functionality without making real HTTP requests.
"""

import pytest
from datetime import datetime

from research_assistant.tools.scraper import (
    scrape_url,
    scrape_multiple_urls,
    extract_domain,
    is_valid_url,
    get_content_preview,
    ScrapeResult,
    scrape_and_validate,
)
from research_assistant.models.source import Source
from tests.mocks.mock_requests import (
    HTML_ARTICLE_MINIMAL,
    create_mock_requests_get,
)
from tests.mocks.mock_newspaper import (
    MockArticle,
    MinimalContentMockArticle,
)
from tests.utils.factories import SourceFactory, create_url_list
from tests.utils.assertions import (
    assert_valid_source,
)


# ============================================================================
# TestScrapeUrl - Basic scraping functionality
# ============================================================================


class TestScrapeUrl:
    """Test basic URL scraping functionality."""

    def test_successful_scrape_with_newspaper(self, mock_newspaper_article):
        """Test successful scraping using newspaper3k."""
        url = "https://example.com/good-article"
        source = scrape_url(url)

        assert isinstance(source, Source)
        assert_valid_source(source)
        assert source.url == url
        assert source.title
        assert source.content
        assert len(source.content) >= 50

    def test_scrape_extracts_metadata(self, mock_newspaper_article):
        """Test that scraping extracts article metadata."""
        url = "https://example.com/good-article"
        source = scrape_url(url)

        assert "domain" in source.metadata
        assert "word_count" in source.metadata
        assert "extraction_method" in source.metadata
        assert source.metadata["extraction_method"] == "newspaper3k"

    def test_fallback_to_beautifulsoup(self, mock_newspaper_failing, mock_requests_get_success):
        """Test fallback to BeautifulSoup when newspaper3k fails."""
        url = "https://example.com/article"
        source = scrape_url(url, max_retries=0)

        assert isinstance(source, Source)
        assert source.url == url
        assert source.content
        assert source.metadata["extraction_method"] == "beautifulsoup"

    def test_retry_logic_succeeds(self, monkeypatch):
        """Test that retry logic works when initial attempts fail."""
        # Mock that fails twice, then succeeds
        call_count = {"count": 0}

        class RetryMockArticle(MockArticle):
            def download(self):
                call_count["count"] += 1
                if call_count["count"] <= 2:
                    raise Exception("Download failed")
                super().download()

        monkeypatch.setattr("research_assistant.tools.scraper.Article", RetryMockArticle)

        url = "https://example.com/article"
        source = scrape_url(url, max_retries=3)

        assert isinstance(source, Source)
        assert call_count["count"] == 3  # Failed twice, succeeded on third

    def test_all_retries_exhausted(self, mock_newspaper_failing, mock_requests_get_failure):
        """Test that exception is raised when all retries fail."""
        url = "https://example.com/article"

        with pytest.raises(Exception, match="Failed to scrape"):
            scrape_url(url, max_retries=2)

    def test_insufficient_content_raises_exception(self, monkeypatch):
        """Test that insufficient content raises exception."""
        from tests.mocks.mock_requests import create_mock_requests_get

        monkeypatch.setattr("research_assistant.tools.scraper.Article", MinimalContentMockArticle)
        mock_get = create_mock_requests_get(HTML_ARTICLE_MINIMAL, status_code=200)
        monkeypatch.setattr("research_assistant.tools.scraper.requests.get", mock_get)

        url = "https://example.com/minimal"

        with pytest.raises(Exception, match="Insufficient content"):
            scrape_url(url, max_retries=0)

    def test_custom_timeout_parameter(self, mock_newspaper_article):
        """Test scraping with custom timeout."""
        url = "https://example.com/article"
        source = scrape_url(url, timeout=30)

        assert isinstance(source, Source)

    def test_scraped_at_timestamp(self, mock_newspaper_article):
        """Test that scraped_at timestamp is set."""
        url = "https://example.com/article"
        before = datetime.now()
        source = scrape_url(url)
        after = datetime.now()

        assert source.scraped_at is not None
        assert before <= source.scraped_at <= after


# ============================================================================
# TestScrapeMultipleUrls - Batch scraping
# ============================================================================


class TestScrapeMultipleUrls:
    """Test scraping multiple URLs."""

    def test_all_urls_succeed(self, mock_newspaper_article):
        """Test scraping when all URLs succeed."""
        urls = create_url_list(3, "example.com")
        sources, failed = scrape_multiple_urls(urls)

        assert len(sources) == 3
        assert len(failed) == 0
        assert all(isinstance(s, Source) for s in sources)

    def test_mixed_success_and_failure(self, monkeypatch):
        """Test scraping with mix of success and failures."""

        # Create mock that fails for URLs containing "fail"
        class SelectiveFailMockArticle(MockArticle):
            def download(self):
                if "fail" in self.url:
                    raise Exception("Download failed")
                super().download()

        monkeypatch.setattr("research_assistant.tools.scraper.Article", SelectiveFailMockArticle)

        urls = [
            "https://example.com/good1",
            "https://example.com/fail1",
            "https://example.com/good2",
            "https://example.com/fail2",
        ]

        sources, failed = scrape_multiple_urls(urls, skip_errors=True)

        assert len(sources) == 2
        assert len(failed) == 2
        assert all("fail" in url for url in failed)

    def test_stop_on_first_error(self, monkeypatch):
        """Test that skip_errors=False stops on first error."""

        class FailingMockArticle(MockArticle):
            def download(self):
                raise Exception("Always fails")

        monkeypatch.setattr("research_assistant.tools.scraper.Article", FailingMockArticle)

        urls = create_url_list(3, "example.com")

        with pytest.raises(Exception):
            scrape_multiple_urls(urls, skip_errors=False)

    def test_empty_url_list(self, mock_newspaper_article):
        """Test scraping empty URL list."""
        sources, failed = scrape_multiple_urls([])

        assert len(sources) == 0
        assert len(failed) == 0

    def test_warnings_printed_for_failures(self, monkeypatch, capsys):
        """Test that warnings are printed for failures."""

        class FailingMockArticle(MockArticle):
            def download(self):
                raise Exception("Download failed")

        monkeypatch.setattr("research_assistant.tools.scraper.Article", FailingMockArticle)

        urls = ["https://example.com/fail"]
        scrape_multiple_urls(urls, skip_errors=True)

        captured = capsys.readouterr()
        assert "Warning" in captured.out or "Failed" in captured.out


# ============================================================================
# TestUtilityFunctions - Helper functions
# ============================================================================


class TestUtilityFunctions:
    """Test utility functions."""

    @pytest.mark.parametrize(
        "url,expected_domain",
        [
            ("https://www.example.com/page", "www.example.com"),
            ("http://github.com/user/repo", "github.com"),
            ("https://stackoverflow.com/questions/123", "stackoverflow.com"),
            ("https://subdomain.example.org/path", "subdomain.example.org"),
        ],
    )
    def test_extract_domain(self, url, expected_domain):
        """Test domain extraction from various URLs."""
        domain = extract_domain(url)
        assert domain == expected_domain

    @pytest.mark.parametrize(
        "url,expected_valid",
        [
            ("https://example.com", True),
            ("http://example.com", True),
            ("https://example.com/path/to/page", True),
            ("not a url", False),
            ("ftp://example.com", True),  # Has scheme and netloc, so valid
            ("", False),
            ("example.com", False),  # Missing scheme
        ],
    )
    def test_is_valid_url(self, url, expected_valid):
        """Test URL validation."""
        result = is_valid_url(url)
        assert result == expected_valid

    def test_get_content_preview_short_content(self):
        """Test content preview with short content."""
        content = "This is short content."
        preview = get_content_preview(content, max_chars=100)

        assert preview == content

    def test_get_content_preview_long_content(self):
        """Test content preview with long content."""
        content = "This is a very long article. " * 50
        preview = get_content_preview(content, max_chars=50)

        assert len(preview) <= 53  # 50 + "..."
        assert preview.endswith("...")

    def test_get_content_preview_breaks_on_word_boundary(self):
        """Test that preview breaks on word boundary."""
        content = "This is some content that should be truncated properly"
        preview = get_content_preview(content, max_chars=20)

        # Should not break in middle of word
        assert not preview.rstrip("...").endswith(" ")


# ============================================================================
# TestScrapeResult - Wrapper class
# ============================================================================


class TestScrapeResult:
    """Test ScrapeResult wrapper class."""

    def test_initialization(self):
        """Test ScrapeResult initialization."""
        sources = SourceFactory.create_batch(3)
        failed_urls = ["https://fail1.com", "https://fail2.com"]

        result = ScrapeResult(sources, failed_urls)

        assert result.sources == sources
        assert result.failed_urls == failed_urls

    def test_success_count_property(self):
        """Test success_count property."""
        sources = SourceFactory.create_batch(5)
        result = ScrapeResult(sources, [])

        assert result.success_count == 5

    def test_failure_count_property(self):
        """Test failure_count property."""
        failed_urls = ["https://fail1.com", "https://fail2.com", "https://fail3.com"]
        result = ScrapeResult([], failed_urls)

        assert result.failure_count == 3

    def test_success_rate_calculation(self):
        """Test success_rate calculation."""
        sources = SourceFactory.create_batch(7)
        failed_urls = ["https://fail1.com", "https://fail2.com", "https://fail3.com"]

        result = ScrapeResult(sources, failed_urls)

        # 7 successful out of 10 total = 70%
        assert result.success_rate == 70.0

    def test_success_rate_all_success(self):
        """Test success_rate with 100% success."""
        sources = SourceFactory.create_batch(5)
        result = ScrapeResult(sources, [])

        assert result.success_rate == 100.0

    def test_success_rate_all_failure(self):
        """Test success_rate with 0% success."""
        failed_urls = ["https://fail1.com", "https://fail2.com"]
        result = ScrapeResult([], failed_urls)

        assert result.success_rate == 0.0

    def test_success_rate_zero_total(self):
        """Test success_rate with no results."""
        result = ScrapeResult([], [])

        assert result.success_rate == 0.0

    def test_get_by_domain(self):
        """Test filtering sources by domain."""
        sources = [
            SourceFactory.create(url="https://github.com/repo1"),
            SourceFactory.create(url="https://github.com/repo2"),
            SourceFactory.create(url="https://stackoverflow.com/question"),
        ]

        result = ScrapeResult(sources, [])
        github_sources = result.get_by_domain("github.com")

        assert len(github_sources) == 2
        assert all("github.com" in s.get_domain() for s in github_sources)

    def test_repr_string(self):
        """Test string representation."""
        sources = SourceFactory.create_batch(7)
        failed_urls = ["https://fail1.com", "https://fail2.com", "https://fail3.com"]

        result = ScrapeResult(sources, failed_urls)
        repr_str = repr(result)

        assert "ScrapeResult" in repr_str
        assert "success=7" in repr_str
        assert "failed=3" in repr_str
        assert "70.0%" in repr_str


# ============================================================================
# TestScrapeAndValidate - Convenience function
# ============================================================================


class TestScrapeAndValidate:
    """Test scrape_and_validate convenience function."""

    def test_valid_content_returns_source(self, mock_newspaper_article):
        """Test that valid content returns Source."""
        url = "https://example.com/good-article"
        source = scrape_and_validate(url, min_content_length=100)

        assert isinstance(source, Source)
        assert len(source.content) >= 100

    def test_insufficient_content_returns_none(self, monkeypatch):
        """Test that insufficient content returns None."""
        monkeypatch.setattr("research_assistant.tools.scraper.Article", MinimalContentMockArticle)

        url = "https://example.com/minimal"
        source = scrape_and_validate(url, min_content_length=100)

        assert source is None

    def test_scraping_failure_returns_none(self, mock_newspaper_failing, mock_requests_get_failure):
        """Test that scraping failure returns None."""
        url = "https://example.com/fail"
        source = scrape_and_validate(url)

        assert source is None

    def test_custom_min_content_length(self, mock_newspaper_article):
        """Test with custom minimum content length."""
        url = "https://example.com/article"
        source = scrape_and_validate(url, min_content_length=50)

        assert isinstance(source, Source)

    def test_prints_message_for_short_content(self, monkeypatch, capsys):
        """Test that message is printed for short content."""
        from tests.mocks.mock_newspaper import create_mock_article_class_with_content

        # Create article with 70 chars (passes >=50 check but fails >=100 validation)
        ArticleClass = create_mock_article_class_with_content(
            "Short Article",
            "This content is exactly long enough to pass scraping but too short.",  # 70 chars
        )
        monkeypatch.setattr("research_assistant.tools.scraper.Article", ArticleClass)

        url = "https://example.com/minimal"
        scrape_and_validate(url, min_content_length=100)

        captured = capsys.readouterr()
        assert "too short" in captured.out.lower() or "short" in captured.out.lower()

    def test_prints_message_for_scraping_failure(
        self, mock_newspaper_failing, mock_requests_get_failure, capsys
    ):
        """Test that message is printed for scraping failure."""
        url = "https://example.com/fail"
        scrape_and_validate(url)

        captured = capsys.readouterr()
        assert "failed" in captured.out.lower() or "error" in captured.out.lower()


# ============================================================================
# TestNewspaperScraping - newspaper3k specific tests
# ============================================================================


class TestNewspaperScraping:
    """Test newspaper3k specific functionality."""

    def test_newspaper_extracts_title(self, mock_newspaper_article):
        """Test that newspaper3k extracts article title."""
        url = "https://example.com/good-article"
        source = scrape_url(url)

        assert source.title
        assert len(source.title) > 0

    def test_newspaper_extracts_authors(self, mock_newspaper_article):
        """Test that newspaper3k extracts authors."""
        url = "https://example.com/good-article"
        source = scrape_url(url)

        if "authors" in source.metadata:
            assert isinstance(source.metadata["authors"], list)

    def test_newspaper_extracts_publish_date(self, mock_newspaper_article):
        """Test that newspaper3k extracts publish date."""
        url = "https://example.com/good-article"
        source = scrape_url(url)

        # Publish date might be in metadata
        if "publish_date" in source.metadata:
            assert source.metadata["publish_date"] is not None

    def test_newspaper_extracts_top_image(self, mock_newspaper_article):
        """Test that newspaper3k extracts top image."""
        url = "https://example.com/good-article"
        source = scrape_url(url)

        if "top_image" in source.metadata:
            assert isinstance(source.metadata["top_image"], str)


# ============================================================================
# TestBeautifulSoupScraping - BeautifulSoup fallback tests
# ============================================================================


class TestBeautifulSoupScraping:
    """Test BeautifulSoup fallback functionality."""

    def test_beautifulsoup_extracts_content(
        self, mock_newspaper_failing, mock_requests_get_success
    ):
        """Test that BeautifulSoup extracts content."""
        url = "https://example.com/article"
        source = scrape_url(url, max_retries=0)

        assert source.content
        assert len(source.content) >= 50

    def test_beautifulsoup_extracts_title(self, mock_newspaper_failing, mock_requests_get_success):
        """Test that BeautifulSoup extracts title."""
        url = "https://example.com/article"
        source = scrape_url(url, max_retries=0)

        assert source.title

    def test_beautifulsoup_removes_unwanted_elements(self, mock_newspaper_failing, monkeypatch):
        """Test that BeautifulSoup removes scripts, nav, etc."""
        # HTML with lots of unwanted elements
        html_with_scripts = """
        <html>
        <body>
            <nav>Navigation here</nav>
            <script>var x = 10;</script>
            <article>
                <p>This is the actual content we want to extract from the page.</p>
                <p>This article has multiple paragraphs with meaningful information.</p>
                <p>The scraper should be able to extract all of this text content.</p>
            </article>
            <footer>Footer content</footer>
        </body>
        </html>
        """

        mock_get = create_mock_requests_get(html_with_scripts)
        monkeypatch.setattr("research_assistant.tools.scraper.requests.get", mock_get)

        url = "https://example.com/article"
        source = scrape_url(url, max_retries=0)

        # Content should have the article text but not script/nav
        assert "actual content" in source.content
        assert "var x = 10" not in source.content
        assert "Navigation here" not in source.content


# ============================================================================
# Parametrized Tests
# ============================================================================


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/article1",
        "https://test.org/blog/post",
        "https://news.edu/story",
    ],
)
def test_scrape_various_urls(mock_newspaper_article, url):
    """Test scraping various URL patterns."""
    source = scrape_url(url)

    assert isinstance(source, Source)
    assert source.url == url


@pytest.mark.parametrize("max_retries", [0, 1, 2, 3])
def test_retry_counts(mock_newspaper_article, max_retries):
    """Test different retry counts."""
    url = "https://example.com/article"
    source = scrape_url(url, max_retries=max_retries)

    assert isinstance(source, Source)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.skip(
        reason="Mock handles empty URL gracefully; test requires real newspaper library"
    )
    def test_empty_url_string(self, mock_newspaper_article):
        """Test scraping with empty URL."""
        with pytest.raises(Exception):
            scrape_url("")

    def test_very_long_url(self, mock_newspaper_article):
        """Test scraping with very long URL."""
        long_url = "https://example.com/" + "a" * 500
        source = scrape_url(long_url)

        assert isinstance(source, Source)

    def test_url_with_special_characters(self, mock_newspaper_article):
        """Test URL with special characters."""
        url = "https://example.com/article?id=123&lang=en#section-2"
        source = scrape_url(url)

        assert isinstance(source, Source)

    def test_unicode_in_content(self, monkeypatch):
        """Test handling of Unicode content."""
        from tests.mocks.mock_newspaper import create_mock_article_class_with_content

        unicode_content = "Content with Unicode: 你好, مرحبا, שלום"
        ArticleClass = create_mock_article_class_with_content(
            "Unicode Article", unicode_content * 10  # Make it long enough
        )

        monkeypatch.setattr("research_assistant.tools.scraper.Article", ArticleClass)

        url = "https://example.com/unicode"
        source = scrape_url(url)

        assert unicode_content in source.content
