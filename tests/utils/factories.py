"""
Test data factories.

Provides convenient functions for creating test objects.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from research_assistant.models.source import Source


# ============================================================================
# Source Factories
# ============================================================================

class SourceFactory:
    """Factory for creating Source objects."""

    @staticmethod
    def create(
        url: str = "https://example.com/article",
        title: str = "Test Article",
        content: str = "Test content for article. " * 20,
        trustworthiness_score: float = 75.0,
        metadata: Optional[dict] = None,
        scraped_at: Optional[datetime] = None
    ) -> Source:
        """
        Create a single Source object.

        Args:
            url: Source URL
            title: Article title
            content: Article content
            trustworthiness_score: Trustworthiness score (0-100)
            metadata: Additional metadata
            scraped_at: Scrape timestamp

        Returns:
            Source instance

        Example:
            >>> source = SourceFactory.create(trustworthiness_score=90.0)
            >>> assert source.is_trustworthy()
        """
        if metadata is None:
            metadata = {
                "word_count": len(content.split()),
            }

        if scraped_at is None:
            scraped_at = datetime(2024, 1, 15, 12, 0, 0)

        return Source(
            url=url,
            title=title,
            content=content,
            trustworthiness_score=trustworthiness_score,
            metadata=metadata,
            scraped_at=scraped_at
        )

    @staticmethod
    def create_batch(count: int, **kwargs) -> List[Source]:
        """
        Create multiple Source objects.

        Args:
            count: Number of sources to create
            **kwargs: Additional arguments passed to create()

        Returns:
            List of Source instances

        Example:
            >>> sources = SourceFactory.create_batch(5, trustworthiness_score=85.0)
            >>> assert len(sources) == 5
        """
        sources = []
        for i in range(count):
            # Add variation to each source
            source_kwargs = kwargs.copy()
            source_kwargs.setdefault("url", f"https://example{i}.com/article")
            source_kwargs.setdefault("title", f"Article {i}")
            source_kwargs.setdefault("content", f"Content for article {i}. " * 20)

            # Vary scrape time (use offset to avoid overflow)
            if "scraped_at" not in source_kwargs:
                from datetime import timedelta
                source_kwargs["scraped_at"] = datetime(2024, 1, 15, 12, 0, 0) + timedelta(seconds=i)

            sources.append(SourceFactory.create(**source_kwargs))

        return sources

    @staticmethod
    def create_trustworthy(
        url: str = "https://stanford.edu/research",
        title: str = "Academic Research Paper",
        content: str = "Well-researched academic content with citations. " * 30
    ) -> Source:
        """
        Create a trustworthy Source (score >= 85).

        Args:
            url: Source URL (defaults to .edu domain)
            title: Article title
            content: Article content

        Returns:
            Trustworthy Source instance

        Example:
            >>> source = SourceFactory.create_trustworthy()
            >>> assert source.is_trustworthy()
            >>> assert source.trustworthiness_score >= 85.0
        """
        return SourceFactory.create(
            url=url,
            title=title,
            content=content,
            trustworthiness_score=95.0,
            metadata={
                "domain": "stanford.edu",
                "authors": ["Dr. Jane Smith", "Dr. John Doe"],
                "word_count": len(content.split()),
                "publish_date": "2024-01-10",
            }
        )

    @staticmethod
    def create_untrustworthy(
        url: str = "https://random-blog.com/opinion",
        title: str = "My Hot Take",
        content: str = "Just my opinion without sources. " * 10
    ) -> Source:
        """
        Create an untrustworthy Source (score < 85).

        Args:
            url: Source URL
            title: Article title
            content: Article content

        Returns:
            Untrustworthy Source instance

        Example:
            >>> source = SourceFactory.create_untrustworthy()
            >>> assert not source.is_trustworthy()
            >>> assert source.trustworthiness_score < 85.0
        """
        return SourceFactory.create(
            url=url,
            title=title,
            content=content,
            trustworthiness_score=45.0,
            metadata={
                "domain": "random-blog.com",
                "word_count": len(content.split()),
            }
        )

    @staticmethod
    def create_with_score_range(
        min_score: float,
        max_score: float,
        count: int = 5
    ) -> List[Source]:
        """
        Create sources with scores in a specific range.

        Args:
            min_score: Minimum trustworthiness score
            max_score: Maximum trustworthiness score
            count: Number of sources to create

        Returns:
            List of Source instances

        Example:
            >>> sources = SourceFactory.create_with_score_range(80.0, 90.0, count=3)
            >>> assert all(80.0 <= s.trustworthiness_score <= 90.0 for s in sources)
        """
        sources = []
        score_step = (max_score - min_score) / (count - 1) if count > 1 else 0

        for i in range(count):
            score = min_score + (i * score_step)
            sources.append(SourceFactory.create(
                url=f"https://example{i}.com/article",
                title=f"Article {i}",
                trustworthiness_score=score
            ))

        return sources


# ============================================================================
# Search Result Factories
# ============================================================================

class SearchResultFactory:
    """Factory for creating search result dictionaries."""

    @staticmethod
    def create(
        title: str = "Test Search Result",
        href: str = "https://example.com/result",
        body: str = "Search result snippet text..."
    ) -> dict:
        """
        Create a single search result dictionary.

        Args:
            title: Result title
            href: Result URL
            body: Result snippet/body

        Returns:
            Search result dict

        Example:
            >>> result = SearchResultFactory.create()
            >>> assert "title" in result
            >>> assert "href" in result
        """
        return {
            "title": title,
            "href": href,
            "body": body
        }

    @staticmethod
    def create_batch(count: int, **kwargs) -> List[dict]:
        """
        Create multiple search results.

        Args:
            count: Number of results to create
            **kwargs: Additional arguments passed to create()

        Returns:
            List of search result dicts

        Example:
            >>> results = SearchResultFactory.create_batch(5)
            >>> assert len(results) == 5
        """
        results = []
        for i in range(count):
            result_kwargs = kwargs.copy()
            result_kwargs.setdefault("title", f"Search Result {i}")
            result_kwargs.setdefault("href", f"https://example{i}.com/result")
            result_kwargs.setdefault("body", f"Snippet for result {i}...")

            results.append(SearchResultFactory.create(**result_kwargs))

        return results

    @staticmethod
    def create_from_domain(domain: str, count: int = 3) -> List[dict]:
        """
        Create search results from a specific domain.

        Args:
            domain: Domain name
            count: Number of results

        Returns:
            List of search result dicts

        Example:
            >>> results = SearchResultFactory.create_from_domain("github.com", 3)
            >>> assert all("github.com" in r["href"] for r in results)
        """
        results = []
        for i in range(count):
            results.append(SearchResultFactory.create(
                title=f"{domain} - Page {i}",
                href=f"https://{domain}/page{i}",
                body=f"Content from {domain}..."
            ))

        return results


# ============================================================================
# HTML Content Factories
# ============================================================================

def create_html_article(
    title: str = "Test Article",
    content_paragraphs: int = 5,
    include_metadata: bool = True
) -> str:
    """
    Create HTML article for testing.

    Args:
        title: Article title
        content_paragraphs: Number of content paragraphs
        include_metadata: Include author/date metadata

    Returns:
        HTML string

    Example:
        >>> html = create_html_article("My Article", content_paragraphs=3)
        >>> assert "My Article" in html
        >>> assert "<article>" in html
    """
    paragraphs = "\n".join([
        f"<p>This is paragraph {i} of the article content. It contains meaningful information about the topic.</p>"
        for i in range(1, content_paragraphs + 1)
    ])

    metadata = ""
    if include_metadata:
        metadata = """
        <div class="meta">
            <span class="author">By Test Author</span>
            <span class="date">January 15, 2024</span>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
    </head>
    <body>
        <article>
            <h1>{title}</h1>
            {metadata}
            {paragraphs}
        </article>
    </body>
    </html>
    """

    return html.strip()


def create_minimal_html(text: str = "Minimal content") -> str:
    """
    Create minimal HTML for testing.

    Args:
        text: Text content

    Returns:
        Minimal HTML string

    Example:
        >>> html = create_minimal_html("Test")
        >>> assert "Test" in html
    """
    return f"""
    <!DOCTYPE html>
    <html>
    <head><title>Minimal</title></head>
    <body><p>{text}</p></body>
    </html>
    """.strip()


# ============================================================================
# Utility Functions
# ============================================================================

def create_test_timestamp(offset_hours: int = 0) -> datetime:
    """
    Create test timestamp with optional offset.

    Args:
        offset_hours: Hours to offset from base time

    Returns:
        Datetime instance

    Example:
        >>> ts1 = create_test_timestamp(0)
        >>> ts2 = create_test_timestamp(1)
        >>> assert ts2 > ts1
    """
    base = datetime(2024, 1, 15, 12, 0, 0)
    return base + timedelta(hours=offset_hours)


def create_url_list(count: int, domain: str = "example.com") -> List[str]:
    """
    Create list of test URLs.

    Args:
        count: Number of URLs
        domain: Base domain

    Returns:
        List of URLs

    Example:
        >>> urls = create_url_list(5, "test.com")
        >>> assert len(urls) == 5
        >>> assert all("test.com" in url for url in urls)
    """
    return [f"https://{domain}/article{i}" for i in range(count)]
