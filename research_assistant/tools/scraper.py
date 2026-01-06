"""
Web scraping tool for extracting article content.

Uses newspaper3k for article extraction with BeautifulSoup fallback.
"""

from typing import Optional
from datetime import datetime
from urllib.parse import urlparse
import requests
from newspaper import Article
from bs4 import BeautifulSoup

from ..models.source import Source
from ..config import settings


def scrape_url(
    url: str,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None
) -> Source:
    """
    Scrape content from a URL.

    Attempts to extract article content using newspaper3k.
    Falls back to BeautifulSoup if newspaper3k fails.

    Args:
        url: URL to scrape
        timeout: Request timeout in seconds (default: from settings)
        max_retries: Maximum retry attempts (default: from settings)

    Returns:
        Source object with scraped content

    Raises:
        Exception: If scraping fails after all retries

    Example:
        >>> source = scrape_url("https://example.com/article")
        >>> print(f"Title: {source.title}")
        >>> print(f"Content length: {len(source.content)}")
    """
    timeout = timeout or settings.scraper_timeout
    max_retries = max_retries or settings.scraper_max_retries

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            # Try newspaper3k first (best for articles)
            source = _scrape_with_newspaper(url, timeout)
            return source

        except Exception as e:
            last_error = e

            # Try BeautifulSoup fallback on last attempt
            if attempt == max_retries:
                try:
                    source = _scrape_with_beautifulsoup(url, timeout)
                    return source
                except Exception as fallback_error:
                    last_error = fallback_error

    # All attempts failed
    raise Exception(f"Failed to scrape {url} after {max_retries + 1} attempts: {last_error}")


def _scrape_with_newspaper(url: str, timeout: int) -> Source:
    """
    Scrape using newspaper3k library.

    Args:
        url: URL to scrape
        timeout: Request timeout

    Returns:
        Source object with extracted content

    Raises:
        Exception: If extraction fails
    """
    article = Article(url)
    article.download()
    article.parse()

    # Extract metadata
    metadata = {
        "domain": urlparse(url).netloc,
        "authors": article.authors,
        "publish_date": str(article.publish_date) if article.publish_date else None,
        "top_image": article.top_image,
        "word_count": len(article.text.split()) if article.text else 0,
        "extraction_method": "newspaper3k"
    }

    source = Source(
        url=url,
        title=article.title or "",
        content=article.text or "",
        metadata=metadata,
        scraped_at=datetime.now()
    )

    # Validate that we got meaningful content
    if not source.content or len(source.content.strip()) < 50:
        raise Exception("Insufficient content extracted")

    return source


def _scrape_with_beautifulsoup(url: str, timeout: int) -> Source:
    """
    Scrape using BeautifulSoup fallback.

    Args:
        url: URL to scrape
        timeout: Request timeout

    Returns:
        Source object with extracted content

    Raises:
        Exception: If extraction fails
    """
    # Fetch the page
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; ResearchAssistant/1.0)'
    }
    response = requests.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()

    # Parse with BeautifulSoup
    soup = BeautifulSoup(response.content, 'lxml')

    # Extract title
    title = ""
    if soup.title:
        title = soup.title.string or ""
    elif soup.find('h1'):
        title = soup.find('h1').get_text(strip=True)

    # Extract main content
    # Remove unwanted elements
    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        element.decompose()

    # Try to find main content area
    content_area = (
        soup.find('article') or
        soup.find('main') or
        soup.find('div', class_='content') or
        soup.find('div', id='content') or
        soup.body
    )

    if content_area:
        # Get text from paragraphs
        paragraphs = content_area.find_all('p')
        content = '\n\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    else:
        content = ""

    # Metadata
    metadata = {
        "domain": urlparse(url).netloc,
        "word_count": len(content.split()) if content else 0,
        "extraction_method": "beautifulsoup"
    }

    source = Source(
        url=url,
        title=title,
        content=content,
        metadata=metadata,
        scraped_at=datetime.now()
    )

    # Validate content
    if not source.content or len(source.content.strip()) < 50:
        raise Exception("Insufficient content extracted with BeautifulSoup")

    return source


def scrape_multiple_urls(
    urls: list[str],
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
    skip_errors: bool = True
) -> tuple[list[Source], list[str]]:
    """
    Scrape multiple URLs.

    Args:
        urls: List of URLs to scrape
        timeout: Request timeout (default: from settings)
        max_retries: Max retry attempts (default: from settings)
        skip_errors: Continue on errors (default: True)

    Returns:
        Tuple of (successful_sources, failed_urls)

    Example:
        >>> urls = ["https://example.com/1", "https://example.com/2"]
        >>> sources, failures = scrape_multiple_urls(urls)
        >>> print(f"Scraped {len(sources)}, failed {len(failures)}")
    """
    sources = []
    failed = []

    for url in urls:
        try:
            source = scrape_url(url, timeout=timeout, max_retries=max_retries)
            sources.append(source)

        except Exception as e:
            print(f"Warning: Failed to scrape {url}: {e}")
            failed.append(url)

            if not skip_errors:
                raise

    return sources, failed


def extract_domain(url: str) -> str:
    """
    Extract domain from URL.

    Args:
        url: Full URL

    Returns:
        Domain name

    Example:
        >>> extract_domain("https://www.example.com/article")
        'www.example.com'
    """
    return urlparse(url).netloc


def is_valid_url(url: str) -> bool:
    """
    Check if URL is valid and accessible.

    Args:
        url: URL to validate

    Returns:
        True if URL is valid

    Example:
        >>> is_valid_url("https://example.com")
        True
        >>> is_valid_url("not a url")
        False
    """
    try:
        parsed = urlparse(url)
        return all([parsed.scheme, parsed.netloc])
    except Exception:
        return False


def get_content_preview(content: str, max_chars: int = 200) -> str:
    """
    Get preview of content.

    Args:
        content: Full content
        max_chars: Maximum characters

    Returns:
        Truncated content with ellipsis

    Example:
        >>> preview = get_content_preview("Long article text...", max_chars=50)
    """
    if len(content) <= max_chars:
        return content
    return content[:max_chars].rsplit(' ', 1)[0] + "..."


class ScrapeResult:
    """
    Wrapper for scraping results with utility methods.
    """

    def __init__(self, sources: list[Source], failed_urls: list[str]):
        """
        Initialize scrape result.

        Args:
            sources: Successfully scraped sources
            failed_urls: URLs that failed to scrape
        """
        self.sources = sources
        self.failed_urls = failed_urls

    @property
    def success_count(self) -> int:
        """Number of successful scrapes"""
        return len(self.sources)

    @property
    def failure_count(self) -> int:
        """Number of failed scrapes"""
        return len(self.failed_urls)

    @property
    def success_rate(self) -> float:
        """Success rate as percentage"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return (self.success_count / total) * 100

    def get_by_domain(self, domain: str) -> list[Source]:
        """
        Get sources from specific domain.

        Args:
            domain: Domain to filter by

        Returns:
            Sources from that domain
        """
        return [s for s in self.sources if domain in s.get_domain()]

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"ScrapeResult(success={self.success_count}, "
            f"failed={self.failure_count}, "
            f"rate={self.success_rate:.1f}%)"
        )


# Convenience function
def scrape_and_validate(
    url: str,
    min_content_length: int = 100
) -> Optional[Source]:
    """
    Scrape URL and validate content length.

    Args:
        url: URL to scrape
        min_content_length: Minimum content length to accept

    Returns:
        Source if valid, None otherwise

    Example:
        >>> source = scrape_and_validate("https://example.com", min_content_length=200)
        >>> if source:
        ...     print("Valid content!")
    """
    try:
        source = scrape_url(url)

        if len(source.content) >= min_content_length:
            return source

        print(f"Content too short for {url}: {len(source.content)} chars")
        return None

    except Exception as e:
        print(f"Scraping failed for {url}: {e}")
        return None
