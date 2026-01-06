"""
Scraper agent for extracting content from URLs.

Orchestrates batch scraping with error handling and progress tracking.
"""

from typing import Dict, Any, List, Optional
from langchain_core.language_models import BaseLLM

from .base import BaseAgent
from ..tools.scraper import scrape_multiple_urls, scrape_url, ScrapeResult
from ..models.source import Source
from ..config import settings


class ScraperAgent(BaseAgent):
    """
    Agent responsible for scraping content from URLs.

    Workflow:
    1. Receive list of URLs to scrape
    2. Execute batch scraping with error handling
    3. Track successful vs failed scrapes
    4. Return Source objects with metadata
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        temperature: float = 0.0,
        verbose: bool = False,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        skip_errors: bool = True,
    ):
        """
        Initialize scraper agent.

        Args:
            llm: Language model instance (unused for scraping, but kept for consistency)
            temperature: LLM temperature (unused)
            verbose: Enable verbose logging
            timeout: Scraping timeout per URL (default: from settings)
            max_retries: Maximum retry attempts (default: from settings)
            skip_errors: Continue scraping on errors (default: True)
        """
        super().__init__(llm=llm, temperature=temperature, verbose=verbose)
        self.timeout = timeout or settings.scraper_timeout
        self.max_retries = max_retries or settings.scraper_max_retries
        self.skip_errors = skip_errors

    def scrape_urls(self, urls: List[str]) -> ScrapeResult:
        """
        Scrape content from multiple URLs.

        Args:
            urls: List of URLs to scrape

        Returns:
            ScrapeResult object with sources and failed URLs

        Example:
            >>> agent = ScraperAgent(verbose=True)
            >>> result = agent.scrape_urls(['https://example.com/1', 'https://example.com/2'])
            >>> print(f"Scraped {result.success_count} sources")
        """
        self.log(f"Scraping {len(urls)} URLs")

        try:
            sources, failed = scrape_multiple_urls(
                urls,
                timeout=self.timeout,
                max_retries=self.max_retries,
                skip_errors=self.skip_errors,
            )

            result = ScrapeResult(sources, failed)

            self.log(
                f"Scraping complete: {result.success_count} succeeded, "
                f"{result.failure_count} failed ({result.success_rate:.1f}% success rate)"
            )

            return result

        except Exception as e:
            self.log(f"Error during batch scraping: {e}", level="ERROR")
            # Return empty result on complete failure
            return ScrapeResult([], urls)

    def scrape_single_url(self, url: str) -> Optional[Source]:
        """
        Scrape a single URL.

        Args:
            url: URL to scrape

        Returns:
            Source object or None if scraping fails

        Example:
            >>> agent = ScraperAgent()
            >>> source = agent.scrape_single_url('https://example.com/article')
            >>> if source:
            ...     print(f"Title: {source.title}")
        """
        self.log(f"Scraping single URL: {url}")

        try:
            source = scrape_url(url, timeout=self.timeout, max_retries=self.max_retries)
            self.log(f"Successfully scraped: {source.title[:50]}...")
            return source

        except Exception as e:
            self.log(f"Failed to scrape {url}: {e}", level="ERROR")
            return None

    def run(self, urls: List[str], min_success_rate: float = 0.0, **kwargs) -> Dict[str, Any]:
        """
        Execute scraping workflow: scrape URLs → track results.

        Args:
            urls: List of URLs to scrape
            min_success_rate: Minimum acceptable success rate (0-100)
            **kwargs: Additional parameters

        Returns:
            Dictionary with:
                - success: bool (True if success_rate >= min_success_rate)
                - sources: List[Source]
                - failed_urls: List[str]
                - success_count: int
                - failure_count: int
                - success_rate: float
                - error: Optional[str]

        Example:
            >>> agent = ScraperAgent(verbose=True)
            >>> result = agent.run(urls=['https://example.com/1', 'https://example.com/2'])
            >>> print(f"Scraped {len(result['sources'])} sources")
        """
        try:
            # Validate inputs
            if not urls:
                return self.handle_error(ValueError("No URLs provided"), context="ScraperAgent.run")

            # Remove duplicates while preserving order
            unique_urls = list(dict.fromkeys(urls))
            if len(unique_urls) < len(urls):
                self.log(f"Removed {len(urls) - len(unique_urls)} duplicate URLs", level="INFO")

            # Execute scraping
            result = self.scrape_urls(unique_urls)

            # Check if minimum success rate met
            success = result.success_rate >= min_success_rate

            if not success:
                self.log(
                    f"Success rate {result.success_rate:.1f}% below minimum {min_success_rate}%",
                    level="WARNING",
                )

            # Return structured results
            return self.create_success_result(
                {
                    "sources": result.sources,
                    "failed_urls": result.failed_urls,
                    "success_count": result.success_count,
                    "failure_count": result.failure_count,
                    "success_rate": result.success_rate,
                    "meets_threshold": success,
                }
            )

        except Exception as e:
            return self.handle_error(e, context="ScraperAgent.run")

    def filter_sources_by_length(
        self, sources: List[Source], min_length: int = 100
    ) -> List[Source]:
        """
        Filter sources by content length.

        Args:
            sources: List of Source objects
            min_length: Minimum content length

        Returns:
            Filtered list of sources

        Example:
            >>> agent = ScraperAgent()
            >>> sources = [...]  # List of Source objects
            >>> long_sources = agent.filter_sources_by_length(sources, min_length=500)
        """
        filtered = [s for s in sources if len(s.content) >= min_length]

        if self.verbose and len(filtered) < len(sources):
            self.log(
                f"Filtered out {len(sources) - len(filtered)} sources with content < {min_length} chars"
            )

        return filtered

    def get_scraping_stats(self, sources: List[Source]) -> Dict[str, Any]:
        """
        Get statistics about scraped sources.

        Args:
            sources: List of Source objects

        Returns:
            Dictionary with scraping statistics

        Example:
            >>> agent = ScraperAgent()
            >>> sources = [...]
            >>> stats = agent.get_scraping_stats(sources)
            >>> print(f"Average content length: {stats['avg_content_length']}")
        """
        if not sources:
            return {
                "total_sources": 0,
                "avg_content_length": 0,
                "min_content_length": 0,
                "max_content_length": 0,
                "total_words": 0,
                "domains": [],
            }

        content_lengths = [len(s.content) for s in sources]
        domains = list(set(s.get_domain() for s in sources))

        return {
            "total_sources": len(sources),
            "avg_content_length": sum(content_lengths) / len(content_lengths),
            "min_content_length": min(content_lengths),
            "max_content_length": max(content_lengths),
            "total_words": sum(s.metadata.get("word_count", 0) for s in sources),
            "domains": domains,
            "domain_count": len(domains),
        }
