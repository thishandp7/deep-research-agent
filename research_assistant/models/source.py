"""
Source data model for research content.

This module defines the Source model used throughout the research assistant
to represent discovered web sources with their content and metadata.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl, ConfigDict


class Source(BaseModel):
    """
    Represents a web source with content and trustworthiness metadata.

    Attributes:
        url: The source URL
        title: Article/page title
        content: Extracted main content (plain text)
        trustworthiness_score: Score from 0-100 (higher = more trustworthy)
        metadata: Additional metadata (author, date, domain, etc.)
        scraped_at: Timestamp when content was scraped

    Example:
        >>> source = Source(
        ...     url="https://example.com/article",
        ...     title="Understanding AI",
        ...     content="Artificial intelligence is...",
        ...     trustworthiness_score=87.5
        ... )
    """

    url: str = Field(
        ...,
        description="Source URL"
    )

    title: str = Field(
        default="",
        description="Article or page title"
    )

    content: str = Field(
        default="",
        description="Extracted main content (plain text)"
    )

    trustworthiness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Trustworthiness score (0-100)"
    )

    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata (author, date, domain, etc.)"
    )

    scraped_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when content was scraped"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "url": "https://example.com/ai-article",
                "title": "Understanding Artificial Intelligence",
                "content": "Artificial intelligence (AI) is the simulation of human intelligence...",
                "trustworthiness_score": 87.5,
                "metadata": {
                    "author": "Dr. Jane Smith",
                    "publish_date": "2024-01-15",
                    "domain": "example.com",
                    "word_count": 1250
                },
                "scraped_at": "2024-01-20T10:30:00"
            }
        }
    )

    def is_trustworthy(self, threshold: float = 85.0) -> bool:
        """
        Check if source meets trustworthiness threshold.

        Args:
            threshold: Minimum score to be considered trustworthy (default: 85.0)

        Returns:
            True if trustworthiness_score >= threshold
        """
        return self.trustworthiness_score >= threshold

    def get_domain(self) -> str:
        """
        Extract domain from URL.

        Returns:
            Domain name (e.g., "example.com")
        """
        from urllib.parse import urlparse
        parsed = urlparse(self.url)
        return parsed.netloc

    def get_content_preview(self, max_chars: int = 200) -> str:
        """
        Get truncated content preview.

        Args:
            max_chars: Maximum characters to return (default: 200)

        Returns:
            Truncated content with ellipsis if needed
        """
        if len(self.content) <= max_chars:
            return self.content
        return self.content[:max_chars] + "..."

    def __str__(self) -> str:
        """String representation"""
        return f"Source(url={self.url}, trust={self.trustworthiness_score:.1f})"

    def __repr__(self) -> str:
        """Detailed representation"""
        return (
            f"Source(url='{self.url}', title='{self.title[:30]}...', "
            f"trust={self.trustworthiness_score:.1f})"
        )


class SourceCollection(BaseModel):
    """
    Collection of sources with utility methods.

    Useful for managing multiple sources together.
    """

    sources: list[Source] = Field(
        default_factory=list,
        description="List of sources"
    )

    def add(self, source: Source) -> None:
        """Add a source to the collection"""
        self.sources.append(source)

    def filter_by_score(self, min_score: float) -> list[Source]:
        """
        Filter sources by minimum trustworthiness score.

        Args:
            min_score: Minimum trustworthiness score

        Returns:
            List of sources with score >= min_score
        """
        return [s for s in self.sources if s.trustworthiness_score >= min_score]

    def get_trustworthy(self, threshold: float = 85.0) -> list[Source]:
        """
        Get trustworthy sources above threshold.

        Args:
            threshold: Minimum score (default: 85.0)

        Returns:
            List of trustworthy sources
        """
        return self.filter_by_score(threshold)

    def get_statistics(self) -> dict:
        """
        Get collection statistics.

        Returns:
            Dictionary with stats (count, avg_score, trustworthy_count)
        """
        if not self.sources:
            return {
                "total_count": 0,
                "average_score": 0.0,
                "trustworthy_count": 0,
                "trustworthy_percentage": 0.0
            }

        scores = [s.trustworthiness_score for s in self.sources]
        trustworthy = self.get_trustworthy()

        return {
            "total_count": len(self.sources),
            "average_score": sum(scores) / len(scores),
            "trustworthy_count": len(trustworthy),
            "trustworthy_percentage": (len(trustworthy) / len(self.sources)) * 100
        }

    def __len__(self) -> int:
        """Length of collection"""
        return len(self.sources)

    def __iter__(self):
        """Iterate over sources"""
        return iter(self.sources)
