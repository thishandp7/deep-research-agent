"""
Mock implementation of newspaper3k Article class.

Simulates newspaper.Article without downloading/parsing real web pages.
"""

from datetime import datetime
from typing import Optional, List


# ============================================================================
# Sample Article Content
# ============================================================================

ARTICLE_CONTENT_GOOD = """
Artificial intelligence (AI) represents a transformative technology that is reshaping how we interact with computers and process information. This comprehensive guide explores the fundamental concepts and practical applications of AI in modern society.

At its core, artificial intelligence refers to computer systems designed to perform tasks that typically require human intelligence. These tasks include visual perception, speech recognition, decision-making, and language translation.

Modern AI systems leverage machine learning algorithms that can identify patterns in large datasets and make predictions or decisions based on that analysis. Deep learning, a subset of machine learning, uses neural networks with multiple layers to process complex information.

AI technology has found applications across numerous industries. In healthcare, AI assists with medical diagnosis and drug discovery. In finance, it powers fraud detection and algorithmic trading systems. Transportation benefits from autonomous vehicles and traffic optimization.

Natural language processing enables chatbots and virtual assistants to understand and respond to human queries. Computer vision allows machines to analyze images and videos for various purposes, from facial recognition to quality control in manufacturing.

As AI systems become more prevalent, important ethical questions arise. Issues of bias in AI algorithms, privacy concerns with data collection, and the potential impact on employment require careful consideration by researchers, policymakers, and society as a whole.

Ensuring AI development proceeds responsibly and benefits all of humanity remains a critical challenge for the coming decades.
"""


ARTICLE_CONTENT_MINIMAL = "This is very short content."


ARTICLE_CONTENT_PYTHON = """
Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum and first released in 1991, Python has become one of the most popular programming languages in the world.

Python's design philosophy emphasizes code readability with significant use of whitespace. The language provides constructs that enable clear programming on both small and large scales.

Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming. It features a dynamic type system and automatic memory management.

The Python Package Index (PyPI) hosts thousands of third-party modules that extend Python's capabilities. Popular frameworks and libraries include Django for web development, NumPy for scientific computing, and TensorFlow for machine learning.

Python is widely used in web development, data analysis, artificial intelligence, scientific computing, automation, and many other domains.
"""


# URL-based content mapping
URL_CONTENT_MAP: dict[str, tuple[str, str]] = {
    "good-article": ("Understanding Artificial Intelligence", ARTICLE_CONTENT_GOOD),
    "minimal": ("Short Article", ARTICLE_CONTENT_MINIMAL),
    "python": ("Python Programming Language", ARTICLE_CONTENT_PYTHON),
    "ai": ("AI Research", ARTICLE_CONTENT_GOOD),
    "example.com": ("Example Article", ARTICLE_CONTENT_GOOD),
}


# ============================================================================
# Mock Article Class
# ============================================================================


class MockArticle:
    """
    Mock newspaper Article class.

    Simulates Article.download() and Article.parse() without real web access.
    """

    def __init__(self, url: str, language: str = "en"):
        """
        Initialize mock article.

        Args:
            url: Article URL
            language: Language code
        """
        self.url = url
        self.language = language

        # Article attributes (populated after parse())
        self.title: str = ""
        self.text: str = ""
        self.authors: List[str] = []
        self.publish_date: Optional[datetime] = None
        self.top_image: str = ""
        self.html: str = ""

        # Internal state
        self._downloaded = False
        self._parsed = False

    def download(self):
        """
        Simulate downloading article HTML.

        Raises:
            Exception: If download fails (based on URL)
        """
        # Simulate failure for certain URLs
        if "fail" in self.url.lower() or "error" in self.url.lower():
            raise Exception(f"Failed to download article from {self.url}")

        # Mark as downloaded
        self._downloaded = True

        # Simulate HTML content
        self.html = "<html><body><h1>Article</h1><p>Content</p></body></html>"

    def parse(self):
        """
        Simulate parsing article content.

        Raises:
            Exception: If not downloaded or parsing fails
        """
        if not self._downloaded:
            raise Exception("Article must be downloaded before parsing")

        # Simulate parsing failure for certain URLs
        if "parse-error" in self.url.lower():
            raise Exception("Failed to parse article")

        # Extract content based on URL
        self._extract_content_from_url()

        # Mark as parsed
        self._parsed = True

    def _extract_content_from_url(self):
        """
        Extract content based on URL pattern.

        Uses URL_CONTENT_MAP for known patterns.
        """
        # Check URL patterns
        for pattern, (title, content) in URL_CONTENT_MAP.items():
            if pattern in self.url.lower():
                self.title = title
                self.text = content
                self._set_metadata_for_pattern(pattern)
                return

        # Default content
        self.title = "Default Article Title"
        self.text = ARTICLE_CONTENT_GOOD
        self.authors = ["Test Author"]
        self.publish_date = datetime(2024, 1, 15, 10, 30, 0)
        self.top_image = "https://example.com/image.jpg"

    def _set_metadata_for_pattern(self, pattern: str):
        """
        Set metadata based on content pattern.

        Args:
            pattern: URL pattern matched
        """
        if pattern == "good-article" or pattern == "ai":
            self.authors = ["Dr. Jane Smith", "Dr. John Doe"]
            self.publish_date = datetime(2024, 1, 15, 14, 0, 0)
            self.top_image = "https://example.com/ai-image.jpg"

        elif pattern == "minimal":
            self.authors = []
            self.publish_date = None
            self.top_image = ""

        elif pattern == "python":
            self.authors = ["Python Developer"]
            self.publish_date = datetime(2024, 1, 10, 9, 0, 0)
            self.top_image = "https://python.org/logo.png"

        else:
            self.authors = ["Test Author"]
            self.publish_date = datetime(2024, 1, 15, 12, 0, 0)
            self.top_image = "https://example.com/default.jpg"


class FailingMockArticle(MockArticle):
    """
    Mock Article that always fails during download.

    Useful for testing error handling and fallback mechanisms.
    """

    def download(self):
        """Always raise an exception."""
        raise Exception("Download always fails in FailingMockArticle")


class ParseFailingMockArticle(MockArticle):
    """
    Mock Article that fails during parsing.

    Download succeeds, but parse() raises an exception.
    """

    def download(self):
        """Succeed."""
        self._downloaded = True

    def parse(self):
        """Always raise an exception."""
        raise Exception("Parsing always fails in ParseFailingMockArticle")


class MinimalContentMockArticle(MockArticle):
    """
    Mock Article that returns insufficient content.

    Used to test minimum content length validation.
    """

    def _extract_content_from_url(self):
        """Return minimal content regardless of URL."""
        self.title = "Minimal"
        self.text = "Too short"
        self.authors = []
        self.publish_date = None
        self.top_image = ""


# ============================================================================
# Factory Functions
# ============================================================================


def create_mock_article_with_content(
    url: str,
    title: str,
    content: str,
    authors: Optional[List[str]] = None,
    publish_date: Optional[datetime] = None,
) -> MockArticle:
    """
    Create a MockArticle with specific content.

    Args:
        url: Article URL
        title: Article title
        content: Article text content
        authors: List of authors
        publish_date: Publication date

    Returns:
        Configured MockArticle instance

    Example:
        >>> article = create_mock_article_with_content(
        ...     "https://test.com/article",
        ...     "Test Title",
        ...     "Test content here.",
        ...     authors=["Author Name"]
        ... )
        >>> article.download()
        >>> article.parse()
        >>> assert article.title == "Test Title"
    """

    class CustomMockArticle(MockArticle):
        def _extract_content_from_url(self):
            self.title = title
            self.text = content
            self.authors = authors or []
            self.publish_date = publish_date
            self.top_image = "https://example.com/image.jpg"

    return CustomMockArticle(url)


def create_mock_article_class_with_content(
    title: str,
    content: str,
    authors: Optional[List[str]] = None,
    publish_date: Optional[datetime] = None,
) -> type[MockArticle]:
    """
    Create a MockArticle class with specific content.

    Useful for monkeypatching newspaper.Article.

    Args:
        title: Article title
        content: Article text content
        authors: List of authors
        publish_date: Publication date

    Returns:
        MockArticle subclass

    Example:
        >>> ArticleClass = create_mock_article_class_with_content(
        ...     "Custom Title", "Custom content"
        ... )
        >>> article = ArticleClass("https://test.com")
        >>> article.download()
        >>> article.parse()
        >>> assert article.title == "Custom Title"
    """

    class CustomMockArticle(MockArticle):
        def _extract_content_from_url(self):
            self.title = title
            self.text = content
            self.authors = authors or []
            self.publish_date = publish_date
            self.top_image = "https://example.com/image.jpg"

    return CustomMockArticle
