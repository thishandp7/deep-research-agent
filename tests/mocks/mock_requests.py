"""
Mock implementation of HTTP requests.

Simulates requests.get() without making real HTTP calls.
"""

from typing import Optional, Callable
import requests


# ============================================================================
# HTML Sample Content
# ============================================================================

HTML_ARTICLE_GOOD = """
<!DOCTYPE html>
<html>
<head>
    <title>Understanding Artificial Intelligence</title>
    <meta charset="utf-8">
</head>
<body>
    <header>
        <nav>Navigation menu here</nav>
    </header>

    <article>
        <h1>Understanding Artificial Intelligence</h1>

        <div class="meta">
            <span class="author">By Dr. Jane Smith</span>
            <span class="date">January 15, 2024</span>
        </div>

        <p>Artificial intelligence (AI) represents a transformative technology that is reshaping how we interact with computers and process information. This comprehensive guide explores the fundamental concepts and practical applications of AI in modern society.</p>

        <h2>What is Artificial Intelligence?</h2>

        <p>At its core, artificial intelligence refers to computer systems designed to perform tasks that typically require human intelligence. These tasks include visual perception, speech recognition, decision-making, and language translation.</p>

        <p>Modern AI systems leverage machine learning algorithms that can identify patterns in large datasets and make predictions or decisions based on that analysis. Deep learning, a subset of machine learning, uses neural networks with multiple layers to process complex information.</p>

        <h2>Applications of AI</h2>

        <p>AI technology has found applications across numerous industries. In healthcare, AI assists with medical diagnosis and drug discovery. In finance, it powers fraud detection and algorithmic trading systems. Transportation benefits from autonomous vehicles and traffic optimization.</p>

        <p>Natural language processing enables chatbots and virtual assistants to understand and respond to human queries. Computer vision allows machines to analyze images and videos for various purposes, from facial recognition to quality control in manufacturing.</p>

        <h2>Ethical Considerations</h2>

        <p>As AI systems become more prevalent, important ethical questions arise. Issues of bias in AI algorithms, privacy concerns with data collection, and the potential impact on employment require careful consideration by researchers, policymakers, and society as a whole.</p>

        <p>Ensuring AI development proceeds responsibly and benefits all of humanity remains a critical challenge for the coming decades.</p>
    </article>

    <footer>
        <p>© 2024 AI Research Institute</p>
    </footer>

    <script>
        // Some analytics script
        console.log("Page loaded");
    </script>
</body>
</html>
"""


HTML_ARTICLE_MINIMAL = """
<!DOCTYPE html>
<html>
<head>
    <title>Short Article</title>
</head>
<body>
    <h1>Brief Note</h1>
    <p>This is very short content.</p>
</body>
</html>
"""


HTML_NO_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>No Real Content</title>
</head>
<body>
    <nav>
        <a href="/">Home</a>
        <a href="/about">About</a>
    </nav>

    <script>
        var x = 10;
        console.log("JavaScript code");
    </script>

    <style>
        body { color: black; }
    </style>

    <footer>
        Copyright 2024
    </footer>
</body>
</html>
"""


HTML_COMPLEX_STRUCTURE = """
<!DOCTYPE html>
<html>
<head>
    <title>Complex Page Structure</title>
</head>
<body>
    <div class="sidebar">
        <p>Sidebar content that should be ignored</p>
    </div>

    <div class="content">
        <article>
            <h1>Main Article Title</h1>
            <p>This is the main content paragraph one.</p>
            <p>This is the main content paragraph two.</p>
        </article>

        <aside>
            <p>Related links and ads</p>
        </aside>
    </div>

    <div class="comments">
        <p>User comments section</p>
    </div>
</body>
</html>
"""


# Map URLs to HTML content for testing
URL_TO_HTML_MAP: dict[str, str] = {
    "good-article": HTML_ARTICLE_GOOD,
    "minimal": HTML_ARTICLE_MINIMAL,
    "no-content": HTML_NO_CONTENT,
    "complex": HTML_COMPLEX_STRUCTURE,
}


# ============================================================================
# Mock Response Class
# ============================================================================

class MockResponse:
    """
    Mock HTTP response object.

    Simulates requests.Response.
    """

    def __init__(
        self,
        content: bytes,
        status_code: int = 200,
        headers: Optional[dict] = None,
        url: str = ""
    ):
        """
        Initialize mock response.

        Args:
            content: Response body as bytes
            status_code: HTTP status code
            headers: Response headers
            url: Request URL
        """
        self.content = content
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.url = url
        self.text = content.decode('utf-8')
        self.encoding = 'utf-8'

    def raise_for_status(self):
        """
        Raise HTTPError for bad status codes.

        Raises:
            requests.HTTPError: If status code >= 400
        """
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code} Error", response=self)


# ============================================================================
# Mock Request Functions
# ============================================================================

def create_mock_requests_get(
    html_content: str,
    status_code: int = 200,
    headers: Optional[dict] = None
) -> Callable:
    """
    Create a mock requests.get function.

    Args:
        html_content: HTML content to return
        status_code: HTTP status code
        headers: Response headers

    Returns:
        Mock get function

    Example:
        >>> mock_get = create_mock_requests_get(HTML_ARTICLE_GOOD, 200)
        >>> response = mock_get("https://example.com")
        >>> assert response.status_code == 200
    """
    def mock_get(url: str, *args, **kwargs) -> MockResponse:
        # Allow URL-based content selection
        for key, content in URL_TO_HTML_MAP.items():
            if key in url:
                html_content_to_use = content
                break
        else:
            html_content_to_use = html_content

        return MockResponse(
            content=html_content_to_use.encode('utf-8'),
            status_code=status_code,
            headers=headers,
            url=url
        )

    return mock_get


def create_failing_requests_get(
    exception: Optional[Exception] = None
) -> Callable:
    """
    Create a mock requests.get that always fails.

    Args:
        exception: Exception to raise (default: generic Exception)

    Returns:
        Mock get function that raises exception

    Example:
        >>> mock_get = create_failing_requests_get()
        >>> mock_get("https://example.com")  # Raises Exception
    """
    if exception is None:
        exception = Exception("HTTP request failed")

    def mock_get(url: str, *args, **kwargs):
        raise exception

    return mock_get


def create_mock_requests_get_with_retry(
    fail_count: int,
    html_content: str,
    status_code: int = 200
) -> Callable:
    """
    Create a mock requests.get that fails a certain number of times.

    Useful for testing retry logic.

    Args:
        fail_count: Number of times to fail before succeeding
        html_content: HTML content to return on success
        status_code: HTTP status code on success

    Returns:
        Mock get function with retry behavior

    Example:
        >>> mock_get = create_mock_requests_get_with_retry(2, HTML_ARTICLE_GOOD)
        >>> # First 2 calls fail, third succeeds
    """
    call_count = {"count": 0}

    def mock_get(url: str, *args, **kwargs):
        call_count["count"] += 1

        if call_count["count"] <= fail_count:
            raise Exception(f"Attempt {call_count['count']} failed")

        return MockResponse(
            content=html_content.encode('utf-8'),
            status_code=status_code,
            url=url
        )

    return mock_get


def create_mock_requests_get_with_map(url_map: dict[str, tuple[str, int]]) -> Callable:
    """
    Create a mock requests.get with URL-based responses.

    Args:
        url_map: Dict mapping URL patterns to (html_content, status_code) tuples

    Returns:
        Mock get function

    Example:
        >>> url_map = {
        ...     "example.com": (HTML_ARTICLE_GOOD, 200),
        ...     "fail.com": ("<html>Error</html>", 404)
        ... }
        >>> mock_get = create_mock_requests_get_with_map(url_map)
    """
    def mock_get(url: str, *args, **kwargs):
        # Find matching URL pattern
        for pattern, (content, status) in url_map.items():
            if pattern in url:
                return MockResponse(
                    content=content.encode('utf-8'),
                    status_code=status,
                    url=url
                )

        # Default: 404
        return MockResponse(
            content=b"<html><body>Not Found</body></html>",
            status_code=404,
            url=url
        )

    return mock_get


# ============================================================================
# Convenience Functions
# ============================================================================

def get_mock_response_good() -> MockResponse:
    """
    Get a mock response with good HTML content.

    Returns:
        MockResponse with HTML_ARTICLE_GOOD
    """
    return MockResponse(
        content=HTML_ARTICLE_GOOD.encode('utf-8'),
        status_code=200
    )


def get_mock_response_minimal() -> MockResponse:
    """
    Get a mock response with minimal HTML content.

    Returns:
        MockResponse with HTML_ARTICLE_MINIMAL
    """
    return MockResponse(
        content=HTML_ARTICLE_MINIMAL.encode('utf-8'),
        status_code=200
    )


def get_mock_response_no_content() -> MockResponse:
    """
    Get a mock response with no meaningful content.

    Returns:
        MockResponse with HTML_NO_CONTENT
    """
    return MockResponse(
        content=HTML_NO_CONTENT.encode('utf-8'),
        status_code=200
    )


def get_mock_response_404() -> MockResponse:
    """
    Get a 404 mock response.

    Returns:
        MockResponse with 404 status
    """
    return MockResponse(
        content=b"<html><body>404 Not Found</body></html>",
        status_code=404
    )
