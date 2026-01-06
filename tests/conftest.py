"""
Global pytest fixtures and configuration.

This file contains fixtures and hooks that are available to all test modules.
"""

import os
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Generator
import pytest


# ============================================================================
# Pytest Hooks
# ============================================================================

def pytest_configure(config):
    """
    Pytest configuration hook - runs before test collection.

    Sets up environment variables and configuration for test runs.
    """
    # Set testing environment variable
    os.environ["TESTING"] = "1"

    # Disable ChromaDB telemetry during tests
    os.environ["ANONYMIZED_TELEMETRY"] = "False"

    # Set test data paths to temporary directories
    os.environ["VECTOR_DB_PATH"] = os.path.join(tempfile.gettempdir(), "test_vector_db")
    os.environ["REPORTS_PATH"] = os.path.join(tempfile.gettempdir(), "test_reports")


def pytest_collection_modifyitems(config, items):
    """
    Modify test items after collection.

    Auto-adds markers based on test location:
    - Tests in tests/unit/ get "unit" marker
    - Tests in tests/integration/ get "integration" marker
    """
    for item in items:
        # Get test file path relative to tests directory
        test_path = Path(item.fspath).relative_to(Path(__file__).parent)

        # Auto-add markers based on directory
        if "unit" in test_path.parts:
            item.add_marker(pytest.mark.unit)
        elif "integration" in test_path.parts:
            item.add_marker(pytest.mark.integration)

        # Auto-add markers based on test name patterns
        if "internet" in item.nodeid.lower():
            item.add_marker(pytest.mark.requires_internet)

        if "ollama" in item.nodeid.lower() or "llm" in item.nodeid.lower():
            item.add_marker(pytest.mark.requires_ollama)


# ============================================================================
# Session-scoped Fixtures (created once per test session)
# ============================================================================

@pytest.fixture(scope="session")
def test_root() -> Path:
    """
    Get the root directory of the test suite.

    Returns:
        Path to tests/ directory
    """
    return Path(__file__).parent


@pytest.fixture(scope="session")
def fixtures_dir(test_root: Path) -> Path:
    """
    Get the fixtures directory.

    Returns:
        Path to tests/fixtures/ directory
    """
    return test_root / "fixtures"


@pytest.fixture(scope="session")
def project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root (parent of tests/)
    """
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def html_samples_dir(fixtures_dir: Path) -> Path:
    """
    Get the HTML samples directory.

    Returns:
        Path to tests/fixtures/html_samples/
    """
    return fixtures_dir / "html_samples"


# ============================================================================
# Function-scoped Fixtures (created for each test)
# ============================================================================

@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test data.

    Automatically cleaned up after test completion.

    Yields:
        Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_env_vars(monkeypatch) -> dict[str, str]:
    """
    Set test-specific environment variables.

    Args:
        monkeypatch: pytest monkeypatch fixture

    Returns:
        Dictionary of environment variables set
    """
    env_vars = {
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "llama3.2:3b",
        "MAX_SOURCES": "10",
        "TRUSTWORTHINESS_THRESHOLD": "85.0",
        "SCRAPER_TIMEOUT": "10",
        "SCRAPER_MAX_RETRIES": "2",
        "MAX_SEARCH_RESULTS_PER_QUERY": "10",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def test_settings(test_env_vars, temp_data_dir: Path):
    """
    Create a Settings instance for testing.

    Uses test environment variables and temporary directories.

    Args:
        test_env_vars: Test environment variables fixture
        temp_data_dir: Temporary directory fixture

    Returns:
        Settings instance configured for testing
    """
    from research_assistant.config import Settings

    # Override paths to use temp directory
    settings = Settings(
        vector_db_path=temp_data_dir / "vector_db",
        reports_path=temp_data_dir / "reports",
    )

    return settings


@pytest.fixture
def mock_timestamp() -> datetime:
    """
    Provide a consistent timestamp for tests.

    Returns:
        Fixed datetime (2024-01-15 12:00:00)
    """
    return datetime(2024, 1, 15, 12, 0, 0)


@pytest.fixture
def sample_urls() -> list[str]:
    """
    Provide sample URLs for testing.

    Returns:
        List of sample URLs
    """
    return [
        "https://example.com/article1",
        "https://example.org/article2",
        "https://test.edu/research",
        "https://news.com/story",
        "https://github.com/user/repo",
    ]


@pytest.fixture
def sample_search_query() -> str:
    """
    Provide a sample search query.

    Returns:
        Sample query string
    """
    return "artificial intelligence ethics"


# ============================================================================
# Auto-applied Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def isolate_unit_tests(request, monkeypatch):
    """
    Isolate unit tests by preventing real external calls.

    This fixture automatically applies to all tests marked with "unit".
    It prevents accidental real HTTP calls by raising errors.

    Args:
        request: pytest request fixture
        monkeypatch: pytest monkeypatch fixture
    """
    # Only apply to unit tests
    if "unit" not in request.keywords:
        return

    # Prevent real HTTP requests
    def raise_on_request(*args, **kwargs):
        raise RuntimeError(
            "Unit tests should not make real HTTP requests! "
            "Use mocks instead (see tests/mocks/)."
        )

    # Mock requests.get and requests.post
    try:
        import requests
        monkeypatch.setattr(requests, "get", raise_on_request)
        monkeypatch.setattr(requests, "post", raise_on_request)
    except ImportError:
        pass  # requests not installed yet


@pytest.fixture(autouse=True)
def reset_default_store():
    """
    Reset the default vector store singleton between tests.

    This ensures tests don't interfere with each other through
    the global _default_store variable.
    """
    # Import here to avoid issues if module not yet created
    try:
        from research_assistant.tools import vector_store
        vector_store._default_store = None
        yield
        vector_store._default_store = None
    except ImportError:
        yield  # Module not ready yet, skip


# ============================================================================
# Pytest Markers Documentation
# ============================================================================

"""
Available pytest markers:

@pytest.mark.unit
    Unit tests - isolated, no external dependencies

@pytest.mark.integration
    Integration tests - multiple components together

@pytest.mark.slow
    Tests that take significant time to run
    Skip with: pytest -m "not slow"

@pytest.mark.requires_internet
    Tests requiring internet connectivity
    Skip with: pytest -m "not requires_internet"

@pytest.mark.requires_ollama
    Tests requiring Ollama to be running
    Skip with: pytest -m "not requires_ollama"

Examples:
    # Run only unit tests
    pytest -m unit

    # Run unit tests excluding slow ones
    pytest -m "unit and not slow"

    # Run all tests except those requiring internet
    pytest -m "not requires_internet"
"""
