"""
Integration test specific fixtures.
"""

import pytest
from pathlib import Path


@pytest.fixture
def integration_temp_dir(temp_data_dir: Path) -> Path:
    """
    Temporary directory for integration tests.

    Returns:
        Path to integration test temp directory
    """
    integration_dir = temp_data_dir / "integration"
    integration_dir.mkdir(parents=True, exist_ok=True)
    return integration_dir


# Import all unit test fixtures for use in integration tests
from tests.unit.conftest import (
    mock_ddgs,
    mock_ddgs_empty,
    mock_newspaper_article,
    mock_newspaper_failing,
    mock_requests_get_success,
    mock_requests_get_failure,
    mock_chroma_client,
    mock_sentence_transformer,
    mock_vector_store_full,
    sample_source,
    sample_sources,
)
