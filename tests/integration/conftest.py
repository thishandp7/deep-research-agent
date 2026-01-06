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
    mock_chroma_client,
    mock_ddgs,
    mock_ddgs_empty,
    mock_newspaper_article,
    mock_newspaper_failing,
    mock_requests_get_failure,
    mock_requests_get_success,
    mock_sentence_transformer,
    mock_vector_store_full,
    sample_source,
    sample_sources,
)

# Import graph-specific fixtures
from tests.unit.graph.conftest import (
    mock_analyzer_agent,
    mock_llm_analysis,
    mock_llm_query_generation,
    mock_llm_report_generation,
    mock_reporter_agent,
    mock_scraper_agent,
    mock_searcher_agent,
    mock_vector_store,
    research_state_with_analyzed_sources,
    research_state_with_queries,
    research_state_with_sources,
    research_state_with_urls,
    sample_research_state,
)
