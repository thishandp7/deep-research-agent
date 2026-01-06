"""
Test fixtures for graph tests.

Provides mocks for LLM, agents, and external dependencies.
"""

import pytest
from unittest.mock import Mock

from research_assistant.models.source import Source
from research_assistant.graph.state import ResearchState, create_initial_state


# ============================================================================
# State Fixtures
# ============================================================================


@pytest.fixture
def sample_research_state() -> ResearchState:
    """Sample research state for testing."""
    return create_initial_state(topic="quantum computing", max_sources=10)


@pytest.fixture
def research_state_with_queries() -> ResearchState:
    """Research state after query generation."""
    state = create_initial_state(topic="quantum computing", max_sources=10)
    state["search_queries"] = [
        "quantum computing basics",
        "quantum algorithms tutorial",
        "quantum computing applications",
    ]
    state["current_step"] = "query_generation_complete"
    return state


@pytest.fixture
def research_state_with_urls() -> ResearchState:
    """Research state after search."""
    state = create_initial_state(topic="quantum computing", max_sources=10)
    state["search_queries"] = ["quantum computing"]
    state["discovered_urls"] = [
        "https://example1.com/article1",
        "https://example2.com/article2",
        "https://example3.com/article3",
    ]
    state["current_step"] = "search_complete"
    return state


@pytest.fixture
def research_state_with_sources() -> ResearchState:
    """Research state after scraping."""
    state = create_initial_state(topic="quantum computing", max_sources=10)
    state["search_queries"] = ["quantum computing"]
    state["discovered_urls"] = ["https://example1.com/article1"]
    state["scraped_sources"] = [
        Source(
            url="https://example1.com/article1",
            title="Quantum Computing Basics",
            content="Quantum computing uses quantum mechanics principles. " * 50,
            metadata={"domain": "example1.com", "word_count": 100},
        )
    ]
    state["current_step"] = "scraping_complete"
    return state


@pytest.fixture
def research_state_with_analyzed_sources() -> ResearchState:
    """Research state after analysis."""
    state = create_initial_state(topic="quantum computing", max_sources=10)
    state["analyzed_sources"] = [
        Source(
            url="https://example1.com/article1",
            title="High Quality Source",
            content="Well researched content. " * 50,
            trustworthiness_score=92.0,
            metadata={"domain": "example1.com"},
        ),
        Source(
            url="https://example2.com/article2",
            title="Medium Quality Source",
            content="Decent content. " * 50,
            trustworthiness_score=78.0,
            metadata={"domain": "example2.com"},
        ),
        Source(
            url="https://example3.com/article3",
            title="Low Quality Source",
            content="Poor content. " * 50,
            trustworthiness_score=45.0,
            metadata={"domain": "example3.com"},
        ),
    ]
    state["current_step"] = "analysis_complete"
    return state


# ============================================================================
# Mock LLM Fixtures
# ============================================================================


@pytest.fixture
def mock_llm_query_generation(monkeypatch):
    """Mock LLM for query generation."""

    def mock_invoke(prompt):
        response = Mock()
        response.content = """quantum computing basics
quantum algorithms
quantum hardware
applications of quantum computing
quantum vs classical computing"""
        return response

    mock_llm = Mock()
    mock_llm.invoke = mock_invoke

    # Mock get_llm to return our mock
    def mock_get_llm(*args, **kwargs):
        return mock_llm

    monkeypatch.setattr("research_assistant.agents.base.get_llm", mock_get_llm)
    return mock_llm


@pytest.fixture
def mock_llm_analysis(monkeypatch):
    """Mock LLM for trustworthiness analysis."""

    def mock_invoke(prompt):
        response = Mock()
        # Return valid JSON response
        response.content = """{
            "score": 85,
            "reasoning": "High quality source with good citations and balanced perspective.",
            "red_flags": [],
            "strengths": ["Well researched", "Credible domain", "Clear citations"]
        }"""
        return response

    mock_llm = Mock()
    mock_llm.invoke = mock_invoke

    def mock_get_llm(*args, **kwargs):
        return mock_llm

    monkeypatch.setattr("research_assistant.agents.base.get_llm", mock_get_llm)
    return mock_llm


@pytest.fixture
def mock_llm_report_generation(monkeypatch):
    """Mock LLM for report generation."""
    call_count = 0

    def mock_invoke(prompt):
        nonlocal call_count
        call_count += 1

        response = Mock()
        if call_count == 1:  # Executive summary
            response.content = "This is an executive summary of quantum computing research."
        else:  # Key findings
            response.content = "1. Finding one\n2. Finding two\n3. Finding three"
        return response

    mock_llm = Mock()
    mock_llm.invoke = mock_invoke

    def mock_get_llm(*args, **kwargs):
        return mock_llm

    monkeypatch.setattr("research_assistant.agents.base.get_llm", mock_get_llm)
    return mock_llm


# ============================================================================
# Agent Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_searcher_agent(monkeypatch, mock_ddgs, mock_llm_query_generation):
    """Mock SearcherAgent methods."""
    from research_assistant.agents import SearcherAgent

    def mock_generate_queries(self, topic):
        return ["quantum computing basics", "quantum algorithms", "quantum applications"]

    def mock_search_urls(self, queries):
        return [f"https://example{i}.com/article{i}" for i in range(1, min(6, len(queries) * 2))]

    monkeypatch.setattr(SearcherAgent, "generate_queries", mock_generate_queries)
    monkeypatch.setattr(SearcherAgent, "search_urls", mock_search_urls)

    yield


@pytest.fixture
def mock_scraper_agent(monkeypatch, mock_newspaper_article, mock_requests_get_success):
    """Mock ScraperAgent methods."""
    from research_assistant.agents import ScraperAgent

    def mock_run(self, urls, **kwargs):
        sources = [
            Source(
                url=url,
                title=f"Article from {url}",
                content=f"Content about quantum computing from {url}. " * 50,
                metadata={"domain": url.split("/")[2], "word_count": 100},
            )
            for url in urls[:3]  # Only scrape first 3
        ]

        return {
            "success": True,
            "sources": sources,
            "failed_urls": urls[3:] if len(urls) > 3 else [],
            "success_count": len(sources),
            "failure_count": max(0, len(urls) - 3),
        }

    monkeypatch.setattr(ScraperAgent, "run", mock_run)

    yield


@pytest.fixture
def mock_analyzer_agent(monkeypatch, mock_llm_analysis):
    """Mock AnalyzerAgent methods."""
    from research_assistant.agents import AnalyzerAgent

    def mock_run(self, sources, topic, **kwargs):
        # Assign varying scores
        for i, source in enumerate(sources):
            # First 40% get high scores, rest get low scores
            if i < len(sources) * 0.4:
                source.trustworthiness_score = 90.0
            else:
                source.trustworthiness_score = 70.0

        return {
            "success": True,
            "sources": sources,
            "total_analyzed": len(sources),
            "trustworthy_count": sum(1 for s in sources if s.trustworthiness_score >= 85),
            "average_score": sum(s.trustworthiness_score for s in sources) / len(sources),
        }

    monkeypatch.setattr(AnalyzerAgent, "run", mock_run)

    yield


@pytest.fixture
def mock_reporter_agent(monkeypatch, mock_llm_report_generation):
    """Mock ReporterAgent methods."""
    from research_assistant.agents import ReporterAgent

    def mock_run(self, topic, sources, **kwargs):
        html = f"""<!DOCTYPE html>
<html>
<head><title>Research Report: {topic}</title></head>
<body>
    <h1>{topic}</h1>
    <p>Found {len(sources)} sources.</p>
</body>
</html>"""

        return {"success": True, "html_report": html, "sources_count": len(sources)}

    monkeypatch.setattr(ReporterAgent, "run", mock_run)

    yield


# ============================================================================
# Vector Store Mock Fixture
# ============================================================================


@pytest.fixture
def mock_vector_store(monkeypatch, mock_chroma_client, mock_sentence_transformer):
    """Mock VectorStore operations."""
    from research_assistant.tools.vector_store import VectorStore

    def mock_add_sources(self, sources):
        # Just store the count, don't actually add to DB
        return len(sources)

    monkeypatch.setattr(VectorStore, "add_sources", mock_add_sources)

    yield
