"""
Unit tests for graph nodes.

Tests individual node functions with mocked dependencies.
"""

from research_assistant.graph.nodes import (
    query_gen_node,
    search_node,
    scraper_node,
    analyzer_node,
    storage_node,
    report_node,
    should_store_sources,
)
from research_assistant.models.source import Source


class TestQueryGenNode:
    """Test query generation node."""

    def test_generates_queries_from_topic(self, sample_research_state, mock_llm_query_generation):
        """Test that queries are generated from topic."""
        result = query_gen_node(sample_research_state)

        assert "search_queries" in result
        assert len(result["search_queries"]) > 0
        assert isinstance(result["search_queries"][0], str)

    def test_updates_current_step(self, sample_research_state, mock_llm_query_generation):
        """Test that current_step is updated."""
        result = query_gen_node(sample_research_state)

        assert result["current_step"] == "query_generation_complete"

    def test_handles_llm_failure_gracefully(self, sample_research_state, monkeypatch):
        """Test error handling when LLM fails."""

        def failing_get_llm(*args, **kwargs):
            raise Exception("LLM connection failed")

        monkeypatch.setattr("research_assistant.agents.base.get_llm", failing_get_llm)

        result = query_gen_node(sample_research_state)

        # Should fallback to topic itself
        assert "search_queries" in result
        assert sample_research_state["topic"] in result["search_queries"]
        assert "errors" in result


class TestSearchNode:
    """Test search node."""

    def test_searches_with_generated_queries(
        self, research_state_with_queries, mock_searcher_agent
    ):
        """Test that search is executed with queries."""
        result = search_node(research_state_with_queries)

        assert "discovered_urls" in result
        assert len(result["discovered_urls"]) > 0

    def test_discovered_urls_are_strings(self, research_state_with_queries, mock_searcher_agent):
        """Test that discovered URLs are strings."""
        result = search_node(research_state_with_queries)

        for url in result["discovered_urls"]:
            assert isinstance(url, str)
            assert url.startswith("http")

    def test_limits_urls_to_max_sources(self, research_state_with_queries, mock_searcher_agent):
        """Test that URLs are limited to max_sources."""
        research_state_with_queries["max_sources"] = 3

        result = search_node(research_state_with_queries)

        assert len(result["discovered_urls"]) <= 3

    def test_updates_current_step(self, research_state_with_queries, mock_searcher_agent):
        """Test that current_step is updated."""
        result = search_node(research_state_with_queries)

        assert result["current_step"] == "search_complete"

    def test_handles_search_failure(self, research_state_with_queries, monkeypatch):
        """Test error handling when search fails."""
        from research_assistant.agents import SearcherAgent

        def failing_search_urls(self, queries):
            raise Exception("Search API failed")

        monkeypatch.setattr(SearcherAgent, "search_urls", failing_search_urls)

        result = search_node(research_state_with_queries)

        assert result["discovered_urls"] == []
        assert "errors" in result


class TestScraperNode:
    """Test scraper node."""

    def test_scrapes_discovered_urls(self, research_state_with_urls, mock_scraper_agent):
        """Test that URLs are scraped."""
        result = scraper_node(research_state_with_urls)

        assert "scraped_sources" in result
        assert len(result["scraped_sources"]) > 0

    def test_scraped_sources_are_source_objects(self, research_state_with_urls, mock_scraper_agent):
        """Test that scraped sources are Source objects."""
        result = scraper_node(research_state_with_urls)

        for source in result["scraped_sources"]:
            assert isinstance(source, Source)
            assert hasattr(source, "url")
            assert hasattr(source, "title")
            assert hasattr(source, "content")

    def test_tracks_failed_urls(self, research_state_with_urls, mock_scraper_agent):
        """Test that failed URLs are tracked."""
        # Add more URLs than mock will succeed
        research_state_with_urls["discovered_urls"] = [
            f"https://example{i}.com" for i in range(1, 6)
        ]

        result = scraper_node(research_state_with_urls)

        # Mock scrapes first 3, fails rest
        assert "failed_urls" in result

    def test_updates_current_step(self, research_state_with_urls, mock_scraper_agent):
        """Test that current_step is updated."""
        result = scraper_node(research_state_with_urls)

        assert result["current_step"] == "scraping_complete"

    def test_handles_no_urls(self, sample_research_state):
        """Test behavior when no URLs to scrape."""
        result = scraper_node(sample_research_state)

        assert result["scraped_sources"] == []
        assert "errors" in result
        assert result["current_step"] == "scraping_skipped_no_urls"


class TestAnalyzerNode:
    """Test analyzer node."""

    def test_analyzes_scraped_sources(self, research_state_with_sources, mock_analyzer_agent):
        """Test that sources are analyzed."""
        result = analyzer_node(research_state_with_sources)

        assert "analyzed_sources" in result
        assert len(result["analyzed_sources"]) > 0

    def test_assigns_trustworthiness_scores(self, research_state_with_sources, mock_analyzer_agent):
        """Test that trustworthiness scores are assigned."""
        result = analyzer_node(research_state_with_sources)

        for source in result["analyzed_sources"]:
            assert source.trustworthiness_score >= 0
            assert source.trustworthiness_score <= 100

    def test_updates_current_step(self, research_state_with_sources, mock_analyzer_agent):
        """Test that current_step is updated."""
        result = analyzer_node(research_state_with_sources)

        assert result["current_step"] == "analysis_complete"

    def test_handles_no_sources(self, sample_research_state):
        """Test behavior when no sources to analyze."""
        result = analyzer_node(sample_research_state)

        assert result["analyzed_sources"] == []
        assert "errors" in result
        assert result["current_step"] == "analysis_skipped_no_sources"

    def test_handles_analysis_failure(self, research_state_with_sources, monkeypatch):
        """Test error handling when analysis fails."""
        from research_assistant.agents import AnalyzerAgent

        def failing_run(self, sources, topic, **kwargs):
            raise Exception("Analysis failed")

        monkeypatch.setattr(AnalyzerAgent, "run", failing_run)

        result = analyzer_node(research_state_with_sources)

        # Should return sources with default scores
        assert "analyzed_sources" in result
        assert "errors" in result


class TestStorageNode:
    """Test storage node."""

    def test_stores_trustworthy_sources(
        self, research_state_with_analyzed_sources, mock_vector_store
    ):
        """Test that trustworthy sources are stored."""
        result = storage_node(research_state_with_analyzed_sources)

        assert "stored_sources" in result
        # State has one source with score 92.0
        assert len(result["stored_sources"]) >= 1

    def test_only_stores_sources_above_threshold(
        self, research_state_with_analyzed_sources, mock_vector_store
    ):
        """Test that only sources >= 85 are stored."""
        result = storage_node(research_state_with_analyzed_sources)

        for source in result["stored_sources"]:
            assert source.trustworthiness_score >= 85.0

    def test_tracks_rejected_sources(self, research_state_with_analyzed_sources, mock_vector_store):
        """Test that rejected sources are tracked."""
        result = storage_node(research_state_with_analyzed_sources)

        assert "rejected_sources" in result
        # State has sources with scores 78 and 45
        assert len(result["rejected_sources"]) >= 2

    def test_rejected_sources_below_threshold(
        self, research_state_with_analyzed_sources, mock_vector_store
    ):
        """Test that rejected sources are below threshold."""
        result = storage_node(research_state_with_analyzed_sources)

        for source in result["rejected_sources"]:
            assert source.trustworthiness_score < 85.0

    def test_updates_current_step(self, research_state_with_analyzed_sources, mock_vector_store):
        """Test that current_step is updated."""
        result = storage_node(research_state_with_analyzed_sources)

        assert result["current_step"] == "storage_complete"

    def test_handles_no_trustworthy_sources(self, sample_research_state, mock_vector_store):
        """Test behavior when no trustworthy sources."""
        # Add only low-scoring sources
        sample_research_state["analyzed_sources"] = [
            Source(url="url1", title="Low", content="Low", trustworthiness_score=50.0)
        ]

        result = storage_node(sample_research_state)

        assert result["stored_sources"] == []
        assert len(result["rejected_sources"]) == 1


class TestReportNode:
    """Test report generation node."""

    def test_generates_html_report(self, research_state_with_analyzed_sources, mock_reporter_agent):
        """Test that HTML report is generated."""
        result = report_node(research_state_with_analyzed_sources)

        assert "report_html" in result
        assert len(result["report_html"]) > 0
        assert "<!DOCTYPE html>" in result["report_html"]

    def test_report_includes_topic(self, research_state_with_analyzed_sources, mock_reporter_agent):
        """Test that report includes topic."""
        result = report_node(research_state_with_analyzed_sources)

        topic = research_state_with_analyzed_sources["topic"]
        assert topic in result["report_html"]

    def test_updates_current_step(self, research_state_with_analyzed_sources, mock_reporter_agent):
        """Test that current_step is updated."""
        result = report_node(research_state_with_analyzed_sources)

        assert result["current_step"] == "report_complete"

    def test_handles_no_sources(self, sample_research_state, mock_reporter_agent):
        """Test report generation when no sources."""
        result = report_node(sample_research_state)

        assert "report_html" in result
        assert len(result["report_html"]) > 0
        # Should generate error report
        assert "No sources" in result["report_html"] or "Error" in result["report_html"]


class TestConditionalRouting:
    """Test conditional routing function."""

    def test_routes_to_storage_when_trustworthy_sources_exist(
        self, research_state_with_analyzed_sources
    ):
        """Test routing to storage when trustworthy sources exist."""
        # State has one source with score 92.0
        route = should_store_sources(research_state_with_analyzed_sources)

        assert route == "storage"

    def test_routes_to_report_when_no_trustworthy_sources(self, sample_research_state):
        """Test routing to report when no trustworthy sources."""
        # Add only low-scoring sources
        sample_research_state["analyzed_sources"] = [
            Source(url="url1", title="Low", content="Low", trustworthiness_score=50.0),
            Source(url="url2", title="Low", content="Low", trustworthiness_score=60.0),
        ]

        route = should_store_sources(sample_research_state)

        assert route == "report"

    def test_routes_to_report_when_no_analyzed_sources(self, sample_research_state):
        """Test routing to report when no analyzed sources."""
        route = should_store_sources(sample_research_state)

        assert route == "report"

    def test_threshold_is_85(self, sample_research_state):
        """Test that trustworthiness threshold is 85."""
        # Source with exactly 85.0 should go to storage
        sample_research_state["analyzed_sources"] = [
            Source(url="url1", title="Boundary", content="Test", trustworthiness_score=85.0)
        ]

        route = should_store_sources(sample_research_state)

        assert route == "storage"

        # Source with 84.9 should go to report
        sample_research_state["analyzed_sources"] = [
            Source(url="url1", title="Below", content="Test", trustworthiness_score=84.9)
        ]

        route = should_store_sources(sample_research_state)

        assert route == "report"
