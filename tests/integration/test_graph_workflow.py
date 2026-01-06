"""
Integration tests for complete graph workflow.

Tests the full research assistant workflow end-to-end with mocked dependencies.
"""

from research_assistant.graph import create_research_graph
from research_assistant.graph.state import create_initial_state
from research_assistant.graph.graph import run_research
from research_assistant.models.source import Source


class TestFullWorkflow:
    """Test complete workflow from start to finish."""

    def test_full_workflow_with_trustworthy_sources(
        self,
        mock_searcher_agent,
        mock_scraper_agent,
        mock_analyzer_agent,
        mock_reporter_agent,
        mock_vector_store,
    ):
        """Test full workflow when trustworthy sources are found."""
        # Create graph
        graph = create_research_graph(verbose=False)

        # Create initial state
        initial_state = create_initial_state(topic="quantum computing", max_sources=5)

        # Execute workflow
        final_state = graph.invoke(initial_state)

        # Verify workflow completed
        assert final_state["current_step"] == "report_complete"

        # Verify all phases executed
        assert len(final_state["search_queries"]) > 0
        assert len(final_state["discovered_urls"]) > 0
        assert len(final_state["scraped_sources"]) > 0
        assert len(final_state["analyzed_sources"]) > 0

        # Verify trustworthy sources were stored
        assert len(final_state["stored_sources"]) > 0

        # Verify report was generated
        assert len(final_state["report_html"]) > 0
        assert "quantum computing" in final_state["report_html"]

    def test_full_workflow_without_trustworthy_sources(
        self,
        mock_searcher_agent,
        mock_scraper_agent,
        mock_reporter_agent,
        mock_vector_store,
        monkeypatch,
    ):
        """Test full workflow when no trustworthy sources found."""
        from research_assistant.agents import AnalyzerAgent

        # Mock analyzer to return low scores
        def mock_run(self, sources, topic, **kwargs):
            for source in sources:
                source.trustworthiness_score = 50.0  # All below threshold

            return {
                "success": True,
                "sources": sources,
                "total_analyzed": len(sources),
                "trustworthy_count": 0,
                "average_score": 50.0,
            }

        monkeypatch.setattr(AnalyzerAgent, "run", mock_run)

        # Create and execute workflow
        graph = create_research_graph(verbose=False)
        initial_state = create_initial_state("test topic", max_sources=3)

        final_state = graph.invoke(initial_state)

        # Verify workflow completed
        assert final_state["current_step"] == "report_complete"

        # Verify no sources were stored
        # Note: rejected_sources is only populated if storage node runs
        # When all sources are untrustworthy, workflow skips storage node
        assert len(final_state["stored_sources"]) == 0
        # analyzed_sources should have low-scoring sources
        assert all(s.trustworthiness_score < 85 for s in final_state["analyzed_sources"])

        # Verify report was still generated
        assert len(final_state["report_html"]) > 0

    def test_workflow_progresses_through_all_steps(
        self,
        mock_searcher_agent,
        mock_scraper_agent,
        mock_analyzer_agent,
        mock_reporter_agent,
        mock_vector_store,
    ):
        """Test that workflow progresses through expected steps."""
        graph = create_research_graph(verbose=False)
        initial_state = create_initial_state("AI ethics", max_sources=3)

        final_state = graph.invoke(initial_state)

        # Check that state accumulated results from each phase
        assert "search_queries" in final_state
        assert "discovered_urls" in final_state
        assert "scraped_sources" in final_state
        assert "analyzed_sources" in final_state
        assert "report_html" in final_state


class TestConditionalRouting:
    """Test conditional routing in workflow."""

    def test_routes_to_storage_with_high_scores(
        self,
        mock_searcher_agent,
        mock_scraper_agent,
        mock_reporter_agent,
        mock_vector_store,
        monkeypatch,
    ):
        """Test that high-scoring sources route through storage."""
        from research_assistant.agents import AnalyzerAgent

        # Mock analyzer to return high scores
        def mock_run(self, sources, topic, **kwargs):
            for source in sources:
                source.trustworthiness_score = 95.0

            return {
                "success": True,
                "sources": sources,
                "total_analyzed": len(sources),
                "trustworthy_count": len(sources),
                "average_score": 95.0,
            }

        monkeypatch.setattr(AnalyzerAgent, "run", mock_run)

        graph = create_research_graph(verbose=False)
        initial_state = create_initial_state("test", max_sources=2)

        final_state = graph.invoke(initial_state)

        # All sources should be stored
        assert len(final_state["stored_sources"]) > 0
        assert len(final_state["rejected_sources"]) == 0

    def test_routes_to_report_with_low_scores(
        self,
        mock_searcher_agent,
        mock_scraper_agent,
        mock_reporter_agent,
        mock_vector_store,
        monkeypatch,
    ):
        """Test that low-scoring sources skip storage."""
        from research_assistant.agents import AnalyzerAgent

        # Mock analyzer to return low scores
        def mock_run(self, sources, topic, **kwargs):
            for source in sources:
                source.trustworthiness_score = 40.0

            return {
                "success": True,
                "sources": sources,
                "total_analyzed": len(sources),
                "trustworthy_count": 0,
                "average_score": 40.0,
            }

        monkeypatch.setattr(AnalyzerAgent, "run", mock_run)

        graph = create_research_graph(verbose=False)
        initial_state = create_initial_state("test", max_sources=2)

        final_state = graph.invoke(initial_state)

        # No sources should be stored
        assert len(final_state["stored_sources"]) == 0
        # Verify all sources have low scores (< 85)
        assert all(s.trustworthiness_score < 85 for s in final_state["analyzed_sources"])


class TestErrorHandling:
    """Test error handling in workflow."""

    def test_workflow_continues_after_search_failure(
        self,
        mock_scraper_agent,
        mock_analyzer_agent,
        mock_reporter_agent,
        mock_vector_store,
        monkeypatch,
    ):
        """Test that workflow handles search failures."""
        from research_assistant.agents import SearcherAgent

        # Mock search to fail
        def failing_search_urls(self, queries):
            raise Exception("Search failed")

        monkeypatch.setattr(SearcherAgent, "search_urls", failing_search_urls)

        graph = create_research_graph(verbose=False)
        initial_state = create_initial_state("test", max_sources=3)

        final_state = graph.invoke(initial_state)

        # Workflow should complete despite failure
        assert "errors" in final_state
        assert len(final_state["errors"]) > 0
        assert len(final_state["discovered_urls"]) == 0

    def test_workflow_generates_error_report_on_failure(self, mock_searcher_agent, monkeypatch):
        """Test that error report is generated on failure."""
        from research_assistant.agents import ScraperAgent

        # Mock scraper to fail
        def failing_run(self, urls, **kwargs):
            raise Exception("Scraping failed")

        monkeypatch.setattr(ScraperAgent, "run", failing_run)

        graph = create_research_graph(verbose=False)
        initial_state = create_initial_state("test", max_sources=2)

        final_state = graph.invoke(initial_state)

        # Should still generate a report
        assert len(final_state["report_html"]) > 0

    def test_multiple_errors_are_tracked(self, monkeypatch):
        """Test that multiple errors are accumulated."""
        from research_assistant.agents import SearcherAgent, ScraperAgent

        # Mock both to fail
        def failing_search(self, queries):
            raise Exception("Search failed")

        def failing_scrape(self, urls, **kwargs):
            raise Exception("Scrape failed")

        monkeypatch.setattr(SearcherAgent, "search_urls", failing_search)
        monkeypatch.setattr(ScraperAgent, "run", failing_scrape)

        graph = create_research_graph(verbose=False)
        initial_state = create_initial_state("test", max_sources=2)

        final_state = graph.invoke(initial_state)

        # Multiple errors should be tracked
        assert len(final_state.get("errors", [])) >= 2


class TestRunResearchHelper:
    """Test run_research helper function."""

    def test_run_research_executes_workflow(
        self,
        mock_searcher_agent,
        mock_scraper_agent,
        mock_analyzer_agent,
        mock_reporter_agent,
        mock_vector_store,
    ):
        """Test that run_research helper works."""
        result = run_research(topic="machine learning", max_sources=5, verbose=False)

        assert result["topic"] == "machine learning"
        assert result["max_sources"] == 5
        assert len(result["report_html"]) > 0

    def test_run_research_returns_complete_state(
        self,
        mock_searcher_agent,
        mock_scraper_agent,
        mock_analyzer_agent,
        mock_reporter_agent,
        mock_vector_store,
    ):
        """Test that run_research returns complete state."""
        result = run_research("test topic", max_sources=3, verbose=False)

        # Verify all state fields present
        assert "topic" in result
        assert "search_queries" in result
        assert "discovered_urls" in result
        assert "scraped_sources" in result
        assert "analyzed_sources" in result
        assert "report_html" in result
        assert "current_step" in result


class TestStateAccumulation:
    """Test that state accumulates correctly across nodes."""

    def test_urls_accumulate_across_searches(
        self,
        mock_searcher_agent,
        mock_scraper_agent,
        mock_analyzer_agent,
        mock_reporter_agent,
        mock_vector_store,
    ):
        """Test that URLs accumulate from multiple queries."""
        graph = create_research_graph(verbose=False)
        initial_state = create_initial_state("test", max_sources=10)

        final_state = graph.invoke(initial_state)

        # Should have URLs from multiple queries
        assert len(final_state["discovered_urls"]) > 0

    def test_sources_accumulate_from_scraping(
        self,
        mock_searcher_agent,
        mock_scraper_agent,
        mock_analyzer_agent,
        mock_reporter_agent,
        mock_vector_store,
    ):
        """Test that sources accumulate from scraping."""
        graph = create_research_graph(verbose=False)
        initial_state = create_initial_state("test", max_sources=5)

        final_state = graph.invoke(initial_state)

        # Should have scraped sources
        assert len(final_state["scraped_sources"]) > 0
        assert all(isinstance(s, Source) for s in final_state["scraped_sources"])


class TestMaxSourcesLimit:
    """Test that max_sources parameter is respected."""

    def test_workflow_respects_max_sources_limit(
        self,
        mock_searcher_agent,
        mock_scraper_agent,
        mock_analyzer_agent,
        mock_reporter_agent,
        mock_vector_store,
    ):
        """Test that workflow doesn't exceed max_sources."""
        max_sources = 3

        graph = create_research_graph(verbose=False)
        initial_state = create_initial_state("test", max_sources=max_sources)

        final_state = graph.invoke(initial_state)

        # Should not exceed max_sources
        assert len(final_state["discovered_urls"]) <= max_sources
        assert len(final_state["scraped_sources"]) <= max_sources


class TestReportGeneration:
    """Test report generation in workflow."""

    def test_report_contains_all_sources(
        self,
        mock_searcher_agent,
        mock_scraper_agent,
        mock_analyzer_agent,
        mock_reporter_agent,
        mock_vector_store,
    ):
        """Test that report includes analyzed sources."""
        graph = create_research_graph(verbose=False)
        initial_state = create_initial_state("AI safety", max_sources=3)

        final_state = graph.invoke(initial_state)

        # Report should mention source count
        assert str(len(final_state["analyzed_sources"])) in final_state["report_html"]

    def test_report_is_valid_html(
        self,
        mock_searcher_agent,
        mock_scraper_agent,
        mock_analyzer_agent,
        mock_reporter_agent,
        mock_vector_store,
    ):
        """Test that generated report is valid HTML."""
        graph = create_research_graph(verbose=False)
        initial_state = create_initial_state("test", max_sources=2)

        final_state = graph.invoke(initial_state)

        html = final_state["report_html"]

        # Basic HTML structure checks
        assert "<!DOCTYPE html>" in html or "<html>" in html
        assert "<body>" in html
        assert "</body>" in html
        assert "</html>" in html
