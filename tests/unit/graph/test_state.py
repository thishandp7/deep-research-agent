"""
Unit tests for ResearchState schema.

Tests state initialization, field types, and helper functions.
"""

from research_assistant.graph.state import create_initial_state
from research_assistant.models.source import Source


class TestCreateInitialState:
    """Test initial state creation."""

    def test_creates_state_with_topic(self):
        """Test that initial state includes topic."""
        state = create_initial_state("quantum computing")
        assert state["topic"] == "quantum computing"

    def test_creates_state_with_max_sources(self):
        """Test that initial state includes max_sources."""
        state = create_initial_state("AI", max_sources=15)
        assert state["max_sources"] == 15

    def test_default_max_sources_is_10(self):
        """Test default max_sources value."""
        state = create_initial_state("topic")
        assert state["max_sources"] == 10

    def test_initializes_empty_lists(self):
        """Test that lists are initialized as empty."""
        state = create_initial_state("topic")

        assert state["search_queries"] == []
        assert state["discovered_urls"] == []
        assert state["scraped_sources"] == []
        assert state["failed_urls"] == []
        assert state["analyzed_sources"] == []
        assert state["stored_sources"] == []
        assert state["rejected_sources"] == []
        assert state["errors"] == []

    def test_initializes_empty_report(self):
        """Test that report_html is empty string."""
        state = create_initial_state("topic")
        assert state["report_html"] == ""

    def test_initializes_current_step(self):
        """Test that current_step is set to initialized."""
        state = create_initial_state("topic")
        assert state["current_step"] == "initialized"


class TestStateFieldTypes:
    """Test that state fields have correct types."""

    def test_topic_is_string(self):
        """Test topic field type."""
        state = create_initial_state("test topic")
        assert isinstance(state["topic"], str)

    def test_max_sources_is_int(self):
        """Test max_sources field type."""
        state = create_initial_state("topic", max_sources=5)
        assert isinstance(state["max_sources"], int)

    def test_search_queries_is_list(self):
        """Test search_queries field type."""
        state = create_initial_state("topic")
        assert isinstance(state["search_queries"], list)

    def test_discovered_urls_is_list(self):
        """Test discovered_urls field type."""
        state = create_initial_state("topic")
        assert isinstance(state["discovered_urls"], list)

    def test_scraped_sources_is_list(self):
        """Test scraped_sources field type."""
        state = create_initial_state("topic")
        assert isinstance(state["scraped_sources"], list)

    def test_analyzed_sources_is_list(self):
        """Test analyzed_sources field type."""
        state = create_initial_state("topic")
        assert isinstance(state["analyzed_sources"], list)

    def test_errors_is_list(self):
        """Test errors field type."""
        state = create_initial_state("topic")
        assert isinstance(state["errors"], list)

    def test_report_html_is_string(self):
        """Test report_html field type."""
        state = create_initial_state("topic")
        assert isinstance(state["report_html"], str)

    def test_current_step_is_string(self):
        """Test current_step field type."""
        state = create_initial_state("topic")
        assert isinstance(state["current_step"], str)


class TestStateManipulation:
    """Test state manipulation operations."""

    def test_can_add_search_queries(self):
        """Test adding search queries to state."""
        state = create_initial_state("topic")
        state["search_queries"] = ["query1", "query2"]

        assert len(state["search_queries"]) == 2
        assert "query1" in state["search_queries"]

    def test_can_add_urls(self):
        """Test adding URLs to state."""
        state = create_initial_state("topic")
        state["discovered_urls"] = ["https://example.com/1", "https://example.com/2"]

        assert len(state["discovered_urls"]) == 2

    def test_can_add_sources(self):
        """Test adding Source objects to state."""
        state = create_initial_state("topic")

        source = Source(
            url="https://example.com/article", title="Test Article", content="Test content"
        )

        state["scraped_sources"] = [source]

        assert len(state["scraped_sources"]) == 1
        assert state["scraped_sources"][0].title == "Test Article"

    def test_can_update_current_step(self):
        """Test updating current_step."""
        state = create_initial_state("topic")
        state["current_step"] = "search_complete"

        assert state["current_step"] == "search_complete"

    def test_can_add_errors(self):
        """Test adding errors to state."""
        state = create_initial_state("topic")
        state["errors"] = ["Error 1", "Error 2"]

        assert len(state["errors"]) == 2
        assert "Error 1" in state["errors"]

    def test_can_set_report_html(self):
        """Test setting report HTML."""
        state = create_initial_state("topic")
        state["report_html"] = "<html><body>Report</body></html>"

        assert "<body>Report</body>" in state["report_html"]


class TestStateProgression:
    """Test state through workflow progression."""

    def test_state_progresses_through_search_phase(self):
        """Test state after search phase."""
        state = create_initial_state("quantum computing")

        # Query generation
        state["search_queries"] = ["quantum basics", "quantum algorithms"]
        state["current_step"] = "query_generation_complete"

        # Search
        state["discovered_urls"] = [
            "https://example1.com",
            "https://example2.com",
            "https://example3.com",
        ]
        state["current_step"] = "search_complete"

        assert len(state["search_queries"]) == 2
        assert len(state["discovered_urls"]) == 3
        assert state["current_step"] == "search_complete"

    def test_state_progresses_through_scraping_phase(self):
        """Test state after scraping phase."""
        state = create_initial_state("topic")

        state["discovered_urls"] = ["url1", "url2", "url3"]

        source1 = Source(url="url1", title="Article 1", content="Content 1")
        source2 = Source(url="url2", title="Article 2", content="Content 2")

        state["scraped_sources"] = [source1, source2]
        state["failed_urls"] = ["url3"]
        state["current_step"] = "scraping_complete"

        assert len(state["scraped_sources"]) == 2
        assert len(state["failed_urls"]) == 1

    def test_state_progresses_through_analysis_phase(self):
        """Test state after analysis phase."""
        state = create_initial_state("topic")

        source1 = Source(
            url="url1", title="High Quality", content="Good", trustworthiness_score=90.0
        )
        source2 = Source(url="url2", title="Low Quality", content="Bad", trustworthiness_score=50.0)

        state["analyzed_sources"] = [source1, source2]
        state["current_step"] = "analysis_complete"

        assert len(state["analyzed_sources"]) == 2
        assert state["analyzed_sources"][0].trustworthiness_score == 90.0

    def test_state_separates_stored_and_rejected(self):
        """Test state separates trustworthy from rejected sources."""
        state = create_initial_state("topic")

        trustworthy = Source(url="url1", title="Good", content="Good", trustworthiness_score=90.0)
        rejected = Source(url="url2", title="Bad", content="Bad", trustworthiness_score=50.0)

        state["stored_sources"] = [trustworthy]
        state["rejected_sources"] = [rejected]

        assert len(state["stored_sources"]) == 1
        assert len(state["rejected_sources"]) == 1
        assert state["stored_sources"][0].trustworthiness_score >= 85.0

    def test_state_contains_final_report(self):
        """Test state contains final report."""
        state = create_initial_state("topic")

        state["report_html"] = "<html><body><h1>Final Report</h1></body></html>"
        state["current_step"] = "report_complete"

        assert len(state["report_html"]) > 0
        assert "<h1>Final Report</h1>" in state["report_html"]


class TestStateErrorTracking:
    """Test error tracking in state."""

    def test_can_accumulate_errors(self):
        """Test that errors can be accumulated."""
        state = create_initial_state("topic")

        state["errors"] = ["Error during search"]
        # Simulate accumulation
        state["errors"] = state["errors"] + ["Error during scraping"]

        assert len(state["errors"]) == 2

    def test_workflow_continues_with_errors(self):
        """Test that workflow can continue despite errors."""
        state = create_initial_state("topic")

        state["errors"] = ["Search failed"]
        state["current_step"] = "search_failed"

        # Workflow can still have other data
        state["discovered_urls"] = []
        state["report_html"] = "<html>Error report</html>"

        assert len(state["errors"]) > 0
        assert state["report_html"] != ""
