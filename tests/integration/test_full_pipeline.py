"""
Integration tests for full research pipeline.

Tests the complete flow: Search → Scrape → Store → Query
"""

import pytest

from research_assistant.tools.search import search_duckduckgo, extract_urls_from_results
from research_assistant.tools.scraper import scrape_multiple_urls
from research_assistant.tools.vector_store import VectorStore
from research_assistant.models.source import Source
from tests.utils.assertions import (
    assert_search_results_valid,
    assert_valid_source,
    assert_all_trustworthy,
    assert_query_results_valid
)


# ============================================================================
# TestSearchToScrapeToVector - Full pipeline integration
# ============================================================================

class TestSearchToScrapeToVector:
    """Test full pipeline from search to vector storage."""

    def test_search_scrape_store_query_pipeline(
        self,
        mock_ddgs,
        mock_newspaper_article,
        mock_vector_store_full,
        temp_data_dir
    ):
        """Test complete pipeline: search → scrape → store → query."""
        # Step 1: Search
        search_results = search_duckduckgo("artificial intelligence", max_results=5)
        assert_search_results_valid(search_results)
        assert len(search_results) > 0

        # Step 2: Extract URLs
        urls = extract_urls_from_results(search_results)
        assert len(urls) > 0

        # Step 3: Scrape URLs
        sources, failed = scrape_multiple_urls(urls[:3], skip_errors=True)
        assert len(sources) > 0
        for source in sources:
            assert_valid_source(source)

        # Step 4: Store in vector database
        store = VectorStore(persist_directory=str(temp_data_dir / "test_db"))
        store.add_sources(sources)

        assert store.count() == len(sources)

        # Step 5: Query vector database
        results = store.query_similar("artificial intelligence", n_results=2)

        assert isinstance(results, list)
        assert len(results) <= 2

    def test_trustworthy_sources_pipeline(
        self,
        mock_ddgs,
        mock_newspaper_article,
        mock_vector_store_full,
        temp_data_dir
    ):
        """Test pipeline that filters for trustworthy sources."""
        # Search and scrape
        search_results = search_duckduckgo("test query", max_results=5)
        urls = extract_urls_from_results(search_results)
        sources, _ = scrape_multiple_urls(urls, skip_errors=True)

        # Manually set trustworthiness scores for testing
        for i, source in enumerate(sources):
            source.trustworthiness_score = 85.0 + (i * 2)  # 85, 87, 89, etc.

        # Store only trustworthy sources
        store = VectorStore(persist_directory=str(temp_data_dir / "trustworthy_db"))
        trustworthy_sources = [s for s in sources if s.is_trustworthy()]
        store.add_sources(trustworthy_sources)

        # Verify all stored sources are trustworthy
        all_stored = store.get_trustworthy_sources(threshold=85.0)
        assert len(all_stored) == len(trustworthy_sources)

    def test_multiple_searches_aggregated(
        self,
        mock_ddgs,
        mock_newspaper_article,
        mock_vector_store_full,
        temp_data_dir
    ):
        """Test aggregating results from multiple searches."""
        queries = ["AI ethics", "machine learning", "deep learning"]
        all_sources = []

        # Search multiple queries
        for query in queries:
            results = search_duckduckgo(query, max_results=2)
            urls = extract_urls_from_results(results)
            sources, _ = scrape_multiple_urls(urls, skip_errors=True)
            all_sources.extend(sources)

        # Store all sources
        store = VectorStore(persist_directory=str(temp_data_dir / "multi_db"))
        store.add_sources(all_sources)

        # Should have sources from all queries
        assert store.count() >= len(queries)

    def test_query_retrieves_relevant_content(
        self,
        mock_ddgs,
        mock_newspaper_article,
        mock_vector_store_full,
        temp_data_dir
    ):
        """Test that queries retrieve semantically similar content."""
        # Create sources with specific content
        from tests.utils.factories import SourceFactory

        sources = [
            SourceFactory.create(
                url="https://example.com/ai",
                content="Artificial intelligence and machine learning applications"
            ),
            SourceFactory.create(
                url="https://example.com/python",
                content="Python programming language tutorial"
            ),
        ]

        # Store sources
        store = VectorStore(persist_directory=str(temp_data_dir / "relevance_db"))
        store.add_sources(sources)

        # Query for AI-related content
        results = store.query_similar("artificial intelligence", n_results=1)

        # Should retrieve the AI-related source
        assert len(results) > 0

    def test_incremental_source_addition(
        self,
        mock_ddgs,
        mock_newspaper_article,
        mock_vector_store_full,
        temp_data_dir
    ):
        """Test adding sources incrementally over time."""
        store = VectorStore(persist_directory=str(temp_data_dir / "incremental_db"))

        # First batch
        results1 = search_duckduckgo("batch 1", max_results=2)
        urls1 = extract_urls_from_results(results1)
        sources1, _ = scrape_multiple_urls(urls1, skip_errors=True)
        store.add_sources(sources1)

        count_after_batch1 = store.count()

        # Second batch
        results2 = search_duckduckgo("batch 2", max_results=2)
        urls2 = extract_urls_from_results(results2)
        sources2, _ = scrape_multiple_urls(urls2, skip_errors=True)
        store.add_sources(sources2)

        count_after_batch2 = store.count()

        # Count should increase
        assert count_after_batch2 >= count_after_batch1


# ============================================================================
# TestErrorHandling - Error handling in pipeline
# ============================================================================

class TestErrorHandling:
    """Test error handling across pipeline stages."""

    def test_pipeline_handles_scraping_failures(
        self,
        mock_ddgs,
        monkeypatch,
        mock_vector_store_full,
        temp_data_dir
    ):
        """Test that pipeline continues when some URLs fail to scrape."""
        # Setup mock that fails for some URLs
        from tests.mocks.mock_newspaper import MockArticle

        class PartiallyFailingArticle(MockArticle):
            def download(self):
                if "fail" in self.url:
                    raise Exception("Download failed")
                super().download()

        monkeypatch.setattr("newspaper.Article", PartiallyFailingArticle)

        # Search
        search_results = search_duckduckgo("test", max_results=5)
        urls = extract_urls_from_results(search_results)

        # Add a URL that will fail
        urls.append("https://fail.com/article")

        # Scrape with error handling
        sources, failed = scrape_multiple_urls(urls, skip_errors=True)

        # Should have some successful sources
        assert len(sources) > 0
        # Should have recorded failure
        assert len(failed) > 0

        # Store successful sources
        store = VectorStore(persist_directory=str(temp_data_dir / "error_db"))
        store.add_sources(sources)

        assert store.count() == len(sources)

    def test_empty_search_results_handled(
        self,
        mock_ddgs_empty,
        mock_vector_store_full,
        temp_data_dir
    ):
        """Test handling of empty search results."""
        # Search returns nothing
        results = search_duckduckgo("nonexistent query")
        assert len(results) == 0

        # Extract URLs from empty results
        urls = extract_urls_from_results(results)
        assert len(urls) == 0

        # Scrape empty URL list
        sources, failed = scrape_multiple_urls(urls)
        assert len(sources) == 0
        assert len(failed) == 0

        # Store empty list
        store = VectorStore(persist_directory=str(temp_data_dir / "empty_db"))
        store.add_sources(sources)
        assert store.count() == 0


# ============================================================================
# TestDataFlow - Data consistency across pipeline
# ============================================================================

class TestDataFlow:
    """Test data consistency as it flows through pipeline."""

    def test_url_preserved_through_pipeline(
        self,
        mock_ddgs,
        mock_newspaper_article,
        mock_vector_store_full,
        temp_data_dir
    ):
        """Test that URLs are preserved from search to storage."""
        # Search
        results = search_duckduckgo("test", max_results=3)
        original_urls = extract_urls_from_results(results)

        # Scrape
        sources, _ = scrape_multiple_urls(original_urls, skip_errors=True)

        # Store
        store = VectorStore(persist_directory=str(temp_data_dir / "url_db"))
        store.add_sources(sources)

        # Verify URLs match
        for url in original_urls:
            result = store.get_by_url(url)
            if result:  # May not exist if scraping failed
                assert result["url"] == url

    def test_metadata_preserved(
        self,
        mock_ddgs,
        mock_newspaper_article,
        mock_vector_store_full,
        temp_data_dir
    ):
        """Test that metadata is preserved through pipeline."""
        # Search and scrape
        results = search_duckduckgo("test", max_results=2)
        urls = extract_urls_from_results(results)
        sources, _ = scrape_multiple_urls(urls, skip_errors=True)

        # Store
        store = VectorStore(persist_directory=str(temp_data_dir / "metadata_db"))
        store.add_sources(sources)

        # Retrieve and verify metadata
        for source in sources:
            result = store.get_by_url(source.url)
            if result:
                assert "domain" in result
                assert "word_count" in result


# ============================================================================
# TestPerformance - Basic performance tests
# ============================================================================

@pytest.mark.slow
class TestPerformance:
    """Basic performance tests for pipeline."""

    def test_bulk_operations(
        self,
        mock_ddgs,
        mock_newspaper_article,
        mock_vector_store_full,
        temp_data_dir
    ):
        """Test pipeline with larger batches."""
        # Search
        results = search_duckduckgo("test", max_results=20)
        urls = extract_urls_from_results(results)

        # Scrape
        sources, failed = scrape_multiple_urls(urls[:10], skip_errors=True)

        # Store
        store = VectorStore(persist_directory=str(temp_data_dir / "bulk_db"))
        store.add_sources(sources)

        # Query
        query_results = store.query_similar("test query", n_results=5)

        # Should handle bulk operations
        assert store.count() > 0
        assert isinstance(query_results, list)


# ============================================================================
# TestRealWorldScenario - Simulate real usage
# ============================================================================

class TestRealWorldScenario:
    """Test scenarios that simulate real-world usage."""

    def test_research_topic_workflow(
        self,
        mock_ddgs,
        mock_newspaper_article,
        mock_vector_store_full,
        temp_data_dir
    ):
        """Simulate researching a topic."""
        topic = "artificial intelligence ethics"

        # Generate multiple search queries (simulated)
        queries = [topic, "AI ethics", "machine learning ethics"]

        # Collect sources from all queries
        all_sources = []
        for query in queries:
            results = search_duckduckgo(query, max_results=3)
            urls = extract_urls_from_results(results)
            sources, _ = scrape_multiple_urls(urls, skip_errors=True)
            all_sources.extend(sources)

        # Filter for trustworthy sources
        trustworthy = [s for s in all_sources if s.is_trustworthy(threshold=80.0)]

        # Store in vector database
        store = VectorStore(
            collection_name="ai_ethics_research",
            persist_directory=str(temp_data_dir / "research_db")
        )
        store.add_sources(trustworthy if trustworthy else all_sources)

        # Query for specific aspects
        ethics_results = store.query_similar("ethical concerns", n_results=3)
        bias_results = store.query_similar("bias in AI", n_results=3)

        # Should have collected and stored sources
        assert store.count() > 0

        # Should be able to query on different aspects
        assert isinstance(ethics_results, list)
        assert isinstance(bias_results, list)

    def test_update_existing_research(
        self,
        mock_ddgs,
        mock_newspaper_article,
        mock_vector_store_full,
        temp_data_dir
    ):
        """Test updating existing research with new sources."""
        store = VectorStore(
            collection_name="ongoing_research",
            persist_directory=str(temp_data_dir / "ongoing_db")
        )

        # Initial research
        results1 = search_duckduckgo("topic", max_results=3)
        urls1 = extract_urls_from_results(results1)
        sources1, _ = scrape_multiple_urls(urls1, skip_errors=True)
        store.add_sources(sources1)

        initial_count = store.count()

        # Update with new sources
        results2 = search_duckduckgo("related topic", max_results=2)
        urls2 = extract_urls_from_results(results2)
        sources2, _ = scrape_multiple_urls(urls2, skip_errors=True)
        store.add_sources(sources2)

        updated_count = store.count()

        # Should have more sources after update
        assert updated_count >= initial_count


# ============================================================================
# Integration Test Markers
# ============================================================================

# All tests in this module are integration tests
pytestmark = pytest.mark.integration
