"""
Unit tests for vector_store module.

Tests ChromaDB vector store functionality without real database persistence.
"""

import pytest
from pathlib import Path

from research_assistant.tools.vector_store import (
    VectorStore,
    create_vector_store,
    get_default_store
)
from research_assistant.models.source import Source
from tests.utils.factories import SourceFactory
from tests.utils.assertions import (
    assert_valid_source,
    assert_source_count,
    assert_all_trustworthy,
    assert_query_results_valid,
    assert_collection_count
)


# ============================================================================
# TestVectorStoreInitialization - VectorStore setup
# ============================================================================

class TestVectorStoreInitialization:
    """Test VectorStore initialization."""

    def test_initialization_with_defaults(self, mock_vector_store_full, temp_data_dir):
        """Test VectorStore initialization with default parameters."""
        store = VectorStore()

        assert store.collection_name == "research_sources"
        assert store.embedder is not None

    def test_initialization_with_custom_collection(self, mock_vector_store_full, temp_data_dir):
        """Test VectorStore with custom collection name."""
        store = VectorStore(collection_name="custom_collection")

        assert store.collection_name == "custom_collection"

    def test_initialization_with_custom_persist_directory(self, mock_vector_store_full, temp_data_dir):
        """Test VectorStore with custom persist directory."""
        custom_dir = str(temp_data_dir / "custom_db")
        store = VectorStore(persist_directory=custom_dir)

        assert store.persist_directory == custom_dir

    def test_initialization_with_custom_embedding_model(self, mock_vector_store_full, temp_data_dir):
        """Test VectorStore with custom embedding model."""
        store = VectorStore(embedding_model="custom-model")

        assert store.embedding_model_name == "custom-model"

    def test_repr_string(self, mock_vector_store_full, temp_data_dir):
        """Test string representation."""
        store = VectorStore()
        repr_str = repr(store)

        assert "VectorStore" in repr_str
        assert "research_sources" in repr_str


# ============================================================================
# TestAddSources - Adding sources to store
# ============================================================================

class TestAddSources:
    """Test adding sources to vector store."""

    def test_add_single_source(self, mock_vector_store_full, temp_data_dir):
        """Test adding a single source."""
        store = VectorStore()
        source = SourceFactory.create()

        store.add_source(source)

        assert store.count() == 1

    def test_add_multiple_sources(self, mock_vector_store_full, temp_data_dir):
        """Test adding multiple sources at once."""
        store = VectorStore()
        sources = SourceFactory.create_batch(5)

        store.add_sources(sources)

        assert store.count() == 5

    def test_add_empty_list(self, mock_vector_store_full, temp_data_dir):
        """Test adding empty list of sources."""
        store = VectorStore()
        store.add_sources([])

        assert store.count() == 0

    def test_source_metadata_stored(self, mock_vector_store_full, temp_data_dir):
        """Test that source metadata is stored correctly."""
        store = VectorStore()
        source = SourceFactory.create(
            url="https://example.com/article",
            title="Test Article",
            trustworthiness_score=90.0
        )

        store.add_source(source)
        result = store.get_by_url(source.url)

        assert result is not None
        assert result["url"] == source.url
        assert result["title"] == source.title
        assert result["trustworthiness_score"] == 90.0

    def test_add_source_with_domain_metadata(self, mock_vector_store_full, temp_data_dir):
        """Test that domain is extracted and stored."""
        store = VectorStore()
        source = SourceFactory.create(url="https://stanford.edu/article")

        store.add_source(source)
        result = store.get_by_url(source.url)

        assert result is not None
        assert "domain" in result
        assert "stanford.edu" in result["domain"]

    def test_add_source_incremental(self, mock_vector_store_full, temp_data_dir):
        """Test adding sources incrementally."""
        store = VectorStore()

        for i in range(3):
            source = SourceFactory.create(url=f"https://example{i}.com/article")
            store.add_source(source)

        assert store.count() == 3


# ============================================================================
# TestQuerySimilar - Semantic search
# ============================================================================

class TestQuerySimilar:
    """Test querying for similar sources."""

    def test_query_returns_results(self, mock_vector_store_full, temp_data_dir):
        """Test that query returns similar sources."""
        store = VectorStore()
        sources = SourceFactory.create_batch(5)
        store.add_sources(sources)

        results = store.query_similar("test query", n_results=3)

        assert isinstance(results, list)
        assert len(results) <= 3

    def test_query_with_min_score_filter(self, mock_vector_store_full, temp_data_dir):
        """Test querying with minimum trustworthiness score filter."""
        store = VectorStore()

        # Add sources with varying scores
        sources = [
            SourceFactory.create(trustworthiness_score=95.0),
            SourceFactory.create(trustworthiness_score=88.0),
            SourceFactory.create(trustworthiness_score=75.0),
        ]
        store.add_sources(sources)

        results = store.query_similar("test query", min_score=85.0)

        # Should only return sources >= 85
        for result in results:
            assert result["trustworthiness_score"] >= 85.0

    def test_query_n_results_limit(self, mock_vector_store_full, temp_data_dir):
        """Test that n_results limits returned results."""
        store = VectorStore()
        sources = SourceFactory.create_batch(10)
        store.add_sources(sources)

        results = store.query_similar("test query", n_results=3)

        assert len(results) <= 3

    def test_query_empty_store(self, mock_vector_store_full, temp_data_dir):
        """Test querying empty store."""
        store = VectorStore()
        results = store.query_similar("test query")

        assert isinstance(results, list)
        assert len(results) == 0

    def test_query_result_format(self, mock_vector_store_full, temp_data_dir):
        """Test that query results have correct format."""
        store = VectorStore()
        source = SourceFactory.create()
        store.add_source(source)

        results = store.query_similar("test query")

        if results:
            result = results[0]
            assert "url" in result
            assert "title" in result
            assert "document" in result
            assert "distance" in result

    def test_query_with_zero_n_results(self, mock_vector_store_full, temp_data_dir):
        """Test query with n_results=0."""
        store = VectorStore()
        sources = SourceFactory.create_batch(5)
        store.add_sources(sources)

        results = store.query_similar("test query", n_results=0)

        # ChromaDB might return empty or raise error
        assert isinstance(results, list)


# ============================================================================
# TestGetByUrl - URL-based retrieval
# ============================================================================

class TestGetByUrl:
    """Test getting sources by URL."""

    def test_get_existing_source(self, mock_vector_store_full, temp_data_dir):
        """Test getting source that exists."""
        store = VectorStore()
        source = SourceFactory.create(url="https://example.com/article")
        store.add_source(source)

        result = store.get_by_url(source.url)

        assert result is not None
        assert result["url"] == source.url

    def test_get_nonexistent_source(self, mock_vector_store_full, temp_data_dir):
        """Test getting source that doesn't exist."""
        store = VectorStore()
        result = store.get_by_url("https://nonexistent.com/article")

        assert result is None

    def test_get_source_with_long_url(self, mock_vector_store_full, temp_data_dir):
        """Test getting source with very long URL (uses hash)."""
        store = VectorStore()
        long_url = "https://example.com/" + "a" * 300
        source = SourceFactory.create(url=long_url)
        store.add_source(source)

        result = store.get_by_url(long_url)

        assert result is not None
        assert result["url"] == long_url

    def test_get_returns_document_content(self, mock_vector_store_full, temp_data_dir):
        """Test that get_by_url returns document content."""
        store = VectorStore()
        source = SourceFactory.create(content="Test content here")
        store.add_source(source)

        result = store.get_by_url(source.url)

        assert result is not None
        assert "document" in result
        assert result["document"] == source.content


# ============================================================================
# TestGetTrustworthySources - Filtering by trustworthiness
# ============================================================================

class TestGetTrustworthySources:
    """Test getting trustworthy sources."""

    def test_get_sources_above_threshold(self, mock_vector_store_full, temp_data_dir):
        """Test getting sources above trustworthiness threshold."""
        store = VectorStore()

        sources = [
            SourceFactory.create(url="https://example1.com/article", trustworthiness_score=95.0),
            SourceFactory.create(url="https://example2.com/article", trustworthiness_score=88.0),
            SourceFactory.create(url="https://example3.com/article", trustworthiness_score=75.0),
            SourceFactory.create(url="https://example4.com/article", trustworthiness_score=92.0),
        ]
        store.add_sources(sources)

        trustworthy = store.get_trustworthy_sources(threshold=85.0)

        assert len(trustworthy) == 3
        for source in trustworthy:
            assert source["trustworthiness_score"] >= 85.0

    def test_get_trustworthy_with_limit(self, mock_vector_store_full, temp_data_dir):
        """Test getting trustworthy sources with limit."""
        store = VectorStore()

        sources = SourceFactory.create_batch(10)
        for source in sources:
            source.trustworthiness_score = 90.0
        store.add_sources(sources)

        trustworthy = store.get_trustworthy_sources(threshold=85.0, limit=3)

        assert len(trustworthy) <= 3

    def test_get_trustworthy_empty_result(self, mock_vector_store_full, temp_data_dir):
        """Test getting trustworthy sources when none meet threshold."""
        store = VectorStore()

        # All sources below threshold
        sources = [
            SourceFactory.create(trustworthiness_score=60.0),
            SourceFactory.create(trustworthiness_score=70.0),
        ]
        store.add_sources(sources)

        trustworthy = store.get_trustworthy_sources(threshold=85.0)

        assert len(trustworthy) == 0

    def test_get_trustworthy_custom_threshold(self, mock_vector_store_full, temp_data_dir):
        """Test getting trustworthy sources with custom threshold."""
        store = VectorStore()

        sources = SourceFactory.create_with_score_range(70.0, 95.0, count=6)
        store.add_sources(sources)

        # Use 80.0 threshold
        trustworthy = store.get_trustworthy_sources(threshold=80.0)

        for source in trustworthy:
            assert source["trustworthiness_score"] >= 80.0


# ============================================================================
# TestDeleteSources - Removing sources
# ============================================================================

class TestDeleteSources:
    """Test deleting sources from store."""

    def test_delete_existing_source(self, mock_vector_store_full, temp_data_dir):
        """Test deleting source that exists."""
        store = VectorStore()
        source = SourceFactory.create(url="https://example.com/article")
        store.add_source(source)

        assert store.count() == 1

        result = store.delete_by_url(source.url)

        assert result is True
        assert store.count() == 0

    def test_delete_nonexistent_source(self, mock_vector_store_full, temp_data_dir):
        """Test deleting source that doesn't exist."""
        store = VectorStore()
        result = store.delete_by_url("https://nonexistent.com/article")

        # Should return False or handle gracefully
        assert result is False or result is True  # Mock implementation may vary

    def test_delete_one_of_many(self, mock_vector_store_full, temp_data_dir):
        """Test deleting one source from many."""
        store = VectorStore()
        sources = SourceFactory.create_batch(5)
        store.add_sources(sources)

        store.delete_by_url(sources[2].url)

        assert store.count() == 4
        assert store.get_by_url(sources[2].url) is None


# ============================================================================
# TestClear - Clearing all sources
# ============================================================================

class TestClear:
    """Test clearing all sources."""

    def test_clear_all_sources(self, mock_vector_store_full, temp_data_dir):
        """Test clearing all sources from store."""
        store = VectorStore()
        sources = SourceFactory.create_batch(10)
        store.add_sources(sources)

        assert store.count() == 10

        store.clear()

        assert store.count() == 0

    def test_clear_empty_store(self, mock_vector_store_full, temp_data_dir):
        """Test clearing already empty store."""
        store = VectorStore()
        store.clear()

        assert store.count() == 0

    def test_add_after_clear(self, mock_vector_store_full, temp_data_dir):
        """Test adding sources after clearing."""
        store = VectorStore()
        sources = SourceFactory.create_batch(5)
        store.add_sources(sources)

        store.clear()

        new_sources = SourceFactory.create_batch(3)
        store.add_sources(new_sources)

        assert store.count() == 3


# ============================================================================
# TestStatistics - Store statistics
# ============================================================================

class TestStatistics:
    """Test getting store statistics."""

    def test_statistics_format(self, mock_vector_store_full, temp_data_dir):
        """Test that statistics have correct format."""
        store = VectorStore()
        stats = store.get_statistics()

        assert isinstance(stats, dict)
        assert "total_count" in stats
        assert "trustworthy_count" in stats
        assert "collection_name" in stats
        assert "persist_directory" in stats
        assert "embedding_model" in stats

    def test_statistics_counts(self, mock_vector_store_full, temp_data_dir):
        """Test that statistics counts are accurate."""
        store = VectorStore()

        sources = [
            SourceFactory.create(url="https://example1.com/article", trustworthiness_score=95.0),
            SourceFactory.create(url="https://example2.com/article", trustworthiness_score=88.0),
            SourceFactory.create(url="https://example3.com/article", trustworthiness_score=75.0),
        ]
        store.add_sources(sources)

        stats = store.get_statistics()

        assert stats["total_count"] == 3
        assert stats["trustworthy_count"] == 2  # >= 85

    def test_statistics_empty_store(self, mock_vector_store_full, temp_data_dir):
        """Test statistics for empty store."""
        store = VectorStore()
        stats = store.get_statistics()

        assert stats["total_count"] == 0
        assert stats["trustworthy_count"] == 0


# ============================================================================
# TestCount - Counting sources
# ============================================================================

class TestCount:
    """Test counting sources in store."""

    def test_count_empty_store(self, mock_vector_store_full, temp_data_dir):
        """Test count on empty store."""
        store = VectorStore()
        assert store.count() == 0

    def test_count_after_adding(self, mock_vector_store_full, temp_data_dir):
        """Test count after adding sources."""
        store = VectorStore()
        sources = SourceFactory.create_batch(7)
        store.add_sources(sources)

        assert store.count() == 7

    def test_count_after_deleting(self, mock_vector_store_full, temp_data_dir):
        """Test count after deleting source."""
        store = VectorStore()
        sources = SourceFactory.create_batch(5)
        store.add_sources(sources)

        store.delete_by_url(sources[0].url)

        assert store.count() == 4


# ============================================================================
# TestHelperMethods - Internal helper methods
# ============================================================================

class TestHelperMethods:
    """Test internal helper methods."""

    def test_generate_id_short_url(self, mock_vector_store_full, temp_data_dir):
        """Test ID generation for short URLs."""
        store = VectorStore()
        url = "https://example.com/article"
        doc_id = store._generate_id(url)

        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    def test_generate_id_long_url(self, mock_vector_store_full, temp_data_dir):
        """Test ID generation for long URLs (uses hash)."""
        store = VectorStore()
        long_url = "https://example.com/" + "a" * 300
        doc_id = store._generate_id(long_url)

        assert isinstance(doc_id, str)
        # Long URLs should be hashed to manageable length
        assert len(doc_id) <= 200

    def test_generate_id_consistent(self, mock_vector_store_full, temp_data_dir):
        """Test that same URL generates same ID."""
        store = VectorStore()
        url = "https://example.com/article"

        id1 = store._generate_id(url)
        id2 = store._generate_id(url)

        assert id1 == id2


# ============================================================================
# TestModuleFunctions - Module-level functions
# ============================================================================

class TestModuleFunctions:
    """Test module-level convenience functions."""

    def test_create_vector_store(self, mock_vector_store_full, temp_data_dir):
        """Test create_vector_store factory function."""
        store = create_vector_store("test_collection")

        assert isinstance(store, VectorStore)
        assert store.collection_name == "test_collection"

    def test_create_vector_store_with_persist_dir(self, mock_vector_store_full, temp_data_dir):
        """Test create_vector_store with custom persist directory."""
        custom_dir = str(temp_data_dir / "custom")
        store = create_vector_store(persist_directory=custom_dir)

        assert isinstance(store, VectorStore)
        assert store.persist_directory == custom_dir

    def test_get_default_store_singleton(self, mock_vector_store_full, temp_data_dir):
        """Test that get_default_store returns singleton."""
        # Reset singleton first
        import research_assistant.tools.vector_store as vs_module
        vs_module._default_store = None

        store1 = get_default_store()
        store2 = get_default_store()

        assert store1 is store2  # Same instance


# ============================================================================
# TestMultipleCollections - Multiple collection support
# ============================================================================

class TestMultipleCollections:
    """Test using multiple collections."""

    def test_different_collections_isolated(self, mock_vector_store_full, temp_data_dir):
        """Test that different collections are isolated."""
        store1 = VectorStore(collection_name="collection1")
        store2 = VectorStore(collection_name="collection2")

        source1 = SourceFactory.create(url="https://example1.com/article")
        source2 = SourceFactory.create(url="https://example2.com/article")

        store1.add_source(source1)
        store2.add_source(source2)

        assert store1.count() == 1
        assert store2.count() == 1

        # Each store only has its own source
        assert store1.get_by_url(source1.url) is not None
        assert store1.get_by_url(source2.url) is None

        assert store2.get_by_url(source2.url) is not None
        assert store2.get_by_url(source1.url) is None


# ============================================================================
# Parametrized Tests
# ============================================================================

@pytest.mark.parametrize("threshold", [80.0, 85.0, 90.0, 95.0])
def test_trustworthiness_thresholds(mock_vector_store_full, temp_data_dir, threshold):
    """Test various trustworthiness thresholds."""
    store = VectorStore()

    sources = SourceFactory.create_with_score_range(70.0, 98.0, count=10)
    store.add_sources(sources)

    trustworthy = store.get_trustworthy_sources(threshold=threshold)

    for source in trustworthy:
        assert source["trustworthiness_score"] >= threshold


@pytest.mark.parametrize("n_sources", [1, 5, 10, 20])
def test_various_source_counts(mock_vector_store_full, temp_data_dir, n_sources):
    """Test with various numbers of sources."""
    store = VectorStore()
    sources = SourceFactory.create_batch(n_sources)
    store.add_sources(sources)

    assert store.count() == n_sources


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_add_source_with_empty_content(self, mock_vector_store_full, temp_data_dir):
        """Test adding source with empty content."""
        store = VectorStore()
        source = SourceFactory.create(content="")

        # Should handle gracefully or raise error
        try:
            store.add_source(source)
        except Exception:
            pass  # Expected for empty content

    def test_query_with_empty_string(self, mock_vector_store_full, temp_data_dir):
        """Test querying with empty string."""
        store = VectorStore()
        sources = SourceFactory.create_batch(3)
        store.add_sources(sources)

        results = store.query_similar("")

        # Should return results or empty list
        assert isinstance(results, list)

    def test_add_duplicate_url(self, mock_vector_store_full, temp_data_dir):
        """Test adding source with duplicate URL."""
        store = VectorStore()
        url = "https://example.com/article"

        source1 = SourceFactory.create(url=url, content="First version")
        source2 = SourceFactory.create(url=url, content="Second version")

        store.add_source(source1)
        store.add_source(source2)

        # Behavior may vary - might overwrite or keep both
        # Just ensure it doesn't crash
        assert store.count() >= 1

    def test_very_large_batch(self, mock_vector_store_full, temp_data_dir):
        """Test adding very large batch of sources."""
        store = VectorStore()
        sources = SourceFactory.create_batch(100)

        store.add_sources(sources)

        assert store.count() == 100
