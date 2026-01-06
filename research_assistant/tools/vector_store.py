"""
Vector store tool using ChromaDB.

Stores and retrieves research sources with semantic search capabilities.
"""

from typing import List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from ..models.source import Source
from ..config import settings


class VectorStore:
    """
    Vector store for research sources using ChromaDB.

    Provides semantic search over stored sources with automatic embedding generation.
    """

    def __init__(
        self,
        collection_name: str = "research_sources",
        persist_directory: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistence (default: from settings)
            embedding_model: Sentence transformer model (default: from settings)

        Example:
            >>> store = VectorStore()
            >>> # Or with custom settings
            >>> store = VectorStore(collection_name="my_research")
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(settings.vector_db_path)
        self.embedding_model_name = embedding_model or settings.embedding_model

        # Initialize ChromaDB client
        self.client = chromadb.Client(ChromaSettings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        ))

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Research assistant source storage"}
        )

        # Initialize embedding model
        self.embedder = SentenceTransformer(self.embedding_model_name)

    def add_source(self, source: Source) -> None:
        """
        Add a single source to the vector store.

        Args:
            source: Source object to add

        Example:
            >>> source = Source(url="https://example.com", content="...")
            >>> store.add_source(source)
        """
        self.add_sources([source])

    def add_sources(self, sources: List[Source]) -> None:
        """
        Add multiple sources to the vector store.

        Args:
            sources: List of Source objects to add

        Example:
            >>> sources = [source1, source2, source3]
            >>> store.add_sources(sources)
        """
        if not sources:
            return

        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []

        for source in sources:
            # Use content for embedding
            documents.append(source.content)

            # Store metadata
            metadatas.append({
                "url": source.url,
                "title": source.title,
                "trustworthiness_score": source.trustworthiness_score,
                "domain": source.get_domain(),
                "scraped_at": str(source.scraped_at) if source.scraped_at else None,
                **source.metadata  # Include additional metadata
            })

            # Use URL as unique ID (hash if too long)
            doc_id = self._generate_id(source.url)
            ids.append(doc_id)

        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query_similar(
        self,
        query: str,
        n_results: int = 5,
        min_score: Optional[float] = None
    ) -> List[dict]:
        """
        Query for similar sources using semantic search.

        Args:
            query: Query text
            n_results: Number of results to return
            min_score: Minimum trustworthiness score filter

        Returns:
            List of result dictionaries with metadata

        Example:
            >>> results = store.query_similar("artificial intelligence", n_results=5)
            >>> for result in results:
            ...     print(f"{result['title']}: {result['url']}")
        """
        # Build where filter if needed
        where = None
        if min_score is not None:
            where = {"trustworthiness_score": {"$gte": min_score}}

        # Query collection
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )

        # Format results
        formatted = []
        if results['metadatas']:
            for i, metadata in enumerate(results['metadatas'][0]):
                result = {
                    **metadata,
                    "document": results['documents'][0][i] if results['documents'] else "",
                    "distance": results['distances'][0][i] if results['distances'] else None
                }
                formatted.append(result)

        return formatted

    def get_by_url(self, url: str) -> Optional[dict]:
        """
        Get source by URL.

        Args:
            url: Source URL

        Returns:
            Source metadata if found, None otherwise

        Example:
            >>> result = store.get_by_url("https://example.com/article")
        """
        doc_id = self._generate_id(url)

        try:
            result = self.collection.get(ids=[doc_id])

            if result['metadatas']:
                return {
                    **result['metadatas'][0],
                    "document": result['documents'][0] if result['documents'] else ""
                }

            return None

        except Exception:
            return None

    def get_trustworthy_sources(
        self,
        threshold: float = 85.0,
        limit: Optional[int] = None
    ) -> List[dict]:
        """
        Get all sources above trustworthiness threshold.

        Args:
            threshold: Minimum trustworthiness score
            limit: Maximum results to return

        Returns:
            List of source metadata

        Example:
            >>> trustworthy = store.get_trustworthy_sources(threshold=85.0)
            >>> print(f"Found {len(trustworthy)} trustworthy sources")
        """
        where = {"trustworthiness_score": {"$gte": threshold}}

        result = self.collection.get(
            where=where,
            limit=limit
        )

        formatted = []
        if result['metadatas']:
            for i, metadata in enumerate(result['metadatas']):
                formatted.append({
                    **metadata,
                    "document": result['documents'][i] if result['documents'] else ""
                })

        return formatted

    def count(self) -> int:
        """
        Get total number of sources in store.

        Returns:
            Number of sources

        Example:
            >>> total = store.count()
            >>> print(f"Store contains {total} sources")
        """
        return self.collection.count()

    def delete_by_url(self, url: str) -> bool:
        """
        Delete source by URL.

        Args:
            url: Source URL to delete

        Returns:
            True if deleted, False if not found

        Example:
            >>> store.delete_by_url("https://example.com/old-article")
        """
        doc_id = self._generate_id(url)

        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception:
            return False

    def clear(self) -> None:
        """
        Clear all sources from the store.

        Warning: This permanently deletes all stored sources!

        Example:
            >>> store.clear()  # Delete everything
        """
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Research assistant source storage"}
        )

    def get_statistics(self) -> dict:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with store statistics

        Example:
            >>> stats = store.get_statistics()
            >>> print(f"Total sources: {stats['total_count']}")
        """
        total = self.count()

        # Get trustworthy count (>= 85)
        trustworthy = self.get_trustworthy_sources(threshold=85.0)

        return {
            "total_count": total,
            "trustworthy_count": len(trustworthy),
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model_name
        }

    def _generate_id(self, url: str) -> str:
        """
        Generate unique ID from URL.

        Args:
            url: Source URL

        Returns:
            Unique ID string

        Note:
            Uses hash for very long URLs to keep ID manageable.
        """
        import hashlib

        # For long URLs, use hash
        if len(url) > 200:
            return hashlib.md5(url.encode()).hexdigest()

        # Otherwise use URL directly (cleaned)
        return url.replace('/', '_').replace(':', '_')[:200]

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"VectorStore(collection='{self.collection_name}', "
            f"sources={self.count()})"
        )


# Convenience function
def create_vector_store(
    collection_name: str = "research_sources",
    persist_directory: Optional[str] = None
) -> VectorStore:
    """
    Create a new vector store instance.

    Args:
        collection_name: Collection name
        persist_directory: Persistence directory

    Returns:
        Configured VectorStore instance

    Example:
        >>> store = create_vector_store("my_research")
    """
    return VectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory
    )


# Global default store instance
_default_store: Optional[VectorStore] = None


def get_default_store() -> VectorStore:
    """
    Get the default vector store instance.

    Creates a singleton instance on first call.

    Returns:
        Default VectorStore instance

    Example:
        >>> from research_assistant.tools.vector_store import get_default_store
        >>> store = get_default_store()
    """
    global _default_store

    if _default_store is None:
        _default_store = VectorStore()

    return _default_store
