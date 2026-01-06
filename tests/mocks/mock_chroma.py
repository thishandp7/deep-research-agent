"""
Mock implementation of ChromaDB.

Provides in-memory document storage without persistence.
"""

from typing import List, Optional, Dict, Any
import hashlib


# ============================================================================
# Mock Collection Class
# ============================================================================

class MockCollection:
    """
    Mock ChromaDB Collection.

    Provides in-memory document storage with basic querying.
    """

    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize mock collection.

        Args:
            name: Collection name
            metadata: Collection metadata
        """
        self.name = name
        self.metadata = metadata or {}

        # In-memory storage
        self._documents: Dict[str, str] = {}  # id -> document
        self._metadatas: Dict[str, dict] = {}  # id -> metadata
        self._embeddings: Dict[str, List[float]] = {}  # id -> embedding (not used in mock)

    def add(
        self,
        documents: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None
    ):
        """
        Add documents to collection.

        Args:
            documents: List of document texts
            metadatas: List of metadata dicts
            ids: List of document IDs
            embeddings: List of embedding vectors (ignored in mock)

        Raises:
            ValueError: If input lists have mismatched lengths
        """
        if not documents:
            return

        # Validate inputs
        n_docs = len(documents)

        if metadatas and len(metadatas) != n_docs:
            raise ValueError("metadatas must match documents length")

        if ids and len(ids) != n_docs:
            raise ValueError("ids must match documents length")

        # Generate IDs if not provided
        if ids is None:
            ids = [self._generate_id(doc, i) for i, doc in enumerate(documents)]

        # Use empty metadata if not provided
        if metadatas is None:
            metadatas = [{} for _ in documents]

        # Store documents
        for doc_id, document, metadata in zip(ids, documents, metadatas):
            self._documents[doc_id] = document
            self._metadatas[doc_id] = metadata

            # Generate dummy embedding
            if embeddings:
                self._embeddings[doc_id] = embeddings[ids.index(doc_id)]
            else:
                self._embeddings[doc_id] = self._generate_dummy_embedding(document)

    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 10,
        where: Optional[dict] = None,
        include: List[str] = ["documents", "metadatas", "distances"]
    ) -> dict:
        """
        Query collection for similar documents.

        Args:
            query_texts: List of query texts
            query_embeddings: List of query embeddings (ignored in mock)
            n_results: Maximum results to return
            where: Metadata filter conditions
            include: What to include in results

        Returns:
            Query results dictionary

        Note:
            This mock uses simple text matching instead of semantic search.
        """
        if not query_texts:
            query_texts = [""]

        # Get all documents
        all_ids = list(self._documents.keys())

        # Apply where filter if provided
        if where:
            all_ids = self._filter_by_where(all_ids, where)

        # For each query, find matching documents
        results = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": []
        }

        for query_text in query_texts:
            # Simple matching: score by text overlap
            scored_docs = []

            for doc_id in all_ids:
                document = self._documents[doc_id]
                score = self._calculate_similarity(query_text, document)
                scored_docs.append((doc_id, score))

            # Sort by score (descending)
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Take top n_results
            top_docs = scored_docs[:n_results]

            # Build results
            query_ids = [doc_id for doc_id, _ in top_docs]
            query_distances = [1.0 - score for _, score in top_docs]  # Convert to distance
            query_documents = [self._documents[doc_id] for doc_id in query_ids]
            query_metadatas = [self._metadatas[doc_id] for doc_id in query_ids]

            results["ids"].append(query_ids)
            results["documents"].append(query_documents if "documents" in include else [])
            results["metadatas"].append(query_metadatas if "metadatas" in include else [])
            results["distances"].append(query_distances if "distances" in include else [])

        return results

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[dict] = None,
        limit: Optional[int] = None,
        include: List[str] = ["documents", "metadatas"]
    ) -> dict:
        """
        Get documents by ID or filter.

        Args:
            ids: List of document IDs
            where: Metadata filter conditions
            limit: Maximum results
            include: What to include in results

        Returns:
            Documents dictionary
        """
        # Get IDs to retrieve
        if ids:
            result_ids = [doc_id for doc_id in ids if doc_id in self._documents]
        else:
            result_ids = list(self._documents.keys())

        # Apply where filter
        if where:
            result_ids = self._filter_by_where(result_ids, where)

        # Apply limit
        if limit:
            result_ids = result_ids[:limit]

        # Build results
        results = {
            "ids": result_ids,
            "documents": [self._documents[doc_id] for doc_id in result_ids] if "documents" in include else [],
            "metadatas": [self._metadatas[doc_id] for doc_id in result_ids] if "metadatas" in include else [],
        }

        return results

    def delete(self, ids: Optional[List[str]] = None, where: Optional[dict] = None):
        """
        Delete documents.

        Args:
            ids: List of document IDs to delete
            where: Metadata filter for deletion
        """
        # Determine which IDs to delete
        if ids:
            ids_to_delete = [doc_id for doc_id in ids if doc_id in self._documents]
        elif where:
            ids_to_delete = self._filter_by_where(list(self._documents.keys()), where)
        else:
            # Delete all if no filter
            ids_to_delete = list(self._documents.keys())

        # Delete documents
        for doc_id in ids_to_delete:
            self._documents.pop(doc_id, None)
            self._metadatas.pop(doc_id, None)
            self._embeddings.pop(doc_id, None)

    def count(self) -> int:
        """
        Get number of documents in collection.

        Returns:
            Document count
        """
        return len(self._documents)

    def _filter_by_where(self, ids: List[str], where: dict) -> List[str]:
        """
        Filter IDs by metadata conditions.

        Args:
            ids: List of document IDs
            where: Filter conditions

        Returns:
            Filtered list of IDs

        Supports:
            - $gte: Greater than or equal
            - $lte: Less than or equal
            - $eq: Equal
            - $ne: Not equal
            - $gt: Greater than
            - $lt: Less than
        """
        filtered = []

        for doc_id in ids:
            metadata = self._metadatas.get(doc_id, {})

            if self._matches_where(metadata, where):
                filtered.append(doc_id)

        return filtered

    def _matches_where(self, metadata: dict, where: dict) -> bool:
        """
        Check if metadata matches where conditions.

        Args:
            metadata: Document metadata
            where: Filter conditions

        Returns:
            True if matches
        """
        for key, condition in where.items():
            if isinstance(condition, dict):
                # Operator-based condition
                value = metadata.get(key)

                if value is None:
                    return False

                for op, target in condition.items():
                    if op == "$gte" and not (value >= target):
                        return False
                    elif op == "$lte" and not (value <= target):
                        return False
                    elif op == "$gt" and not (value > target):
                        return False
                    elif op == "$lt" and not (value < target):
                        return False
                    elif op == "$eq" and not (value == target):
                        return False
                    elif op == "$ne" and not (value != target):
                        return False
            else:
                # Direct equality
                if metadata.get(key) != condition:
                    return False

        return True

    def _calculate_similarity(self, query: str, document: str) -> float:
        """
        Calculate simple text similarity.

        Args:
            query: Query text
            document: Document text

        Returns:
            Similarity score (0-1)
        """
        # Simple word overlap scoring
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())

        if not query_words or not doc_words:
            return 0.0

        overlap = len(query_words & doc_words)
        union = len(query_words | doc_words)

        return overlap / union if union > 0 else 0.0

    def _generate_dummy_embedding(self, text: str) -> List[float]:
        """
        Generate dummy embedding vector.

        Args:
            text: Input text

        Returns:
            Dummy 384-dimensional vector
        """
        # Use hash for deterministic but meaningless embedding
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(hash_val >> i) % 100 / 100.0 for i in range(384)]

    def _generate_id(self, document: str, index: int) -> str:
        """
        Generate document ID.

        Args:
            document: Document text
            index: Document index

        Returns:
            Generated ID
        """
        hash_val = hashlib.md5(document.encode()).hexdigest()
        return f"doc_{index}_{hash_val[:8]}"


# ============================================================================
# Mock Client Class
# ============================================================================

class MockChromaClient:
    """
    Mock ChromaDB Client.

    Manages collections in memory.
    """

    def __init__(self, settings: Optional[Any] = None):
        """
        Initialize mock client.

        Args:
            settings: ChromaDB settings (ignored in mock)
        """
        self.settings = settings
        self._collections: Dict[str, MockCollection] = {}

    def create_collection(
        self,
        name: str,
        metadata: Optional[dict] = None
    ) -> MockCollection:
        """
        Create a new collection.

        Args:
            name: Collection name
            metadata: Collection metadata

        Returns:
            Created collection

        Raises:
            ValueError: If collection already exists
        """
        if name in self._collections:
            raise ValueError(f"Collection {name} already exists")

        collection = MockCollection(name, metadata)
        self._collections[name] = collection
        return collection

    def get_collection(self, name: str) -> MockCollection:
        """
        Get existing collection.

        Args:
            name: Collection name

        Returns:
            Collection

        Raises:
            ValueError: If collection doesn't exist
        """
        if name not in self._collections:
            raise ValueError(f"Collection {name} does not exist")

        return self._collections[name]

    def get_or_create_collection(
        self,
        name: str,
        metadata: Optional[dict] = None
    ) -> MockCollection:
        """
        Get or create collection.

        Args:
            name: Collection name
            metadata: Collection metadata

        Returns:
            Collection
        """
        try:
            return self.get_collection(name)
        except ValueError:
            return self.create_collection(name, metadata)

    def delete_collection(self, name: str):
        """
        Delete collection.

        Args:
            name: Collection name
        """
        self._collections.pop(name, None)

    def list_collections(self) -> List[str]:
        """
        List all collection names.

        Returns:
            List of collection names
        """
        return list(self._collections.keys())
