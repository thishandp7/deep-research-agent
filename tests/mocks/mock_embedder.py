"""
Mock implementation of SentenceTransformer.

Generates deterministic embeddings without downloading models.
"""

from typing import List, Union, Optional
import hashlib
import numpy as np


# ============================================================================
# Mock SentenceTransformer Class
# ============================================================================

class MockSentenceTransformer:
    """
    Mock SentenceTransformer for testing.

    Generates deterministic embeddings based on text hash without requiring
    actual model downloads.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize mock sentence transformer.

        Args:
            model_name: Model name (stored but not used)
        """
        self.model_name = model_name
        self.embedding_dimension = 384  # Standard dimension for all-MiniLM-L6-v2
        self.encode_count = 0  # Track number of encode calls

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True
    ) -> Union[np.ndarray, List[List[float]]]:
        """
        Generate mock embeddings for sentences.

        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size (ignored)
            show_progress_bar: Show progress (ignored)
            convert_to_numpy: Return numpy array vs list

        Returns:
            Embeddings as numpy array or list
        """
        self.encode_count += 1

        # Handle single sentence
        if isinstance(sentences, str):
            embedding = self._generate_embedding(sentences)
            return np.array(embedding) if convert_to_numpy else embedding

        # Handle list of sentences
        embeddings = [self._generate_embedding(sent) for sent in sentences]

        if convert_to_numpy:
            return np.array(embeddings)
        else:
            return embeddings

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate deterministic embedding for text.

        Uses MD5 hash to create consistent embeddings for same text.

        Args:
            text: Input text

        Returns:
            384-dimensional embedding vector
        """
        # Generate hash from text
        hash_bytes = hashlib.md5(text.encode('utf-8')).digest()

        # Extend hash to create 384 dimensions
        embedding = []

        for i in range(self.embedding_dimension):
            # Use different byte positions and operations for variety
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] + i) % 256

            # Normalize to [-1, 1] range
            normalized = (value - 128) / 128.0
            embedding.append(normalized)

        return embedding

    def get_sentence_embedding_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Embedding dimension (384)
        """
        return self.embedding_dimension

    def similarity(self, embeddings1: List[List[float]], embeddings2: List[List[float]]) -> np.ndarray:
        """
        Calculate cosine similarity between embeddings.

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings

        Returns:
            Similarity matrix
        """
        # Convert to numpy if needed
        if not isinstance(embeddings1, np.ndarray):
            embeddings1 = np.array(embeddings1)
        if not isinstance(embeddings2, np.ndarray):
            embeddings2 = np.array(embeddings2)

        # Calculate cosine similarity
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)

        normalized1 = embeddings1 / norm1
        normalized2 = embeddings2 / norm2

        return np.dot(normalized1, normalized2.T)


# ============================================================================
# Utility Functions
# ============================================================================

def create_mock_embedder(model_name: str = "all-MiniLM-L6-v2") -> MockSentenceTransformer:
    """
    Create a mock sentence transformer.

    Args:
        model_name: Model name to use

    Returns:
        MockSentenceTransformer instance

    Example:
        >>> embedder = create_mock_embedder()
        >>> embedding = embedder.encode("test sentence")
        >>> assert len(embedding) == 384
    """
    return MockSentenceTransformer(model_name)


def get_deterministic_embedding(text: str, dimension: int = 384) -> List[float]:
    """
    Get deterministic embedding for text.

    Useful for testing when you need consistent embeddings.

    Args:
        text: Input text
        dimension: Embedding dimension

    Returns:
        Embedding vector

    Example:
        >>> emb1 = get_deterministic_embedding("test")
        >>> emb2 = get_deterministic_embedding("test")
        >>> assert emb1 == emb2  # Same text -> same embedding
    """
    embedder = MockSentenceTransformer()
    embedder.embedding_dimension = dimension
    return embedder._generate_embedding(text)


def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (-1 to 1)

    Example:
        >>> vec1 = [1.0, 0.0, 0.0]
        >>> vec2 = [1.0, 0.0, 0.0]
        >>> sim = calculate_cosine_similarity(vec1, vec2)
        >>> assert sim == 1.0
    """
    # Convert to numpy
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    # Calculate cosine similarity
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


# ============================================================================
# Test Data Generators
# ============================================================================

def generate_random_embeddings(n: int, dimension: int = 384, seed: Optional[int] = None) -> List[List[float]]:
    """
    Generate random embeddings for testing.

    Args:
        n: Number of embeddings
        dimension: Embedding dimension
        seed: Random seed for reproducibility

    Returns:
        List of embedding vectors

    Example:
        >>> embeddings = generate_random_embeddings(5, dimension=384, seed=42)
        >>> assert len(embeddings) == 5
        >>> assert len(embeddings[0]) == 384
    """
    if seed is not None:
        np.random.seed(seed)

    embeddings = []
    for _ in range(n):
        embedding = np.random.randn(dimension).tolist()
        embeddings.append(embedding)

    return embeddings


def generate_similar_embeddings(base_text: str, variations: int = 5) -> List[List[float]]:
    """
    Generate embeddings for similar texts.

    Useful for testing similarity search.

    Args:
        base_text: Base text
        variations: Number of variations

    Returns:
        List of embeddings

    Example:
        >>> embeddings = generate_similar_embeddings("artificial intelligence", 3)
        >>> assert len(embeddings) == 3
    """
    embedder = MockSentenceTransformer()

    texts = [base_text]
    for i in range(1, variations):
        # Add small variation to text
        texts.append(f"{base_text} variant {i}")

    return [embedder._generate_embedding(text) for text in texts]
