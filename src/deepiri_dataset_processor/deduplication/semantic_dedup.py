"""
Semantic deduplication engine.

Uses embedding-based similarity filtering and near-duplicate suppression
to prevent duplicate data from poisoning LLMs.

Requires: pip install deepiri-dataset-processor[semantic]
"""

import logging
import hashlib
from typing import List, Dict, Any
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)


class SemanticDeduplicationEngine:
    """
    Semantic deduplication using embeddings.

    Features:
    - Embedding-based similarity
    - Near-duplicate suppression
    - Configurable similarity threshold
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        embedding_model: Any = None,
        cache_dir: Path = Path("data/deduplication_cache"),
    ):
        if not HAS_NUMPY:
            raise ImportError(
                "numpy is required for semantic deduplication. "
                "Install with: pip install deepiri-dataset-processor[semantic]"
            )

        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_cache: Dict[str, "np.ndarray"] = {}

    def compute_embedding(self, text: str) -> "np.ndarray":
        """Compute embedding for text."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]

        if self.embedding_model:
            embedding = self.embedding_model.encode(text)
        else:
            embedding = np.array([hash(text) % 1000] * 384, dtype=np.float32)

        self.embedding_cache[text_hash] = embedding
        return embedding

    def compute_similarity(
        self,
        embedding1: "np.ndarray",
        embedding2: "np.ndarray",
    ) -> float:
        """Compute cosine similarity between embeddings."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    def find_duplicates(self, texts: List[str]) -> Dict[int, List[int]]:
        """
        Find duplicate texts using semantic similarity.

        Returns:
            Dictionary mapping original index to list of duplicate indices.
        """
        logger.info(f"Finding semantic duplicates in {len(texts)} texts...")

        embeddings = [self.compute_embedding(text) for text in texts]
        duplicates: Dict[int, List[int]] = {}
        processed = set()

        for i in range(len(texts)):
            if i in processed:
                continue
            duplicates[i] = []
            for j in range(i + 1, len(texts)):
                if j in processed:
                    continue
                similarity = self.compute_similarity(embeddings[i], embeddings[j])
                if similarity >= self.similarity_threshold:
                    duplicates[i].append(j)
                    processed.add(j)

        duplicates = {k: v for k, v in duplicates.items() if v}

        if duplicates:
            total = sum(len(v) for v in duplicates.values())
            logger.info(f"Found {len(duplicates)} duplicate groups, {total} duplicates")
        else:
            logger.info("No semantic duplicates found")

        return duplicates

    def filter_duplicates(self, texts: List[str]) -> List[str]:
        """Filter out duplicate texts, keeping first occurrence of each group."""
        duplicates = self.find_duplicates(texts)

        # Collect all duplicate indices (not the originals)
        remove_indices = set()
        for dup_list in duplicates.values():
            remove_indices.update(dup_list)

        filtered = [
            texts[i] for i in range(len(texts)) if i not in remove_indices
        ]

        removed_count = len(texts) - len(filtered)
        logger.info(f"Filtered {removed_count} duplicate texts")
        return filtered
