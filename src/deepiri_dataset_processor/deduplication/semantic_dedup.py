"""
Semantic deduplication engine.

Uses embedding-based similarity filtering with sentence-transformers when
available, plus optional LSH bucketing for scale.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_sentence_transformer(model_name: str = _DEFAULT_MODEL) -> Any:
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as exc:
        logger.warning("sentence-transformers unavailable: %s", exc)
        return None


class SemanticDeduplicationEngine:
    """Semantic deduplication using embeddings with optional LSH bucketing."""

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        embedding_model: Any = None,
        cache_dir: Path = Path("data/deduplication_cache"),
        *,
        model_name: str = _DEFAULT_MODEL,
        use_lsh: bool = True,
        lsh_bands: int = 20,
    ):
        if not HAS_NUMPY:
            raise ImportError("numpy is required for semantic deduplication")

        self.similarity_threshold = similarity_threshold
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.use_lsh = use_lsh
        self.lsh_bands = lsh_bands

        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = _load_sentence_transformer(model_name)

    def compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]

        if self.embedding_model is not None:
            embedding = np.asarray(self.embedding_model.encode(text), dtype=np.float32)
        else:
            embedding = np.array([hash(text) % 1000] * 384, dtype=np.float32)

        self.embedding_cache[text_hash] = embedding
        return embedding

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        dot_product = float(np.dot(embedding1, embedding2))
        norm1 = float(np.linalg.norm(embedding1))
        norm2 = float(np.linalg.norm(embedding2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def _lsh_bucket(self, embedding: np.ndarray) -> int:
        """Simple LSH bucket from embedding sign bits."""
        bits = (embedding[: self.lsh_bands] > 0).astype(int)
        return int("".join(str(b) for b in bits), 2)

    def find_duplicates(self, texts: List[str]) -> Dict[int, List[int]]:
        """Find duplicate texts using semantic similarity."""
        logger.info("Finding semantic duplicates in %d texts...", len(texts))
        embeddings = [self.compute_embedding(text) for text in texts]
        duplicates: Dict[int, List[int]] = {}
        processed: Set[int] = set()

        if self.use_lsh and len(texts) > 50:
            buckets: Dict[int, List[int]] = {}
            for idx, emb in enumerate(embeddings):
                bucket = self._lsh_bucket(emb)
                buckets.setdefault(bucket, []).append(idx)
            candidate_pairs: List[tuple[int, int]] = []
            for indices in buckets.values():
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        candidate_pairs.append((indices[i], indices[j]))
        else:
            candidate_pairs = [
                (i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))
            ]

        for i, j in candidate_pairs:
            if i in processed or j in processed:
                continue
            similarity = self.compute_similarity(embeddings[i], embeddings[j])
            if similarity >= self.similarity_threshold:
                duplicates.setdefault(i, []).append(j)
                processed.add(j)

        duplicates = {k: v for k, v in duplicates.items() if v}
        if duplicates:
            total = sum(len(v) for v in duplicates.values())
            logger.info("Found %d duplicate groups, %d duplicates", len(duplicates), total)
        return duplicates

    def filter_duplicates(self, texts: List[str]) -> List[str]:
        """Filter out duplicate texts, keeping first occurrence of each group."""
        duplicates = self.find_duplicates(texts)
        remove_indices: Set[int] = set()
        for dup_list in duplicates.values():
            remove_indices.update(dup_list)
        filtered = [texts[i] for i in range(len(texts)) if i not in remove_indices]
        logger.info("Filtered %d duplicate texts", len(texts) - len(filtered))
        return filtered
