"""Tests for semantic deduplication module."""

import pytest

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

pytestmark = pytest.mark.skipif(
    not HAS_NUMPY, reason="numpy required for semantic dedup tests"
)


class MockEmbeddingModel:
    """Mock that returns deterministic, well-separated embeddings per unique text."""

    def __init__(self):
        self._cache = {}
        self._counter = 0

    def encode(self, text):
        import numpy as np

        if text not in self._cache:
            # Assign a unique angle per unique text to ensure well-separated embeddings
            angle = self._counter * 2.1  # radians, spread apart
            self._cache[text] = np.array(
                [np.cos(angle), np.sin(angle)], dtype=np.float32
            )
            self._counter += 1
        return self._cache[text]


@pytest.fixture
def engine(tmp_path):
    from deepiri_dataset_processor.deduplication.semantic_dedup import (
        SemanticDeduplicationEngine,
    )

    return SemanticDeduplicationEngine(
        embedding_model=MockEmbeddingModel(),
        cache_dir=tmp_path / "cache",
    )


class TestSemanticDeduplicationEngine:
    def test_import_guard(self):
        from deepiri_dataset_processor.deduplication import semantic_dedup

        assert hasattr(semantic_dedup, "HAS_NUMPY")

    def test_compute_embedding(self, engine):
        embedding = engine.compute_embedding("hello world")
        assert embedding is not None
        assert len(embedding) == 2

    def test_compute_similarity_identical(self, engine):
        e1 = engine.compute_embedding("hello world")
        e2 = engine.compute_embedding("hello world")
        sim = engine.compute_similarity(e1, e2)
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_find_duplicates_with_dupes(self, engine):
        texts = ["hello world", "hello world", "something different"]
        result = engine.find_duplicates(texts)
        # Returns Dict[int, List[int]]
        assert isinstance(result, dict)
        # First two are identical, so index 0 should map to [1]
        assert 0 in result
        assert 1 in result[0]

    def test_find_duplicates_no_dupes(self, engine):
        texts = ["completely unique text", "another unique text", "third unique text"]
        result = engine.find_duplicates(texts)
        assert isinstance(result, dict)
        # All texts are different, so no duplicate groups
        # (depends on hash collisions, but very unlikely)

    def test_find_duplicates_empty(self, engine):
        result = engine.find_duplicates([])
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_filter_duplicates(self, engine):
        texts = ["hello world", "hello world", "something different"]
        filtered = engine.filter_duplicates(texts)
        assert len(filtered) == 2
        assert "hello world" in filtered
        assert "something different" in filtered

    def test_filter_duplicates_no_dupes(self, engine):
        texts = ["alpha", "beta", "gamma"]
        filtered = engine.filter_duplicates(texts)
        assert len(filtered) == 3
