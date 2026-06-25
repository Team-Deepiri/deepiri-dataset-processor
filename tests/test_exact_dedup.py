"""Tests for exact deduplication module."""

from deepiri_dataset_processor.deduplication.exact_dedup import ExactDeduplicator


class TestExactDeduplicator:
    def setup_method(self):
        self.dedup = ExactDeduplicator()

    def test_find_duplicates_with_dupes(self):
        texts = ["hello", "world", "hello", "foo", "world"]
        report = self.dedup.find_duplicates(texts)
        assert report["duplicates_detected"] is True
        assert report["duplicate_groups"] == 2
        assert report["total_duplicate_instances"] == 2

    def test_find_duplicates_no_dupes(self):
        texts = ["hello", "world", "foo"]
        report = self.dedup.find_duplicates(texts)
        assert report["duplicates_detected"] is False
        assert report["duplicate_groups"] == 0
        assert report["total_duplicate_instances"] == 0

    def test_find_duplicates_empty(self):
        report = self.dedup.find_duplicates([])
        assert report["duplicates_detected"] is False
        assert report["duplicate_rate"] == 0.0

    def test_find_duplicates_single_item(self):
        report = self.dedup.find_duplicates(["hello"])
        assert report["duplicates_detected"] is False

    def test_find_duplicates_all_same(self):
        texts = ["same", "same", "same"]
        report = self.dedup.find_duplicates(texts)
        assert report["duplicates_detected"] is True
        assert report["duplicate_groups"] == 1
        assert report["total_duplicate_instances"] == 2

    def test_filter_duplicates(self):
        texts = ["hello", "world", "hello", "foo", "world"]
        unique = self.dedup.filter_duplicates(texts)
        assert len(unique) == 3
        assert unique == ["hello", "world", "foo"]

    def test_filter_duplicates_preserves_order(self):
        texts = ["c", "b", "a", "b", "c"]
        unique = self.dedup.filter_duplicates(texts)
        assert unique == ["c", "b", "a"]

    def test_filter_duplicates_empty(self):
        assert self.dedup.filter_duplicates([]) == []

    def test_duplicate_rate(self):
        texts = ["a", "b", "a", "b"]
        report = self.dedup.find_duplicates(texts)
        assert report["duplicate_rate"] == 0.5
