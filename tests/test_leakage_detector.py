"""Tests for data leakage detection module."""

from deepiri_dataset_processor.safety.leakage_detector import DataLeakageDetector


class TestDataLeakageDetector:
    def setup_method(self):
        # ngram_size=2 for simpler test data
        self.detector = DataLeakageDetector(ngram_size=2)

    def test_extract_ngrams(self):
        text = "the quick brown fox"
        ngrams = self.detector.extract_ngrams(text)
        # Returns Set[str] of space-joined ngrams
        assert "the quick" in ngrams
        assert "quick brown" in ngrams
        assert "brown fox" in ngrams

    def test_extract_ngrams_single_word(self):
        ngrams = self.detector.extract_ngrams("hello")
        assert len(ngrams) == 0

    def test_extract_ngrams_empty(self):
        ngrams = self.detector.extract_ngrams("")
        assert len(ngrams) == 0

    def test_detect_exact_duplicates(self):
        texts = ["hello world", "foo bar", "hello world", "test data"]
        result = self.detector.detect_exact_duplicates(texts)
        assert result["duplicates_detected"] is True
        assert result["duplicate_groups"] == 1
        assert result["total_duplicate_instances"] == 1

    def test_detect_exact_duplicates_no_dupes(self):
        texts = ["hello world", "foo bar", "test data"]
        result = self.detector.detect_exact_duplicates(texts)
        assert result["duplicates_detected"] is False
        assert result["total_duplicate_instances"] == 0

    def test_detect_exact_duplicates_empty(self):
        result = self.detector.detect_exact_duplicates([])
        assert result["duplicates_detected"] is False

    def test_detect_train_eval_contamination(self):
        train = ["the quick brown fox jumps over the lazy dog"]
        eval_data = ["the quick brown fox jumps over the lazy dog"]
        result = self.detector.detect_train_eval_contamination(train, eval_data)
        assert "contamination_rate" in result
        assert result["contamination_detected"] is True

    def test_detect_train_eval_no_contamination(self):
        train = ["alpha beta gamma"]
        eval_data = ["completely different text here"]
        result = self.detector.detect_train_eval_contamination(train, eval_data)
        assert result["contamination_detected"] is False

    def test_detect_near_duplicates(self):
        texts = [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox leaps over the lazy dog",
            "completely unrelated text here",
        ]
        result = self.detector.detect_near_duplicates(texts, similarity_threshold=0.5)
        assert "near_duplicates_detected" in result

    def test_detect_memorization_patterns(self):
        # Need many repeated ngrams to hit min_frequency=10
        texts = ["hello world test"] * 15 + ["unique text here"]
        result = self.detector.detect_memorization_patterns(texts, min_frequency=10)
        assert isinstance(result, dict)
        assert "patterns_detected" in result
