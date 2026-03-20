"""Tests for text cleaning module."""

from deepiri_dataset_processor.cleaning.text_cleaner import TextCleaner, clean_text_document


class TestTextCleaner:
    def setup_method(self):
        # Use min_length=5 for testing so short strings aren't filtered
        self.cleaner = TextCleaner(min_length=5)

    def test_clean_removes_extra_whitespace(self):
        text = "hello   world   this is a test string"
        result = self.cleaner.clean(text)
        assert result is not None
        assert "   " not in result

    def test_clean_strips_text(self):
        text = "  hello world this is a long test string  "
        result = self.cleaner.clean(text)
        assert result is not None
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_clean_returns_none_for_empty(self):
        result = self.cleaner.clean("")
        assert result is None

    def test_clean_returns_none_for_short(self):
        cleaner = TextCleaner(min_length=100)
        result = cleaner.clean("short")
        assert result is None

    def test_clean_batch(self):
        texts = [
            "hello world this is a valid text",
            "another valid text for testing",
            "yet another good text for batch",
        ]
        results = self.cleaner.clean_batch(texts)
        assert len(results) > 0

    def test_clean_batch_filters_short(self):
        cleaner = TextCleaner(min_length=100)
        texts = ["short", "also short", "nope"]
        results = cleaner.clean_batch(texts)
        assert len(results) == 0

    def test_remove_duplicates(self):
        texts = ["hello", "world", "hello", "foo", "world"]
        unique = self.cleaner.remove_duplicates(texts)
        assert len(unique) == 3
        assert unique[0] == "hello"
        assert unique[1] == "world"
        assert unique[2] == "foo"

    def test_remove_duplicates_empty(self):
        assert self.cleaner.remove_duplicates([]) == []

    def test_clean_text_document_convenience(self):
        text = "  hello   world  this is a test document for our purposes  "
        result = clean_text_document(text, min_length=10)
        assert result is not None
        assert isinstance(result, str)
        assert "   " not in result

    def test_clean_text_document_returns_none_for_short(self):
        result = clean_text_document("hi", min_length=50)
        assert result is None
