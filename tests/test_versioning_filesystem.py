"""Tests for filesystem-based dataset versioning."""

import json
from pathlib import Path

from deepiri_dataset_processor.versioning.filesystem import DatasetVersioningSystem


class TestDatasetVersioningSystem:
    def test_compute_checksum_file(self, tmp_path):
        versioning = DatasetVersioningSystem(metadata_dir=tmp_path / "meta")
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        checksum = versioning.compute_dataset_checksum(f)
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex

    def test_compute_checksum_deterministic(self, tmp_path):
        versioning = DatasetVersioningSystem(metadata_dir=tmp_path / "meta")
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        c1 = versioning.compute_dataset_checksum(f)
        c2 = versioning.compute_dataset_checksum(f)
        assert c1 == c2

    def test_compute_checksum_different_content(self, tmp_path):
        versioning = DatasetVersioningSystem(metadata_dir=tmp_path / "meta")
        f1 = tmp_path / "a.txt"
        f1.write_text("hello")
        f2 = tmp_path / "b.txt"
        f2.write_text("world")
        assert versioning.compute_dataset_checksum(f1) != versioning.compute_dataset_checksum(f2)

    def test_compute_checksum_directory(self, tmp_path):
        versioning = DatasetVersioningSystem(metadata_dir=tmp_path / "meta")
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "a.txt").write_text("hello")
        (data_dir / "b.txt").write_text("world")
        checksum = versioning.compute_dataset_checksum(data_dir)
        assert isinstance(checksum, str)
        assert len(checksum) == 64

    def test_count_samples_jsonl(self, tmp_path):
        versioning = DatasetVersioningSystem(metadata_dir=tmp_path / "meta")
        f = tmp_path / "data.jsonl"
        f.write_text('{"text": "a"}\n{"text": "b"}\n{"text": "c"}\n')
        result = versioning.count_samples_and_tokens(f)
        assert result["sample_count"] == 3
        assert result["token_count"] == 0  # No tokenizer

    def test_count_samples_dir(self, tmp_path):
        versioning = DatasetVersioningSystem(metadata_dir=tmp_path / "meta")
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        f = data_dir / "train.jsonl"
        f.write_text('{"text": "a"}\n{"text": "b"}\n')
        result = versioning.count_samples_and_tokens(data_dir)
        assert result["sample_count"] == 2

    def test_create_dataset_version(self, tmp_path):
        versioning = DatasetVersioningSystem(metadata_dir=tmp_path / "meta")
        data_dir = tmp_path / "dataset"
        data_dir.mkdir()
        f = data_dir / "train.jsonl"
        f.write_text('{"text": "hello"}\n{"text": "world"}\n')

        meta = versioning.create_dataset_version(
            dataset_path=data_dir,
            dataset_id="test_ds",
        )
        assert meta["dataset_id"] == "test_ds"
        assert meta["sample_count"] == 2
        assert "checksum" in meta
        assert "version" in meta

    def test_create_dataset_version_with_parents(self, tmp_path):
        versioning = DatasetVersioningSystem(metadata_dir=tmp_path / "meta")
        data_dir = tmp_path / "dataset"
        data_dir.mkdir()
        f = data_dir / "train.jsonl"
        f.write_text('{"text": "hello"}\n')

        meta = versioning.create_dataset_version(
            dataset_path=data_dir,
            dataset_id="test_ds",
            parent_versions=["v0.9"],
        )
        assert meta["parent_versions"] == ["v0.9"]

    def test_verify_dataset_integrity(self, tmp_path):
        versioning = DatasetVersioningSystem(metadata_dir=tmp_path / "meta")
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        checksum = versioning.compute_dataset_checksum(f)
        assert versioning.verify_dataset_integrity(f, checksum) is True

    def test_verify_dataset_integrity_mismatch(self, tmp_path):
        versioning = DatasetVersioningSystem(metadata_dir=tmp_path / "meta")
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        assert versioning.verify_dataset_integrity(f, "wrong_checksum") is False

    def test_get_dataset_lineage(self, tmp_path):
        versioning = DatasetVersioningSystem(metadata_dir=tmp_path / "meta")
        data_dir = tmp_path / "dataset"
        data_dir.mkdir()
        (data_dir / "train.jsonl").write_text('{"text": "a"}\n')

        versioning.create_dataset_version(dataset_path=data_dir, dataset_id="ds1")
        lineage = versioning.get_dataset_lineage("ds1")
        assert len(lineage) == 1

    def test_get_dataset_lineage_empty(self, tmp_path):
        versioning = DatasetVersioningSystem(metadata_dir=tmp_path / "meta")
        lineage = versioning.get_dataset_lineage("nonexistent")
        assert lineage == []

    def test_create_version_empty_dir(self, tmp_path):
        versioning = DatasetVersioningSystem(metadata_dir=tmp_path / "meta")
        data_dir = tmp_path / "empty"
        data_dir.mkdir()
        meta = versioning.create_dataset_version(
            dataset_path=data_dir,
            dataset_id="empty_ds",
        )
        assert meta["sample_count"] == 0
