"""Tests for pipeline presets and advanced stages."""
import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "data.jsonl"
    records = [
        {"text": "This is a long enough training document for the cleaner to accept it."},
        {"text": "This is a long enough training document for the cleaner to accept it."},
        {"text": "Another unique document that passes minimum length requirements easily."},
    ]
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def test_training_preset_dedup(tmp_path: Path, sample_jsonl: Path):
    from deepiri_dataset_processor.pipeline.presets import training_preset
    from deepiri_dataset_processor.streaming.chunked_jsonl import load_jsonl_records

    records = load_jsonl_records(sample_jsonl)
    pipeline = training_preset(required_fields=["text"])
    result = pipeline.run(records)
    assert result.success
    out = result.processed_data.data
    assert len(out) == 2


def test_feedback_preset(tmp_path: Path):
    from deepiri_dataset_processor.pipeline.presets import feedback_preset

    records = [{"text": "correction one"}, {"text": "correction one"}, {"text": "two"}]
    pipeline = feedback_preset()
    result = pipeline.run(records)
    assert result.success
    assert len(result.processed_data.data) == 2


def test_run_streaming(tmp_path: Path, sample_jsonl: Path):
    from deepiri_dataset_processor.pipeline.presets import training_preset

    pipeline = training_preset(required_fields=["text"])
    result = pipeline.run_streaming(sample_jsonl, chunk_size=2)
    assert result.success
    assert len(result.processed_data.data) == 2


def test_text_cleaning_stage():
    from deepiri_dataset_processor.pipeline.advanced_stages import TextCleaningStage

    stage = TextCleaningStage(config={"min_length": 10})
    records = [{"text": "  hello   world  "}, {"text": "short"}]
    result = stage.process(records)
    assert result.success
    assert len(result.processed_data.data) == 1


def test_exact_dedup_stage():
    from deepiri_dataset_processor.pipeline.advanced_stages import ExactDedupStage

    stage = ExactDedupStage()
    records = [{"text": "same"}, {"text": "same"}, {"text": "diff"}]
    result = stage.process(records)
    assert result.success
    assert len(result.processed_data.data) == 2


def test_manifest_stage(tmp_path: Path):
    from deepiri_dataset_processor.pipeline.advanced_stages import ManifestStage

    out = tmp_path / "out.jsonl"
    stage = ManifestStage(config={"output_path": str(out), "dataset_id": "testds"})
    records = [{"text": "hello world example text"}]
    result = stage.process(records)
    assert result.success
    assert "manifest" in result.processed_data.metadata


def test_iter_jsonl_chunks(tmp_path: Path, sample_jsonl: Path):
    from deepiri_dataset_processor.streaming.chunked_jsonl import iter_jsonl_chunks

    chunks = list(iter_jsonl_chunks(sample_jsonl, chunk_size=2))
    assert len(chunks) >= 1
    assert sum(len(c) for c in chunks) == 3


def test_quality_all_dimensions(checker=None):
    from deepiri_dataset_processor.quality.checker import QualityChecker

    qc = QualityChecker()
    data = [
        {"text": "hello", "label": 1, "id": "a", "created_at": "2026-01-01"},
        {"text": "world", "label": 2, "id": "b", "created_at": "2026-02-01"},
        {"text": "foo", "label": 3, "id": "c", "created_at": "2026-03-01"},
        {"text": "bar", "label": 4, "id": "d", "created_at": "2026-04-01"},
        {"text": "baz", "label": 5, "id": "e", "created_at": "2026-05-01"},
    ]
    report = qc.check_quality(data, dataset_id="test")
    dims = set(report.dimension_scores.keys())
    assert "completeness" in dims
    assert "timeliness" in dims
    assert "accuracy" in dims
    assert "integrity" in dims


def test_build_manifest_column_stats(tmp_path: Path, sample_jsonl: Path):
    from deepiri_dataset_processor.manifest import build_manifest

    manifest = build_manifest(sample_jsonl, dataset_id="ds1")
    if hasattr(manifest, "metadata"):
        meta = manifest.metadata
    else:
        meta = manifest.get("metadata", {})
    assert "column_stats" in meta


def test_semantic_dedup_with_mock_embedding():
    import numpy as np

    from deepiri_dataset_processor.deduplication.semantic_dedup import (
        SemanticDeduplicationEngine,
    )

    class MockModel:
        def encode(self, text):
            if "duplicate" in text:
                return np.ones(384, dtype=np.float32)
            return np.zeros(384, dtype=np.float32)

    engine = SemanticDeduplicationEngine(
        embedding_model=MockModel(), similarity_threshold=0.99, use_lsh=False
    )
    texts = ["this is duplicate content", "this is duplicate content", "unique text"]
    filtered = engine.filter_duplicates(texts)
    assert len(filtered) == 2
