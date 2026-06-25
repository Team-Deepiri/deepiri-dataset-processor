"""Tests for dataset manifest generation."""

import json
from pathlib import Path

from deepiri_dataset_processor.manifest import build_manifest


class TestBuildManifest:
    def test_build_manifest_dict(self, tmp_path):
        dataset_dir = tmp_path / "train"
        dataset_dir.mkdir()
        jsonl = dataset_dir / "data.jsonl"
        jsonl.write_text(
            json.dumps({"text": "hello", "label": 1}) + "\n"
            + json.dumps({"text": "world", "label": 2}) + "\n"
        )

        manifest = build_manifest(
            dataset_dir,
            dataset_id="test-dataset",
            version="1.0.0",
            metadata={"source": "unit-test"},
        )

        if hasattr(manifest, "model_dump"):
            payload = manifest.model_dump()
        else:
            payload = manifest

        assert payload["id"] == "test-dataset"
        assert payload["version"] == "1.0.0"
        assert payload["row_count"] == 2
        assert len(payload["content_hash"]) == 64
        assert payload["schema"]["format"] == "jsonl"
        assert "text" in payload["schema"]["fields"]
        assert payload["produced_by"].startswith("deepiri-dataset-processor")
        assert payload["metadata"]["source"] == "unit-test"
        assert Path(payload["path"]).exists()

    def test_build_manifest_missing_path(self, tmp_path):
        missing = tmp_path / "does-not-exist"
        try:
            build_manifest(missing)
            raised = False
        except FileNotFoundError:
            raised = True
        assert raised

    def test_build_manifest_jsonl_file(self, tmp_path):
        jsonl = tmp_path / "samples.jsonl"
        jsonl.write_text('{"text": "sample"}\n')

        manifest = build_manifest(jsonl, dataset_id="samples")
        payload = manifest.model_dump() if hasattr(manifest, "model_dump") else manifest
        assert payload["row_count"] == 1
        assert payload["id"] == "samples"
