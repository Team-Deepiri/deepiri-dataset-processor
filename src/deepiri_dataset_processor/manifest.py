"""Build DatasetManifest objects compatible with deepiri-modelkit training contracts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Union

from deepiri_dataset_processor.versioning.filesystem import DatasetVersioningSystem

try:
    from deepiri_modelkit.contracts.training import DatasetManifest

    _HAS_MODELKIT = True
except ImportError:
    _HAS_MODELKIT = False
    DatasetManifest = None  # type: ignore[misc, assignment]

ManifestResult = Union["DatasetManifest", Dict[str, Any]]


def _infer_schema(dataset_path: Path, max_rows: int = 100) -> Dict[str, Any]:
    """Infer a lightweight schema description from JSONL samples."""
    fields: Dict[str, set[str]] = {}

    def _collect(record: dict[str, Any]) -> None:
        for key, value in record.items():
            fields.setdefault(key, set()).add(type(value).__name__)

    path = Path(dataset_path)
    rows_read = 0

    if path.suffix == ".jsonl":
        files = [path]
    elif path.is_dir():
        files = sorted(path.rglob("*.jsonl"))
    else:
        files = []

    for file_path in files:
        with open(file_path, encoding="utf-8") as handle:
            for line in handle:
                if rows_read >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    _collect(data)
                    rows_read += 1
        if rows_read >= max_rows:
            break

    if not fields:
        return {"fields": {}, "format": "unknown"}

    return {
        "fields": {name: sorted(types) for name, types in fields.items()},
        "format": "jsonl",
    }


def build_manifest(
    dataset_path: Path | str,
    *,
    dataset_id: str | None = None,
    version: str | None = None,
    produced_by: str = "deepiri-dataset-processor@0.2.0",
    metadata: Dict[str, Any] | None = None,
) -> ManifestResult:
    """
    Build an immutable dataset manifest for training orchestration.

    Returns a modelkit ``DatasetManifest`` when ``deepiri-modelkit`` is installed,
    otherwise a compatible dictionary with the same keys.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    versioning = DatasetVersioningSystem()
    content_hash = versioning.compute_dataset_checksum(path)
    row_count = versioning.count_samples_and_tokens(path)["sample_count"]

    manifest_data = {
        "id": dataset_id or path.stem,
        "version": version or datetime.now(timezone.utc).strftime("%Y.%m.%d"),
        "path": str(path.resolve()),
        "content_hash": content_hash,
        "row_count": row_count,
        "schema": _infer_schema(path),
        "produced_by": produced_by,
        "metadata": metadata or {},
    }

    if _HAS_MODELKIT:
        return DatasetManifest(**manifest_data)

    return manifest_data
