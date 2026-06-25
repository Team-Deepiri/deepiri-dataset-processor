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


def _column_stats(dataset_path: Path, max_rows: int = 1000) -> Dict[str, Any]:
    """Compute lightweight column statistics from JSONL samples."""
    import json
    from collections import defaultdict

    numeric: Dict[str, List[float]] = defaultdict(list)
    text_lengths: List[int] = []
    rows_read = 0

    files = (
        [dataset_path]
        if dataset_path.suffix == ".jsonl"
        else sorted(dataset_path.rglob("*.jsonl"))
    )
    for file_path in files:
        with open(file_path, encoding="utf-8") as handle:
            for line in handle:
                if rows_read >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                for key, value in record.items():
                    if isinstance(value, (int, float)):
                        numeric[key].append(float(value))
                    elif key == "text" and isinstance(value, str):
                        text_lengths.append(len(value))
                rows_read += 1
        if rows_read >= max_rows:
            break

    stats: Dict[str, Any] = {}
    for col, values in numeric.items():
        if values:
            stats[col] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
    if text_lengths:
        stats["text_length"] = {
            "min": min(text_lengths),
            "max": max(text_lengths),
            "mean": sum(text_lengths) / len(text_lengths),
        }
    return stats


def build_manifest(
    dataset_path: Path | str,
    *,
    dataset_id: str | None = None,
    version: str | None = None,
    produced_by: str = "deepiri-dataset-processor@0.3.0",
    metadata: Dict[str, Any] | None = None,
    license_info: str | None = None,
    provenance: str | None = None,
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
        "metadata": {
            **(metadata or {}),
            "column_stats": _column_stats(path),
            "license": license_info,
            "provenance": provenance,
        },
    }

    if _HAS_MODELKIT:
        return DatasetManifest(**manifest_data)

    return manifest_data
