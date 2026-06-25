# deepiri-dataset-processor

Reusable data preprocessing library for ML/AI projects: clean, dedup, safety-check, version, and manifest generation.

## Installation

```bash
poetry install
# or with optional extras:
poetry install --all-extras
```

## Features

- **Text Cleaning** — Boilerplate removal, URL filtering, length checks, whitespace normalization
- **Exact Deduplication** — Hash-based duplicate detection (stdlib only)
- **Semantic Deduplication** — Embedding-based near-duplicate detection (optional: numpy)
- **Data Leakage Detection** — N-gram overlap, train/eval contamination, memorization patterns
- **Dataset Versioning** — Filesystem checksums + lineage tracking, or SQLAlchemy-backed DB versioning
- **Pipeline Stages** — Composable preprocessing stages with validation
- **Manifest Generation** — `build_manifest()` producing modelkit-compatible `DatasetManifest` objects

## Quick Start

```python
from deepiri_dataset_processor.cleaning import TextCleaner
from deepiri_dataset_processor.manifest import build_manifest
from deepiri_dataset_processor.safety import DataLeakageDetector
from deepiri_dataset_processor.deduplication import ExactDeduplicator

cleaner = TextCleaner(min_length=50, max_urls=5)
cleaned = cleaner.clean_batch(raw_texts)

manifest = build_manifest("data/processed/train.jsonl", dataset_id="corpus-v1")
```

## License

Apache-2.0
