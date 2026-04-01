# deepiri-dataset-processor

Reusable data preprocessing library cleaning, deduping, quality checking, and versioning datasets.

## Installation

Core modules (zero dependencies):

```bash
pip install deepiri-dataset-processor
```

With optional extras:

```bash
# Semantic deduplication (requires numpy + sentence-transformers)
pip install deepiri-dataset-processor[semantic]

# Database-backed versioning (requires sqlalchemy + pydantic)
pip install deepiri-dataset-processor[versioning]

# Quality checks (requires numpy + pandas)
pip install deepiri-dataset-processor[quality]

# Everything
pip install deepiri-dataset-processor[all]
```

## Features

- **Text Cleaning** — Boilerplate removal, URL filtering, length checks, whitespace normalization
- **Exact Deduplication** — Hash-based duplicate detection (stdlib only)
- **Semantic Deduplication** — Embedding-based near-duplicate detection (optional: numpy)
- **Data Leakage Detection** — N-gram overlap, train/eval contamination, memorization patterns
- **Dataset Versioning** — Filesystem checksums + lineage tracking, or SQLAlchemy-backed DB versioning
- **Pipeline Stages** — Composable preprocessing stages with validation

## Quick Start

```python
from deepiri_dataset_processor.cleaning import TextCleaner
from deepiri_dataset_processor.safety import DataLeakageDetector
from deepiri_dataset_processor.deduplication import ExactDeduplicator

# Clean text
cleaner = TextCleaner(min_length=50, max_urls=5)
cleaned = cleaner.clean_batch(raw_texts)

# Detect duplicates
dedup = ExactDeduplicator()
report = dedup.find_duplicates(cleaned)

# Check for train/eval leakage
detector = DataLeakageDetector(ngram_size=5)
report = detector.detect_train_eval_contamination(train_texts, eval_texts)
```

## License

MIT
