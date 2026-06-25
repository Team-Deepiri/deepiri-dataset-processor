"""
deepiri-dataset-processor: Production dataset preprocessing library.

Core modules:
    - cleaning: Text cleaning and normalization
    - deduplication: Exact and semantic deduplication
    - safety: Data leakage detection
    - quality: 7-dimension quality scoring
    - versioning: Filesystem and database versioning
    - pipeline: Composable preprocessing stages and presets
    - streaming: Chunked JSONL processing
    - manifest: DatasetManifest generation
"""

__version__ = "0.3.0"

from deepiri_dataset_processor.cleaning.text_cleaner import (
    TextCleaner,
    clean_text_document,
)
from deepiri_dataset_processor.deduplication.exact_dedup import ExactDeduplicator
from deepiri_dataset_processor.deduplication.semantic_dedup import (
    SemanticDeduplicationEngine,
)
from deepiri_dataset_processor.manifest import build_manifest
from deepiri_dataset_processor.pipeline.advanced_stages import (
    ExactDedupStage,
    LeakageCheckStage,
    ManifestStage,
    QualityScoringStage,
    SemanticDedupStage,
    TextCleaningStage,
    VersioningStage,
)
from deepiri_dataset_processor.pipeline.base import (
    PreprocessingStage,
    ProcessedData,
    StageResult,
    ValidationResult,
)
from deepiri_dataset_processor.pipeline.orchestrator import DatasetPipeline
from deepiri_dataset_processor.pipeline.presets import (
    feedback_preset,
    production_preset,
    training_preset,
)
from deepiri_dataset_processor.pipeline.stages import (
    DataCleaningStage,
    DataTransformationStage,
    DataValidationStage,
)
from deepiri_dataset_processor.quality.checker import (
    QualityChecker,
    QualityConfig,
    QualityReport,
)
from deepiri_dataset_processor.safety.leakage_detector import DataLeakageDetector
from deepiri_dataset_processor.streaming.chunked_jsonl import (
    iter_jsonl_chunks,
    load_jsonl_records,
    write_jsonl_records,
)
from deepiri_dataset_processor.versioning.filesystem import DatasetVersioningSystem

__all__ = [
    "__version__",
    "TextCleaner",
    "clean_text_document",
    "ExactDeduplicator",
    "SemanticDeduplicationEngine",
    "build_manifest",
    "QualityChecker",
    "QualityConfig",
    "QualityReport",
    "DatasetVersioningSystem",
    "PreprocessingStage",
    "ProcessedData",
    "StageResult",
    "ValidationResult",
    "DatasetPipeline",
    "DataCleaningStage",
    "DataValidationStage",
    "DataTransformationStage",
    "TextCleaningStage",
    "ExactDedupStage",
    "SemanticDedupStage",
    "LeakageCheckStage",
    "QualityScoringStage",
    "ManifestStage",
    "VersioningStage",
    "training_preset",
    "feedback_preset",
    "production_preset",
    "DataLeakageDetector",
    "iter_jsonl_chunks",
    "load_jsonl_records",
    "write_jsonl_records",
]
