"""
deepiri-dataset-processor: Reusable data preprocessing, dedup, safety-check, and versioning.

Core modules (zero dependencies):
    - cleaning: Text cleaning and normalization
    - safety: Data leakage detection
    - pipeline: Preprocessing pipeline framework
    - deduplication.exact_dedup: Hash-based exact deduplication

Optional modules (require extras):
    - deduplication.semantic_dedup: pip install deepiri-dataset-processor[semantic]
    - versioning.database: pip install deepiri-dataset-processor[versioning]
    - quality.checker: pip install deepiri-dataset-processor[quality]
"""

from deepiri_dataset_processor.cleaning.text_cleaner import (
    TextCleaner,
    clean_text_document,
)
from deepiri_dataset_processor.deduplication.exact_dedup import ExactDeduplicator
from deepiri_dataset_processor.pipeline.base import (
    PreprocessingStage,
    ProcessedData,
    StageResult,
    ValidationResult,
)
from deepiri_dataset_processor.pipeline.orchestrator import DatasetPipeline
from deepiri_dataset_processor.pipeline.stages import (
    DataCleaningStage,
    DataTransformationStage,
    DataValidationStage,
)
from deepiri_dataset_processor.safety.leakage_detector import DataLeakageDetector

__all__ = [
    "TextCleaner",
    "clean_text_document",
    "ExactDeduplicator",
    "PreprocessingStage",
    "ProcessedData",
    "StageResult",
    "ValidationResult",
    "DatasetPipeline",
    "DataCleaningStage",
    "DataValidationStage",
    "DataTransformationStage",
    "DataLeakageDetector",
]
