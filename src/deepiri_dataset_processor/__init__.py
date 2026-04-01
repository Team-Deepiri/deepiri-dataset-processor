"""
deepiri-dataset-processor: Reusable data preprocessing, dedup, safety-check, and versioning.

Core modules (zero dependencies beyond declared deps):
    - cleaning: Text cleaning and normalization
    - safety: Data leakage detection
    - pipeline: Preprocessing pipeline (stages, DatasetPipeline, PipelineOrchestrator; DAG uses networkx)
    - deduplication.exact_dedup: Hash-based exact deduplication

Optional modules (require extras):
    - deduplication.semantic_dedup: pip install deepiri-dataset-processor[semantic]
    - versioning.database: pip install deepiri-dataset-processor[versioning]
    - quality: pip install deepiri-dataset-processor[quality]
"""

__version__ = "0.1.0"

from deepiri_dataset_processor.cleaning.text_cleaner import (
    TextCleaner,
    clean_text_document,
)
from deepiri_dataset_processor.deduplication.exact_dedup import ExactDeduplicator
from deepiri_dataset_processor.pipeline import (
    DataCleaningStage,
    DataLoadingStage,
    DataRoutingStage,
    DataTransformationStage,
    DataValidationStage,
    DatasetPipeline,
    LabelValidationStage,
    PipelineOrchestrator,
    PreprocessingStage,
    ProcessedData,
    StageResult,
    ValidationResult,
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
    "PipelineOrchestrator",
    "DataCleaningStage",
    "DataLoadingStage",
    "DataRoutingStage",
    "DataValidationStage",
    "LabelValidationStage",
    "DataTransformationStage",
    "DataLeakageDetector",
]
