"""Pre-configured pipeline presets for common dataset processing workflows."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .advanced_stages import (
    ExactDedupStage,
    LeakageCheckStage,
    ManifestStage,
    QualityScoringStage,
    SemanticDedupStage,
    TextCleaningStage,
    VersioningStage,
)
from .orchestrator import DatasetPipeline
from .stages import DataValidationStage


def training_preset(
    *,
    required_fields: Optional[List[str]] = None,
    text_field: str = "text",
    dataset_id: Optional[str] = None,
) -> DatasetPipeline:
    """Clean + exact dedup + validate + manifest (standard training prep)."""
    fields = required_fields or ["text"]
    return DatasetPipeline(
        [
            TextCleaningStage(config={"text_field": text_field}),
            ExactDedupStage(config={"text_field": text_field}),
            DataValidationStage(config={"required_fields": fields}),
            ManifestStage(config={"dataset_id": dataset_id}),
        ]
    )


def feedback_preset(
    *,
    text_field: str = "text",
    dataset_id: Optional[str] = None,
) -> DatasetPipeline:
    """Light clean + dedup for small correction batches."""
    return DatasetPipeline(
        [
            TextCleaningStage(config={"text_field": text_field, "min_length": 1}),
            ExactDedupStage(config={"text_field": text_field}),
            ManifestStage(config={"dataset_id": dataset_id}),
        ]
    )


def production_preset(
    *,
    required_fields: Optional[List[str]] = None,
    text_field: str = "text",
    eval_texts: Optional[List[str]] = None,
    dataset_id: Optional[str] = None,
    dataset_name: Optional[str] = None,
    quality_threshold: float = 0.8,
    semantic_dedup: bool = True,
) -> DatasetPipeline:
    """Full chain: clean, dedup, semantic dedup, leakage, quality, manifest, version."""
    fields = required_fields or ["text"]
    stages: List[Any] = [
        TextCleaningStage(config={"text_field": text_field}),
        ExactDedupStage(config={"text_field": text_field}),
    ]
    if semantic_dedup:
        stages.append(SemanticDedupStage(config={"text_field": text_field}))
    if eval_texts:
        stages.append(
            LeakageCheckStage(
                config={"text_field": text_field, "eval_texts": eval_texts}
            )
        )
    stages.extend(
        [
            DataValidationStage(config={"required_fields": fields}),
            QualityScoringStage(
                config={"quality_threshold": quality_threshold, "dataset_id": dataset_id}
            ),
            ManifestStage(config={"dataset_id": dataset_id}),
            VersioningStage(config={"dataset_name": dataset_name}),
        ]
    )
    return DatasetPipeline(stages)
