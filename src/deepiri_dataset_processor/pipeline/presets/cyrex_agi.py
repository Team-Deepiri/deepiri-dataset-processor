"""Cyrex AGI training export presets."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from deepiri_dataset_processor.export.cyrex_bridge import batch_to_jsonl_records
from deepiri_dataset_processor.pipeline.advanced_stages import (
    ExactDedupStage,
    LeakageCheckStage,
    ManifestStage,
    QualityScoringStage,
    TextCleaningStage,
)
from deepiri_dataset_processor.pipeline.orchestrator import DatasetPipeline
from deepiri_dataset_processor.pipeline.stages import DataValidationStage
from deepiri_dataset_processor.streaming.chunked_jsonl import write_jsonl_records


def cyrex_reckoning_export_preset(
    *,
    dataset_id: Optional[str] = None,
    quality_threshold: float = 0.65,
    eval_texts: Optional[List[str]] = None,
) -> DatasetPipeline:
    """Clean + dedup + quality gate for reckoning/correction Helox rows."""
    stages: List[Any] = [
        TextCleaningStage(config={"text_field": "text", "min_length": 8}),
        ExactDedupStage(config={"text_field": "text"}),
        DataValidationStage(
            config={"required_fields": ["instruction", "output", "text", "category"]}
        ),
    ]
    if eval_texts:
        stages.append(
            LeakageCheckStage(config={"text_field": "text", "eval_texts": eval_texts})
        )
    stages.extend(
        [
            QualityScoringStage(
                config={"quality_threshold": quality_threshold, "dataset_id": dataset_id}
            ),
            ManifestStage(config={"dataset_id": dataset_id or "cyrex-reckoning"}),
        ]
    )
    return DatasetPipeline(stages)


def cyrex_visual_grounding_preset(
    *,
    dataset_id: Optional[str] = None,
) -> DatasetPipeline:
    """Light pipeline for Elkedel live-scene visual grounding samples."""
    return DatasetPipeline(
        [
            TextCleaningStage(config={"text_field": "text", "min_length": 4}),
            ExactDedupStage(config={"text_field": "text"}),
            DataValidationStage(
                config={"required_fields": ["instruction", "output", "category"]}
            ),
            ManifestStage(config={"dataset_id": dataset_id or "elkedel-visual"}),
        ]
    )


def export_training_jsonl(
    records: List[Dict[str, Any]],
    output_path: str,
    *,
    preset: Optional[DatasetPipeline] = None,
) -> Dict[str, Any]:
    """Run preset (optional) and write JSONL; returns manifest stats."""
    normalized = batch_to_jsonl_records(records)
    if preset is not None:
        stage_result = preset.run(normalized)
        if stage_result.success and stage_result.processed_data is not None:
            payload = stage_result.processed_data.data
            if isinstance(payload, list):
                normalized = payload
    path = write_jsonl_records(output_path, normalized)
    return {
        "path": str(path),
        "record_count": len(normalized),
    }
