"""
Dataset pipeline orchestrator.

Chains preprocessing stages and runs them in sequence with validation.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from .base import PreprocessingStage, ProcessedData, StageResult, ValidationResult

logger = logging.getLogger(__name__)


class DatasetPipeline:
    """
    Composable pipeline that chains preprocessing stages.

    Example::

        pipeline = DatasetPipeline([
            DataCleaningStage(),
            DataValidationStage(config={"required_fields": ["text"]}),
            DataTransformationStage(),
        ])
        result = pipeline.run(data)
    """

    def __init__(self, stages: List[PreprocessingStage]):
        self.stages = stages

    def run(self, data: Any) -> StageResult:
        """Run all stages in sequence, passing output of each to the next."""
        current_data = data
        result: StageResult | None = None

        for stage in self.stages:
            start = time.perf_counter()
            result = stage.process(current_data)
            elapsed = time.perf_counter() - start
            result.execution_time = elapsed

            if not result.success:
                logger.error(
                    f"Pipeline failed at stage '{stage.get_name()}': {result.error}"
                )
                return result

            current_data = result.processed_data
            logger.debug(
                f"Stage '{stage.get_name()}' completed in {elapsed:.3f}s"
            )

        if result is None:
            processed = (
                current_data
                if isinstance(current_data, ProcessedData)
                else ProcessedData(data=current_data)
            )
            return StageResult(success=True, processed_data=processed)

        return result

    def validate_all(self, data: Any) -> List[ValidationResult]:
        """Run validation on all stages without processing."""
        results = []
        current_data = data

        for stage in self.stages:
            validation = stage.validate(current_data)
            results.append(validation)

            if not validation.is_valid:
                logger.warning(
                    f"Validation failed at stage '{stage.get_name()}': "
                    f"{validation.errors}"
                )

        return results

    def get_stage_names(self) -> List[str]:
        """Get ordered list of stage names."""
        return [s.get_name() for s in self.stages]

    def run_streaming(
        self,
        path: Any,
        *,
        chunk_size: int = 1000,
        output_path: Optional[Any] = None,
    ) -> StageResult:
        """Process a JSONL file in chunks without loading the full corpus."""
        from pathlib import Path

        from ..streaming.chunked_jsonl import iter_jsonl_chunks, write_jsonl_records

        source = Path(path)
        all_records: List[Dict[str, Any]] = []
        last_result: StageResult | None = None

        for chunk in iter_jsonl_chunks(source, chunk_size=chunk_size):
            result = self.run(chunk)
            last_result = result
            if not result.success:
                return result
            chunk_data = result.processed_data
            if hasattr(chunk_data, "data"):
                all_records.extend(chunk_data.data)
            elif isinstance(chunk_data, list):
                all_records.extend(chunk_data)

        if output_path:
            write_jsonl_records(output_path, all_records)

        from .base import ProcessedData

        processed = ProcessedData(data=all_records, metadata={})
        if last_result and last_result.processed_data:
            processed.metadata = getattr(last_result.processed_data, "metadata", {})

        return StageResult(success=True, processed_data=processed)
