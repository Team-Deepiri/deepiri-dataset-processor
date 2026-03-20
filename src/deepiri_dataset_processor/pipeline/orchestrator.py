"""
Dataset pipeline orchestrator.

Chains preprocessing stages and runs them in sequence with validation.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from .base import PreprocessingStage, StageResult, ValidationResult

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
