"""
Base classes for preprocessing pipeline stages.

Provides the PreprocessingStage ABC and related dataclasses that all
pipeline stages inherit from.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Default label ID range for classification models (31 categories: 0-30)
DEFAULT_MIN_LABEL_ID = 0
DEFAULT_MAX_LABEL_ID = 30


@dataclass
class ProcessedData:
    """
    Standardized output format for processed data.
    This class holds the result of any preprocessing stage.
    """
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    schema_version: Optional[str] = None


@dataclass
class ValidationResult:
    """
    Results from validation checks.
    This class holds the outcome of data validation operations.
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    quality_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class StageResult:
    """
    Results from stage execution.
    This class holds the complete result of running a preprocessing stage.
    """
    success: bool
    processed_data: Optional[ProcessedData] = None
    validation_result: Optional[ValidationResult] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None
    stage_name: Optional[str] = None


class PreprocessingStage(ABC):
    """
    Abstract base class for all preprocessing stages.
    All preprocessing stages must inherit from this class.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    def process(self, data: Any) -> StageResult:
        """Process the input data."""
        pass

    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """Validate the input data."""
        pass

    def get_dependencies(self) -> List[str]:
        """Get list of stage names this stage depends on."""
        return []

    def get_name(self) -> str:
        """Get the name of this stage."""
        return self.name

    def _extract_data_and_metadata(self, data: Any) -> tuple[Any, Dict[str, Any]]:
        """Extract actual data and metadata from ProcessedData or raw data."""
        if isinstance(data, ProcessedData):
            return data.data, data.metadata
        else:
            return data, {}

    def _extract_data(self, data: Any) -> Any:
        """Extract actual data from ProcessedData or raw data."""
        if isinstance(data, ProcessedData):
            return data.data
        else:
            return data

    def _process_items(
        self,
        actual_data: Any,
        process_func: Callable[[Any], Any],
        allow_empty: bool = False,
    ) -> Any:
        """Process data items, handling both lists and single items."""
        if actual_data is None:
            if allow_empty:
                return None
            raise ValueError(f"Cannot process None data in stage '{self.name}'")

        if isinstance(actual_data, list):
            if len(actual_data) == 0 and not allow_empty:
                raise ValueError(f"Cannot process empty list in stage '{self.name}'")
            return [process_func(item) for item in actual_data]
        else:
            return process_func(actual_data)

    def _check_data_not_none(self, actual_data: Any, context: str = "validate") -> None:
        """Standard check: None data always fails."""
        if actual_data is None:
            raise ValueError(f"Cannot {context} None data in stage '{self.name}'")

    def _validate_items(
        self,
        actual_data: Any,
        validate_func: Callable[[Any], tuple[List[str], List[str]]],
        empty_error: Optional[str] = None,
    ) -> tuple[List[str], List[str]]:
        """Validate data items using a diagnostic function."""
        errors = []
        warnings = []

        if actual_data == [] or actual_data == {}:
            error_msg = empty_error or f"Cannot validate empty data in stage '{self.name}'"
            errors.append(error_msg)
            return errors, warnings

        if isinstance(actual_data, list):
            for i, item in enumerate(actual_data):
                item_errors, item_warnings = validate_func(item)
                for err in item_errors:
                    errors.append(f"Item at index {i}: {err}")
                for warn in item_warnings:
                    warnings.append(f"Item at index {i}: {warn}")
        elif isinstance(actual_data, dict):
            item_errors, item_warnings = validate_func(actual_data)
            errors.extend(item_errors)
            warnings.extend(item_warnings)
        else:
            errors.append(
                f"Data must be a dictionary or list, got {type(actual_data).__name__}"
            )

        return errors, warnings

    def _create_validation_result(
        self,
        errors: List[str],
        warnings: List[str],
    ) -> ValidationResult:
        """Create a ValidationResult from errors and warnings."""
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _create_result(
        self,
        processed_data: Any,
        original_metadata: Dict[str, Any],
        metadata_updates: Dict[str, Any],
        success: bool = True,
        error: Optional[str] = None,
    ) -> StageResult:
        """Create a StageResult with processed data wrapped in ProcessedData."""
        if success:
            processed_data_obj = ProcessedData(
                data=processed_data,
                metadata={**original_metadata, **metadata_updates},
            )
            return StageResult(
                success=True,
                processed_data=processed_data_obj,
                stage_name=self.name,
            )
        else:
            return StageResult(
                success=False,
                error=error,
                stage_name=self.name,
            )
