"""
Concrete preprocessing stage implementations.

Provides DataCleaningStage, DataValidationStage, DataTransformationStage,
and other stages for common data preprocessing operations.
"""

from typing import Any, Dict, List, Optional
from .base import (
    PreprocessingStage,
    StageResult,
    ProcessedData,
    ValidationResult,
    DEFAULT_MIN_LABEL_ID,
    DEFAULT_MAX_LABEL_ID,
)


class DataCleaningStage(PreprocessingStage):
    """Stage for cleaning data: removing spaces, nulls, and duplicates."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="data_cleaning", config=config)

    def get_dependencies(self) -> List[str]:
        return ["data_loading"]

    def process(self, data: Any) -> StageResult:
        try:
            actual_data, original_metadata = self._extract_data_and_metadata(data)
            cleaned_items = self._process_items(actual_data, self._clean_single_item)

            if isinstance(cleaned_items, list):
                cleaned_data = self._remove_duplicates(cleaned_items)
            else:
                cleaned_data = cleaned_items

            return self._create_result(
                processed_data=cleaned_data,
                original_metadata=original_metadata,
                metadata_updates={"cleaned": True, "cleaning_stage": self.name},
            )
        except Exception as e:
            return self._create_result(
                processed_data=None,
                original_metadata={},
                metadata_updates={},
                success=False,
                error=f"Data cleaning failed: {str(e)}",
            )

    def _clean_single_item(self, item: Dict) -> Dict:
        if not isinstance(item, dict):
            return item
        cleaned = item.copy()
        if "text" in cleaned and isinstance(cleaned["text"], str):
            cleaned["text"] = " ".join(cleaned["text"].split())
        cleaned = {k: v for k, v in cleaned.items() if v is not None and v != ""}
        return cleaned

    def _remove_duplicates(self, items: List[Dict]) -> List[Dict]:
        seen = []
        unique_items = []
        for item in items:
            item_tuple = tuple(sorted(item.items()))
            if item_tuple not in seen:
                seen.append(item_tuple)
                unique_items.append(item)
        return unique_items

    def validate(self, data: Any) -> ValidationResult:
        errors = []
        warnings = []
        actual_data = self._extract_data(data)
        try:
            self._check_data_not_none(actual_data, context="validate")
        except ValueError as e:
            errors.append(str(e))
            return self._create_validation_result(errors, warnings)

        item_errors, item_warnings = self._validate_items(
            actual_data,
            self._validate_cleaning_diagnostics,
            empty_error=f"Cannot validate empty data in stage '{self.name}'",
        )
        errors.extend(item_errors)
        warnings.extend(item_warnings)
        return self._create_validation_result(errors, warnings)

    def _validate_cleaning_diagnostics(self, item: Dict) -> tuple[List[str], List[str]]:
        errors = []
        warnings = []
        if not isinstance(item, dict):
            errors.append(f"Item must be a dictionary, got {type(item).__name__}")
            return errors, warnings
        if "text" not in item:
            warnings.append("Data missing 'text' field - may not be cleanable")
        else:
            if not isinstance(item["text"], str):
                warnings.append("'text' field is not a string")
            elif item["text"].strip() == "":
                warnings.append("'text' field is empty or only whitespace")
        return errors, warnings


class DataValidationStage(PreprocessingStage):
    """Stage for validating data has required fields and meets criteria."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="data_validation", config=config)
        self.required_fields = (
            config.get("required_fields", ["text", "label"]) if config else ["text", "label"]
        )

    def get_dependencies(self) -> List[str]:
        return ["data_cleaning"]

    def process(self, data: Any) -> StageResult:
        try:
            actual_data, original_metadata = self._extract_data_and_metadata(data)
            validated_data = self._process_items(actual_data, self._validate_single_item)
            return self._create_result(
                processed_data=validated_data,
                original_metadata=original_metadata,
                metadata_updates={"validated": True, "validation_stage": self.name},
            )
        except Exception as e:
            return self._create_result(
                processed_data=None,
                original_metadata={},
                metadata_updates={},
                success=False,
                error=f"Data validation failed: {str(e)}",
            )

    def _validate_single_item(self, item: Dict) -> Dict:
        if not isinstance(item, dict):
            raise ValueError(f"Data item must be a dictionary, got {type(item).__name__}")
        for fld in self.required_fields:
            if fld not in item:
                raise ValueError(f"Required field '{fld}' is missing")
        if "text" in item:
            if not isinstance(item["text"], str):
                raise ValueError("'text' field must be a string")
            if item["text"].strip() == "":
                raise ValueError("'text' field cannot be empty")
        return item

    def validate(self, data: Any) -> ValidationResult:
        errors = []
        warnings = []
        actual_data = self._extract_data(data)
        try:
            self._check_data_not_none(actual_data, context="validate")
        except ValueError as e:
            errors.append(str(e))
            return self._create_validation_result(errors, warnings)

        item_errors, item_warnings = self._validate_items(
            actual_data,
            self._validate_item_diagnostics,
            empty_error="Data is empty - nothing to validate",
        )
        errors.extend(item_errors)
        warnings.extend(item_warnings)
        return self._create_validation_result(errors, warnings)

    def _validate_item_diagnostics(self, item: Dict) -> tuple[List[str], List[str]]:
        errors = []
        warnings = []
        if not isinstance(item, dict):
            errors.append(f"Item must be a dictionary, got {type(item).__name__}")
            return errors, warnings
        for fld in self.required_fields:
            if fld not in item:
                errors.append(f"Required field '{fld}' is missing")
        if "text" in item:
            if not isinstance(item["text"], str):
                errors.append("'text' field must be a string")
            elif item["text"].strip() == "":
                errors.append("'text' field cannot be empty")
        return errors, warnings


class DataTransformationStage(PreprocessingStage):
    """Stage for transforming data: normalization and final transformations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="data_transformation", config=config)
        self.normalize_lowercase = config.get("normalize_lowercase", True) if config else True
        self.normalize_trim = config.get("normalize_trim", True) if config else True

    def get_dependencies(self) -> List[str]:
        return ["label_validation"]

    def process(self, data: Any) -> StageResult:
        try:
            actual_data, original_metadata = self._extract_data_and_metadata(data)
            transformed_data = self._process_items(actual_data, self._transform_single_item)
            return self._create_result(
                processed_data=transformed_data,
                original_metadata=original_metadata,
                metadata_updates={"transformed": True, "transformation_stage": self.name},
            )
        except Exception as e:
            return self._create_result(
                processed_data=None,
                original_metadata={},
                metadata_updates={},
                success=False,
                error=f"Data transformation failed: {str(e)}",
            )

    def _transform_single_item(self, item: Dict) -> Dict:
        if not isinstance(item, dict):
            raise ValueError(f"Data item must be a dictionary, got {type(item).__name__}")
        transformed_item = item.copy()
        if "text" in transformed_item and isinstance(transformed_item["text"], str):
            text = transformed_item["text"]
            if self.normalize_trim:
                text = text.strip()
            if self.normalize_lowercase:
                text = text.lower()
            transformed_item["text"] = text
        return transformed_item

    def validate(self, data: Any) -> ValidationResult:
        errors = []
        warnings = []
        actual_data = self._extract_data(data)
        try:
            self._check_data_not_none(actual_data, context="validate")
        except ValueError as e:
            errors.append(str(e))
            return self._create_validation_result(errors, warnings)

        item_errors, item_warnings = self._validate_items(
            actual_data,
            self._validate_transformation_diagnostics,
            empty_error="Data is empty - nothing to transform",
        )
        errors.extend(item_errors)
        warnings.extend(item_warnings)
        return self._create_validation_result(errors, warnings)

    def _validate_transformation_diagnostics(self, item: Dict) -> tuple[List[str], List[str]]:
        errors = []
        warnings = []
        if not isinstance(item, dict):
            errors.append(f"Item must be a dictionary, got {type(item).__name__}")
            return errors, warnings
        if "text" not in item:
            warnings.append("'text' field is missing - nothing to normalize")
            return errors, warnings
        text = item["text"]
        if not isinstance(text, str):
            warnings.append(
                f"'text' field is not a string (got {type(text).__name__}) - cannot normalize"
            )
        elif text.strip() == "":
            warnings.append("'text' field is empty or only whitespace")
        return errors, warnings
