"""
Concrete preprocessing stage implementations.

Provides stages for loading, cleaning, validation, label routing, label checks,
and transformation for common dataset preprocessing pipelines.
"""

from typing import Any, Dict, List, Optional

from .base import (
    DEFAULT_MAX_LABEL_ID,
    DEFAULT_MIN_LABEL_ID,
    PreprocessingStage,
    StageResult,
    ValidationResult,
)


class DataLoadingStage(PreprocessingStage):
    """Load data from sources (JSONL, JSON, etc.); pass-through when data is provided."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="data_loading", config=config)
        self.source = config.get("source") if config else None
        self.format = config.get("format", "jsonl") if config else "jsonl"

    def get_dependencies(self) -> List[str]:
        return []

    def process(self, data: Any) -> StageResult:
        try:
            if data is not None:
                actual_data, original_metadata = self._extract_data_and_metadata(data)
                metadata_updates = {"source": self.source, "format": self.format}
                return self._create_result(
                    processed_data=actual_data,
                    original_metadata=original_metadata,
                    metadata_updates=metadata_updates,
                )
            raise NotImplementedError(
                "Data loading from source not yet implemented. "
                "Please provide initial_data to the pipeline."
            )
        except Exception as e:
            return self._create_result(
                processed_data=None,
                original_metadata={},
                metadata_updates={},
                success=False,
                error=f"Data loading failed: {str(e)}",
            )

    def validate(self, data: Any) -> ValidationResult:
        errors = []
        warnings = []
        if not self.source:
            warnings.append("No data source configured")
        supported_formats = ["jsonl", "json", "csv", "parquet"]
        if self.format not in supported_formats:
            errors.append(
                f"Unsupported format: {self.format}. Supported: {supported_formats}"
            )
        return self._create_validation_result(errors, warnings)


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
            elif len(item["text"].strip()) < 3:
                warnings.append("'text' field is very short (less than 3 characters)")
        if "label" in item:
            if item["label"] is None:
                warnings.append("'label' field is None")
            elif isinstance(item["label"], str) and item["label"].strip() == "":
                warnings.append("'label' field is empty string")
        return errors, warnings


class DataRoutingStage(PreprocessingStage):
    """Map labels to integer IDs for training; adds ``label_id``."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="data_routing", config=config)
        self.label_mapping = config.get("label_mapping", {}) if config else {}

    def get_dependencies(self) -> List[str]:
        return ["data_validation"]

    def process(self, data: Any) -> StageResult:
        try:
            actual_data, original_metadata = self._extract_data_and_metadata(data)
            routed_data = self._process_items(actual_data, self._map_label_to_id)
            return self._create_result(
                processed_data=routed_data,
                original_metadata=original_metadata,
                metadata_updates={"routed": True, "routing_stage": self.name},
            )
        except Exception as e:
            return self._create_result(
                processed_data=None,
                original_metadata={},
                metadata_updates={},
                success=False,
                error=f"Data routing failed: {str(e)}",
            )

    def _map_label_to_id(self, item: Dict) -> Dict:
        if not isinstance(item, dict):
            raise ValueError(f"Data item must be a dictionary, got {type(item).__name__}")
        routed_item = item.copy()
        if "label" not in routed_item:
            raise ValueError("Required field 'label' is missing")
        label = routed_item["label"]
        if isinstance(label, str):
            label_id = self._get_label_id(label)
            routed_item["label_id"] = label_id
        elif isinstance(label, (int, float)):
            label_id = int(label)
            if label_id < DEFAULT_MIN_LABEL_ID or label_id > DEFAULT_MAX_LABEL_ID:
                raise ValueError(
                    f"Label ID {label_id} is out of valid range "
                    f"[{DEFAULT_MIN_LABEL_ID}, {DEFAULT_MAX_LABEL_ID}]"
                )
            routed_item["label_id"] = label_id
        else:
            raise ValueError(f"Label must be string or numeric, got {type(label).__name__}")
        return routed_item

    def _get_label_id(self, label_str: str) -> int:
        label_str = label_str.strip().lower()
        if label_str in self.label_mapping:
            return self.label_mapping[label_str]
        raise ValueError(
            f"Unknown label '{label_str}'. Valid labels: {list(self.label_mapping.keys())}"
        )

    def validate(self, data: Any) -> ValidationResult:
        errors = []
        warnings = []
        actual_data = self._extract_data(data)
        try:
            self._check_data_not_none(actual_data, context="validate")
        except ValueError as e:
            errors.append(str(e))
            return self._create_validation_result(errors, warnings)
        if not self.label_mapping:
            warnings.append("Label mapping is empty - string labels cannot be mapped")
        item_errors, item_warnings = self._validate_items(
            actual_data,
            self._validate_label_routing,
            empty_error="Data is empty - nothing to route",
        )
        errors.extend(item_errors)
        warnings.extend(item_warnings)
        return self._create_validation_result(errors, warnings)

    def _validate_label_routing(self, item: Dict) -> tuple[List[str], List[str]]:
        errors = []
        warnings = []
        if not isinstance(item, dict):
            errors.append(f"Item must be a dictionary, got {type(item).__name__}")
            return errors, warnings
        if "label" not in item:
            errors.append("Required field 'label' is missing for routing")
            return errors, warnings
        label = item["label"]
        if isinstance(label, str):
            label_str = label.strip().lower()
            if label_str == "":
                errors.append("Label is empty string")
            elif not self.label_mapping:
                errors.append(f"Label '{label}' cannot be mapped - label mapping is empty")
            elif label_str not in self.label_mapping:
                valid_labels = list(self.label_mapping.keys())
                errors.append(
                    f"Label '{label}' not found in mapping. Valid labels: {valid_labels}"
                )
        elif isinstance(label, (int, float)):
            label_id = int(label)
            if label_id < DEFAULT_MIN_LABEL_ID or label_id > DEFAULT_MAX_LABEL_ID:
                errors.append(
                    f"Numeric label {label_id} is out of valid range "
                    f"[{DEFAULT_MIN_LABEL_ID}, {DEFAULT_MAX_LABEL_ID}]"
                )
            else:
                warnings.append(
                    f"Label is already numeric ({label_id}) - routing will validate range only"
                )
        elif label is None:
            errors.append("Label is None - cannot be routed")
        else:
            errors.append(f"Label must be string or numeric, got {type(label).__name__}")
        return errors, warnings


class LabelValidationStage(PreprocessingStage):
    """Validate ``label_id`` after routing (range and type)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="label_validation", config=config)
        self.min_label_id = (
            config.get("min_label_id", DEFAULT_MIN_LABEL_ID) if config else DEFAULT_MIN_LABEL_ID
        )
        self.max_label_id = (
            config.get("max_label_id", DEFAULT_MAX_LABEL_ID) if config else DEFAULT_MAX_LABEL_ID
        )

    def get_dependencies(self) -> List[str]:
        return ["data_routing"]

    def process(self, data: Any) -> StageResult:
        try:
            actual_data, original_metadata = self._extract_data_and_metadata(data)
            validated_data = self._process_items(actual_data, self._validate_label)
            return self._create_result(
                processed_data=validated_data,
                original_metadata=original_metadata,
                metadata_updates={
                    "label_validated": True,
                    "label_validation_stage": self.name,
                },
            )
        except Exception as e:
            return self._create_result(
                processed_data=None,
                original_metadata={},
                metadata_updates={},
                success=False,
                error=f"Label validation failed: {str(e)}",
            )

    def _validate_label(self, item: Dict) -> Dict:
        if not isinstance(item, dict):
            raise ValueError(f"Data item must be a dictionary, got {type(item).__name__}")
        if "label_id" not in item:
            raise ValueError(
                "Required field 'label_id' is missing (should be created by routing stage)"
            )
        label_id = item["label_id"]
        if not isinstance(label_id, (int, float)):
            raise ValueError(f"label_id must be numeric, got {type(label_id).__name__}")
        label_id = int(label_id)
        if label_id < self.min_label_id or label_id > self.max_label_id:
            raise ValueError(
                f"Label ID {label_id} is out of valid range "
                f"[{self.min_label_id}, {self.max_label_id}]"
            )
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
            self._validate_label_diagnostics,
            empty_error="Data is empty - nothing to validate",
        )
        errors.extend(item_errors)
        warnings.extend(item_warnings)
        return self._create_validation_result(errors, warnings)

    def _validate_label_diagnostics(self, item: Dict) -> tuple[List[str], List[str]]:
        errors = []
        warnings = []
        if not isinstance(item, dict):
            errors.append(f"Item must be a dictionary, got {type(item).__name__}")
            return errors, warnings
        if "label_id" not in item:
            errors.append(
                "Required field 'label_id' is missing (should be created by routing stage)"
            )
            return errors, warnings
        label_id = item["label_id"]
        if not isinstance(label_id, (int, float)):
            errors.append(f"label_id must be numeric, got {type(label_id).__name__}")
            return errors, warnings
        label_id = int(label_id)
        if label_id < self.min_label_id or label_id > self.max_label_id:
            errors.append(
                f"Label ID {label_id} is out of valid range "
                f"[{self.min_label_id}, {self.max_label_id}]"
            )
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
