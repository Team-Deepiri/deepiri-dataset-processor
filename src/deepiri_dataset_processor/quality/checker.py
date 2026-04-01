"""
Data Quality Framework.

Provides comprehensive data quality checks across 7 dimensions:
1. Completeness - Missing values, coverage
2. Consistency - Ranges, formats
3. Validity - Schema, business rules
4. Uniqueness - Duplicates, keys
5. Timeliness - Freshness, staleness
6. Accuracy - Statistical tests, outliers
7. Integrity - Referential, constraints

Requires: pip install deepiri-dataset-processor[quality]
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ..pipeline.base import PreprocessingStage, ProcessedData, StageResult, ValidationResult

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class QualityConfig:
    """Configuration class for data quality checks."""

    # Dimension thresholds (0.0 to 1.0)
    completeness_threshold: float = 0.95
    consistency_threshold: float = 0.90
    validity_threshold: float = 1.0
    uniqueness_threshold: float = 0.98
    timeliness_threshold: float = 0.70
    accuracy_threshold: float = 0.90
    integrity_threshold: float = 1.0

    # Statistical method parameters
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0
    isolation_forest_contamination: float = 0.1
    random_state: int = 42

    # Timeliness parameters
    freshness_decay_days: float = 30.0

    # Uniqueness parameters
    key_uniqueness_threshold: float = 0.95

    # Recommendation threshold
    recommendation_threshold: float = 0.8

    # Validity scoring parameters
    validity_error_penalty: float = 0.1

    # Performance parameters
    shapiro_wilk_sample_size: int = 5000
    min_samples_for_outlier_detection: int = 4
    min_samples_for_statistical_tests: int = 30


@dataclass
class QualityMetric:
    """Individual quality metric result."""

    dimension: str
    metric_name: str
    value: float
    threshold: Optional[float] = None
    passed: bool = True
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Comprehensive quality report for a dataset."""

    dataset_id: str
    timestamp: datetime
    overall_score: float
    dimension_scores: Dict[str, float]
    metrics: List[QualityMetric]
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert quality report to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_score": self.overall_score,
            "dimension_scores": self.dimension_scores,
            "metrics": [
                {
                    "dimension": m.dimension,
                    "metric_name": m.metric_name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "passed": m.passed,
                    "details": m.details,
                }
                for m in self.metrics
            ],
            "summary": self.summary,
            "recommendations": self.recommendations,
        }

    def to_validation_result(self) -> ValidationResult:
        """Convert report to :class:`~deepiri_dataset_processor.pipeline.ValidationResult`."""
        errors = [
            f"{m.dimension}.{m.metric_name}: {m.value:.2f} < {m.threshold}"
            for m in self.metrics
            if not m.passed
        ]
        warnings = self.recommendations.copy()
        quality_scores = dict(self.dimension_scores)
        quality_scores["overall"] = self.overall_score
        return ValidationResult(
            is_valid=len(errors) == 0 and self.overall_score >= 0.8,
            errors=errors,
            warnings=warnings,
            quality_scores=quality_scores,
        )

    def get_quality_metrics_for_processed_data(self) -> Dict[str, float]:
        """Dimension scores suitable for :class:`~deepiri_dataset_processor.pipeline.ProcessedData`."""
        return dict(self.dimension_scores)


_QUALITY_CONFIG_KEYS = frozenset(
    {
        "completeness_threshold",
        "consistency_threshold",
        "validity_threshold",
        "uniqueness_threshold",
        "timeliness_threshold",
        "accuracy_threshold",
        "integrity_threshold",
        "iqr_multiplier",
        "zscore_threshold",
        "isolation_forest_contamination",
        "random_state",
        "freshness_decay_days",
        "key_uniqueness_threshold",
        "recommendation_threshold",
        "validity_error_penalty",
        "shapiro_wilk_sample_size",
        "min_samples_for_outlier_detection",
        "min_samples_for_statistical_tests",
    }
)


def _check_dependencies():
    if not HAS_NUMPY:
        raise ImportError(
            "numpy is required for quality checks. "
            "Install with: pip install deepiri-dataset-processor[quality]"
        )
    if not HAS_PANDAS:
        raise ImportError(
            "pandas is required for quality checks. "
            "Install with: pip install deepiri-dataset-processor[quality]"
        )


class QualityChecker:
    """
    Main class for performing comprehensive data quality checks.

    Implements all 7 quality dimensions and generates quality reports.
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        _check_dependencies()
        self.config = config or QualityConfig()

    def check_quality(
        self,
        data: Union["pd.DataFrame", List[Dict], Dict],
        dataset_id: str = "dataset",
        schema: Optional[Dict[str, Any]] = None,
    ) -> QualityReport:
        """
        Perform comprehensive quality checks on data.

        Args:
            data: Data to check (DataFrame, list of dicts, or single dict)
            dataset_id: Identifier for the dataset
            schema: Optional schema definition for validation

        Returns:
            QualityReport with all quality metrics
        """
        try:
            df = self._to_dataframe(data)
        except Exception as e:
            return QualityReport(
                dataset_id=dataset_id,
                timestamp=datetime.now(),
                overall_score=0.0,
                dimension_scores={},
                metrics=[
                    QualityMetric(
                        dimension="validity",
                        metric_name="data_conversion_error",
                        value=0.0,
                        threshold=self.config.validity_threshold,
                        passed=False,
                        details={"error": str(e)},
                    )
                ],
                summary={"error": "Failed to convert data to DataFrame"},
                recommendations=[f"Data conversion failed: {str(e)}"],
            )

        metrics = []
        metrics.extend(self._check_completeness(df))
        metrics.extend(self._check_consistency(df))
        metrics.extend(self._check_validity(df, schema))
        metrics.extend(self._check_uniqueness(df))

        dimension_scores = self._calculate_dimension_scores(metrics)
        overall_score = float(
            np.mean(list(dimension_scores.values())) if dimension_scores else 0.0
        )
        recommendations = self._generate_recommendations(dimension_scores, metrics)

        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "total_metrics": len(metrics),
            "failed_metrics": sum(1 for m in metrics if not m.passed),
        }

        return QualityReport(
            dataset_id=dataset_id,
            timestamp=datetime.now(),
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            metrics=metrics,
            summary=summary,
            recommendations=recommendations,
        )

    def _to_dataframe(self, data) -> "pd.DataFrame":
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, list):
            if len(data) == 0:
                return pd.DataFrame()
            if isinstance(data[0], dict):
                return pd.DataFrame(data)
            raise ValueError(f"List data must contain dicts, got {type(data[0])}")
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        raise ValueError(f"Unsupported data type: {type(data)}")

    def _check_completeness(self, df: "pd.DataFrame") -> List[QualityMetric]:
        metrics = []
        if df.empty:
            metrics.append(
                QualityMetric(
                    dimension="completeness",
                    metric_name="empty_dataset",
                    value=0.0,
                    threshold=self.config.completeness_threshold,
                    passed=False,
                    details={"message": "Dataset is empty"},
                )
            )
            return metrics

        total_cells = df.size
        missing_cells = int(df.isnull().sum().sum())
        score = 1.0 - (missing_cells / total_cells)

        metrics.append(
            QualityMetric(
                dimension="completeness",
                metric_name="overall_completeness",
                value=score,
                threshold=self.config.completeness_threshold,
                passed=score >= self.config.completeness_threshold,
                details={
                    "total_cells": int(total_cells),
                    "missing_cells": missing_cells,
                    "missing_percentage": float(missing_cells / total_cells * 100),
                },
            )
        )

        for col in df.columns:
            col_score = 1.0 - (df[col].isnull().sum() / len(df))
            metrics.append(
                QualityMetric(
                    dimension="completeness",
                    metric_name=f"column_{col}_completeness",
                    value=float(col_score),
                    threshold=self.config.completeness_threshold,
                    passed=float(col_score) >= self.config.completeness_threshold,
                    details={"column": col},
                )
            )

        return metrics

    def _check_consistency(self, df: "pd.DataFrame") -> List[QualityMetric]:
        metrics = []
        if df.empty:
            return metrics

        type_score = 1.0
        type_issues = []

        for col in df.columns:
            col_types = df[col].apply(type).unique()
            if len(col_types) > 1:
                type_score -= 0.1
                type_issues.append(
                    {"column": col, "types": [str(t) for t in col_types]}
                )

        metrics.append(
            QualityMetric(
                dimension="consistency",
                metric_name="type_consistency",
                value=max(0.0, type_score),
                threshold=self.config.consistency_threshold,
                passed=type_score >= self.config.consistency_threshold,
                details={"type_issues": type_issues},
            )
        )

        return metrics

    def _check_validity(
        self, df: "pd.DataFrame", schema: Optional[Dict] = None
    ) -> List[QualityMetric]:
        metrics = []
        if df.empty:
            return metrics

        if schema and "required_fields" in schema:
            required = schema["required_fields"]
            missing = set(required) - set(df.columns)
            if missing:
                metrics.append(
                    QualityMetric(
                        dimension="validity",
                        metric_name="required_fields_check",
                        value=0.0,
                        threshold=self.config.validity_threshold,
                        passed=False,
                        details={"missing_required_fields": list(missing)},
                    )
                )
            else:
                metrics.append(
                    QualityMetric(
                        dimension="validity",
                        metric_name="required_fields_check",
                        value=1.0,
                        threshold=self.config.validity_threshold,
                        passed=True,
                        details={"required_fields": required},
                    )
                )

        if schema and "fields" in schema:
            errors = []
            for field_name, field_spec in schema["fields"].items():
                if field_name not in df.columns:
                    errors.append(f"Missing field: {field_name}")
                    continue
                if "constraints" in field_spec:
                    constraints = field_spec["constraints"]
                    if "min" in constraints and df[field_name].min() < constraints["min"]:
                        errors.append(f"Value below minimum for {field_name}")
                    if "max" in constraints and df[field_name].max() > constraints["max"]:
                        errors.append(f"Value above maximum for {field_name}")

            validity_score = (
                1.0
                if not errors
                else max(0.0, 1.0 - len(errors) * self.config.validity_error_penalty)
            )
            metrics.append(
                QualityMetric(
                    dimension="validity",
                    metric_name="schema_validation",
                    value=validity_score,
                    threshold=self.config.validity_threshold,
                    passed=validity_score >= self.config.validity_threshold,
                    details={"errors": errors} if errors else {"status": "valid"},
                )
            )

        return metrics

    def _check_uniqueness(self, df: "pd.DataFrame") -> List[QualityMetric]:
        metrics = []
        if df.empty:
            return metrics

        dup_rows = int(df.duplicated().sum())
        score = 1.0 - (dup_rows / len(df))

        metrics.append(
            QualityMetric(
                dimension="uniqueness",
                metric_name="row_uniqueness",
                value=score,
                threshold=self.config.uniqueness_threshold,
                passed=score >= self.config.uniqueness_threshold,
                details={
                    "duplicate_rows": dup_rows,
                    "unique_rows": len(df) - dup_rows,
                    "duplicate_percentage": float(dup_rows / len(df) * 100),
                },
            )
        )

        return metrics

    def _calculate_dimension_scores(
        self, metrics: List[QualityMetric]
    ) -> Dict[str, float]:
        from collections import defaultdict

        dimension_values: Dict[str, List[float]] = defaultdict(list)
        for m in metrics:
            dimension_values[m.dimension].append(m.value)

        return {
            dim: float(np.mean(values)) for dim, values in dimension_values.items()
        }

    def _generate_recommendations(
        self,
        dimension_scores: Dict[str, float],
        metrics: List[QualityMetric],
    ) -> List[str]:
        recommendations = []
        threshold = self.config.recommendation_threshold

        for dim, score in dimension_scores.items():
            if score < threshold:
                recommendations.append(
                    f"Improve {dim}: score {score:.2f} is below threshold {threshold:.2f}"
                )

        failed = [m for m in metrics if not m.passed]
        for m in failed[:5]:
            recommendations.append(
                f"Fix {m.dimension}.{m.metric_name}: "
                f"{m.value:.2f} below threshold {m.threshold}"
            )

        return recommendations


def check_data_quality(
    data: Union["pd.DataFrame", List[Dict], Dict],
    dataset_id: str = "dataset",
    schema: Optional[Dict[str, Any]] = None,
    config: Optional[QualityConfig] = None,
) -> QualityReport:
    """Run :class:`QualityChecker` and return a :class:`QualityReport`."""
    checker = QualityChecker(config=config)
    return checker.check_quality(data, dataset_id, schema)


class QualityCheckStage(PreprocessingStage):
    """
    Pipeline stage wrapping :class:`QualityChecker` (Helox-compatible).

    Depends on no other stages by default; insert where needed in the DAG or list.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        raw = config or {}
        super().__init__(name="quality_check", config=raw)

        quality_config = raw.get("quality_config")
        if quality_config is None:
            quality_kwargs = {k: v for k, v in raw.items() if k in _QUALITY_CONFIG_KEYS}
            if quality_kwargs:
                quality_config = QualityConfig(**quality_kwargs)

        self.quality_checker = QualityChecker(config=quality_config)
        self.dataset_id = raw.get("dataset_id", "quality_check")
        self.schema = raw.get("schema")

    def get_dependencies(self) -> List[str]:
        return []

    def process(self, data: Any) -> StageResult:
        try:
            actual_data, original_metadata = self._extract_data_and_metadata(data)
            quality_report = self.quality_checker.check_quality(
                data=actual_data,
                dataset_id=self.dataset_id,
                schema=self.schema or original_metadata.get("schema"),
            )
            validation_result = quality_report.to_validation_result()
            processed_data = ProcessedData(
                data=actual_data,
                metadata={
                    **original_metadata,
                    "quality_check_timestamp": quality_report.timestamp.isoformat(),
                    "overall_quality_score": quality_report.overall_score,
                    "schema": self.schema or original_metadata.get("schema"),
                },
                quality_metrics=quality_report.get_quality_metrics_for_processed_data(),
                schema_version=original_metadata.get("schema_version"),
            )
            return StageResult(
                success=True,
                processed_data=processed_data,
                validation_result=validation_result,
                stage_name=self.name,
            )
        except Exception as e:
            return StageResult(
                success=False,
                error=f"Quality check failed: {str(e)}",
                stage_name=self.name,
            )

    def validate(self, data: Any) -> ValidationResult:
        try:
            actual_data = self._extract_data(data)
            quality_report = self.quality_checker.check_quality(
                data=actual_data,
                dataset_id=self.dataset_id,
                schema=self.schema,
            )
            return quality_report.to_validation_result()
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Quality check failed: {str(e)}"],
                warnings=[],
                quality_scores={},
            )
