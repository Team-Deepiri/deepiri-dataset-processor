"""Pipeline stages that wrap core dataset-processor modules."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..cleaning.text_cleaner import TextCleaner
from ..deduplication.exact_dedup import ExactDeduplicator
from ..deduplication.semantic_dedup import SemanticDeduplicationEngine
from ..manifest import build_manifest
from ..quality.checker import QualityChecker
from ..safety.leakage_detector import DataLeakageDetector
from ..versioning.filesystem import DatasetVersioningSystem
from .base import PreprocessingStage, StageResult, ValidationResult
from .stages import DataValidationStage


def _to_records(data: Any) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from .base import ProcessedData

    if isinstance(data, ProcessedData):
        actual = data.data
        metadata = dict(data.metadata or {})
        if isinstance(actual, list):
            return actual, metadata
        if isinstance(actual, ProcessedData):
            return _to_records(actual)
    if hasattr(data, "data"):
        metadata = getattr(data, "metadata", {}) or {}
        actual = data.data
    elif isinstance(data, dict) and "data" in data:
        metadata = data.get("metadata", {})
        actual = data["data"]
    else:
        metadata = {}
        actual = data

    if isinstance(actual, list):
        return actual, metadata
    if isinstance(actual, Path) and actual.suffix == ".jsonl":
        records = []
        with open(actual, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records, metadata
    if isinstance(actual, Path) and actual.is_dir():
        records = []
        for file_path in sorted(actual.rglob("*.jsonl")):
            with open(file_path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records, metadata
    raise ValueError(f"Unsupported pipeline data type: {type(actual)}")


def _wrap_records(records: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Any:
    from .base import ProcessedData

    return ProcessedData(data=records, metadata=metadata)


class TextCleaningStage(PreprocessingStage):
    """Wrap TextCleaner for pipeline use."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="text_cleaning", config=config or {})
        self.text_field = self.config.get("text_field", "text")
        self.cleaner = TextCleaner(
            min_length=self.config.get("min_length", 50),
            max_urls=self.config.get("max_urls", 5),
        )

    def process(self, data: Any) -> StageResult:
        try:
            records, metadata = _to_records(data)
            cleaned: List[Dict[str, Any]] = []
            for record in records:
                if not isinstance(record, dict):
                    continue
                item = dict(record)
                if self.text_field in item and isinstance(item[self.text_field], str):
                    result = self.cleaner.clean(item[self.text_field])
                    if result is None:
                        continue
                    item[self.text_field] = result
                cleaned.append(item)
            metadata["text_cleaned"] = True
            return self._create_result(
                processed_data=cleaned,
                original_metadata=metadata,
                metadata_updates={"text_cleaned": True},
            )
        except Exception as exc:
            return self._create_result(
                processed_data=None,
                original_metadata={},
                metadata_updates={},
                success=False,
                error=str(exc),
            )

    def validate(self, data: Any) -> ValidationResult:
        return DataValidationStage(
            config={"required_fields": [self.text_field]}
        ).validate(data)


class ExactDedupStage(PreprocessingStage):
    """Exact deduplication on a text field."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="exact_dedup", config=config or {})
        self.text_field = self.config.get("text_field", "text")
        self.dedup = ExactDeduplicator()

    def process(self, data: Any) -> StageResult:
        try:
            records, metadata = _to_records(data)
            texts = [
                str(r.get(self.text_field, ""))
                for r in records
                if isinstance(r, dict) and self.text_field in r
            ]
            unique_texts = self.dedup.filter_duplicates(texts)
            unique_set = set(unique_texts)
            filtered = [
                r
                for r in records
                if isinstance(r, dict) and str(r.get(self.text_field, "")) in unique_set
            ]
            seen: set[str] = set()
            deduped: List[Dict[str, Any]] = []
            for record in filtered:
                text = str(record.get(self.text_field, ""))
                if text in seen:
                    continue
                seen.add(text)
                deduped.append(record)
            metadata["exact_dedup_removed"] = len(records) - len(deduped)
            return self._create_result(
                processed_data=deduped,
                original_metadata=metadata,
                metadata_updates={"exact_dedup": True},
            )
        except Exception as exc:
            return self._create_result(
                processed_data=None,
                original_metadata={},
                metadata_updates={},
                success=False,
                error=str(exc),
            )

    def validate(self, data: Any) -> ValidationResult:
        return ValidationResult(is_valid=True)


class SemanticDedupStage(PreprocessingStage):
    """Semantic deduplication using sentence-transformers when available."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="semantic_dedup", config=config or {})
        self.text_field = self.config.get("text_field", "text")
        self.engine = SemanticDeduplicationEngine(
            similarity_threshold=self.config.get("similarity_threshold", 0.95),
            embedding_model=self.config.get("embedding_model"),
            use_lsh=self.config.get("use_lsh", True),
        )

    def process(self, data: Any) -> StageResult:
        try:
            records, metadata = _to_records(data)
            texts = [str(r.get(self.text_field, "")) for r in records]
            unique_texts = self.engine.filter_duplicates(texts)
            unique_set = set(unique_texts)
            filtered = [
                r for r in records if str(r.get(self.text_field, "")) in unique_set
            ]
            seen: set[str] = set()
            deduped: List[Dict[str, Any]] = []
            for record in filtered:
                text = str(record.get(self.text_field, ""))
                if text in seen:
                    continue
                seen.add(text)
                deduped.append(record)
            metadata["semantic_dedup_removed"] = len(records) - len(deduped)
            return self._create_result(
                processed_data=deduped,
                original_metadata=metadata,
                metadata_updates={"semantic_dedup": True},
            )
        except Exception as exc:
            return self._create_result(
                processed_data=None,
                original_metadata={},
                metadata_updates={},
                success=False,
                error=str(exc),
            )

    def validate(self, data: Any) -> ValidationResult:
        return ValidationResult(is_valid=True)


class LeakageCheckStage(PreprocessingStage):
    """Train/eval leakage detection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="leakage_check", config=config or {})
        self.text_field = self.config.get("text_field", "text")
        self.eval_texts = self.config.get("eval_texts", [])
        self.detector = DataLeakageDetector(
            overlap_threshold=self.config.get("overlap_threshold", 0.8)
        )

    def process(self, data: Any) -> StageResult:
        try:
            records, metadata = _to_records(data)
            train_texts = [str(r.get(self.text_field, "")) for r in records]
            report = self.detector.detect_train_eval_contamination(
                train_texts, self.eval_texts
            )
            metadata["leakage_report"] = report
            if report.get("contamination_detected"):
                return self._create_result(
                    processed_data=None,
                    original_metadata=metadata,
                    metadata_updates={},
                    success=False,
                    error=f"Leakage detected: {report}",
                )
            return self._create_result(
                processed_data=records,
                original_metadata=metadata,
                metadata_updates={"leakage_checked": True},
            )
        except Exception as exc:
            return self._create_result(
                processed_data=None,
                original_metadata={},
                metadata_updates={},
                success=False,
                error=str(exc),
            )

    def validate(self, data: Any) -> ValidationResult:
        return ValidationResult(is_valid=True)


class QualityScoringStage(PreprocessingStage):
    """Run QualityChecker and gate on threshold."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="quality_scoring", config=config or {})
        self.threshold = self.config.get("quality_threshold", 0.8)
        self.dataset_id = self.config.get("dataset_id", "dataset")
        self.checker = QualityChecker()

    def process(self, data: Any) -> StageResult:
        try:
            records, metadata = _to_records(data)
            report = self.checker.check_quality(records, dataset_id=self.dataset_id)
            metadata["quality_report"] = report.to_dict()
            if report.overall_score < self.threshold:
                return self._create_result(
                    processed_data=None,
                    original_metadata=metadata,
                    metadata_updates={},
                    success=False,
                    error=f"Quality score {report.overall_score:.2f} below {self.threshold}",
                )
            return self._create_result(
                processed_data=records,
                original_metadata=metadata,
                metadata_updates={"quality_passed": True},
            )
        except Exception as exc:
            return self._create_result(
                processed_data=None,
                original_metadata={},
                metadata_updates={},
                success=False,
                error=str(exc),
            )

    def validate(self, data: Any) -> ValidationResult:
        return ValidationResult(is_valid=True)


class ManifestStage(PreprocessingStage):
    """Write dataset manifest to metadata."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="manifest", config=config or {})
        self.dataset_id = self.config.get("dataset_id")
        self.output_path = self.config.get("output_path")

    def process(self, data: Any) -> StageResult:
        try:
            records, metadata = _to_records(data)
            if self.output_path:
                out = Path(self.output_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "w", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest = build_manifest(
                    out,
                    dataset_id=self.dataset_id,
                    produced_by="deepiri-dataset-processor@0.3.0",
                    metadata=metadata.get("quality_report"),
                )
            else:
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
                ) as tmp:
                    for record in records:
                        tmp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    tmp_path = tmp.name
                manifest = build_manifest(
                    tmp_path,
                    dataset_id=self.dataset_id,
                    produced_by="deepiri-dataset-processor@0.3.0",
                    metadata=metadata.get("quality_report"),
                )
            if hasattr(manifest, "model_dump"):
                metadata["manifest"] = manifest.model_dump(mode="json")
            else:
                metadata["manifest"] = manifest
            return self._create_result(
                processed_data=records,
                original_metadata=metadata,
                metadata_updates={"manifest_built": True},
            )
        except Exception as exc:
            return self._create_result(
                processed_data=None,
                original_metadata={},
                metadata_updates={},
                success=False,
                error=str(exc),
            )

    def validate(self, data: Any) -> ValidationResult:
        return ValidationResult(is_valid=True)


class VersioningStage(PreprocessingStage):
    """Create a filesystem dataset version snapshot."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="versioning", config=config or {})
        self.dataset_name = self.config.get("dataset_name", "dataset")
        self.versioning = DatasetVersioningSystem()

    def process(self, data: Any) -> StageResult:
        try:
            records, metadata = _to_records(data)
            if self.config.get("output_path"):
                path = Path(self.config["output_path"])
            else:
                import tempfile

                path = Path(tempfile.mkstemp(suffix=".jsonl")[1])
                with open(path, "w", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            version_meta = self.versioning.create_dataset_version(
                path, self.dataset_name
            )
            metadata["version"] = version_meta
            return self._create_result(
                processed_data=records,
                original_metadata=metadata,
                metadata_updates={"versioned": True},
            )
        except Exception as exc:
            return self._create_result(
                processed_data=None,
                original_metadata={},
                metadata_updates={},
                success=False,
                error=str(exc),
            )

    def validate(self, data: Any) -> ValidationResult:
        return ValidationResult(is_valid=True)
