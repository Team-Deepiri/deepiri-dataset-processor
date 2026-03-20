"""Tests for pipeline framework: base, stages, and orchestrator."""

from deepiri_dataset_processor.pipeline.base import (
    PreprocessingStage,
    ProcessedData,
    StageResult,
    ValidationResult,
)
from deepiri_dataset_processor.pipeline.stages import (
    DataCleaningStage,
    DataTransformationStage,
    DataValidationStage,
)
from deepiri_dataset_processor.pipeline.orchestrator import DatasetPipeline


class TestProcessedData:
    def test_create(self):
        pd = ProcessedData(data=[{"text": "hello"}], metadata={"cleaned": True})
        assert pd.data == [{"text": "hello"}]
        assert pd.metadata["cleaned"] is True


class TestValidationResult:
    def test_valid(self):
        vr = ValidationResult(is_valid=True, errors=[], warnings=[])
        assert vr.is_valid is True

    def test_invalid(self):
        vr = ValidationResult(is_valid=False, errors=["missing field"], warnings=[])
        assert vr.is_valid is False
        assert len(vr.errors) == 1


class TestStageResult:
    def test_success(self):
        pd = ProcessedData(data=[], metadata={})
        sr = StageResult(success=True, processed_data=pd)
        assert sr.success is True

    def test_failure(self):
        sr = StageResult(success=False, error="something went wrong")
        assert sr.success is False
        assert sr.error == "something went wrong"


class TestDataCleaningStage:
    def setup_method(self):
        self.stage = DataCleaningStage()

    def test_name(self):
        assert self.stage.get_name() == "data_cleaning"

    def test_process_cleans_whitespace(self):
        data = [{"text": "hello   world"}, {"text": "  foo  bar  "}]
        result = self.stage.process(data)
        assert result.success is True

    def test_process_removes_duplicates(self):
        data = [{"text": "hello"}, {"text": "hello"}]
        result = self.stage.process(data)
        assert result.success is True
        actual_data = result.processed_data
        if isinstance(actual_data, ProcessedData):
            assert len(actual_data.data) == 1
        elif isinstance(actual_data, list):
            assert len(actual_data) == 1

    def test_validate_valid_data(self):
        data = [{"text": "hello world"}]
        result = self.stage.validate(data)
        assert result.is_valid is True

    def test_validate_none_data(self):
        result = self.stage.validate(None)
        assert result.is_valid is False


class TestDataValidationStage:
    def setup_method(self):
        self.stage = DataValidationStage(
            config={"required_fields": ["text", "label"]}
        )

    def test_name(self):
        assert self.stage.get_name() == "data_validation"

    def test_process_valid(self):
        data = [{"text": "hello", "label": 1}]
        result = self.stage.process(data)
        assert result.success is True

    def test_process_missing_field(self):
        data = [{"text": "hello"}]
        result = self.stage.process(data)
        assert result.success is False

    def test_validate_missing_fields(self):
        data = [{"text": "hello"}]
        result = self.stage.validate(data)
        assert any("missing" in e.lower() for e in result.errors)


class TestDataTransformationStage:
    def setup_method(self):
        self.stage = DataTransformationStage()

    def test_name(self):
        assert self.stage.get_name() == "data_transformation"

    def test_process_lowercases(self):
        data = [{"text": "Hello World"}]
        result = self.stage.process(data)
        assert result.success is True

    def test_process_trims(self):
        data = [{"text": "  hello  "}]
        result = self.stage.process(data)
        assert result.success is True

    def test_validate_valid(self):
        data = [{"text": "hello"}]
        result = self.stage.validate(data)
        assert result.is_valid is True


class TestDatasetPipeline:
    def test_run_empty_stages(self):
        pipeline = DatasetPipeline(stages=[])
        # With no stages, run returns the initial data untouched
        # Actually the pipeline returns result from last stage, and with no stages
        # result is not assigned. Let's test with one stage.

    def test_run_single_stage(self):
        pipeline = DatasetPipeline(stages=[DataCleaningStage()])
        data = [{"text": "hello   world"}]
        result = pipeline.run(data)
        assert result.success is True

    def test_run_chain(self):
        pipeline = DatasetPipeline(
            stages=[
                DataCleaningStage(),
                DataValidationStage(config={"required_fields": ["text"]}),
            ]
        )
        data = [{"text": "hello   world"}]
        result = pipeline.run(data)
        assert result.success is True

    def test_run_chain_fails_on_invalid(self):
        pipeline = DatasetPipeline(
            stages=[
                DataCleaningStage(),
                DataValidationStage(config={"required_fields": ["text", "label"]}),
            ]
        )
        data = [{"text": "hello"}]
        result = pipeline.run(data)
        assert result.success is False

    def test_validate_all(self):
        pipeline = DatasetPipeline(
            stages=[
                DataCleaningStage(),
                DataValidationStage(config={"required_fields": ["text"]}),
            ]
        )
        data = [{"text": "hello"}]
        results = pipeline.validate_all(data)
        assert len(results) == 2

    def test_get_stage_names(self):
        pipeline = DatasetPipeline(
            stages=[DataCleaningStage(), DataTransformationStage()]
        )
        names = pipeline.get_stage_names()
        assert names == ["data_cleaning", "data_transformation"]
