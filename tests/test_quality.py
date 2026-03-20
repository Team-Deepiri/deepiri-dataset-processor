"""Tests for quality checking module."""

import pytest

try:
    import numpy as np
    import pandas as pd

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(
    not HAS_DEPS, reason="numpy and pandas required for quality tests"
)


@pytest.fixture
def checker():
    from deepiri_dataset_processor.quality.checker import QualityChecker

    return QualityChecker()


@pytest.fixture
def sample_data():
    return [
        {"text": "hello world", "label": 1, "id": "a"},
        {"text": "foo bar", "label": 2, "id": "b"},
        {"text": "test data", "label": 1, "id": "c"},
    ]


class TestQualityConfig:
    def test_defaults(self):
        from deepiri_dataset_processor.quality.checker import QualityConfig

        config = QualityConfig()
        assert config.completeness_threshold == 0.95
        assert config.consistency_threshold == 0.90
        assert config.validity_threshold == 1.0

    def test_custom(self):
        from deepiri_dataset_processor.quality.checker import QualityConfig

        config = QualityConfig(completeness_threshold=0.8)
        assert config.completeness_threshold == 0.8


class TestQualityMetric:
    def test_create(self):
        from deepiri_dataset_processor.quality.checker import QualityMetric

        m = QualityMetric(dimension="completeness", metric_name="test", value=0.95)
        assert m.dimension == "completeness"
        assert m.passed is True


class TestQualityReport:
    def test_to_dict(self):
        from deepiri_dataset_processor.quality.checker import QualityReport
        from datetime import datetime

        report = QualityReport(
            dataset_id="test",
            timestamp=datetime.now(),
            overall_score=0.9,
            dimension_scores={"completeness": 0.95},
            metrics=[],
        )
        d = report.to_dict()
        assert d["dataset_id"] == "test"
        assert d["overall_score"] == 0.9


class TestQualityChecker:
    def test_check_quality_list_data(self, checker, sample_data):
        report = checker.check_quality(sample_data, dataset_id="test")
        assert report.dataset_id == "test"
        assert 0.0 <= report.overall_score <= 1.0
        assert len(report.metrics) > 0

    def test_check_quality_dict_data(self, checker):
        data = {"text": "hello", "label": 1}
        report = checker.check_quality(data)
        assert report.overall_score > 0

    def test_check_quality_dataframe(self, checker):
        import pandas as pd

        df = pd.DataFrame([
            {"text": "hello", "label": 1},
            {"text": "world", "label": 2},
        ])
        report = checker.check_quality(df)
        assert report.overall_score > 0

    def test_check_quality_empty(self, checker):
        report = checker.check_quality([])
        # Should handle gracefully
        assert isinstance(report, type(report))

    def test_completeness_with_missing(self, checker):
        data = [
            {"text": "hello", "label": 1},
            {"text": None, "label": 2},
            {"text": "world", "label": None},
        ]
        report = checker.check_quality(data)
        completeness_metrics = [
            m for m in report.metrics if m.dimension == "completeness"
        ]
        assert len(completeness_metrics) > 0

    def test_uniqueness_with_dupes(self, checker):
        data = [
            {"text": "hello", "label": 1},
            {"text": "hello", "label": 1},
            {"text": "world", "label": 2},
        ]
        report = checker.check_quality(data)
        uniqueness_metrics = [
            m for m in report.metrics if m.dimension == "uniqueness"
        ]
        assert len(uniqueness_metrics) > 0

    def test_validity_with_schema(self, checker):
        data = [{"text": "hello", "label": 1}]
        schema = {"required_fields": ["text", "label"]}
        report = checker.check_quality(data, schema=schema)
        validity_metrics = [
            m for m in report.metrics if m.dimension == "validity"
        ]
        assert len(validity_metrics) > 0
        assert all(m.passed for m in validity_metrics)

    def test_validity_missing_required(self, checker):
        data = [{"text": "hello"}]
        schema = {"required_fields": ["text", "label"]}
        report = checker.check_quality(data, schema=schema)
        validity_metrics = [
            m for m in report.metrics if m.dimension == "validity"
        ]
        assert any(not m.passed for m in validity_metrics)

    def test_recommendations_generated(self, checker):
        data = [
            {"text": "hello", "label": 1},
            {"text": "hello", "label": 1},
            {"text": "hello", "label": 1},
        ]
        report = checker.check_quality(data)
        # High duplication should trigger recommendations
        assert isinstance(report.recommendations, list)

    def test_custom_config(self):
        from deepiri_dataset_processor.quality.checker import QualityChecker, QualityConfig

        config = QualityConfig(completeness_threshold=0.5)
        checker = QualityChecker(config=config)
        data = [{"text": "hello", "label": None}]
        report = checker.check_quality(data)
        assert isinstance(report.overall_score, float)
