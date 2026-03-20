"""Tests for database-backed dataset versioning."""

import json
import pytest

try:
    import sqlalchemy
    import pydantic

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(
    not HAS_DEPS,
    reason="sqlalchemy and pydantic required for database versioning tests",
)


@pytest.fixture
def manager():
    from deepiri_dataset_processor.versioning.database import DatasetVersionManager

    return DatasetVersionManager(
        db_url="sqlite:///:memory:",
        storage_backend="local",
    )


@pytest.fixture
def sample_data(tmp_path):
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    f = data_dir / "train.jsonl"
    f.write_text('{"text": "hello"}\n{"text": "world"}\n')
    return data_dir


class TestDatasetVersionManager:
    def test_create_version(self, manager, sample_data, tmp_path):
        from deepiri_dataset_processor.versioning.models import DatasetType

        meta = manager.create_version(
            dataset_name="test_ds",
            dataset_type=DatasetType.LEASE_ABSTRACTION,
            data_path=sample_data,
            version="1.0.0",
            created_by="test",
        )
        assert meta.dataset_name == "test_ds"
        assert meta.version == "1.0.0"
        assert meta.total_samples == 2
        assert meta.file_count == 1

    def test_get_version(self, manager, sample_data, tmp_path):
        from deepiri_dataset_processor.versioning.models import DatasetType

        manager.create_version(
            dataset_name="test_ds",
            dataset_type=DatasetType.LEASE_ABSTRACTION,
            data_path=sample_data,
            version="1.0.0",
        )
        result = manager.get_version("test_ds", "1.0.0")
        assert result is not None
        assert result.version == "1.0.0"

    def test_get_version_not_found(self, manager):
        result = manager.get_version("nonexistent", "1.0.0")
        assert result is None

    def test_list_versions(self, manager, sample_data, tmp_path):
        from deepiri_dataset_processor.versioning.models import DatasetType

        manager.create_version(
            dataset_name="test_ds",
            dataset_type=DatasetType.LEASE_ABSTRACTION,
            data_path=sample_data,
            version="1.0.0",
        )
        manager.create_version(
            dataset_name="test_ds",
            dataset_type=DatasetType.LEASE_ABSTRACTION,
            data_path=sample_data,
            version="1.0.1",
            parent_version="1.0.0",
        )
        versions = manager.list_versions("test_ds")
        assert len(versions) == 2

    def test_get_latest_version(self, manager, sample_data, tmp_path):
        from deepiri_dataset_processor.versioning.models import DatasetType

        manager.create_version(
            dataset_name="test_ds",
            dataset_type=DatasetType.LEASE_ABSTRACTION,
            data_path=sample_data,
            version="1.0.0",
        )
        manager.create_version(
            dataset_name="test_ds",
            dataset_type=DatasetType.LEASE_ABSTRACTION,
            data_path=sample_data,
            version="1.0.1",
        )
        latest = manager.get_latest_version("test_ds")
        assert latest is not None

    def test_compare_versions(self, manager, sample_data, tmp_path):
        from deepiri_dataset_processor.versioning.models import DatasetType

        manager.create_version(
            dataset_name="test_ds",
            dataset_type=DatasetType.LEASE_ABSTRACTION,
            data_path=sample_data,
            version="1.0.0",
        )
        manager.create_version(
            dataset_name="test_ds",
            dataset_type=DatasetType.LEASE_ABSTRACTION,
            data_path=sample_data,
            version="1.0.1",
        )
        comparison = manager.compare_versions("test_ds", "1.0.0", "1.0.1")
        assert "sample_difference" in comparison

    def test_auto_version(self, manager, sample_data, tmp_path):
        from deepiri_dataset_processor.versioning.models import DatasetType

        meta = manager.create_version(
            dataset_name="test_ds",
            dataset_type=DatasetType.LEASE_ABSTRACTION,
            data_path=sample_data,
        )
        assert meta.version == "1.0.0"

        meta2 = manager.create_version(
            dataset_name="test_ds",
            dataset_type=DatasetType.LEASE_ABSTRACTION,
            data_path=sample_data,
        )
        assert meta2.version == "1.0.1"


class TestDatasetType:
    def test_enum_values(self):
        from deepiri_dataset_processor.versioning.models import DatasetType

        assert DatasetType.LEASE_ABSTRACTION.value == "lease_abstraction"
        assert DatasetType.CONTRACT_INTELLIGENCE.value == "contract_intelligence"
