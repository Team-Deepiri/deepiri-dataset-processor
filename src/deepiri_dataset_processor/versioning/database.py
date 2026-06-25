"""
Dataset Version Manager with database backend.

Requires: pip install deepiri-dataset-processor[versioning]
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import (
    DatasetType,
    HAS_PYDANTIC,
    HAS_SQLALCHEMY,
    _check_pydantic,
    _check_sqlalchemy,
)

if HAS_SQLALCHEMY:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from .models import Base, DatasetVersion

if HAS_PYDANTIC:
    from .models import DatasetVersionMetadata


class DatasetVersionManager:
    """
    Manages dataset versions with a database backend.

    Features:
    - Create new dataset versions
    - Track version lineage
    - Validate dataset integrity
    - Query version history
    - Compare versions
    """

    def __init__(
        self,
        db_url: str,
        storage_backend: str = "s3",
        storage_config: Optional[Dict[str, Any]] = None,
    ):
        _check_sqlalchemy()
        _check_pydantic()

        connect_args = {}
        if db_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False, "timeout": 60}

        self.engine = create_engine(
            db_url,
            connect_args=connect_args,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

        try:
            Base.metadata.create_all(self.engine)
        except Exception as e:
            print(f"Warning: Could not create tables automatically: {e}")

        self.Session = sessionmaker(bind=self.engine)
        self.storage_backend = storage_backend
        self.storage_config = storage_config or {}

    def create_version(
        self,
        dataset_name: str,
        dataset_type: DatasetType,
        data_path: Path,
        version: Optional[str] = None,
        parent_version: Optional[str] = None,
        change_summary: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
    ) -> "DatasetVersionMetadata":
        """Create a new dataset version."""
        if version is None:
            version = self._get_next_version(dataset_name, dataset_type)

        stats = self._calculate_statistics(data_path)
        data_checksum = self._calculate_data_checksum(data_path)
        metadata_checksum = self._calculate_metadata_checksum(
            dataset_name, version, stats, metadata or {}
        )
        change_type = self._determine_change_type(
            dataset_name, dataset_type, parent_version, stats
        )
        storage_path = self._store_dataset(dataset_name, version, data_path)

        version_metadata = DatasetVersionMetadata(
            dataset_name=dataset_name,
            version=version,
            dataset_type=dataset_type,
            storage_path=storage_path,
            storage_backend=self.storage_backend,
            total_samples=stats["total_samples"],
            file_count=stats["file_count"],
            total_size_bytes=stats["total_size_bytes"],
            data_checksum=data_checksum,
            metadata_checksum=metadata_checksum,
            parent_version=parent_version,
            change_summary=change_summary,
            change_type=change_type,
            tags=tags or [],
            metadata=metadata or {},
            created_by=created_by,
        )

        self._save_version_to_db(version_metadata)
        return version_metadata

    def get_version(
        self,
        dataset_name: str,
        version: str,
        dataset_type: Optional[DatasetType] = None,
    ) -> Optional["DatasetVersionMetadata"]:
        """Get version metadata."""
        session = self.Session()
        try:
            query = session.query(DatasetVersion).filter(
                DatasetVersion.dataset_name == dataset_name,
                DatasetVersion.version == version,
            )
            if dataset_type:
                query = query.filter(
                    DatasetVersion.dataset_type == dataset_type.value
                )
            record = query.first()
            return self._db_to_metadata(record) if record else None
        finally:
            session.close()

    def list_versions(
        self,
        dataset_name: Optional[str] = None,
        dataset_type: Optional[DatasetType] = None,
        limit: int = 100,
    ) -> List["DatasetVersionMetadata"]:
        """List all versions, optionally filtered."""
        session = self.Session()
        try:
            query = session.query(DatasetVersion)
            if dataset_name:
                query = query.filter(DatasetVersion.dataset_name == dataset_name)
            if dataset_type:
                query = query.filter(
                    DatasetVersion.dataset_type == dataset_type.value
                )
            versions = (
                query.order_by(DatasetVersion.created_at.desc()).limit(limit).all()
            )
            return [self._db_to_metadata(v) for v in versions]
        finally:
            session.close()

    def get_latest_version(
        self,
        dataset_name: str,
        dataset_type: Optional[DatasetType] = None,
    ) -> Optional["DatasetVersionMetadata"]:
        """Get latest version of a dataset."""
        versions = self.list_versions(dataset_name, dataset_type, limit=1)
        return versions[0] if versions else None

    def compare_versions(
        self,
        dataset_name: str,
        version1: str,
        version2: str,
    ) -> Dict[str, Any]:
        """Compare two versions of a dataset."""
        v1 = self.get_version(dataset_name, version1)
        v2 = self.get_version(dataset_name, version2)

        if not v1 or not v2:
            raise ValueError("One or both versions not found")

        return {
            "version1": v1.version,
            "version2": v2.version,
            "sample_difference": v2.total_samples - v1.total_samples,
            "file_difference": v2.file_count - v1.file_count,
            "size_difference_bytes": v2.total_size_bytes - v1.total_size_bytes,
            "change_type": v2.change_type,
            "change_summary": v2.change_summary,
        }

    def validate_version(
        self,
        dataset_name: str,
        version: str,
    ) -> Dict[str, Any]:
        """Validate dataset version integrity."""
        version_meta = self.get_version(dataset_name, version)
        if not version_meta:
            raise ValueError(f"Version {version} not found")

        data_path = Path(version_meta.storage_path)
        current_checksum = self._calculate_data_checksum(data_path)
        is_valid = current_checksum == version_meta.data_checksum

        return {
            "is_valid": is_valid,
            "expected_checksum": version_meta.data_checksum,
            "actual_checksum": current_checksum,
            "validation_timestamp": datetime.utcnow().isoformat(),
        }

    def _get_next_version(self, dataset_name: str, dataset_type: DatasetType) -> str:
        latest = self.get_latest_version(dataset_name, dataset_type)
        if not latest:
            return "1.0.0"

        raw = latest.version.strip().lstrip("v")
        parts = raw.split(".")
        try:
            major = int(parts[0]) if len(parts) > 0 else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
        except (ValueError, IndexError):
            return "1.0.0"
        return f"{major}.{minor}.{patch + 1}"

    def _calculate_statistics(self, data_path: Path) -> Dict[str, Any]:
        total_samples = 0
        file_count = 0
        total_size = 0

        for file_path in data_path.rglob("*"):
            if file_path.is_file():
                file_count += 1
                total_size += file_path.stat().st_size

                if file_path.suffix == ".jsonl":
                    with open(file_path, "r") as f:
                        total_samples += sum(1 for _ in f)
                elif file_path.suffix == ".json":
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            total_samples += len(data)
                        else:
                            total_samples += 1

        return {
            "total_samples": total_samples,
            "file_count": file_count,
            "total_size_bytes": total_size,
        }

    def _calculate_data_checksum(self, data_path: Path) -> str:
        hasher = hashlib.sha256()
        for file_path in sorted(data_path.rglob("*")):
            if file_path.is_file():
                with open(file_path, "rb") as f:
                    hasher.update(f.read())
        return hasher.hexdigest()

    def _calculate_metadata_checksum(
        self, dataset_name: str, version: str, stats: Dict, metadata: Dict
    ) -> str:
        metadata_str = json.dumps(
            {
                "dataset_name": dataset_name,
                "version": version,
                "stats": stats,
                "metadata": metadata,
            },
            sort_keys=True,
        )
        return hashlib.sha256(metadata_str.encode()).hexdigest()

    def _determine_change_type(
        self,
        dataset_name: str,
        dataset_type: DatasetType,
        parent_version: Optional[str],
        stats: Dict[str, Any],
    ) -> str:
        if not parent_version:
            return "MAJOR"

        parent = self.get_version(dataset_name, parent_version, dataset_type)
        if not parent:
            return "MAJOR"

        sample_diff = stats["total_samples"] - parent.total_samples
        sample_change_pct = (
            abs(sample_diff) / parent.total_samples if parent.total_samples > 0 else 0
        )

        if sample_change_pct > 0.5:
            return "MAJOR"
        elif sample_change_pct > 0.1:
            return "MINOR"
        else:
            return "PATCH"

    def _store_dataset(self, dataset_name: str, version: str, data_path: Path) -> str:
        if self.storage_backend == "local":
            import shutil

            storage_path = Path(f"./datasets/{dataset_name}/{version}")
            storage_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(data_path, storage_path, dirs_exist_ok=True)
            return str(storage_path.absolute())
        elif self.storage_backend == "s3":
            raise NotImplementedError(
                "S3 storage is not implemented yet. Use storage_backend='local'."
            )
        else:
            raise ValueError(f"Unsupported storage backend: {self.storage_backend}")

    def _save_version_to_db(self, metadata: "DatasetVersionMetadata"):
        session = self.Session()
        try:
            record = DatasetVersion(
                dataset_name=metadata.dataset_name,
                version=metadata.version,
                dataset_type=metadata.dataset_type.value,
                storage_path=metadata.storage_path,
                storage_backend=metadata.storage_backend,
                total_samples=metadata.total_samples,
                file_count=metadata.file_count,
                total_size_bytes=metadata.total_size_bytes,
                data_checksum=metadata.data_checksum,
                metadata_checksum=metadata.metadata_checksum,
                parent_version=metadata.parent_version,
                change_summary=metadata.change_summary,
                change_type=metadata.change_type,
                quality_score=(
                    str(metadata.quality_score) if metadata.quality_score else None
                ),
                validation_status=metadata.validation_status,
                validation_errors=metadata.validation_errors,
                tags=metadata.tags,
                dataset_metadata=metadata.metadata,
                created_by=metadata.created_by,
            )
            session.add(record)
            session.commit()
        finally:
            session.close()

    def _db_to_metadata(self, db_record: "DatasetVersion") -> "DatasetVersionMetadata":
        return DatasetVersionMetadata(
            dataset_name=db_record.dataset_name,
            version=db_record.version,
            dataset_type=DatasetType(db_record.dataset_type),
            storage_path=db_record.storage_path,
            storage_backend=db_record.storage_backend,
            total_samples=db_record.total_samples,
            file_count=db_record.file_count,
            total_size_bytes=db_record.total_size_bytes,
            data_checksum=db_record.data_checksum,
            metadata_checksum=db_record.metadata_checksum,
            parent_version=db_record.parent_version,
            change_summary=db_record.change_summary,
            change_type=db_record.change_type,
            quality_score=(
                float(db_record.quality_score) if db_record.quality_score else None
            ),
            validation_status=db_record.validation_status,
            validation_errors=db_record.validation_errors,
            tags=db_record.tags or [],
            metadata=db_record.dataset_metadata or {},
            created_at=db_record.created_at,
            created_by=db_record.created_by,
        )
