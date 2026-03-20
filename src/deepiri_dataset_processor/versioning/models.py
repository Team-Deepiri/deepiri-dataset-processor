"""
Dataset versioning ORM models and Pydantic schemas.

Requires: pip install deepiri-dataset-processor[versioning]
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum

try:
    from pydantic import BaseModel, Field

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

try:
    from sqlalchemy import Column, String, Integer, DateTime, JSON, Text
    from sqlalchemy.orm import declarative_base

    HAS_SQLALCHEMY = True
    Base = declarative_base()
except ImportError:
    HAS_SQLALCHEMY = False
    Base = None


class DatasetType(str, Enum):
    """Types of datasets for language intelligence."""

    LEASE_ABSTRACTION = "lease_abstraction"
    CONTRACT_INTELLIGENCE = "contract_intelligence"
    OBLIGATION_DEPENDENCY = "obligation_dependency"
    REGULATORY_LANGUAGE = "regulatory_language"
    CLAUSE_EVOLUTION = "clause_evolution"


def _check_sqlalchemy():
    if not HAS_SQLALCHEMY:
        raise ImportError(
            "sqlalchemy is required for database versioning. "
            "Install with: pip install deepiri-dataset-processor[versioning]"
        )


def _check_pydantic():
    if not HAS_PYDANTIC:
        raise ImportError(
            "pydantic is required for dataset version metadata. "
            "Install with: pip install deepiri-dataset-processor[versioning]"
        )


if HAS_SQLALCHEMY:

    class DatasetVersion(Base):
        """Database model for dataset versions."""

        __tablename__ = "dataset_versions"

        id = Column(Integer, primary_key=True)
        dataset_name = Column(String(255), nullable=False, index=True)
        version = Column(String(50), nullable=False)
        dataset_type = Column(String(50), nullable=False)

        # Storage information
        storage_path = Column(String(500), nullable=False)
        storage_backend = Column(String(50), default="s3")

        # Data statistics
        total_samples = Column(Integer, nullable=False)
        file_count = Column(Integer, nullable=False)
        total_size_bytes = Column(Integer, nullable=False)

        # Checksums for integrity
        data_checksum = Column(String(64), nullable=False)
        metadata_checksum = Column(String(64), nullable=False)

        # Version metadata
        parent_version = Column(String(50), nullable=True)
        change_summary = Column(Text, nullable=True)
        change_type = Column(String(50))

        # Quality metrics
        quality_score = Column(String(20), nullable=True)
        validation_status = Column(String(50), default="PENDING")
        validation_errors = Column(JSON, nullable=True)

        # Metadata
        tags = Column(JSON, default=[])
        dataset_metadata = Column(JSON, default={})

        # Timestamps
        created_at = Column(DateTime, default=datetime.utcnow)
        created_by = Column(String(255), nullable=True)

else:
    DatasetVersion = None


if HAS_PYDANTIC:

    class DatasetVersionMetadata(BaseModel):
        """Pydantic model for dataset version metadata."""

        dataset_name: str
        version: str
        dataset_type: DatasetType
        storage_path: str
        storage_backend: str = "s3"

        total_samples: int
        file_count: int
        total_size_bytes: int

        data_checksum: str
        metadata_checksum: str

        parent_version: Optional[str] = None
        change_summary: Optional[str] = None
        change_type: str = "PATCH"

        quality_score: Optional[float] = None
        validation_status: str = "PENDING"
        validation_errors: Optional[List[Dict[str, Any]]] = None

        tags: List[str] = []
        metadata: Dict[str, Any] = {}

        created_at: datetime = Field(default_factory=datetime.utcnow)
        created_by: Optional[str] = None

else:
    DatasetVersionMetadata = None
