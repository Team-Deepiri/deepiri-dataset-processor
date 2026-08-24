"""Cyrex AGI ↔ Helox training record bridge."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional


def reckoning_record_to_training(
    record: Mapping[str, Any],
    *,
    document_id: str,
    artifact_id: Optional[str] = None,
    producer: str = "cyrex.reckoning",
) -> Dict[str, Any]:
    """Map a reckoning PredictionRecord dict to a structured Helox training row."""
    field_name = str(record.get("field_name", "unknown"))
    status = str(record.get("status", "unknown"))
    actual = record.get("actual_value")
    prior = record.get("predicted_mean") or record.get("predicted_range")
    instruction = (
        f"Document {document_id}: field '{field_name}' reckoning status={status}. "
        f"Prior={prior}, actual={actual}."
    )
    return {
        "instruction": instruction,
        "input": str(prior or ""),
        "output": str(actual or ""),
        "text": instruction,
        "category": "reckoning",
        "quality_score": 0.85 if status == "anomalous" else 0.7,
        "producer": producer,
        "metadata": {
            "document_id": document_id,
            "artifact_id": artifact_id,
            "field_name": field_name,
            "status": status,
            "sigma_delta": record.get("sigma_delta"),
            "source": "cyrex.reckoning",
        },
    }


def correction_to_training(
    *,
    document_id: str,
    field_name: str,
    corrected_value: Any,
    original_value: Any = None,
    actor_id: str = "unknown",
    artifact_id: Optional[str] = None,
) -> Dict[str, Any]:
    instruction = f"Correct field '{field_name}' for document {document_id}"
    return {
        "instruction": instruction,
        "input": str(original_value or ""),
        "output": str(corrected_value),
        "text": f"{instruction}\n{corrected_value}",
        "category": "correction",
        "quality_score": 1.0,
        "producer": "correction_writer",
        "metadata": {
            "document_id": document_id,
            "artifact_id": artifact_id,
            "field_name": field_name,
            "actor_id": actor_id,
            "source": "cyrex.correction",
        },
    }


def visual_observation_to_training(
    trace: Mapping[str, Any],
    *,
    document_id: str,
    scene_hash: str = "elkedel-live-scene-v1",
) -> Dict[str, Any]:
    """Map an Elkedel eyes identity trace to a visual grounding training row."""
    label = str(trace.get("label") or "object")
    identity = str(trace.get("trace_id") or trace.get("identity_id") or "unknown")
    raw_strength = trace.get("strength", 0.5)
    try:
        strength = float(raw_strength) if raw_strength is not None else 0.5
    except (TypeError, ValueError):
        strength = 0.5
    ts = trace.get("last_seen_ms") or trace.get("first_seen_ms") or 0
    instruction = (
        f"Live scene {document_id}: identify {label} (identity {identity}) "
        f"at frame_ts_{ts}."
    )
    return {
        "instruction": instruction,
        "input": f"scene:{scene_hash} ts:{ts}",
        "output": label,
        "text": instruction,
        "category": "visual_grounding",
        "quality_score": min(1.0, max(0.4, strength)),
        "producer": "elkedel.eyes",
        "metadata": {
            "document_id": document_id,
            "identity_id": identity,
            "label": label,
            "strength": strength,
            "n_observations": trace.get("n_observations"),
            "source": "elkedel.visual",
        },
    }


def batch_to_jsonl_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize records for JSONL export (ensure ``text`` field)."""
    out: List[Dict[str, Any]] = []
    for rec in records:
        row = dict(rec)
        if not row.get("text"):
            parts = [row.get("instruction"), row.get("input"), row.get("output")]
            row["text"] = "\n\n".join(str(p) for p in parts if p)
        row.setdefault("exported_at", datetime.now(timezone.utc).isoformat())
        out.append(row)
    return out
