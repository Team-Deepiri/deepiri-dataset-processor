"""Chunked JSONL readers and writers for memory-efficient processing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Union

PathLike = Union[str, Path]


def iter_jsonl_chunks(
    path: PathLike,
    *,
    chunk_size: int = 1000,
) -> Generator[List[Dict[str, Any]], None, None]:
    """Yield lists of records from a JSONL file or directory."""
    source = Path(path)
    files: List[Path]
    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = sorted(source.rglob("*.jsonl"))
    else:
        raise FileNotFoundError(source)

    chunk: List[Dict[str, Any]] = []
    for file_path in files:
        with open(file_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if isinstance(record, dict):
                    chunk.append(record)
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
    if chunk:
        yield chunk


def load_jsonl_records(path: PathLike) -> List[Dict[str, Any]]:
    """Load all JSONL records into memory."""
    records: List[Dict[str, Any]] = []
    for chunk in iter_jsonl_chunks(path, chunk_size=5000):
        records.extend(chunk)
    return records


def write_jsonl_records(path: PathLike, records: Iterable[Dict[str, Any]]) -> Path:
    """Write records to a JSONL file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output
