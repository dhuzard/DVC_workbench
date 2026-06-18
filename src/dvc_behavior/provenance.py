"""Provenance manifest helpers for reproducible DVC exports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from datetime import timezone as _dt_timezone
import hashlib
import math
from pathlib import Path
from typing import Any

import pandas as pd

from dvc_behavior import __version__

__all__ = [
    "MANIFEST_VERSION",
    "build_provenance_manifest",
]

MANIFEST_VERSION = 1


def build_provenance_manifest(
    input_files: Iterable[Any],
    selected_config: Mapping[str, Any] | None = None,
    row_counts: Mapping[str, Any] | None = None,
    tables: Mapping[str, pd.DataFrame | None] | None = None,
    app_version: str | None = None,
    processing_timestamp: str | None = None,
) -> dict[str, Any]:
    """Build a serialisable provenance manifest for processed exports.

    ``input_files`` accepts paths, ``(name, bytes)`` pairs, bytes-like values,
    file-like objects, or mappings with ``name``/``filename`` and
    ``content``/``data``/``bytes`` entries.
    """
    manifest_row_counts: dict[str, int | None] = {}
    if row_counts:
        manifest_row_counts.update(_normalise_row_counts(row_counts))
    if tables:
        manifest_row_counts.update(
            {name: (len(table) if table is not None else None) for name, table in tables.items()}
        )

    return {
        "manifest_version": MANIFEST_VERSION,
        "app_version": app_version or __version__,
        "processing_timestamp": processing_timestamp or datetime.now(_dt_timezone.utc).isoformat(),
        "input_files": [
            _build_file_record(input_file, index)
            for index, input_file in enumerate(input_files, start=1)
        ],
        "selected_config": _make_manifest_safe(dict(selected_config or {})),
        "row_counts": manifest_row_counts,
    }


def _normalise_row_counts(row_counts: Mapping[str, Any]) -> dict[str, int | None]:
    normalised: dict[str, int | None] = {}
    for name, value in row_counts.items():
        if value is None:
            normalised[name] = None
        elif isinstance(value, pd.DataFrame):
            normalised[name] = len(value)
        else:
            normalised[name] = int(value)
    return normalised


def _build_file_record(input_file: Any, index: int) -> dict[str, Any]:
    name, data = _coerce_named_bytes(input_file, index)
    return {
        "name": name,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _coerce_named_bytes(input_file: Any, index: int) -> tuple[str, bytes]:
    if isinstance(input_file, (str, Path)):
        path = Path(input_file)
        return path.name, _read_path_bytes(path)

    if isinstance(input_file, Mapping):
        name = (
            input_file.get("name")
            or input_file.get("filename")
            or input_file.get("path")
            or f"input_{index}"
        )
        if "path" in input_file and not any(k in input_file for k in ("content", "data", "bytes")):
            return Path(str(name)).name, _read_path_bytes(Path(input_file["path"]))
        for key in ("content", "data", "bytes"):
            if key in input_file:
                return Path(str(name)).name, _as_bytes(input_file[key])
        raise TypeError("Input file mapping must include content, data, bytes, or path")

    if isinstance(input_file, (tuple, list)) and len(input_file) == 2:
        name, content = input_file
        return Path(str(name)).name, _as_bytes(content)

    inferred_name = getattr(input_file, "name", f"input_{index}")
    return Path(str(inferred_name)).name, _as_bytes(input_file)


def _read_path_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


def _as_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        return value.encode("utf-8")
    if hasattr(value, "getvalue"):
        return _as_bytes(value.getvalue())
    if hasattr(value, "read"):
        return _read_file_like(value)
    raise TypeError(f"Unsupported input file content type: {type(value)!r}")


def _read_file_like(value: Any) -> bytes:
    position = None
    if hasattr(value, "tell"):
        try:
            position = value.tell()
        except Exception:
            position = None
    try:
        data = value.read()
    finally:
        if position is not None and hasattr(value, "seek"):
            try:
                value.seek(position)
            except Exception:
                pass
    return _as_bytes(data)


def _make_manifest_safe(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _make_manifest_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_manifest_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted((_make_manifest_safe(v) for v in obj), key=repr)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj
