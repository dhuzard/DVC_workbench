"""File I/O utilities — load CSVs from paths or uploaded bytes."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd


def read_csv_flexible(source: Any, **kwargs: Any) -> pd.DataFrame:
    """
    Read a CSV from a file path, pathlib.Path, bytes, or BytesIO object.
    Extra kwargs are forwarded to pd.read_csv.
    """
    if isinstance(source, (str, Path)):
        return pd.read_csv(source, low_memory=False, **kwargs)
    if isinstance(source, bytes):
        return pd.read_csv(io.BytesIO(source), low_memory=False, **kwargs)
    # Already a file-like object
    return pd.read_csv(source, low_memory=False, **kwargs)


def list_example_files(examples_dir: Path) -> dict[str, list[Path]]:
    """
    Scan the examples directory and return two lists:
    metric_files and event_files (heuristic: files named *event* go in events).
    """
    if not examples_dir.is_dir():
        return {"metric": [], "event": []}

    metric: list[Path] = []
    event: list[Path] = []

    for p in sorted(examples_dir.glob("*.csv")):
        if "event" in p.name.lower():
            event.append(p)
        else:
            metric.append(p)

    return {"metric": metric, "event": event}
