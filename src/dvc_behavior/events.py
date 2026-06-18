"""
DVC event CSV parsing.

Expected CSV columns:
  group, day, hour, minute, relativeTime, timestamp, cage, rack, position, event
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import EVENT_CATEGORY_MAP

__all__ = [
    "parse_event_csv",
    "combine_event_dfs",
    "get_unique_event_types",
]


_EVENT_COLS_REQUIRED = {"group", "timestamp", "event"}
_EVENT_COLS_OPTIONAL = {"day", "hour", "minute", "relativeTime", "cage", "rack", "position"}


def _categorise_event(raw: str) -> str:
    return EVENT_CATEGORY_MAP.get(str(raw).strip().upper(), "other")


def parse_event_csv(
    source: Any,
    source_file: str = "",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load a DVC event CSV and return a clean event table + warnings.

    Returns
    -------
    (event_df, warnings)
    """
    from .io import read_csv_flexible

    warns: list[str] = []
    try:
        df = read_csv_flexible(source)
    except Exception as exc:
        warns.append(f"Cannot read event file '{source_file}': {exc}")
        return pd.DataFrame(), warns

    missing = _EVENT_COLS_REQUIRED - set(df.columns)
    if missing:
        warns.append(
            f"Event file '{source_file}' is missing required columns: {missing}. "
            "File will be skipped."
        )
        return pd.DataFrame(), warns

    df = df.copy()

    # Normalise event label
    df["raw_event_label"] = df["event"].astype(str).str.strip()
    df["event_type"] = df["raw_event_label"].str.upper()
    df["event_category"] = df["event_type"].apply(_categorise_event)

    # subject_id from cage column when present
    if "cage" in df.columns:
        df["subject_id"] = df["cage"].astype(str).str.strip()
    else:
        df["subject_id"] = pd.NA

    df["group_id"] = df["group"].astype(str).str.strip() if "group" in df.columns else pd.NA
    df["source_file"] = source_file

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")

    df["timestamp_utc"] = df["timestamp"].apply(
        lambda t: (
            t.tz_convert("UTC")
            if t is not pd.NaT and t.tzinfo is not None
            else (t.tz_localize("UTC") if t is not pd.NaT else pd.NaT)
        )
    )
    df["timestamp_local"] = df["timestamp"]  # re-localised with user TZ later

    df["relative_time_seconds"] = pd.to_numeric(
        df.get("relativeTime", pd.Series(dtype=float)), errors="coerce"
    )

    rack_col = df["rack"].astype(str) if "rack" in df.columns else pd.Series(pd.NA, index=df.index)
    pos_col = (
        df["position"].astype(str) if "position" in df.columns else pd.Series(pd.NA, index=df.index)
    )

    n_bad = df["timestamp"].isna().sum()
    if n_bad:
        warns.append(f"Event file '{source_file}': {n_bad} rows have unparseable timestamps.")

    out = pd.DataFrame(
        {
            "source_file": df["source_file"],
            "group_id": df["group_id"],
            "subject_id": df["subject_id"],
            "event_type": df["event_type"],
            "timestamp": df["timestamp"],
            "timestamp_utc": df["timestamp_utc"],
            "timestamp_local": df["timestamp_local"],
            "relative_time_seconds": df["relative_time_seconds"],
            "rack": rack_col.values,
            "position": pos_col.values,
            "raw_event_label": df["raw_event_label"],
            "event_category": df["event_category"],
        }
    )

    return out, warns


def combine_event_dfs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def get_unique_event_types(event_df: pd.DataFrame) -> list[str]:
    if event_df.empty or "event_type" not in event_df.columns:
        return []
    return sorted(event_df["event_type"].dropna().unique().tolist())
