"""
Wide-format DVC metric CSV → long tidy table.

DVC exports one file per metric, with columns:
  day, hour, minute, relativeTime,
  {group}_TIMESTAMP, {group}_AVG, {group}_SEM, {group}_QRT, {group}_SAMPLES,
  {group}_{subject_col}, ...   (one per cage/animal)

Multiple group blocks can appear in the same file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import GROUP_META_SUFFIXES

__all__ = [
    "detect_group_prefixes",
    "get_group_meta_col_names",
    "get_subject_columns",
    "extract_subject_id",
    "parse_timestamp_series",
    "to_utc",
    "detect_native_bin_seconds",
    "infer_metric_name",
    "wide_to_long",
    "load_metric_csv",
    "combine_long_dfs",
]


# ---------------------------------------------------------------------------
# Group / column detection
# ---------------------------------------------------------------------------


def detect_group_prefixes(columns: list[str]) -> list[str]:
    """Return group prefixes inferred from columns ending with _TIMESTAMP."""
    return [col[: -len("_TIMESTAMP")] for col in columns if col.endswith("_TIMESTAMP")]


def get_group_meta_col_names(prefix: str) -> set[str]:
    return {f"{prefix}{suf}" for suf in GROUP_META_SUFFIXES}


def get_subject_columns(columns: list[str], prefix: str) -> list[str]:
    """Return individual-subject columns for a group block (not metadata cols)."""
    meta = get_group_meta_col_names(prefix)
    return [c for c in columns if c.startswith(f"{prefix}_") and c not in meta]


def extract_subject_id(col: str, group_prefix: str) -> str:
    """
    Remove the leading ``{group_prefix}_`` from a column name.

    Examples
    --------
    C57_C57_2       -> C57_2
    3_S_C_3_S_C_9   -> 3_S_C_9
    70Q_WT_70Q_WT8  -> 70Q_WT8
    """
    sep = group_prefix + "_"
    if col.startswith(sep):
        return col[len(sep) :]
    return col


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    """Parse a series of timestamp strings into timezone-aware pd.Timestamps."""
    return pd.to_datetime(series, utc=False, errors="coerce")


def to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts is pd.NaT or ts is None:
        return pd.NaT
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def detect_native_bin_seconds(ts_series: pd.Series) -> tuple[float | None, bool]:
    """
    Return (median_diff_seconds, is_irregular).
    is_irregular is True when std > 10 % of |median|.
    """
    ts = ts_series.dropna()
    if len(ts) < 2:
        return None, False
    diffs_s = ts.diff().dropna().dt.total_seconds()
    median = float(diffs_s.median())
    std = float(diffs_s.std()) if len(diffs_s) > 1 else 0.0
    irregular = std > 0.1 * abs(median) if median != 0 else False
    return median, irregular


# ---------------------------------------------------------------------------
# Metric name inference
# ---------------------------------------------------------------------------


def infer_metric_name(source_file: str) -> str:
    stem = Path(source_file).stem
    for suffix in ("_loc__index_smoothed", "_loc_index_smoothed", "_index_smoothed", "_smoothed"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


# ---------------------------------------------------------------------------
# Core conversion: wide → long
# ---------------------------------------------------------------------------


def wide_to_long(
    df: pd.DataFrame,
    source_file: str,
    metric_name: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Convert a wide-format DVC metric DataFrame to a long tidy table.

    Returns
    -------
    (long_df, warnings)
    """
    if metric_name is None:
        metric_name = infer_metric_name(source_file)

    columns = df.columns.tolist()
    prefixes = detect_group_prefixes(columns)
    warns: list[str] = []

    if not prefixes:
        warns.append(f"{source_file}: no *_TIMESTAMP column found — cannot detect group blocks.")
        return pd.DataFrame(), warns

    all_rows: list[dict[str, Any]] = []

    for prefix in prefixes:
        ts_col = f"{prefix}_TIMESTAMP"
        avg_col = f"{prefix}_AVG"
        sem_col = f"{prefix}_SEM"
        samples_col = f"{prefix}_SAMPLES"

        subject_cols = get_subject_columns(columns, prefix)
        if not subject_cols:
            warns.append(f"{source_file}: group '{prefix}' has no individual subject columns.")

        # Parse timestamps for this group
        raw_ts = (
            parse_timestamp_series(df[ts_col])
            if ts_col in df.columns
            else pd.Series([pd.NaT] * len(df), index=df.index)
        )

        native_bin, irregular = detect_native_bin_seconds(raw_ts)
        if irregular:
            warns.append(
                f"{source_file}: group '{prefix}' has irregular timestamp intervals "
                f"(median ≈ {native_bin:.0f} s)."
            )

        # Pre-extract scalar series for speed
        avg_s = (
            df[avg_col].astype(float, errors="ignore")
            if avg_col in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        sem_s = (
            df[sem_col].astype(float, errors="ignore")
            if sem_col in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        samples_s = (
            df[samples_col] if samples_col in df.columns else pd.Series(np.nan, index=df.index)
        )
        rel_s = (
            pd.to_numeric(df["relativeTime"], errors="coerce")
            if "relativeTime" in df.columns
            else pd.Series(np.nan, index=df.index)
        )

        for subject_col in subject_cols:
            subject_id = extract_subject_id(subject_col, prefix)
            values = pd.to_numeric(df[subject_col], errors="coerce")

            for idx in df.index:
                ts = raw_ts.iat[idx] if isinstance(idx, int) else raw_ts[idx]
                ts_utc = to_utc(ts) if ts is not pd.NaT else pd.NaT

                all_rows.append(
                    {
                        "source_file": source_file,
                        "metric_name": metric_name,
                        "group_id": prefix,
                        "subject_id": subject_id,
                        "source_column": subject_col,
                        "timestamp": ts,
                        "timestamp_utc": ts_utc,
                        "timestamp_local": ts,  # re-localized later with user TZ
                        "day": df["day"].iat[idx] if "day" in df.columns else np.nan,
                        "hour": df["hour"].iat[idx] if "hour" in df.columns else np.nan,
                        "minute": df["minute"].iat[idx] if "minute" in df.columns else np.nan,
                        "relative_time_seconds": rel_s.iat[idx],
                        "native_bin_seconds": native_bin,
                        "value": values.iat[idx],
                        "group_avg": avg_s.iat[idx],
                        "group_sem": sem_s.iat[idx],
                        "samples": samples_s.iat[idx],
                        "is_group_average": False,
                    }
                )

    long_df = pd.DataFrame(all_rows)
    return long_df, warns


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_metric_csv(
    source: Any,
    source_file: str = "",
    metric_name: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load a DVC metric CSV and return (long_df, warnings).
    ``source`` may be a file path, bytes, or file-like object.
    """
    from .io import read_csv_flexible

    warns: list[str] = []
    try:
        df = read_csv_flexible(source)
    except Exception as exc:
        warns.append(f"Cannot read '{source_file}': {exc}")
        return pd.DataFrame(), warns

    long_df, parse_warns = wide_to_long(df, source_file, metric_name)
    warns.extend(parse_warns)
    return long_df, warns


def combine_long_dfs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate multiple long DataFrames, reset index."""
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)
