"""Per-subject data quality diagnostics for long-format DVC metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "QUALITY_REPORT_COLUMNS",
    "build_quality_report",
    "compute_quality_report",
    "summarize_quality",
]


QUALITY_REPORT_COLUMNS = [
    "subject_id",
    "metric_name",
    "group_id",
    "n_rows",
    "missing_value_count",
    "missing_value_rate",
    "missing_timestamp_count",
    "duplicate_timestamp_count",
    "median_interval_seconds",
    "max_gap_seconds",
    "long_gap_threshold_seconds",
    "long_gap_count",
    "irregular_interval_flag",
    "negative_value_count",
    "zero_variance_flag",
]


def build_quality_report(
    df: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    metric_col: str = "metric_name",
    timestamp_col: str = "timestamp_utc",
    value_col: str = "value",
    tolerance_fraction: float = 0.10,
    long_gap_multiplier: float = 3.0,
    long_gap_threshold_seconds: float | None = None,
) -> pd.DataFrame:
    """Return one quality diagnostic row per subject and metric.

    Missing values are counted after numeric coercion of ``value_col``.
    Duplicate timestamps are counted within each subject/metric stream after the
    first occurrence. Interval diagnostics ignore missing timestamps and repeated
    duplicates so duplicate rows do not mask the underlying sampling cadence.
    """
    _validate_parameters(tolerance_fraction, long_gap_multiplier, long_gap_threshold_seconds)
    _validate_required_columns(df, {subject_col, metric_col, timestamp_col, value_col})

    if df is None or df.empty:
        return _empty_report()

    rows: list[dict] = []
    for keys, group in df.groupby([subject_col, metric_col], dropna=False, sort=True):
        subject_id, metric_name = keys
        values = pd.to_numeric(group[value_col], errors="coerce")
        timestamps = pd.to_datetime(group[timestamp_col], utc=True, errors="coerce")
        intervals = _timestamp_intervals_seconds(timestamps)
        median_interval = float(intervals.median()) if not intervals.empty else np.nan
        max_gap = float(intervals.max()) if not intervals.empty else np.nan
        gap_threshold = _long_gap_threshold(
            median_interval,
            long_gap_multiplier,
            long_gap_threshold_seconds,
        )

        rows.append(
            {
                "subject_id": subject_id,
                "metric_name": metric_name,
                "group_id": _group_id(group),
                "n_rows": int(len(group)),
                "missing_value_count": int(values.isna().sum()),
                "missing_value_rate": float(values.isna().mean()) if len(values) else np.nan,
                "missing_timestamp_count": int(timestamps.isna().sum()),
                "duplicate_timestamp_count": int(
                    timestamps.dropna().duplicated(keep="first").sum()
                ),
                "median_interval_seconds": median_interval,
                "max_gap_seconds": max_gap,
                "long_gap_threshold_seconds": gap_threshold,
                "long_gap_count": _long_gap_count(intervals, gap_threshold),
                "irregular_interval_flag": _is_irregular_interval(intervals, tolerance_fraction),
                "negative_value_count": int((values < 0).sum()),
                "zero_variance_flag": _has_zero_variance(values),
            }
        )

    return pd.DataFrame(rows, columns=QUALITY_REPORT_COLUMNS)


def compute_quality_report(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Alias for :func:`build_quality_report`."""
    return build_quality_report(df, **kwargs)


def summarize_quality(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Alias for :func:`build_quality_report`."""
    return build_quality_report(df, **kwargs)


def _empty_report() -> pd.DataFrame:
    return pd.DataFrame(columns=QUALITY_REPORT_COLUMNS)


def _validate_required_columns(df: pd.DataFrame | None, required: set[str]) -> None:
    columns = set(df.columns) if df is not None else set()
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"quality report requires columns: {', '.join(missing)}")


def _validate_parameters(
    tolerance_fraction: float,
    long_gap_multiplier: float,
    long_gap_threshold_seconds: float | None,
) -> None:
    if tolerance_fraction < 0:
        raise ValueError("tolerance_fraction must be non-negative.")
    if long_gap_multiplier <= 0:
        raise ValueError("long_gap_multiplier must be positive.")
    if long_gap_threshold_seconds is not None and long_gap_threshold_seconds <= 0:
        raise ValueError("long_gap_threshold_seconds must be positive when provided.")


def _timestamp_intervals_seconds(timestamps: pd.Series) -> pd.Series:
    ordered_unique = timestamps.dropna().drop_duplicates().sort_values()
    return ordered_unique.diff().dropna().dt.total_seconds()


def _long_gap_threshold(
    median_interval_seconds: float,
    long_gap_multiplier: float,
    long_gap_threshold_seconds: float | None,
) -> float:
    if long_gap_threshold_seconds is not None:
        return float(long_gap_threshold_seconds)
    if np.isnan(median_interval_seconds) or median_interval_seconds <= 0:
        return np.nan
    return float(long_gap_multiplier * median_interval_seconds)


def _long_gap_count(intervals: pd.Series, threshold_seconds: float) -> int:
    if intervals.empty or np.isnan(threshold_seconds):
        return 0
    return int((intervals > threshold_seconds).sum())


def _is_irregular_interval(intervals: pd.Series, tolerance_fraction: float) -> bool:
    if intervals.empty:
        return False
    median = float(intervals.median())
    if median == 0:
        return False
    std = float(intervals.std()) if len(intervals) > 1 else 0.0
    return bool(std > tolerance_fraction * abs(median))


def _has_zero_variance(values: pd.Series) -> bool:
    observed = values.dropna()
    return bool(len(observed) >= 2 and observed.nunique(dropna=True) == 1)


def _group_id(group: pd.DataFrame) -> object:
    if "group_id" not in group.columns:
        return pd.NA
    observed = group["group_id"].dropna().unique()
    if len(observed) != 1:
        return pd.NA
    return observed[0]
