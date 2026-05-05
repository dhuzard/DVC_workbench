"""
Optional temporal aggregation of the processed long_df.

Aggregation happens AFTER exclusion/alignment/baseline, so excluded rows
are already marked.  We aggregate by time bin, preserving metadata columns.
"""

from __future__ import annotations

import pandas as pd


_NUMERIC_COLS_TO_AGG = [
    "value",
    "group_avg",
    "group_sem",
    "baseline_corrected_value",
    "baseline_percent_change",
    "time_from_event_seconds",
    "time_from_event_hours",
    "experimental_day",
]

_METADATA_COLS = [
    "source_file",
    "metric_name",
    "group_id",
    "subject_id",
    # merged metadata
    "animal_id",
    "cage_id",
    "group_label",
    "treatment_group",
    "treatment",
    "genotype",
    "sex",
    "strain",
    "cohort",
    "batch",
    "rack",
    "position",
    # baseline / alignment scalar (constant per subject)
    "native_bin_seconds",
    "baseline_value",
    "baseline_n_bins",
    "baseline_expected_bins",
    "baseline_coverage",
    "baseline_valid",
    "alignment_event_type",
    "alignment_timestamp",
    "timestamp_local",
    "light_dark_phase",
    "zeitgeber_time_hours",
    "baseline_imputed",
    "baseline_override",
]


def aggregate(
    df: pd.DataFrame,
    bin_seconds: int | None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Aggregate long_df to a coarser bin size.

    Parameters
    ----------
    bin_seconds : target bin size in seconds.  None → return df unchanged.

    Returns
    -------
    (aggregated_df, warnings)
    """
    warns: list[str] = []

    if bin_seconds is None:
        return df, warns

    if "timestamp_utc" not in df.columns or df["timestamp_utc"].isna().all():
        warns.append("No valid timestamps; cannot aggregate.")
        return df, warns

    # Check native bin size
    if "native_bin_seconds" in df.columns:
        native = df["native_bin_seconds"].dropna()
        if not native.empty:
            native_val = float(native.mode().iloc[0])
            if native_val > 0 and bin_seconds < native_val:
                warns.append(
                    f"Requested bin size ({bin_seconds} s) is smaller than native bin size "
                    f"({native_val:.0f} s). Keeping native bin size."
                )
                return df, warns

    # Floor timestamps to the desired bin
    freq = f"{bin_seconds}s"
    try:
        ts_binned = df["timestamp_utc"].dt.floor(freq)
    except Exception as exc:
        warns.append(f"Aggregation failed: {exc}")
        return df, warns

    df = df.copy()
    df["_bin"] = ts_binned

    group_keys = ["_bin", "metric_name", "group_id", "subject_id"]
    group_keys = [k for k in group_keys if k in df.columns]

    agg_parts: list[pd.DataFrame] = []

    # Numeric mean columns
    numeric_cols = [c for c in _NUMERIC_COLS_TO_AGG if c in df.columns]
    if numeric_cols:
        num_agg = df.groupby(group_keys)[numeric_cols].mean().reset_index()
        agg_parts.append(num_agg)

    # Metadata — take first value per group (they're constant within subject/metric)
    meta_cols = [c for c in _METADATA_COLS if c in df.columns and c not in group_keys]
    if meta_cols:
        meta_agg = df.groupby(group_keys)[meta_cols].first().reset_index()
        agg_parts.append(meta_agg.drop(columns=[k for k in group_keys if k in meta_agg.columns], errors="ignore"))

    # Exclusion summary per bin
    if "is_excluded" in df.columns:
        excl_agg = df.groupby(group_keys)["is_excluded"].any().reset_index()
        count_agg = df.groupby(group_keys).size().reset_index(name="_n_bins")
        excl_pct = df.groupby(group_keys)["is_excluded"].mean().reset_index(name="_pct_excluded")
        agg_parts += [
            excl_agg.drop(columns=[k for k in group_keys if k in excl_agg.columns], errors="ignore"),
            count_agg.drop(columns=[k for k in group_keys if k in count_agg.columns], errors="ignore"),
            excl_pct.drop(columns=[k for k in group_keys if k in excl_pct.columns], errors="ignore"),
        ]

    if not agg_parts:
        warns.append("No columns to aggregate.")
        return df.drop(columns=["_bin"], errors="ignore"), warns

    # Join all aggregated pieces
    result = agg_parts[0]
    for part in agg_parts[1:]:
        result = pd.concat([result, part], axis=1)

    # Rename _bin to timestamp_utc
    result = result.rename(columns={"_bin": "timestamp_utc"})
    result["native_bin_seconds"] = bin_seconds

    # Re-add light/dark from localised timestamp if available
    return result, warns
