"""
Baseline calculation per (subject_id, metric_name).

Baseline window is defined in hours relative to the alignment event
(time_from_event_hours).  Requires alignment to have been run first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_BASELINE_EPSILON",
    "compute_baseline",
]

# Minimum absolute baseline value for which a percent-change is considered
# numerically stable. Light-phase locomotion baselines are frequently
# near-zero; dividing by them makes ``baseline_percent_change`` explode to
# thousands of percent and dominate group means. When the absolute baseline is
# below this floor we set the percent change to NaN and flag the row as
# unstable instead. The default is small enough not to affect typical activity
# counts but large enough to catch the degenerate near-zero case.
DEFAULT_BASELINE_EPSILON = 1e-3


def compute_baseline(
    df: pd.DataFrame,
    start_hours: float = -72.0,
    end_hours: float = -24.0,
    method: str = "mean",
    exclude_excluded: bool = True,
    min_coverage: float = 0.7,
    impute_from_group_mean: bool = False,
    baseline_overrides: pd.DataFrame | None = None,
    epsilon: float = DEFAULT_BASELINE_EPSILON,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Compute per-(subject_id, metric_name) baseline values and attach them to df.

    Parameters
    ----------
    df               : long_df with alignment columns (time_from_event_hours)
    start_hours      : baseline window start (relative to alignment, typically negative)
    end_hours        : baseline window end   (relative to alignment, typically negative)
    method           : 'mean' (only option in MVP)
    exclude_excluded : if True, excluded rows are ignored in baseline calc
    min_coverage     : minimum fraction of expected bins that must be present
    impute_from_group_mean : if True, invalid/missing baselines are filled from
        valid group/metric baselines and marked with baseline_imputed=True
    baseline_overrides : optional table with subject_id, metric_name, baseline_value
        for manual per-animal baseline overrides
    epsilon          : small-denominator floor for the percent-change guard.
        When ``abs(baseline_value) < epsilon`` the percent change is undefined
        (near-zero baselines make it explode), so ``baseline_percent_change``
        is set to NaN and ``baseline_percent_change_unstable`` is set to True
        for those rows. The absolute ``baseline_corrected_value`` is still
        computed because it is well-defined even near zero. Defaults to
        ``DEFAULT_BASELINE_EPSILON``.

    Returns
    -------
    (df_annotated, baseline_summary, warnings)
    """
    warns: list[str] = []
    df = df.copy()

    has_alignment = (
        "time_from_event_hours" in df.columns and not df["time_from_event_hours"].isna().all()
    )

    if not has_alignment:
        warns.append(
            "Alignment has not been applied; baseline calculation skipped. "
            "Run alignment first or use absolute-time mode."
        )
        _add_empty_baseline_cols(df)
        return df, pd.DataFrame(), warns

    # Rows in the baseline window
    t = df["time_from_event_hours"]
    in_window = (t >= start_hours) & (t < end_hours)

    if exclude_excluded and "is_excluded" in df.columns:
        in_window = in_window & ~df["is_excluded"]

    baseline_df = df[in_window].copy()

    # Expected bins per (subject, metric) based on native_bin_seconds
    if "native_bin_seconds" in df.columns:
        # Use mode of native_bin_seconds across the df (avoid NaN issues)
        bins_s_mode = pd.to_numeric(df["native_bin_seconds"], errors="coerce").dropna()
        native_bin = float(bins_s_mode.mode().iloc[0]) if not bins_s_mode.empty else None
    else:
        native_bin = None

    window_duration_h = abs(end_hours - start_hours)
    if native_bin and native_bin > 0:
        expected_bins = window_duration_h * 3600.0 / native_bin
    else:
        expected_bins = np.nan
        warns.append(
            "native_bin_seconds is unavailable or invalid; baseline_expected_bins and "
            "baseline_coverage will be NaN."
        )

    # Group by (subject_id, metric_name) and compute baseline
    group_keys = ["subject_id", "metric_name"]
    if not all(k in baseline_df.columns for k in group_keys):
        warns.append("subject_id or metric_name missing; cannot compute per-subject baseline.")
        _add_empty_baseline_cols(df)
        return df, pd.DataFrame(), warns

    summary_rows: list[dict] = []

    baseline_map: dict[tuple[str, str], dict] = {}

    all_pairs = df[group_keys].drop_duplicates()
    group_lookup_cols = [c for c in ["subject_id", "metric_name", "group_id"] if c in df.columns]
    group_lookup = df[group_lookup_cols].drop_duplicates(group_keys).set_index(group_keys)

    for _, pair in all_pairs.iterrows():
        sid = pair["subject_id"]
        metric = pair["metric_name"]
        grp = baseline_df[
            (baseline_df["subject_id"].astype(str) == str(sid))
            & (baseline_df["metric_name"].astype(str) == str(metric))
        ]
        values = pd.to_numeric(grp["value"], errors="coerce").dropna()
        n_bins = len(values)
        group_id = None
        if "group_id" in group_lookup.columns:
            try:
                group_id = group_lookup.loc[(sid, metric), "group_id"]
            except Exception:
                group_id = None

        if not np.isnan(expected_bins):
            coverage = n_bins / expected_bins
        else:
            coverage = np.nan

        valid = n_bins > 0 and (np.isnan(coverage) or coverage >= min_coverage)

        if n_bins == 0:
            bval = np.nan
            warns.append(
                f"Subject '{sid}', metric '{metric}': no data in baseline window "
                f"[{start_hours}h, {end_hours}h]; baseline set to NaN."
            )
        elif method == "mean":
            bval = float(values.mean())
        else:
            bval = float(values.mean())  # fallback

        baseline_map[(str(sid), str(metric))] = {
            "baseline_value": bval,
            "baseline_n_bins": n_bins,
            "baseline_expected_bins": expected_bins,
            "baseline_coverage": coverage,
            "baseline_valid": valid,
            "group_id": group_id,
            "baseline_imputed": False,
            "baseline_override": False,
        }

        summary_rows.append(
            {
                "subject_id": sid,
                "metric_name": metric,
                "group_id": group_id,
                "baseline_start_hours": start_hours,
                "baseline_end_hours": end_hours,
                "baseline_value": bval,
                "baseline_n_bins": n_bins,
                "baseline_expected_bins": expected_bins,
                "baseline_coverage": round(float(coverage), 3) if not np.isnan(coverage) else None,
                "baseline_valid": valid,
                "baseline_imputed": False,
                "baseline_override": False,
            }
        )

    _apply_baseline_overrides(baseline_map, summary_rows, baseline_overrides)
    if impute_from_group_mean:
        _impute_invalid_baselines_from_group_mean(baseline_map, summary_rows, warns)

    # Merge back into df
    def _lookup(row: pd.Series, field: str):
        key = (str(row["subject_id"]), str(row["metric_name"]))
        entry = baseline_map.get(key, {})
        return entry.get(field, np.nan if field not in ("baseline_valid",) else False)

    df["baseline_value"] = df.apply(lambda r: _lookup(r, "baseline_value"), axis=1)
    df["baseline_n_bins"] = df.apply(lambda r: _lookup(r, "baseline_n_bins"), axis=1)
    df["baseline_expected_bins"] = df.apply(lambda r: _lookup(r, "baseline_expected_bins"), axis=1)
    df["baseline_coverage"] = df.apply(lambda r: _lookup(r, "baseline_coverage"), axis=1)
    df["baseline_valid"] = df.apply(lambda r: _lookup(r, "baseline_valid"), axis=1)
    df["baseline_imputed"] = df.apply(lambda r: _lookup(r, "baseline_imputed"), axis=1)
    df["baseline_override"] = df.apply(lambda r: _lookup(r, "baseline_override"), axis=1)

    bv = pd.to_numeric(df["baseline_value"], errors="coerce")
    val = pd.to_numeric(df["value"], errors="coerce")

    df["baseline_corrected_value"] = val - bv
    # Percent change: 100 * (value - baseline) / baseline, with a
    # near-zero guard. Exact-zero is not enough: near-zero baselines (common
    # for light-phase locomotion) make the ratio explode. When the absolute
    # baseline is below `epsilon` we do not divide; the percent change is NaN
    # and the row is flagged unstable. The absolute corrected value above
    # remains well-defined and intact.
    stable = bv.abs() >= epsilon
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(stable, 100.0 * (val - bv) / bv, np.nan)
    df["baseline_percent_change"] = pct
    # Only finite near-zero baselines are "unstable"; a NaN baseline is simply
    # missing, not near-zero, so it stays False (the default).
    unstable = bv.notna() & (bv.abs() < epsilon)
    df["baseline_percent_change_unstable"] = unstable.to_numpy()

    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    return df, summary_df, warns


def _apply_baseline_overrides(
    baseline_map: dict[tuple[str, str], dict],
    summary_rows: list[dict],
    overrides: pd.DataFrame | None,
) -> None:
    if overrides is None or overrides.empty:
        return
    required = {"subject_id", "metric_name", "baseline_value"}
    if not required <= set(overrides.columns):
        return

    for _, row in overrides.iterrows():
        sid = str(row.get("subject_id", "")).strip()
        metric = str(row.get("metric_name", "")).strip()
        value = pd.to_numeric(row.get("baseline_value"), errors="coerce")
        if not sid or not metric or pd.isna(value):
            continue
        key = (sid, metric)
        if key not in baseline_map:
            continue
        baseline_map[key]["baseline_value"] = float(value)
        baseline_map[key]["baseline_valid"] = True
        baseline_map[key]["baseline_imputed"] = False
        baseline_map[key]["baseline_override"] = True
        for summary in summary_rows:
            if str(summary["subject_id"]) == sid and str(summary["metric_name"]) == metric:
                summary["baseline_value"] = float(value)
                summary["baseline_valid"] = True
                summary["baseline_imputed"] = False
                summary["baseline_override"] = True


def _impute_invalid_baselines_from_group_mean(
    baseline_map: dict[tuple[str, str], dict],
    summary_rows: list[dict],
    warns: list[str],
) -> None:
    valid_values: dict[tuple[str, str], list[float]] = {}
    for row in summary_rows:
        group_id = row.get("group_id")
        metric = row.get("metric_name")
        value = row.get("baseline_value")
        if row.get("baseline_valid") and pd.notna(value):
            valid_values.setdefault((str(group_id), str(metric)), []).append(float(value))

    group_means = {key: float(np.mean(values)) for key, values in valid_values.items() if values}

    for row in summary_rows:
        if row.get("baseline_valid") and pd.notna(row.get("baseline_value")):
            continue
        key = (str(row.get("group_id")), str(row.get("metric_name")))
        imputed = group_means.get(key)
        if imputed is None:
            continue
        map_key = (str(row["subject_id"]), str(row["metric_name"]))
        baseline_map[map_key]["baseline_value"] = imputed
        baseline_map[map_key]["baseline_valid"] = True
        baseline_map[map_key]["baseline_imputed"] = True
        row["baseline_value"] = imputed
        row["baseline_valid"] = True
        row["baseline_imputed"] = True
        warns.append(
            f"Subject '{row['subject_id']}', metric '{row['metric_name']}': baseline imputed from group mean."
        )


def _add_empty_baseline_cols(df: pd.DataFrame) -> None:
    for col in (
        "baseline_value",
        "baseline_n_bins",
        "baseline_expected_bins",
        "baseline_coverage",
        "baseline_valid",
        "baseline_imputed",
        "baseline_override",
        "baseline_corrected_value",
        "baseline_percent_change",
    ):
        df[col] = np.nan
    # Boolean flag column defaults to False (no rows are near-zero-unstable
    # when no baseline was computed).
    df["baseline_percent_change_unstable"] = False
