"""
Post-export analysis helpers for processed DVC time series.

These functions operate on the tidy ``processed_timeseries`` table produced by
the preprocessing pipeline.  They are intentionally table-oriented so their
outputs can be exported as CSVs or plotted by the app without depending on
Streamlit.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_GROUP_COLS = ("group_id", "metric_name")
EXPLORATORY_STATS_DISCLAIMER = (
    "These results are for orientation only. Consult a statistician for confirmatory analysis."
)


def summarize_circadian_cosinor(
    df: pd.DataFrame,
    value_col: str = "value",
    zt_col: str = "zeitgeber_time_hours",
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
    period_hours: float = 24.0,
    zt_bin_hours: float = 1.0,
    min_points: int = 6,
    subject_col: str = "subject_id",
    fit_level: str = "subject",
    photoperiod_hours: float = 12.0,
    exclude_excluded: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Fit a 24 h cosinor model per group/metric and return rhythm parameters.

    The model is ``MESOR + amplitude * cos(2*pi*(ZT - acrophase_ZT)/period)``.
    ``scipy.optimize.curve_fit`` is used when available; otherwise a linear
    least-squares cosinor fit is used.

    Parameters
    ----------
    fit_level:
        Controls the residuals used for the rhythm-detection p-value and the
        confidence intervals (``p_rhythm`` and the ``*_ci_*`` columns).
        ``"subject"`` (default) aggregates to per-subject-per-ZT-bin means
        before computing the inference, which gives more honest residual degrees
        of freedom and therefore meaningful confidence intervals.  It is only
        used when ``subject_col`` is present in the input; otherwise the function
        falls back to the legacy ``"binned"`` behaviour (one mean per ZT bin,
        pooled across subjects).  ``"binned"`` forces the legacy behaviour
        regardless of ``subject_col``.

        For backward compatibility the legacy point-estimate columns
        (``MESOR``, ``amplitude``, ``acrophase_ZT``, ``R2``) are ALWAYS computed
        from the pooled per-ZT-bin means, independent of ``fit_level``; only the
        added inference columns reflect the chosen ``fit_level`` (reported in the
        ``fit_level`` column).
    photoperiod_hours:
        Length of the light phase (hours).  ZT in ``[0, photoperiod_hours)`` is
        labelled ``"light"`` and the remainder ``"dark"``.  Defaults to 12.0
        (12:12 LD).  Set to e.g. 16.0 or 8.0 for non-12:12 designs.

    Returns
    -------
    A DataFrame with the legacy columns (``MESOR``, ``amplitude``,
    ``acrophase_ZT``, ``R2``, ``phase``, ``n_points``, ``n_observations``,
    ``fit_method``) plus rhythm-detection / uncertainty columns:
    ``p_rhythm`` (F-test that amplitude != 0 vs intercept-only model),
    ``amplitude_ci_low`` / ``amplitude_ci_high`` (95% CI of amplitude),
    ``acrophase_ci_low`` / ``acrophase_ci_high`` (95% CI of acrophase in ZT
    hours, via the delta method; NaN when unstable), and ``fit_level``.
    """
    warns: list[str] = []
    data = _analysis_frame(df, value_col, exclude_excluded)
    keys = _existing_cols(data, group_cols)

    if data.empty:
        warns.append("No valid rows available for circadian cosinor analysis.")
        return pd.DataFrame(), warns
    if zt_col not in data.columns:
        zt_col = _add_decimal_hour_from_timestamp(data, warns)
        if zt_col not in data.columns:
            warns.append("No zeitgeber or timestamp column available for cosinor analysis.")
            return pd.DataFrame(), warns
    if not keys:
        warns.append("No grouping columns found; fitting one global cosinor.")

    use_subject_level = fit_level == "subject" and subject_col in data.columns
    if fit_level == "subject" and not use_subject_level:
        warns.append(
            f"fit_level='subject' requested but '{subject_col}' is missing; "
            "falling back to binned cosinor fit."
        )

    rows: list[dict[str, Any]] = []
    grouped = data.groupby(keys, dropna=False) if keys else [((), data)]
    for group_key, grp in grouped:
        group_values = _group_key_values(keys, group_key)
        cols = [zt_col, value_col]
        if use_subject_level:
            cols = [zt_col, value_col, subject_col]
        clean = grp[cols].copy()
        clean[zt_col] = pd.to_numeric(clean[zt_col], errors="coerce") % period_hours
        clean[value_col] = pd.to_numeric(clean[value_col], errors="coerce")
        clean = clean.dropna(subset=[zt_col, value_col])
        if not clean.empty and zt_bin_hours > 0:
            clean["_zt_bin"] = (np.floor(clean[zt_col] / zt_bin_hours) * zt_bin_hours) % period_hours
            # Legacy point estimates always use pooled per-ZT-bin means.
            fit_data = clean.groupby("_zt_bin", dropna=False)[value_col].mean().reset_index()
            fit_x_col = "_zt_bin"
            if use_subject_level:
                infer_data = (
                    clean.groupby(["_zt_bin", subject_col], dropna=False)[value_col]
                    .mean()
                    .reset_index()
                )
            else:
                infer_data = fit_data
            infer_x_col = "_zt_bin"
        else:
            fit_data = clean
            fit_x_col = zt_col
            infer_data = clean
            infer_x_col = zt_col

        effective_level = "subject" if use_subject_level else "binned"
        if len(fit_data) < min_points or fit_data[fit_x_col].nunique() < 3:
            rows.append(
                {
                    **group_values,
                    "MESOR": np.nan,
                    "amplitude": np.nan,
                    "acrophase_ZT": np.nan,
                    "R2": np.nan,
                    "p_rhythm": np.nan,
                    "amplitude_ci_low": np.nan,
                    "amplitude_ci_high": np.nan,
                    "acrophase_ci_low": np.nan,
                    "acrophase_ci_high": np.nan,
                    "phase": None,
                    "n_points": int(len(fit_data)),
                    "n_observations": int(len(clean)),
                    "fit_level": effective_level,
                    "fit_method": "insufficient_data",
                }
            )
            continue

        x = fit_data[fit_x_col].to_numpy(dtype=float)
        y = fit_data[value_col].to_numpy(dtype=float)
        fit = _fit_cosinor(x, y, period_hours)
        infer_x = infer_data[infer_x_col].to_numpy(dtype=float)
        infer_y = infer_data[value_col].to_numpy(dtype=float)
        inference = _cosinor_inference(infer_x, infer_y, period_hours, warns)
        rows.append(
            {
                **group_values,
                "MESOR": fit["mesor"],
                "amplitude": fit["amplitude"],
                "acrophase_ZT": fit["acrophase"],
                "R2": fit["r2"],
                "p_rhythm": inference["p_rhythm"],
                "amplitude_ci_low": inference["amplitude_ci_low"],
                "amplitude_ci_high": inference["amplitude_ci_high"],
                "acrophase_ci_low": inference["acrophase_ci_low"],
                "acrophase_ci_high": inference["acrophase_ci_high"],
                "phase": _phase_from_zt(fit["acrophase"], photoperiod_hours),
                "n_points": int(len(fit_data)),
                "n_observations": int(len(clean)),
                "fit_level": effective_level,
                "fit_method": fit["method"],
            }
        )

    return pd.DataFrame(rows), warns


def summarize_light_dark(
    df: pd.DataFrame,
    value_col: str = "value",
    phase_col: str = "light_dark_phase",
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
    subject_col: str = "subject_id",
    exclude_excluded: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Summarize mean +/- SEM by light/dark phase per group/metric.

    Means and SEMs are computed over per-subject phase means when
    ``subject_col`` is available.  ``dark_light_ratio`` is the dark phase group
    mean divided by the light phase group mean.
    """
    warns: list[str] = []
    data = _analysis_frame(df, value_col, exclude_excluded)
    keys = _existing_cols(data, group_cols)

    if data.empty:
        warns.append("No valid rows available for light/dark summary.")
        return pd.DataFrame(), warns
    if phase_col not in data.columns:
        warns.append(f"Missing '{phase_col}'; cannot summarize light vs dark.")
        return pd.DataFrame(), warns

    data = data[data[phase_col].isin(["light", "dark"])].copy()
    if data.empty:
        warns.append("No rows labelled as light or dark.")
        return pd.DataFrame(), warns

    subject_keys = [*keys, phase_col]
    if subject_col in data.columns:
        subject_keys.append(subject_col)
        subject_means = (
            data.groupby(subject_keys, dropna=False)[value_col]
            .mean()
            .reset_index(name="subject_mean")
        )
        summary_source = subject_means
        summary_value = "subject_mean"
        observation_counts = (
            data.groupby([*keys, phase_col], dropna=False)
            .size()
            .reset_index(name="n_observations")
        )
    else:
        summary_source = data.rename(columns={value_col: "subject_mean"})
        summary_value = "subject_mean"
        observation_counts = (
            data.groupby([*keys, phase_col], dropna=False)
            .size()
            .reset_index(name="n_observations")
        )

    summary = (
        summary_source.groupby([*keys, phase_col], dropna=False)[summary_value]
        .agg(mean="mean", sd=lambda s: s.std(ddof=1), n_subjects="count")
        .reset_index()
    )
    summary["sem"] = summary.apply(lambda r: _sem_from_sd(r["sd"], r["n_subjects"]), axis=1)
    summary = summary.merge(observation_counts, on=[*keys, phase_col], how="left")

    ratios = _dark_light_ratios(summary, keys, phase_col)
    summary = summary.merge(ratios, on=keys, how="left") if keys else _merge_global_ratio(summary, ratios)
    summary = summary.rename(columns={phase_col: "phase"})
    return _ordered_summary_columns(summary, [*keys, "phase"]), warns


def summarize_time_bins(
    df: pd.DataFrame,
    bin_size: str | float | int = "1D",
    relative_to: str = "alignment",
    value_col: str = "value",
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
    subject_col: str = "subject_id",
    timestamp_col: str = "timestamp_utc",
    relative_time_col: str = "time_from_event_hours",
    exclude_excluded: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Summarize group means by daily, weekly, or custom time bins.

    ``relative_to='alignment'`` bins ``time_from_event_hours`` in hours.
    ``relative_to='absolute'`` bins timestamps by calendar day/week.
    """
    warns: list[str] = []
    data = _analysis_frame(df, value_col, exclude_excluded)
    keys = _existing_cols(data, group_cols)

    if data.empty:
        warns.append("No valid rows available for time-bin summary.")
        return pd.DataFrame(), warns

    data = data.copy()
    if relative_to == "alignment":
        if relative_time_col not in data.columns:
            warns.append(f"Missing '{relative_time_col}'; cannot summarize alignment-relative bins.")
            return pd.DataFrame(), warns
        bin_hours = _bin_size_to_hours(bin_size)
        rel = pd.to_numeric(data[relative_time_col], errors="coerce")
        data = data[rel.notna()].copy()
        rel = rel.loc[data.index]
        data["time_bin_start"] = np.floor(rel / bin_hours) * bin_hours
        data["time_bin_end"] = data["time_bin_start"] + bin_hours
        data["time_bin_label"] = data["time_bin_start"].map(_format_hour_bin)
        bin_cols = ["time_bin_start", "time_bin_end", "time_bin_label"]
        bin_unit = "hours_from_alignment"
    elif relative_to == "absolute":
        if timestamp_col not in data.columns:
            warns.append(f"Missing '{timestamp_col}'; cannot summarize absolute time bins.")
            return pd.DataFrame(), warns
        ts = pd.to_datetime(data[timestamp_col], errors="coerce")
        data = data[ts.notna()].copy()
        ts = ts.loc[data.index]
        data["time_bin_start"] = _absolute_bin_start(ts, bin_size)
        data["time_bin_end"] = _absolute_bin_end(data["time_bin_start"], bin_size)
        data["time_bin_label"] = data["time_bin_start"].astype(str)
        bin_cols = ["time_bin_start", "time_bin_end", "time_bin_label"]
        bin_unit = "absolute_time"
    else:
        raise ValueError("relative_to must be 'alignment' or 'absolute'.")

    if data.empty:
        warns.append("No rows with valid bin coordinates.")
        return pd.DataFrame(), warns

    subject_keys = [*keys, *bin_cols]
    if subject_col in data.columns:
        subject_keys.append(subject_col)
        subject_means = (
            data.groupby(subject_keys, dropna=False)[value_col]
            .mean()
            .reset_index(name="subject_mean")
        )
        summary_source = subject_means
        value_for_summary = "subject_mean"
    else:
        summary_source = data.rename(columns={value_col: "subject_mean"})
        value_for_summary = "subject_mean"

    summary = (
        summary_source.groupby([*keys, *bin_cols], dropna=False)[value_for_summary]
        .agg(mean="mean", sd=lambda s: s.std(ddof=1), n_subjects="count")
        .reset_index()
    )
    summary["sem"] = summary.apply(lambda r: _sem_from_sd(r["sd"], r["n_subjects"]), axis=1)
    counts = data.groupby([*keys, *bin_cols], dropna=False).size().reset_index(name="n_observations")
    summary = summary.merge(counts, on=[*keys, *bin_cols], how="left")
    summary["bin_size"] = str(bin_size)
    summary["relative_to"] = relative_to
    summary["bin_unit"] = bin_unit
    return _ordered_summary_columns(summary, [*keys, *bin_cols]), warns


def summarize_daily(
    df: pd.DataFrame,
    relative_to: str = "alignment",
    **kwargs: Any,
) -> tuple[pd.DataFrame, list[str]]:
    """Convenience wrapper for 1-day time-bin summaries."""
    return summarize_time_bins(df, bin_size="1D", relative_to=relative_to, **kwargs)


def summarize_weekly(
    df: pd.DataFrame,
    relative_to: str = "alignment",
    **kwargs: Any,
) -> tuple[pd.DataFrame, list[str]]:
    """Convenience wrapper for 1-week time-bin summaries."""
    return summarize_time_bins(df, bin_size="1W", relative_to=relative_to, **kwargs)


def compute_auc_per_animal(
    df: pd.DataFrame,
    start: float | pd.Timestamp | str | None = None,
    end: float | pd.Timestamp | str | None = None,
    x_col: str | None = None,
    value_col: str = "value",
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
    subject_col: str = "subject_id",
    exclude_excluded: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute trapezoidal AUC per animal and metric.

    If ``x_col`` is not provided, ``time_from_event_hours`` is preferred.  When
    timestamps are used, AUC is calculated against elapsed hours from the first
    included timestamp for each animal.
    """
    warns: list[str] = []
    data = _analysis_frame(df, value_col, exclude_excluded)
    keys = _existing_cols(data, group_cols)

    if data.empty:
        warns.append("No valid rows available for AUC calculation.")
        return pd.DataFrame(), warns
    if subject_col not in data.columns:
        warns.append(f"Missing '{subject_col}'; cannot compute per-animal AUC.")
        return pd.DataFrame(), warns

    x_col = x_col or _default_auc_x_col(data)
    if x_col is None:
        warns.append("No time coordinate available for AUC calculation.")
        return pd.DataFrame(), warns

    data = data.copy()
    x_is_datetime = _is_datetime_like(data[x_col])
    if x_is_datetime:
        raw_x = pd.to_datetime(data[x_col], errors="coerce")
        start_value = pd.to_datetime(start, errors="coerce") if start is not None else None
        end_value = pd.to_datetime(end, errors="coerce") if end is not None else None
    else:
        raw_x = pd.to_numeric(data[x_col], errors="coerce")
        start_value = float(start) if start is not None else None
        end_value = float(end) if end is not None else None

    mask = raw_x.notna()
    if start_value is not None:
        mask &= raw_x >= start_value
    if end_value is not None:
        mask &= raw_x <= end_value
    data = data[mask].copy()
    raw_x = raw_x.loc[data.index]

    if data.empty:
        warns.append("No rows fall inside the requested AUC window.")
        return pd.DataFrame(), warns

    data["_auc_x_raw"] = raw_x
    rows: list[dict[str, Any]] = []
    for group_key, grp in data.groupby([*keys, subject_col], dropna=False):
        key_values = _group_key_values([*keys, subject_col], group_key)
        clean = grp[["_auc_x_raw", value_col]].copy().dropna()
        if clean.empty:
            continue
        clean = clean.sort_values("_auc_x_raw")

        if x_is_datetime:
            x0 = clean["_auc_x_raw"].iloc[0]
            x = (clean["_auc_x_raw"] - x0).dt.total_seconds().to_numpy(dtype=float) / 3600.0
            x_start = clean["_auc_x_raw"].iloc[0]
            x_end = clean["_auc_x_raw"].iloc[-1]
            x_unit = "elapsed_hours"
        else:
            x = clean["_auc_x_raw"].to_numpy(dtype=float)
            x_start = float(x[0])
            x_end = float(x[-1])
            x_unit = f"{x_col}"

        y = clean[value_col].to_numpy(dtype=float)
        auc = float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))
        rows.append(
            {
                **key_values,
                "auc": auc,
                "n_points": int(len(clean)),
                "x_col": x_col,
                "x_unit": x_unit,
                "x_start": x_start,
                "x_end": x_end,
            }
        )

    return pd.DataFrame(rows), warns


def quick_exploratory_stats(
    df: pd.DataFrame,
    value_col: str = "value",
    group_col: str = "group_id",
    block_cols: Sequence[str] | None = None,
    subject_col: str = "subject_id",
    exclude_excluded: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Run quick exploratory normality and nonparametric group comparisons.

    Uses ``scipy.stats`` when available.  If SciPy is missing, returns a
    structured warning table instead of raising.

    For every two-group comparison row the table also carries estimation-first
    columns: ``median_difference`` (median of group 2 minus median of group 1,
    where the two group labels are taken in sorted order as in ``comparison``),
    ``diff_ci_low`` / ``diff_ci_high`` (bootstrap 95% CI of that median
    difference), ``effect_size_ci_low`` / ``effect_size_ci_high`` (bootstrap 95%
    CI of the rank-biserial effect size), ``min_possible_p`` (smallest two-sided
    Mann-Whitney p achievable for the two group sizes, ``2 / C(n1+n2, n1)``), and
    ``small_n_warning`` (True when ``min_possible_p > 0.05`` -- the test cannot
    reach significance).  Bootstraps use 2000 resamples with a fixed seed
    (``np.random.default_rng(0)``) so results are deterministic.  The small-n
    condition is reported through the ``small_n_warning`` column rather than the
    returned ``warns`` list (which stays reserved for table-level problems).
    """
    warns: list[str] = []
    # Per-comparison "cannot reach significance" notes.  These are surfaced
    # canonically through the ``small_n_warning`` column on each comparison row
    # (and as human-readable strings here) but are intentionally kept out of the
    # returned ``warns`` list so that ``warns`` stays reserved for table-level
    # problems, preserving the established API contract for existing callers.
    small_n_warnings: list[str] = []
    data = _analysis_frame(df, value_col, exclude_excluded)
    if data.empty:
        warns.append("No valid rows available for exploratory statistics.")
        return pd.DataFrame(), warns
    if group_col not in data.columns:
        warns.append(f"Missing '{group_col}'; cannot compare groups.")
        return pd.DataFrame(), warns

    block_cols = list(block_cols) if block_cols is not None else _default_block_cols(data, group_col)
    block_cols = _existing_cols(data, block_cols)
    stats_mod = _load_scipy_stats()

    values = _subject_level_values(data, value_col, group_col, block_cols, subject_col)
    grouped_blocks = values.groupby(block_cols, dropna=False) if block_cols else [((), values)]

    if stats_mod is None:
        warns.append("scipy.stats is not available; exploratory p-values were not computed.")
        rows = []
        for block_key, block in grouped_blocks:
            group_names = [str(g) for g in block[group_col].dropna().unique()]
            rows.append(
                {
                    **_group_key_values(block_cols, block_key),
                    "test": "scipy.stats unavailable",
                    "comparison": " vs ".join(group_names),
                    "group": None,
                    "n_groups": len(group_names),
                    "n_total": int(block[value_col].notna().sum()),
                    "statistic": np.nan,
                    "effect_size": np.nan,
                    "effect_size_name": None,
                    "p_value": np.nan,
                    "p_label": None,
                    "q_value": np.nan,
                    "q_label": None,
                    "median_difference": np.nan,
                    "diff_ci_low": np.nan,
                    "diff_ci_high": np.nan,
                    "effect_size_ci_low": np.nan,
                    "effect_size_ci_high": np.nan,
                    "min_possible_p": np.nan,
                    "small_n_warning": False,
                    "scipy_available": False,
                    "exploratory": True,
                    "disclaimer": EXPLORATORY_STATS_DISCLAIMER,
                }
            )
        return pd.DataFrame(rows), warns

    rows = []
    for block_key, block in grouped_blocks:
        block_values = _group_key_values(block_cols, block_key)
        group_arrays = []
        group_names = []
        for group_name, grp in block.groupby(group_col, dropna=False):
            arr = pd.to_numeric(grp[value_col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(arr) == 0:
                continue
            group_arrays.append(arr)
            group_names.append(str(group_name))

            stat = p_value = np.nan
            test_name = "Shapiro-Wilk"
            if 3 <= len(arr) <= 5000:
                stat, p_value = stats_mod.shapiro(arr)
            rows.append(
                {
                    **block_values,
                    "test": test_name,
                    "comparison": str(group_name),
                    "group": str(group_name),
                    "n_groups": 1,
                    "n_total": int(len(arr)),
                    "statistic": float(stat) if pd.notna(stat) else np.nan,
                    "effect_size": np.nan,
                    "effect_size_name": None,
                    "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                    "p_label": _p_label(p_value),
                    "q_value": np.nan,
                    "q_label": None,
                    "median_difference": np.nan,
                    "diff_ci_low": np.nan,
                    "diff_ci_high": np.nan,
                    "effect_size_ci_low": np.nan,
                    "effect_size_ci_high": np.nan,
                    "min_possible_p": np.nan,
                    "small_n_warning": False,
                    "scipy_available": True,
                    "exploratory": True,
                    "disclaimer": EXPLORATORY_STATS_DISCLAIMER,
                }
            )

        median_difference = np.nan
        diff_ci_low = diff_ci_high = np.nan
        effect_size_ci_low = effect_size_ci_high = np.nan
        min_possible_p = np.nan
        small_n_warning = False
        if len(group_arrays) == 2:
            stat, p_value = stats_mod.mannwhitneyu(group_arrays[0], group_arrays[1], alternative="two-sided")
            test_name = "Mann-Whitney U"
            effect_size = _rank_biserial_from_u(stat, len(group_arrays[0]), len(group_arrays[1]))
            effect_size_name = "rank-biserial"
            comparison_label = " vs ".join(group_names)
            est = _two_group_estimation(group_arrays[0], group_arrays[1])
            median_difference = est["median_difference"]
            diff_ci_low = est["diff_ci_low"]
            diff_ci_high = est["diff_ci_high"]
            effect_size_ci_low = est["effect_size_ci_low"]
            effect_size_ci_high = est["effect_size_ci_high"]
            min_possible_p = _min_possible_mannwhitney_p(
                len(group_arrays[0]), len(group_arrays[1])
            )
            small_n_warning = bool(np.isfinite(min_possible_p) and min_possible_p > 0.05)
            if small_n_warning:
                small_n_warnings.append(
                    f"Comparison '{comparison_label}' has too few subjects "
                    f"(n={len(group_arrays[0])} vs {len(group_arrays[1])}); the Mann-Whitney "
                    f"test cannot reach p<0.05 (smallest achievable p={min_possible_p:.3g})."
                )
        elif len(group_arrays) > 2:
            stat, p_value = stats_mod.kruskal(*group_arrays)
            test_name = "Kruskal-Wallis"
            effect_size = _kruskal_epsilon_squared(stat, group_arrays)
            effect_size_name = "epsilon-squared"
        else:
            stat = p_value = np.nan
            test_name = "group comparison skipped"
            effect_size = np.nan
            effect_size_name = None

        rows.append(
            {
                **block_values,
                "test": test_name,
                "comparison": " vs ".join(group_names),
                "group": None,
                "n_groups": len(group_arrays),
                "n_total": int(sum(len(arr) for arr in group_arrays)),
                "statistic": float(stat) if pd.notna(stat) else np.nan,
                "effect_size": effect_size,
                "effect_size_name": effect_size_name,
                "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                "p_label": _p_label(p_value),
                "q_value": np.nan,
                "q_label": None,
                "median_difference": median_difference,
                "diff_ci_low": diff_ci_low,
                "diff_ci_high": diff_ci_high,
                "effect_size_ci_low": effect_size_ci_low,
                "effect_size_ci_high": effect_size_ci_high,
                "min_possible_p": min_possible_p,
                "small_n_warning": small_n_warning,
                "scipy_available": True,
                "exploratory": True,
                "disclaimer": EXPLORATORY_STATS_DISCLAIMER,
            }
        )

    return _add_fdr_q_values(pd.DataFrame(rows)), warns


def summarize_nonparametric_circadian(
    df: pd.DataFrame,
    value_col: str = "value",
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
    subject_col: str = "subject_id",
    timestamp_col: str = "timestamp_local",
    bin_hours: float = 1.0,
    exclude_excluded: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute non-parametric circadian (actigraphy) descriptors per subject.

    Builds an hourly-binned activity profile for each subject from
    ``timestamp_local`` (falling back to ``timestamp_utc``) and returns the
    standard van Someren descriptors:

    * ``IS`` -- interdaily stability: variance of the average 24 h profile over
      the overall variance, ``[0, 1]`` (1 = perfectly stable day-to-day).
    * ``IV`` -- intradaily variability: mean squared first difference over the
      overall variance (higher = more fragmented).
    * ``M10`` / ``L5`` -- mean activity of the most-active 10 h and least-active
      5 h consecutive windows of the average 24 h profile.
    * ``RA`` -- relative amplitude ``(M10 - L5) / (M10 + L5)``.

    One row per subject is returned with ``group_id`` / ``metric_name`` /
    ``subject_id`` plus ``IS``, ``IV``, ``RA``, ``M10``, ``L5``, ``n_days``,
    ``fit_method`` and ``notes``.  Warns and returns an empty frame when no
    usable timestamp column is present.
    """
    warns: list[str] = []
    data = _analysis_frame(df, value_col, exclude_excluded)
    keys = _existing_cols(data, group_cols)
    if data.empty:
        warns.append("No valid rows available for non-parametric circadian analysis.")
        return pd.DataFrame(), warns

    ts_col = _pick_timestamp_col(data, timestamp_col)
    if ts_col is None:
        warns.append("No usable timestamp column for non-parametric circadian analysis.")
        return pd.DataFrame(), warns
    if subject_col not in data.columns:
        warns.append(f"Missing '{subject_col}'; cannot compute per-subject descriptors.")
        return pd.DataFrame(), warns
    if bin_hours <= 0:
        warns.append("bin_hours must be positive.")
        return pd.DataFrame(), warns

    data = data.copy()
    data["_ts"] = pd.to_datetime(data[ts_col], errors="coerce")
    data = data[data["_ts"].notna()].copy()
    if data.empty:
        warns.append("No rows with valid timestamps for non-parametric circadian analysis.")
        return pd.DataFrame(), warns

    rows: list[dict[str, Any]] = []
    for group_key, grp in data.groupby([*keys, subject_col], dropna=False):
        key_values = _group_key_values([*keys, subject_col], group_key)
        profile = _binned_activity_series(grp, "_ts", value_col, bin_hours)
        if profile is None or profile.empty:
            continue
        metrics = _nonparametric_circadian_metrics(profile, bin_hours)
        rows.append(
            {
                **key_values,
                "IS": metrics["IS"],
                "IV": metrics["IV"],
                "RA": metrics["RA"],
                "M10": metrics["M10"],
                "L5": metrics["L5"],
                "n_days": metrics["n_days"],
                "fit_method": "van_someren",
                "notes": metrics["notes"],
            }
        )

    if not rows:
        warns.append("No subjects produced a usable activity profile.")
    return pd.DataFrame(rows), warns


def summarize_activity_bouts(
    df: pd.DataFrame,
    value_col: str = "value",
    threshold: str | float = "auto",
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
    subject_col: str = "subject_id",
    timestamp_col: str = "timestamp_local",
    exclude_excluded: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Activity bout / fragmentation descriptors per subject and metric.

    Each time bin is classified active vs inactive: ``threshold="auto"`` uses the
    per-subject median of ``value_col`` (a bin is active when ``value >
    threshold``); a float threshold may be supplied instead.  Bout lengths are
    converted to minutes using the median timestamp spacing for that subject.

    Returns per subject/metric: ``n_active_bouts``, ``mean_active_bout_minutes``,
    ``max_active_bout_minutes``, ``n_inactive_bouts``,
    ``mean_inactive_bout_minutes``, ``total_active_minutes``,
    ``fraction_time_active``, ``daily_total`` (mean per-day sum of ``value_col``)
    and the ``threshold`` used.  Warns when no timestamp column is available.
    """
    warns: list[str] = []
    data = _analysis_frame(df, value_col, exclude_excluded)
    keys = _existing_cols(data, group_cols)
    if data.empty:
        warns.append("No valid rows available for activity-bout analysis.")
        return pd.DataFrame(), warns

    ts_col = _pick_timestamp_col(data, timestamp_col)
    if ts_col is None:
        warns.append("No usable timestamp column for activity-bout analysis.")
        return pd.DataFrame(), warns
    if subject_col not in data.columns:
        warns.append(f"Missing '{subject_col}'; cannot compute per-subject bouts.")
        return pd.DataFrame(), warns

    data = data.copy()
    data["_ts"] = pd.to_datetime(data[ts_col], errors="coerce")
    data = data[data["_ts"].notna()].copy()
    if data.empty:
        warns.append("No rows with valid timestamps for activity-bout analysis.")
        return pd.DataFrame(), warns

    rows: list[dict[str, Any]] = []
    for group_key, grp in data.groupby([*keys, subject_col], dropna=False):
        key_values = _group_key_values([*keys, subject_col], group_key)
        clean = grp[["_ts", value_col]].dropna().sort_values("_ts")
        if len(clean) < 2:
            continue
        values = clean[value_col].to_numpy(dtype=float)
        ts = clean["_ts"]
        diffs = ts.diff().dt.total_seconds().to_numpy()[1:]
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        bin_minutes = float(np.median(diffs)) / 60.0 if len(diffs) else np.nan

        thr = float(np.median(values)) if threshold == "auto" else float(threshold)
        active = values > thr
        active_runs = _run_lengths(active, True)
        inactive_runs = _run_lengths(active, False)

        daily_total = _mean_daily_sum(ts, values)
        total_active_minutes = (
            float(active.sum()) * bin_minutes if np.isfinite(bin_minutes) else np.nan
        )
        rows.append(
            {
                **key_values,
                "n_active_bouts": int(len(active_runs)),
                "mean_active_bout_minutes": _runs_to_minutes_mean(active_runs, bin_minutes),
                "max_active_bout_minutes": (
                    float(max(active_runs)) * bin_minutes
                    if active_runs and np.isfinite(bin_minutes)
                    else np.nan
                ),
                "n_inactive_bouts": int(len(inactive_runs)),
                "mean_inactive_bout_minutes": _runs_to_minutes_mean(inactive_runs, bin_minutes),
                "total_active_minutes": total_active_minutes,
                "fraction_time_active": float(active.mean()),
                "daily_total": daily_total,
                "threshold": thr,
                "bin_minutes": bin_minutes,
            }
        )

    if not rows:
        warns.append("No subjects produced usable activity-bout statistics.")
    return pd.DataFrame(rows), warns


def compare_window_summaries(
    df: pd.DataFrame,
    windows: Sequence[tuple[str, float, float]],
    value_col: str = "value",
    group_col: str = "group_id",
    subject_col: str = "subject_id",
    metric_col: str = "metric_name",
    x_col: str = "time_from_event_hours",
    statistic: str = "mean",
    exclude_excluded: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Window-summary group contrast (autocorrelation-safe alternative to per-bin).

    Each window ``(label, start, end)`` (in ``x_col`` units, half-open
    ``[start, end)``) is reduced to ONE value per subject -- the mean of
    ``value_col`` for ``statistic="mean"`` or the trapezoidal AUC over ``x_col``
    for ``statistic="auc"``.  A single exploratory group comparison
    (Mann-Whitney for two groups, Kruskal-Wallis for more) is then run per window
    per metric, with rank-biserial effect size and a bootstrap CI of the median
    difference, and Benjamini-Hochberg FDR is applied ACROSS windows (within each
    metric).  Returns one row per window per metric.  The reduction used is
    recorded in the ``window_statistic`` column; ``statistic`` holds the U/H test
    statistic (matching :func:`quick_exploratory_stats`).
    """
    warns: list[str] = []
    data = _analysis_frame(df, value_col, exclude_excluded)
    if data.empty:
        warns.append("No valid rows available for window-summary contrast.")
        return pd.DataFrame(), warns
    for required in (group_col, x_col):
        if required not in data.columns:
            warns.append(f"Missing '{required}'; cannot run window-summary contrast.")
            return pd.DataFrame(), warns
    if statistic not in ("mean", "auc"):
        warns.append("statistic must be 'mean' or 'auc'.")
        return pd.DataFrame(), warns
    if not windows:
        warns.append("No windows supplied for window-summary contrast.")
        return pd.DataFrame(), warns

    stats_mod = _load_scipy_stats()
    data = data.copy()
    data["_x"] = pd.to_numeric(data[x_col], errors="coerce")
    metric_keys = [metric_col] if metric_col in data.columns else []
    rows: list[dict[str, Any]] = []

    if metric_keys:
        grouped_metrics = data.groupby(metric_col, dropna=False)
    else:
        grouped_metrics = [((), data)]
    for metric_key, metric_block in grouped_metrics:
        metric_values = _group_key_values(metric_keys, metric_key)
        for label, start, end in windows:
            window = metric_block[
                metric_block["_x"].notna()
                & (metric_block["_x"] >= float(start))
                & (metric_block["_x"] < float(end))
            ]
            reduced = _reduce_window_per_subject(
                window, value_col, group_col, subject_col, statistic
            )
            row = _window_comparison_row(reduced, value_col, group_col, stats_mod)
            row = {
                **metric_values,
                "window_label": label,
                "window_start": float(start),
                "window_end": float(end),
                "window_statistic": statistic,
                **row,
                "exploratory": True,
                "disclaimer": EXPLORATORY_STATS_DISCLAIMER,
            }
            rows.append(row)

    if stats_mod is None:
        warns.append("scipy.stats is not available; window-summary p-values were not computed.")

    out = pd.DataFrame(rows)
    if out.empty:
        return out, warns
    # BH-FDR across windows within each metric.
    if metric_keys:
        parts = [
            _add_fdr_q_values(block.copy())
            for _, block in out.groupby(metric_keys, dropna=False, sort=False)
        ]
        out = pd.concat(parts, ignore_index=True)
    else:
        out = _add_fdr_q_values(out)
    return out, warns


def estimate_period(
    df: pd.DataFrame,
    value_col: str = "value",
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
    subject_col: str = "subject_id",
    timestamp_col: str = "timestamp_local",
    min_period_h: float = 18.0,
    max_period_h: float = 30.0,
    exclude_excluded: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Estimate the dominant period per subject via a Lomb-Scargle periodogram.

    Operates on each subject's ``(elapsed_hours, value)`` series and searches for
    the strongest period in ``[min_period_h, max_period_h]`` using
    ``scipy.signal.lombscargle`` (mean-subtracted, ``precenter`` style).  Useful
    for free-running / tau designs.  Returns per subject:
    ``estimated_period_hours``, ``peak_power``, ``n_points`` and ``method``.
    When SciPy's periodogram is unavailable the estimate is NaN with a warning.
    """
    warns: list[str] = []
    data = _analysis_frame(df, value_col, exclude_excluded)
    keys = _existing_cols(data, group_cols)
    if data.empty:
        warns.append("No valid rows available for period estimation.")
        return pd.DataFrame(), warns

    ts_col = _pick_timestamp_col(data, timestamp_col)
    if ts_col is None:
        warns.append("No usable timestamp column for period estimation.")
        return pd.DataFrame(), warns
    if subject_col not in data.columns:
        warns.append(f"Missing '{subject_col}'; cannot estimate per-subject period.")
        return pd.DataFrame(), warns
    if not (0.0 < min_period_h < max_period_h):
        warns.append("Require 0 < min_period_h < max_period_h for period estimation.")
        return pd.DataFrame(), warns

    lombscargle = _load_lombscargle()
    if lombscargle is None:
        warns.append("scipy.signal.lombscargle is not available; period estimates are NaN.")

    data = data.copy()
    data["_ts"] = pd.to_datetime(data[ts_col], errors="coerce")
    data = data[data["_ts"].notna()].copy()

    rows: list[dict[str, Any]] = []
    for group_key, grp in data.groupby([*keys, subject_col], dropna=False):
        key_values = _group_key_values([*keys, subject_col], group_key)
        clean = grp[["_ts", value_col]].dropna().sort_values("_ts")
        n_points = int(len(clean))
        period = np.nan
        peak_power = np.nan
        if n_points >= 4 and lombscargle is not None:
            t0 = clean["_ts"].iloc[0]
            t = (clean["_ts"] - t0).dt.total_seconds().to_numpy(dtype=float) / 3600.0
            y = clean[value_col].to_numpy(dtype=float)
            period, peak_power = _lombscargle_peak(
                lombscargle, t, y, min_period_h, max_period_h
            )
        rows.append(
            {
                **key_values,
                "estimated_period_hours": period,
                "peak_power": peak_power,
                "n_points": n_points,
                "method": "lomb_scargle" if lombscargle is not None else "unavailable",
            }
        )

    return pd.DataFrame(rows), warns


def _pick_timestamp_col(data: pd.DataFrame, preferred: str) -> str | None:
    for col in (preferred, "timestamp_local", "timestamp_utc"):
        if col in data.columns and pd.to_datetime(data[col], errors="coerce").notna().any():
            return col
    return None


def _binned_activity_series(
    grp: pd.DataFrame,
    ts_col: str,
    value_col: str,
    bin_hours: float,
) -> pd.Series | None:
    """Return an evenly hourly-binned activity series indexed by bin start."""
    clean = grp[[ts_col, value_col]].dropna().sort_values(ts_col)
    if clean.empty:
        return None
    freq = pd.to_timedelta(bin_hours, unit="h")
    binned = (
        clean.set_index(ts_col)[value_col]
        .resample(freq, origin="epoch")
        .mean()
    )
    # Fill internal gaps so the regular grid used by IS/IV is complete.
    binned = binned.reindex(
        pd.date_range(binned.index.min(), binned.index.max(), freq=freq)
    )
    if binned.notna().sum() < 2:
        return None
    return binned


def _nonparametric_circadian_metrics(profile: pd.Series, bin_hours: float) -> dict[str, Any]:
    """Compute IS/IV/RA/M10/L5 from a regularly binned activity series."""
    nan = float("nan")
    notes = ""
    values = profile.to_numpy(dtype=float)
    valid = values[np.isfinite(values)]
    n = len(values)
    overall_mean = float(np.nanmean(values)) if np.isfinite(np.nanmean(values)) else nan
    overall_var = float(np.nanmean((values - overall_mean) ** 2)) if n else nan

    bins_per_day = int(round(24.0 / bin_hours))
    n_days = float(np.isfinite(values).sum() / bins_per_day) if bins_per_day else nan

    # Interdaily stability: variance of the average daily profile / overall var.
    is_value = nan
    if bins_per_day >= 2 and overall_var and overall_var > 0:
        phase = np.arange(n) % bins_per_day
        hourly_means = np.array(
            [np.nanmean(values[phase == h]) for h in range(bins_per_day)]
        )
        if np.isfinite(hourly_means).all():
            between_var = float(np.mean((hourly_means - overall_mean) ** 2))
            is_value = float(between_var / overall_var)

    # Intradaily variability: mean squared first difference / overall var.
    iv_value = nan
    if overall_var and overall_var > 0:
        diffs = np.diff(values)
        diffs = diffs[np.isfinite(diffs)]
        if len(diffs):
            iv_value = float(np.mean(diffs**2) / overall_var)

    m10, l5, ra = _m10_l5_ra(valid, bin_hours)
    if not np.isfinite(is_value):
        notes = "IS undefined (insufficient days/variance)"
    return {
        "IS": is_value,
        "IV": iv_value,
        "RA": ra,
        "M10": m10,
        "L5": l5,
        "n_days": n_days,
        "notes": notes,
    }


def _m10_l5_ra(values: np.ndarray, bin_hours: float) -> tuple[float, float, float]:
    """Most-active 10 h (M10), least-active 5 h (L5) and relative amplitude."""
    nan = float("nan")
    if len(values) == 0:
        return nan, nan, nan
    win10 = max(1, int(round(10.0 / bin_hours)))
    win5 = max(1, int(round(5.0 / bin_hours)))
    m10 = _rolling_window_extreme(values, win10, want_max=True)
    l5 = _rolling_window_extreme(values, win5, want_max=False)
    if not (np.isfinite(m10) and np.isfinite(l5)) or (m10 + l5) == 0:
        ra = nan
    else:
        ra = float((m10 - l5) / (m10 + l5))
    return m10, l5, ra


def _rolling_window_extreme(values: np.ndarray, window: int, want_max: bool) -> float:
    if len(values) < window or window <= 0:
        if len(values) == 0:
            return float("nan")
        window = len(values)
    means = np.convolve(values, np.ones(window) / window, mode="valid")
    if len(means) == 0:
        return float("nan")
    return float(np.max(means) if want_max else np.min(means))


def _run_lengths(mask: np.ndarray, target: bool) -> list[int]:
    """Lengths of consecutive runs where ``mask == target``."""
    runs: list[int] = []
    count = 0
    for value in mask:
        if bool(value) == target:
            count += 1
        elif count:
            runs.append(count)
            count = 0
    if count:
        runs.append(count)
    return runs


def _runs_to_minutes_mean(runs: list[int], bin_minutes: float) -> float:
    if not runs or not np.isfinite(bin_minutes):
        return float("nan")
    return float(np.mean(runs)) * bin_minutes


def _mean_daily_sum(ts: pd.Series, values: np.ndarray) -> float:
    frame = pd.DataFrame({"_day": ts.dt.floor("D").to_numpy(), "_v": values})
    daily = frame.groupby("_day")["_v"].sum()
    if daily.empty:
        return float("nan")
    return float(daily.mean())


def _reduce_window_per_subject(
    window: pd.DataFrame,
    value_col: str,
    group_col: str,
    subject_col: str,
    statistic: str,
) -> pd.DataFrame:
    """Reduce a window to one value per subject (mean or trapezoidal AUC)."""
    if window.empty:
        return pd.DataFrame(columns=[group_col, "_reduced"])
    has_subject = subject_col in window.columns
    keys = [group_col] + ([subject_col] if has_subject else [])
    out_rows = []
    for key, grp in window.groupby(keys, dropna=False):
        key_values = _group_key_values(keys, key)
        clean = grp[["_x", value_col]].dropna()
        if clean.empty:
            continue
        if statistic == "auc" and len(clean) >= 2:
            clean = clean.sort_values("_x")
            x = clean["_x"].to_numpy(dtype=float)
            y = clean[value_col].to_numpy(dtype=float)
            reduced = (
                float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))
            )
        else:
            reduced = float(clean[value_col].mean())
        out_rows.append({**key_values, "_reduced": reduced})
    return pd.DataFrame(out_rows)


def _window_comparison_row(
    reduced: pd.DataFrame,
    value_col: str,
    group_col: str,
    stats_mod: Any | None,
) -> dict[str, Any]:
    """Run a single exploratory group comparison on per-subject reduced values."""
    nan = float("nan")
    base = {
        "test": "group comparison skipped",
        "comparison": "",
        "n_groups": 0,
        "n_total": 0,
        "statistic": nan,
        "effect_size": nan,
        "effect_size_name": None,
        "p_value": nan,
        "p_label": None,
        "q_value": nan,
        "q_label": None,
        "median_difference": nan,
        "diff_ci_low": nan,
        "diff_ci_high": nan,
        "scipy_available": stats_mod is not None,
    }
    if reduced.empty or group_col not in reduced.columns:
        return base

    group_arrays = []
    group_names = []
    for group_name, grp in reduced.groupby(group_col, dropna=False):
        arr = pd.to_numeric(grp["_reduced"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(arr) == 0:
            continue
        group_arrays.append(arr)
        group_names.append(str(group_name))

    base["comparison"] = " vs ".join(group_names)
    base["n_groups"] = len(group_arrays)
    base["n_total"] = int(sum(len(a) for a in group_arrays))
    if stats_mod is None or len(group_arrays) < 2:
        return base

    if len(group_arrays) == 2:
        stat, p_value = stats_mod.mannwhitneyu(
            group_arrays[0], group_arrays[1], alternative="two-sided"
        )
        base["test"] = "Mann-Whitney U"
        base["effect_size"] = _rank_biserial_from_u(
            stat, len(group_arrays[0]), len(group_arrays[1])
        )
        base["effect_size_name"] = "rank-biserial"
        est = _two_group_estimation(group_arrays[0], group_arrays[1])
        base["median_difference"] = est["median_difference"]
        base["diff_ci_low"] = est["diff_ci_low"]
        base["diff_ci_high"] = est["diff_ci_high"]
    else:
        stat, p_value = stats_mod.kruskal(*group_arrays)
        base["test"] = "Kruskal-Wallis"
        base["effect_size"] = _kruskal_epsilon_squared(stat, group_arrays)
        base["effect_size_name"] = "epsilon-squared"

    base["statistic"] = float(stat) if pd.notna(stat) else nan
    base["p_value"] = float(p_value) if pd.notna(p_value) else nan
    base["p_label"] = _p_label(p_value)
    return base


def _load_lombscargle() -> Any | None:
    try:
        signal = importlib.import_module("scipy.signal")
    except Exception:
        return None
    return getattr(signal, "lombscargle", None)


def _lombscargle_peak(
    lombscargle: Any,
    t: np.ndarray,
    y: np.ndarray,
    min_period_h: float,
    max_period_h: float,
    n_freq: int = 2000,
) -> tuple[float, float]:
    """Return (period_hours, power) of the strongest Lomb-Scargle peak in range."""
    nan = float("nan")
    if len(t) < 4:
        return nan, nan
    y = y - float(np.mean(y))
    if not np.any(y != 0):
        return nan, nan
    periods = np.linspace(min_period_h, max_period_h, n_freq)
    angular = 2.0 * np.pi / periods
    try:
        power = lombscargle(t.astype(float), y.astype(float), angular, normalize=True)
    except Exception:
        return nan, nan
    if not np.any(np.isfinite(power)):
        return nan, nan
    idx = int(np.nanargmax(power))
    return float(periods[idx]), float(power[idx])


def _analysis_frame(df: pd.DataFrame, value_col: str, exclude_excluded: bool) -> pd.DataFrame:
    data = df.copy()
    if value_col not in data.columns:
        return pd.DataFrame()
    if exclude_excluded and "is_excluded" in data.columns:
        data = data[~data["is_excluded"].fillna(False)].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    return data[data[value_col].notna()].copy()


def _existing_cols(df: pd.DataFrame, cols: Sequence[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _group_key_values(keys: Sequence[str], group_key: Any) -> dict[str, Any]:
    if not keys:
        return {}
    if len(keys) == 1:
        group_key = (group_key,)
    return dict(zip(keys, group_key, strict=False))


def _fit_cosinor(x: np.ndarray, y: np.ndarray, period_hours: float) -> dict[str, float | str]:
    scipy_fit = _fit_cosinor_scipy(x, y, period_hours)
    if scipy_fit is not None:
        return scipy_fit
    return _fit_cosinor_linear(x, y, period_hours)


def _fit_cosinor_scipy(
    x: np.ndarray,
    y: np.ndarray,
    period_hours: float,
) -> dict[str, float | str] | None:
    try:
        optimize = importlib.import_module("scipy.optimize")
    except Exception:
        return None

    linear = _fit_cosinor_linear(x, y, period_hours)
    if not np.isfinite(linear["amplitude"]):
        return None

    def model(zt: np.ndarray, mesor: float, amplitude: float, acrophase: float) -> np.ndarray:
        return mesor + amplitude * np.cos(2.0 * np.pi * (zt - acrophase) / period_hours)

    try:
        params, _ = optimize.curve_fit(
            model,
            x,
            y,
            p0=[linear["mesor"], max(linear["amplitude"], 0.0), linear["acrophase"]],
            bounds=([-np.inf, 0.0, 0.0], [np.inf, np.inf, period_hours]),
            maxfev=10000,
        )
    except Exception:
        return linear

    y_hat = model(x, *params)
    return {
        "mesor": float(params[0]),
        "amplitude": float(params[1]),
        "acrophase": float(params[2] % period_hours),
        "r2": _r_squared(y, y_hat),
        "method": "scipy_curve_fit",
    }


def _fit_cosinor_linear(x: np.ndarray, y: np.ndarray, period_hours: float) -> dict[str, float | str]:
    radians = 2.0 * np.pi * x / period_hours
    design = np.column_stack([np.ones_like(radians), np.cos(radians), np.sin(radians)])
    mesor, beta_cos, beta_sin = np.linalg.lstsq(design, y, rcond=None)[0]
    amplitude = math.sqrt(beta_cos**2 + beta_sin**2)
    acrophase = (math.atan2(beta_sin, beta_cos) * period_hours / (2.0 * np.pi)) % period_hours
    y_hat = design @ np.array([mesor, beta_cos, beta_sin])
    return {
        "mesor": float(mesor),
        "amplitude": float(amplitude),
        "acrophase": float(acrophase),
        "r2": _r_squared(y, y_hat),
        "method": "linear_least_squares",
    }


def _cosinor_inference(
    x: np.ndarray,
    y: np.ndarray,
    period_hours: float,
    warns: list[str] | None = None,
) -> dict[str, float]:
    """
    Closed-form inference for the linear cosinor model.

    Fits ``y = mesor + beta_cos*cos(w x) + beta_sin*sin(w x)`` by ordinary least
    squares and derives, from the residual covariance of the design matrix:

    * ``p_rhythm`` -- two-degree-of-freedom F-test of the null hypothesis that
      both ``beta_cos`` and ``beta_sin`` are zero (zero-amplitude rhythm) versus
      the intercept-only model.
    * ``amplitude_ci_low`` / ``amplitude_ci_high`` -- 95% CI for the amplitude
      ``sqrt(beta_cos^2 + beta_sin^2)`` via the delta method.
    * ``acrophase_ci_low`` / ``acrophase_ci_high`` -- 95% CI for the acrophase
      (ZT hours) via the delta method on ``atan2(beta_sin, beta_cos)``.

    The mesor/amplitude/acrophase/r2 from the same linear fit are also returned.
    SciPy's F and t distributions are used when available; otherwise p-values and
    CIs are returned as NaN with a warning.
    """
    nan = float("nan")
    result = {
        "mesor": nan,
        "amplitude": nan,
        "acrophase": nan,
        "r2": nan,
        "p_rhythm": nan,
        "amplitude_ci_low": nan,
        "amplitude_ci_high": nan,
        "acrophase_ci_low": nan,
        "acrophase_ci_high": nan,
    }
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    n_params = 3
    if n < n_params + 1:
        return result

    radians = 2.0 * np.pi * x / period_hours
    design = np.column_stack([np.ones_like(radians), np.cos(radians), np.sin(radians)])
    try:
        xtx_inv = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return result
    beta = xtx_inv @ design.T @ y
    mesor, beta_cos, beta_sin = beta
    y_hat = design @ beta
    residuals = y - y_hat
    amplitude = math.hypot(beta_cos, beta_sin)
    acrophase = (math.atan2(beta_sin, beta_cos) * period_hours / (2.0 * np.pi)) % period_hours

    result["mesor"] = float(mesor)
    result["amplitude"] = float(amplitude)
    result["acrophase"] = float(acrophase)
    result["r2"] = _r_squared(y, y_hat)

    dof = n - n_params
    ss_res = float(np.sum(residuals**2))
    sigma2 = ss_res / dof if dof > 0 else nan

    stats_mod = _load_scipy_stats()
    if stats_mod is None:
        if warns is not None:
            warns.append(
                "scipy.stats is not available; cosinor p_rhythm and confidence "
                "intervals were not computed."
            )
        return result
    if not np.isfinite(sigma2) or sigma2 <= 0 or amplitude == 0:
        return result

    # F-test: full (3-param) vs reduced (intercept-only) model.
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    df_num = 2  # cos + sin terms
    if ss_res > 0 and dof > 0:
        f_stat = ((ss_tot - ss_res) / df_num) / (ss_res / dof)
        result["p_rhythm"] = float(stats_mod.f.sf(f_stat, df_num, dof))

    cov = sigma2 * xtx_inv
    var_cos = float(cov[1, 1])
    var_sin = float(cov[2, 2])
    cov_cs = float(cov[1, 2])
    t_crit = float(stats_mod.t.ppf(0.975, dof))

    # Amplitude delta method: A = sqrt(bc^2 + bs^2).
    grad_amp = np.array([beta_cos / amplitude, beta_sin / amplitude])
    var_amp = (
        grad_amp[0] ** 2 * var_cos
        + grad_amp[1] ** 2 * var_sin
        + 2.0 * grad_amp[0] * grad_amp[1] * cov_cs
    )
    if var_amp > 0:
        se_amp = math.sqrt(var_amp)
        result["amplitude_ci_low"] = float(amplitude - t_crit * se_amp)
        result["amplitude_ci_high"] = float(amplitude + t_crit * se_amp)

    # Acrophase delta method: theta = atan2(bs, bc); d theta / d bc = -bs/A^2,
    # d theta / d bs = bc/A^2.  Convert radians -> ZT hours.
    a2 = amplitude**2
    if a2 > 0:
        grad_theta = np.array([-beta_sin / a2, beta_cos / a2])
        var_theta = (
            grad_theta[0] ** 2 * var_cos
            + grad_theta[1] ** 2 * var_sin
            + 2.0 * grad_theta[0] * grad_theta[1] * cov_cs
        )
        if var_theta > 0:
            se_theta = math.sqrt(var_theta)
            scale = period_hours / (2.0 * np.pi)
            half_width = t_crit * se_theta * scale
            # Only report when the interval is narrower than a full cycle.
            if half_width < period_hours / 2.0:
                result["acrophase_ci_low"] = float((acrophase - half_width) % period_hours)
                result["acrophase_ci_high"] = float((acrophase + half_width) % period_hours)
    return result


def _r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def _phase_from_zt(zt: float, photoperiod_hours: float = 12.0) -> str | None:
    if not np.isfinite(zt):
        return None
    return "light" if 0.0 <= (zt % 24.0) < photoperiod_hours else "dark"


def _add_decimal_hour_from_timestamp(data: pd.DataFrame, warns: list[str]) -> str:
    ts_col = "timestamp_local" if "timestamp_local" in data.columns else "timestamp_utc"
    if ts_col not in data.columns:
        return ""
    ts = pd.to_datetime(data[ts_col], errors="coerce")
    if ts.notna().sum() == 0:
        return ""
    data["_decimal_hour"] = ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0
    warns.append("Using decimal clock hour for cosinor because zeitgeber_time_hours is missing.")
    return "_decimal_hour"


def _sem_from_sd(sd: float, n: int) -> float:
    if n <= 1 or pd.isna(sd):
        return np.nan
    return float(sd) / math.sqrt(int(n))


def _dark_light_ratios(summary: pd.DataFrame, keys: list[str], phase_col: str) -> pd.DataFrame:
    index_cols = keys if keys else ["_global"]
    tmp = summary.copy()
    if not keys:
        tmp["_global"] = "all"
    pivot = tmp.pivot_table(index=index_cols, columns=phase_col, values="mean", aggfunc="first").reset_index()
    light = pd.to_numeric(pivot.get("light"), errors="coerce")
    dark = pd.to_numeric(pivot.get("dark"), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        pivot["dark_light_ratio"] = np.where(light != 0, dark / light, np.nan)
    return pivot[[*index_cols, "dark_light_ratio"]]


def _merge_global_ratio(summary: pd.DataFrame, ratios: pd.DataFrame) -> pd.DataFrame:
    ratio = ratios["dark_light_ratio"].iloc[0] if not ratios.empty else np.nan
    out = summary.copy()
    out["dark_light_ratio"] = ratio
    return out


def _ordered_summary_columns(df: pd.DataFrame, leading_cols: Sequence[str]) -> pd.DataFrame:
    metric_cols = ["mean", "sem", "sd", "n_subjects", "n_observations"]
    ordered = [c for c in [*leading_cols, *metric_cols] if c in df.columns]
    ordered.extend(c for c in df.columns if c not in ordered)
    return df[ordered]


def _bin_size_to_hours(bin_size: str | float | int) -> float:
    if isinstance(bin_size, (int, float)):
        if float(bin_size) <= 0:
            raise ValueError("bin_size must be positive.")
        return float(bin_size)
    text = str(bin_size).strip().upper()
    if text.endswith("H"):
        return float(text[:-1])
    if text.endswith("D"):
        return float(text[:-1]) * 24.0
    if text.endswith("W"):
        return float(text[:-1]) * 24.0 * 7.0
    return float(text)


def _format_hour_bin(start_hour: float) -> str:
    if float(start_hour).is_integer():
        return f"{int(start_hour)}h"
    return f"{start_hour:g}h"


def _absolute_bin_start(ts: pd.Series, bin_size: str | float | int) -> pd.Series:
    text = str(bin_size).strip().upper()
    if text.endswith("W"):
        return ts.dt.tz_localize(None).dt.to_period("W").dt.start_time
    if text.endswith("D"):
        return ts.dt.floor(f"{int(float(text[:-1]))}D")
    if text.endswith("H"):
        return ts.dt.floor(f"{int(float(text[:-1]))}h")
    return ts.dt.floor(f"{int(_bin_size_to_hours(bin_size))}h")


def _absolute_bin_end(starts: pd.Series, bin_size: str | float | int) -> pd.Series:
    hours = _bin_size_to_hours(bin_size)
    return starts + pd.to_timedelta(hours, unit="h")


def _default_auc_x_col(data: pd.DataFrame) -> str | None:
    for col in ("time_from_event_hours", "timestamp_utc", "timestamp_local", "relative_time_seconds"):
        if col in data.columns:
            return col
    return None


def _is_datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dropna().empty:
        return False
    sample = series.dropna().iloc[0]
    return isinstance(sample, pd.Timestamp) or "datetime" in type(sample).__name__.lower()


def _load_scipy_stats() -> Any | None:
    try:
        return importlib.import_module("scipy.stats")
    except Exception:
        return None


def _default_block_cols(data: pd.DataFrame, group_col: str) -> list[str]:
    candidates = ["metric_name", "time_bin_start", "time_bin_label", "phase"]
    return [c for c in candidates if c in data.columns and c != group_col]


def _subject_level_values(
    data: pd.DataFrame,
    value_col: str,
    group_col: str,
    block_cols: Sequence[str],
    subject_col: str,
) -> pd.DataFrame:
    keys = [*block_cols, group_col]
    if subject_col in data.columns:
        keys.append(subject_col)
    return data.groupby(keys, dropna=False)[value_col].mean().reset_index()


def _two_group_estimation(
    group1: np.ndarray,
    group2: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """
    Estimation-first summary for a two-group comparison.

    Returns the observed median difference (``median(group2) - median(group1)``)
    plus bootstrap (percentile) 95% confidence intervals for both that median
    difference and the rank-biserial effect size.  Resampling is stratified
    within each group and seeded for determinism.
    """
    nan = float("nan")
    result = {
        "median_difference": nan,
        "diff_ci_low": nan,
        "diff_ci_high": nan,
        "effect_size_ci_low": nan,
        "effect_size_ci_high": nan,
    }
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    n1, n2 = len(g1), len(g2)
    if n1 == 0 or n2 == 0:
        return result

    result["median_difference"] = float(np.median(g2) - np.median(g1))

    rng = np.random.default_rng(seed)
    diff_samples = np.empty(n_boot, dtype=float)
    effect_samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        b1 = g1[rng.integers(0, n1, size=n1)]
        b2 = g2[rng.integers(0, n2, size=n2)]
        diff_samples[i] = np.median(b2) - np.median(b1)
        effect_samples[i] = _rank_biserial_brute(b1, b2)

    lo_d, hi_d = np.percentile(diff_samples, [2.5, 97.5])
    lo_e, hi_e = np.percentile(effect_samples, [2.5, 97.5])
    result["diff_ci_low"] = float(lo_d)
    result["diff_ci_high"] = float(hi_d)
    result["effect_size_ci_low"] = float(lo_e)
    result["effect_size_ci_high"] = float(hi_e)
    return result


def _rank_biserial_brute(group1: np.ndarray, group2: np.ndarray) -> float:
    """Rank-biserial correlation computed directly from pairwise dominance.

    Defined as ``P(x1 > x2) - P(x1 < x2)`` over all pairs, matching the
    ``2*U/(n1*n2) - 1`` convention used by :func:`_rank_biserial_from_u` for the
    ``group1`` vs ``group2`` orientation (positive => group1 tends to exceed
    group2).
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return np.nan
    diff = group1[:, None] - group2[None, :]
    wins = float(np.sum(diff > 0))
    ties = float(np.sum(diff == 0))
    u = wins + 0.5 * ties
    return float(2.0 * u / (n1 * n2) - 1.0)


def _min_possible_mannwhitney_p(n1: int, n2: int) -> float:
    """Smallest achievable two-sided Mann-Whitney p for the given group sizes.

    The most extreme rank configuration yields ``p = 2 / C(n1+n2, n1)``.
    """
    if n1 <= 0 or n2 <= 0:
        return np.nan
    return float(min(1.0, 2.0 / math.comb(n1 + n2, n1)))


def _rank_biserial_from_u(u_statistic: float, n_left: int, n_right: int) -> float:
    denominator = int(n_left) * int(n_right)
    if denominator <= 0 or pd.isna(u_statistic):
        return np.nan
    return float(2.0 * float(u_statistic) / denominator - 1.0)


def _kruskal_epsilon_squared(h_statistic: float, group_arrays: Sequence[np.ndarray]) -> float:
    n_total = int(sum(len(arr) for arr in group_arrays))
    n_groups = len(group_arrays)
    if n_groups < 2 or n_total <= n_groups or pd.isna(h_statistic):
        return np.nan
    epsilon_squared = (float(h_statistic) - n_groups + 1.0) / (n_total - n_groups)
    return float(np.clip(epsilon_squared, 0.0, 1.0))


def _add_fdr_q_values(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    if "q_value" not in out.columns:
        out["q_value"] = np.nan
    if "q_label" not in out.columns:
        out["q_label"] = None
    if "p_value" not in out.columns or "n_groups" not in out.columns:
        return out

    p_values = pd.to_numeric(out["p_value"], errors="coerce")
    n_groups = pd.to_numeric(out["n_groups"], errors="coerce")
    comparison_mask = n_groups.ge(2) & np.isfinite(p_values)
    if not comparison_mask.any():
        return out

    q_values = _benjamini_hochberg(p_values.loc[comparison_mask].to_numpy(dtype=float))
    out.loc[comparison_mask, "q_value"] = q_values
    out.loc[comparison_mask, "q_label"] = [_p_label(q) for q in q_values]
    return out


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    if len(p_values) == 0:
        return p_values

    order = np.argsort(p_values)
    ranked_p = p_values[order]
    ranks = np.arange(1, len(ranked_p) + 1, dtype=float)
    adjusted_ranked = ranked_p * len(ranked_p) / ranks
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)

    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def _p_label(p_value: float | None) -> str | None:
    if p_value is None or pd.isna(p_value):
        return None
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


__all__ = [
    "EXPLORATORY_STATS_DISCLAIMER",
    "compare_window_summaries",
    "compute_auc_per_animal",
    "estimate_period",
    "quick_exploratory_stats",
    "summarize_activity_bouts",
    "summarize_circadian_cosinor",
    "summarize_daily",
    "summarize_light_dark",
    "summarize_nonparametric_circadian",
    "summarize_time_bins",
    "summarize_weekly",
]
