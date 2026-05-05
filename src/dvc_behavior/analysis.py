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
    exclude_excluded: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Fit a 24 h cosinor model per group/metric and return rhythm parameters.

    The model is ``MESOR + amplitude * cos(2*pi*(ZT - acrophase_ZT)/period)``.
    ``scipy.optimize.curve_fit`` is used when available; otherwise a linear
    least-squares cosinor fit is used.
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

    rows: list[dict[str, Any]] = []
    grouped = data.groupby(keys, dropna=False) if keys else [((), data)]
    for group_key, grp in grouped:
        group_values = _group_key_values(keys, group_key)
        clean = grp[[zt_col, value_col]].copy()
        clean[zt_col] = pd.to_numeric(clean[zt_col], errors="coerce") % period_hours
        clean[value_col] = pd.to_numeric(clean[value_col], errors="coerce")
        clean = clean.dropna()
        if not clean.empty and zt_bin_hours > 0:
            clean["_zt_bin"] = (np.floor(clean[zt_col] / zt_bin_hours) * zt_bin_hours) % period_hours
            fit_data = clean.groupby("_zt_bin", dropna=False)[value_col].mean().reset_index()
            fit_x_col = "_zt_bin"
        else:
            fit_data = clean
            fit_x_col = zt_col

        if len(fit_data) < min_points or fit_data[fit_x_col].nunique() < 3:
            rows.append(
                {
                    **group_values,
                    "MESOR": np.nan,
                    "amplitude": np.nan,
                    "acrophase_ZT": np.nan,
                    "R2": np.nan,
                    "phase": None,
                    "n_points": int(len(fit_data)),
                    "n_observations": int(len(clean)),
                    "fit_method": "insufficient_data",
                }
            )
            continue

        x = fit_data[fit_x_col].to_numpy(dtype=float)
        y = fit_data[value_col].to_numpy(dtype=float)
        fit = _fit_cosinor(x, y, period_hours)
        rows.append(
            {
                **group_values,
                "MESOR": fit["mesor"],
                "amplitude": fit["amplitude"],
                "acrophase_ZT": fit["acrophase"],
                "R2": fit["r2"],
                "phase": _phase_from_zt(fit["acrophase"]),
                "n_points": int(len(fit_data)),
                "n_observations": int(len(clean)),
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
    """
    warns: list[str] = []
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
                    "p_value": np.nan,
                    "p_label": None,
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
                    "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                    "p_label": _p_label(p_value),
                    "scipy_available": True,
                    "exploratory": True,
                    "disclaimer": EXPLORATORY_STATS_DISCLAIMER,
                }
            )

        if len(group_arrays) == 2:
            stat, p_value = stats_mod.mannwhitneyu(group_arrays[0], group_arrays[1], alternative="two-sided")
            test_name = "Mann-Whitney U"
        elif len(group_arrays) > 2:
            stat, p_value = stats_mod.kruskal(*group_arrays)
            test_name = "Kruskal-Wallis"
        else:
            stat = p_value = np.nan
            test_name = "group comparison skipped"

        rows.append(
            {
                **block_values,
                "test": test_name,
                "comparison": " vs ".join(group_names),
                "group": None,
                "n_groups": len(group_arrays),
                "n_total": int(sum(len(arr) for arr in group_arrays)),
                "statistic": float(stat) if pd.notna(stat) else np.nan,
                "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                "p_label": _p_label(p_value),
                "scipy_available": True,
                "exploratory": True,
                "disclaimer": EXPLORATORY_STATS_DISCLAIMER,
            }
        )

    return pd.DataFrame(rows), warns


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


def _r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def _phase_from_zt(zt: float) -> str | None:
    if not np.isfinite(zt):
        return None
    return "light" if 0.0 <= (zt % 24.0) < 12.0 else "dark"


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
    "compute_auc_per_animal",
    "quick_exploratory_stats",
    "summarize_circadian_cosinor",
    "summarize_daily",
    "summarize_light_dark",
    "summarize_time_bins",
    "summarize_weekly",
]
