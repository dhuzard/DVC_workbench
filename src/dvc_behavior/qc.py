"""
QC visualisations using Plotly.

All functions return a plotly Figure object.
"""

from __future__ import annotations

import importlib
import math

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


_EXCLUSION_BAND_COLOR = "rgba(255, 100, 100, 0.18)"
_DARK_PHASE_COLOR = "rgba(50, 50, 80, 0.12)"

_NORMAL_Z_975 = 1.959963984540054


def confidence_halfwidth(sd: float, n: float, level: float = 0.95) -> float:
    """Return the t-based confidence-interval half-width for a group mean.

    ``sd`` is the sample standard deviation (ddof=1) across the experimental
    units (subjects) and ``n`` is the number of units. Uses Student's t when
    SciPy is available, falling back to the normal approximation. Returns NaN
    when ``n`` is too small (< 2) for an interval to be defined.
    """
    if n is None or not np.isfinite(n) or n <= 1:
        return float("nan")
    if sd is None or not np.isfinite(sd):
        return float("nan")
    sem = float(sd) / math.sqrt(int(n))
    try:
        stats = importlib.import_module("scipy.stats")
        crit = float(stats.t.ppf(0.5 + level / 2.0, int(n) - 1))
    except Exception:
        crit = _NORMAL_Z_975
    return crit * sem



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_exclusion_bands(
    df: pd.DataFrame,
    subject_id: str,
) -> list[dict]:
    """Return list of {x0, x1} dicts for excluded periods of a subject."""
    if "is_excluded" not in df.columns or "timestamp_local" not in df.columns:
        return []

    sub = df[df["subject_id"] == subject_id].copy()
    sub = sub[sub["is_excluded"]].sort_values("timestamp_local")
    if sub.empty:
        return []

    bands: list[dict] = []
    ts = sub["timestamp_local"].tolist()
    if not ts:
        return []

    start = ts[0]
    prev = ts[0]
    for t in ts[1:]:
        if (t - prev).total_seconds() > 7200:  # gap > 2 h → new band
            bands.append({"x0": start, "x1": prev})
            start = t
        prev = t
    bands.append({"x0": start, "x1": prev})
    return bands


def _add_exclusion_bands(fig: go.Figure, bands: list[dict]) -> None:
    for band in bands:
        fig.add_vrect(
            x0=band["x0"],
            x1=band["x1"],
            fillcolor=_EXCLUSION_BAND_COLOR,
            layer="below",
            line_width=0,
        )


# ---------------------------------------------------------------------------
# Raw timeseries by subject
# ---------------------------------------------------------------------------

def plot_raw_timeseries(
    df: pd.DataFrame,
    metric_name: str,
    subjects: list[str] | None = None,
    x_col: str = "timestamp_local",
    show_exclusions: bool = True,
) -> go.Figure:
    """Plot raw (unaligned) timeseries for selected subjects."""
    sub_df = df[df["metric_name"] == metric_name].copy() if "metric_name" in df.columns else df.copy()

    if subjects:
        sub_df = sub_df[sub_df["subject_id"].isin(subjects)]

    if sub_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No data for metric: {metric_name}")
        return fig

    fig = px.line(
        sub_df,
        x=x_col,
        y="value",
        color="subject_id",
        title=f"Raw timeseries – {metric_name}",
        labels={"value": metric_name, x_col: "Time", "subject_id": "Subject"},
    )
    fig.update_traces(line=dict(width=1.2))

    if show_exclusions:
        all_subs = sub_df["subject_id"].unique()
        for sid in all_subs:
            for band in _get_exclusion_bands(sub_df, sid):
                fig.add_vrect(
                    x0=band["x0"],
                    x1=band["x1"],
                    fillcolor=_EXCLUSION_BAND_COLOR,
                    layer="below",
                    line_width=0,
                    annotation_text="excl.",
                    annotation_position="top left",
                )
            break  # only add bands once (they overlap anyway in single-subject view)

    fig.update_layout(hovermode="x unified", template="plotly_white")
    return fig


# ---------------------------------------------------------------------------
# Aligned timeseries by subject
# ---------------------------------------------------------------------------

def plot_aligned_timeseries(
    df: pd.DataFrame,
    metric_name: str,
    subjects: list[str] | None = None,
    y_col: str = "value",
    show_exclusions: bool = True,
    show_baseline_window: bool = True,
    baseline_start_h: float | None = None,
    baseline_end_h: float | None = None,
) -> go.Figure:
    """Plot timeseries aligned to event (x = time_from_event_hours)."""
    sub_df = df[df["metric_name"] == metric_name].copy() if "metric_name" in df.columns else df.copy()

    if "time_from_event_hours" not in sub_df.columns or sub_df["time_from_event_hours"].isna().all():
        return go.Figure(layout=dict(title="Alignment not applied — no time_from_event_hours column."))

    if subjects:
        sub_df = sub_df[sub_df["subject_id"].isin(subjects)]

    if sub_df.empty:
        return go.Figure(layout=dict(title=f"No data for metric: {metric_name}"))

    fig = px.line(
        sub_df.sort_values("time_from_event_hours"),
        x="time_from_event_hours",
        y=y_col,
        color="subject_id",
        title=f"Aligned timeseries – {metric_name} ({y_col})",
        labels={
            y_col: metric_name,
            "time_from_event_hours": "Time from event (h)",
            "subject_id": "Subject",
        },
    )
    fig.update_traces(line=dict(width=1.2))

    if show_exclusions and "is_excluded" in sub_df.columns:
        excl = sub_df[sub_df["is_excluded"]]
        if not excl.empty:
            fig.add_scatter(
                x=excl["time_from_event_hours"],
                y=excl[y_col],
                mode="markers",
                marker=dict(color="red", size=4, opacity=0.5, symbol="x"),
                name="Excluded",
                showlegend=True,
            )

    if show_baseline_window and baseline_start_h is not None and baseline_end_h is not None:
        fig.add_vrect(
            x0=baseline_start_h,
            x1=baseline_end_h,
            fillcolor="rgba(0,180,0,0.1)",
            layer="below",
            line_width=0,
            annotation_text="Baseline",
            annotation_position="top left",
        )

    fig.add_vline(x=0, line_dash="dash", line_color="grey", annotation_text="J0")
    fig.update_layout(hovermode="x unified", template="plotly_white")
    return fig


# ---------------------------------------------------------------------------
# Group mean aligned timeseries
# ---------------------------------------------------------------------------

def plot_group_mean_timeseries(
    df: pd.DataFrame,
    metric_name: str,
    y_col: str = "value",
    groups: list[str] | None = None,
    band: str = "ci95",
    min_n: int = 3,
) -> go.Figure:
    """Plot group-mean aligned timeseries with a confidence band.

    ``band`` selects the shaded interval: ``"ci95"`` (Student's t 95% CI over
    subjects, the default), ``"sem"`` (standard error), or ``"none"``. Means and
    intervals are computed over **per-subject** means when ``subject_id`` is
    present, so the band reflects between-animal variability rather than the
    number of raw bins. Time bins with fewer than ``min_n`` subjects are marked.
    """
    sub_df = df[df["metric_name"] == metric_name].copy() if "metric_name" in df.columns else df.copy()

    if "time_from_event_hours" not in sub_df.columns or sub_df["time_from_event_hours"].isna().all():
        return go.Figure(layout=dict(title="Alignment not applied."))
    if y_col not in sub_df.columns:
        return go.Figure(layout=dict(title=f"No data column: {y_col}"))

    # Exclude excluded rows from group mean
    if "is_excluded" in sub_df.columns:
        sub_df = sub_df[~sub_df["is_excluded"]]

    if groups:
        if "group_id" not in sub_df.columns:
            return go.Figure(layout=dict(title="No group_id column available."))
        sub_df = sub_df[sub_df["group_id"].isin(groups)]

    if sub_df.empty:
        return go.Figure(layout=dict(title="No data after exclusion filter."))

    label_col = "group_label" if "group_label" in sub_df.columns else "group_id"
    required_cols = ["time_from_event_hours", y_col, label_col]
    sub_df = sub_df.dropna(subset=required_cols)
    if sub_df.empty:
        return go.Figure(layout=dict(title="No plottable group-mean data."))

    # Round time to nearest bin for grouping
    if "native_bin_seconds" in sub_df.columns:
        nb = sub_df["native_bin_seconds"].dropna()
        native_h = float(nb.mode().iloc[0]) / 3600.0 if not nb.empty else 1.0
    else:
        native_h = 1.0

    sub_df["_t_bin"] = (sub_df["time_from_event_hours"] / native_h).round() * native_h

    # Aggregate to the experimental unit (subject) before summarizing, so the
    # band reflects between-animal variability rather than raw-bin count.
    has_subjects = "subject_id" in sub_df.columns
    if has_subjects:
        unit = (
            sub_df.groupby(["_t_bin", label_col, "subject_id"])[y_col]
            .mean()
            .reset_index()
        )
    else:
        unit = sub_df[["_t_bin", label_col, y_col]].copy()

    grp_stats = (
        unit.groupby(["_t_bin", label_col])[y_col]
        .agg(grp_mean="mean", grp_sd=lambda s: s.std(ddof=1), n="count")
        .reset_index()
    )
    if grp_stats.empty:
        return go.Figure(layout=dict(title="No group-mean data after aggregation."))

    grp_stats["grp_sem"] = grp_stats.apply(
        lambda r: (r["grp_sd"] / math.sqrt(int(r["n"]))) if r["n"] and r["n"] > 1 else float("nan"),
        axis=1,
    )
    grp_stats["grp_ci95"] = grp_stats.apply(
        lambda r: confidence_halfwidth(r["grp_sd"], r["n"]), axis=1
    )

    band = (band or "ci95").lower()
    half_col = {"ci95": "grp_ci95", "sem": "grp_sem", "none": None}.get(band, "grp_ci95")
    band_label = {"ci95": "95% CI", "sem": "SEM", "none": "mean only"}.get(band, "95% CI")
    unit_label = "subjects" if has_subjects else "observations"

    fig = go.Figure()
    color_cycle = px.colors.qualitative.Plotly
    for i, (grp_label, grp_data) in enumerate(grp_stats.groupby(label_col)):
        color = color_cycle[i % len(color_cycle)]
        grp_data = grp_data.sort_values("_t_bin")

        if half_col is not None:
            half = grp_data[half_col].fillna(0.0)
            upper = grp_data["grp_mean"] + half
            lower = grp_data["grp_mean"] - half
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([grp_data["_t_bin"], grp_data["_t_bin"][::-1]]),
                    y=pd.concat([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor=color.replace("rgb", "rgba").replace(")", ", 0.2)"),
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        # Hollow markers flag bins with too few units for a reliable interval.
        low_n = grp_data["n"] < min_n
        fig.add_trace(
            go.Scatter(
                x=grp_data["_t_bin"],
                y=grp_data["grp_mean"],
                mode="lines+markers",
                name=str(grp_label),
                line=dict(color=color, width=2),
                marker=dict(
                    color=np.where(low_n, "rgba(255,255,255,0)", color),
                    line=dict(color=color, width=1),
                    size=6,
                ),
                customdata=grp_data["n"],
                hovertemplate=(
                    f"%{{y:.3g}} ± {band_label}<br>n=%{{customdata}} {unit_label}<extra>"
                    f"{grp_label}</extra>"
                ),
            )
        )

    fig.add_vline(x=0, line_dash="dash", line_color="grey", annotation_text="J0")
    fig.update_layout(
        title=f"Group mean ± {band_label} – {metric_name} ({y_col})",
        xaxis_title="Time from event (h)",
        yaxis_title=metric_name,
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def plot_baseline_quality_heatmap(baseline_summary: pd.DataFrame) -> go.Figure:
    """Plot baseline coverage by subject and metric."""
    if baseline_summary is None or baseline_summary.empty:
        return go.Figure(layout=dict(title="No baseline summary available."))
    required = {"subject_id", "metric_name", "baseline_coverage"}
    if not required <= set(baseline_summary.columns):
        return go.Figure(layout=dict(title="Baseline coverage columns are unavailable."))

    pivot = baseline_summary.pivot_table(
        index="subject_id",
        columns="metric_name",
        values="baseline_coverage",
        aggfunc="first",
    )
    if pivot.empty:
        return go.Figure(layout=dict(title="No baseline coverage values available."))

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(dtype=float),
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            zmin=0,
            zmax=1,
            colorscale="RdYlGn",
            colorbar=dict(title="Coverage"),
        )
    )
    fig.update_layout(
        title="Baseline coverage by subject and metric",
        xaxis_title="Metric",
        yaxis_title="Subject",
        template="plotly_white",
    )
    return fig


def detect_irregular_bins(
    df: pd.DataFrame,
    tolerance_fraction: float = 0.10,
) -> pd.DataFrame:
    """Flag subject/metric streams whose timestamp intervals are irregular."""
    required = {"subject_id", "metric_name", "timestamp_utc"}
    if df is None or df.empty or not required <= set(df.columns):
        return pd.DataFrame(
            columns=[
                "subject_id", "metric_name", "group_id", "median_interval_seconds",
                "std_interval_seconds", "n_intervals", "irregular_bins",
            ]
        )

    rows = []
    group_keys = [c for c in ["subject_id", "metric_name", "group_id"] if c in df.columns]
    for keys, grp in df.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_keys, keys, strict=False))
        ts = pd.to_datetime(grp["timestamp_utc"], utc=True, errors="coerce").dropna().sort_values()
        diffs = ts.diff().dropna().dt.total_seconds()
        if diffs.empty:
            median = std = float("nan")
            irregular = False
        else:
            median = float(diffs.median())
            std = float(diffs.std()) if len(diffs) > 1 else 0.0
            irregular = bool(median and std > tolerance_fraction * abs(median))
        rows.append(
            {
                **key_values,
                "median_interval_seconds": median,
                "std_interval_seconds": std,
                "n_intervals": int(len(diffs)),
                "irregular_bins": irregular,
            }
        )
    return pd.DataFrame(rows)
