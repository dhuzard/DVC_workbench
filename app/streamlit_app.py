"""
DVC Behavioral Preprocessing Workbench – Streamlit GUI.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
from contextlib import contextmanager
from difflib import SequenceMatcher
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_APP_DIR = Path(__file__).parent
_ROOT = _APP_DIR.parent
sys.path.insert(0, str(_APP_DIR))
sys.path.insert(0, str(_ROOT / "src"))

from components.metadata_tables import group_column_config, subject_column_config
from components.workflow import (
    render_contextual_help,
    render_data_flow_diagram,
    render_output_glossary,
    render_progress_stepper,
)

from dvc_behavior import (  # noqa: E402
    aggregation,
    alignment,
    analysis,
    baseline,
    config as cfg,
    events as ev_mod,
    exclusions,
    export as exp_mod,
    light_dark,
    metadata as meta_mod,
    parsing,
    provenance,
    quality,
    qc,
    reporting,
    schemas,
)
_NO_EVENT_FILE = "No event file"


@contextmanager
def _working_status(title: str, detail: str):
    """Show a short, consistent visual status while app work is running."""
    status = st.status(title, expanded=True)
    status.write(detail)
    try:
        yield
    except Exception:
        status.update(label=f"{title} failed", state="error", expanded=True)
        raise
    else:
        status.update(label=f"{title} complete", state="complete", expanded=False)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict[str, Any] = {
        # raw file bytes: list of (name, bytes)
        "metric_files": [],
        "event_files": [],
        # parsed dataframes
        "long_df": None,
        "event_df": None,
        # metadata tables
        "study_meta": cfg.STUDY_METADATA_DEFAULTS.copy(),
        "subject_meta": None,
        "group_meta": None,
        "treatment_schedule": pd.DataFrame(
            columns=["animal_id", "event_type", "timestamp", "dose", "unit", "route", "notes"]
        ),
        "facility_events": pd.DataFrame(
            columns=["timestamp_start", "timestamp_end", "event_type", "affected_groups", "notes"]
        ),
        "baseline_overrides": pd.DataFrame(columns=["subject_id", "metric_name", "baseline_value", "notes"]),
        # processing config
        "timezone": cfg.DEFAULT_TIMEZONE,
        "light_on": cfg.DEFAULT_LIGHT_ON,
        "light_off": cfg.DEFAULT_LIGHT_OFF,
        "exclusion_rules": deepcopy(cfg.DEFAULT_EXCLUSION_RULES),
        "alignment_cfg": deepcopy(cfg.DEFAULT_ALIGNMENT),
        "baseline_cfg": deepcopy(cfg.DEFAULT_BASELINE),
        "aggregation_bin": None,
        # results
        "processed_df": None,
        "exclusion_log": None,
        "baseline_summary": None,
        "all_warnings": [],
        "schema_validation_enabled": False,
        "analysis_tables": {},
        "analysis_figures": {},
        "analysis_warning": "",
        "quality_report": None,
        "raw_event_file_mapping": {},
        "raw_preview_plot_requested": False,
        "raw_preview_plot_settings": {},
        "scroll_to_top": False,
        # pipeline timestamp for cache-busting display
        "pipeline_run_ts": None,
        "export_zip_ready": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Helper: parse all uploaded metric & event files
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_load_metric_csv(name: str, data: bytes) -> tuple[pd.DataFrame, list[str]]:
    return parsing.load_metric_csv(io.BytesIO(data), source_file=name)


@st.cache_data(show_spinner=False)
def _cached_parse_event_csv(name: str, data: bytes) -> tuple[pd.DataFrame, list[str]]:
    return ev_mod.parse_event_csv(io.BytesIO(data), source_file=name)


def _parse_all_files() -> None:
    """Re-parse all metric and event files and store in session_state."""
    warns: list[str] = []
    total_bytes = sum(len(data) for _, data in st.session_state.metric_files + st.session_state.event_files)
    progress = st.progress(0, text="Parsing uploaded files...") if total_bytes > 1_000_000 else None
    total_files = len(st.session_state.metric_files) + len(st.session_state.event_files)
    parsed_files = 0

    # ---- metric files ----
    long_dfs: list[pd.DataFrame] = []
    for name, data in st.session_state.metric_files:
        df, w = _cached_load_metric_csv(name, data)
        warns.extend(w)
        if not df.empty:
            long_dfs.append(df)
        parsed_files += 1
        if progress and total_files:
            progress.progress(parsed_files / total_files, text=f"Parsed {parsed_files}/{total_files} files")

    st.session_state.long_df = parsing.combine_long_dfs(long_dfs) if long_dfs else None
    if st.session_state.schema_validation_enabled:
        warns.extend(
            schemas.validate_long_df(
                st.session_state.long_df
                if st.session_state.long_df is not None
                else pd.DataFrame()
            )
        )

    # ---- event files ----
    ev_dfs: list[pd.DataFrame] = []
    for name, data in st.session_state.event_files:
        df, w = _cached_parse_event_csv(name, data)
        warns.extend(w)
        if not df.empty:
            ev_dfs.append(df)
        parsed_files += 1
        if progress and total_files:
            progress.progress(parsed_files / total_files, text=f"Parsed {parsed_files}/{total_files} files")

    st.session_state.event_df = ev_mod.combine_event_dfs(ev_dfs) if ev_dfs else None
    if st.session_state.schema_validation_enabled:
        warns.extend(
            schemas.validate_event_df(
                st.session_state.event_df
                if st.session_state.event_df is not None
                else pd.DataFrame()
            )
        )
    if progress:
        progress.empty()

    # Build metadata templates from detected subjects/groups
    if st.session_state.long_df is not None:
        long_df = st.session_state.long_df
        # Only rebuild if not already defined (avoid overwriting user edits)
        if st.session_state.subject_meta is None:
            st.session_state.subject_meta = meta_mod.build_subject_metadata_template(long_df)
        if st.session_state.group_meta is None:
            st.session_state.group_meta = meta_mod.build_group_metadata_template(long_df)

    st.session_state.all_warnings = warns
    st.session_state.processed_df = None  # invalidate downstream results
    st.session_state.exclusion_log = None
    st.session_state.baseline_summary = None
    st.session_state.quality_report = None
    st.session_state.export_zip_ready = False


def _facility_events_as_event_df(facility_events: pd.DataFrame | None) -> pd.DataFrame:
    """Convert the facility event editor table into exclusion-compatible rows."""
    if facility_events is None or facility_events.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in facility_events.iterrows():
        start = pd.to_datetime(row.get("timestamp_start"), utc=True, errors="coerce")
        end = pd.to_datetime(row.get("timestamp_end"), utc=True, errors="coerce")
        if pd.isna(start):
            continue
        rows.append(
            {
                "source_file": "manual_facility_events",
                "group_id": str(row.get("affected_groups", "")).strip() or pd.NA,
                "subject_id": pd.NA,
                "event_scope": "facility" if not str(row.get("affected_groups", "")).strip() else "group",
                "event_type": "FACILITY_EVENT",
                "timestamp": start,
                "timestamp_utc": start,
                "timestamp_end": end if not pd.isna(end) else start,
                "timestamp_end_utc": end if not pd.isna(end) else start,
                "timestamp_local": start,
                "relative_time_seconds": pd.NA,
                "rack": pd.NA,
                "position": pd.NA,
                "raw_event_label": row.get("event_type", "FACILITY_EVENT"),
                "event_category": "facility",
                "notes": row.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def _alignment_is_configured() -> bool:
    aln_cfg = st.session_state.alignment_cfg
    return bool(aln_cfg.get("event_type") or aln_cfg.get("fallback_timestamp"))


def _add_empty_baseline_columns(df: pd.DataFrame) -> None:
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
        df[col] = pd.NA


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def _run_pipeline() -> None:
    """
    Execute the full pipeline on long_df and store results in session_state.

    Steps:
      1. Merge metadata
      2. Localise timestamps + annotate light/dark
      3. Apply exclusions
      4. Align to event
      5. Compute baseline when alignment is configured
      6. (optional) aggregate
    """
    long_df = st.session_state.long_df
    if long_df is None or long_df.empty:
        st.error("No parsed data found. Load files first.")
        return

    warns: list[str] = list(st.session_state.all_warnings)
    df = long_df.copy()

    # 1. Merge metadata
    if st.session_state.subject_meta is not None:
        df = meta_mod.merge_subject_metadata(df, st.session_state.subject_meta)
    if st.session_state.group_meta is not None:
        df = meta_mod.merge_group_metadata(df, st.session_state.group_meta)

    tz = st.session_state.timezone
    light_on = st.session_state.light_on
    light_off = st.session_state.light_off

    # 2. Localise + light/dark annotation
    df, ld_warns = light_dark.add_light_dark_columns(df, tz, light_on, light_off)
    warns.extend(ld_warns)

    # 3. Exclusions
    event_df = st.session_state.event_df
    facility_df = _facility_events_as_event_df(st.session_state.facility_events)
    if facility_df is not None and not facility_df.empty:
        event_df = (
            pd.concat([event_df, facility_df], ignore_index=True)
            if event_df is not None and not event_df.empty
            else facility_df
        )
    log_df = pd.DataFrame()
    if event_df is not None and not event_df.empty:
        # Localise event timestamps too
        if "timestamp_utc" in event_df.columns:
            try:
                event_df = event_df.copy()
                event_df["timestamp_local"] = event_df["timestamp_utc"].dt.tz_convert(tz)
            except Exception:
                pass

        win_df = exclusions.compute_exclusion_windows(
            event_df, st.session_state.exclusion_rules
        )
        df, log_df = exclusions.apply_exclusions(df, win_df)
    else:
        df["is_excluded"] = False
        df["exclusion_reason"] = ""
        df["flag_reason"] = ""

    st.session_state.exclusion_log = log_df

    # 4. Alignment
    aln_cfg = st.session_state.alignment_cfg
    event_type = aln_cfg.get("event_type")
    fallback_ts = aln_cfg.get("fallback_timestamp")

    if event_type:
        df, aln_warns = alignment.align_to_event(
            df,
            event_df if event_df is not None else pd.DataFrame(),
            event_type=event_type,
            scope=aln_cfg.get("scope", "subject"),
            fallback_timestamp=fallback_ts if fallback_ts else None,
            alignment_label=aln_cfg.get("label", "J0"),
        )
        warns.extend(aln_warns)
    elif fallback_ts:
        df, aln_warns = alignment.align_to_manual_timestamp(
            df, fallback_ts, alignment_label=aln_cfg.get("label", "J0")
        )
        warns.extend(aln_warns)
    else:
        warns.append(
            "No alignment event or manual timestamp configured; "
            "time_from_event columns will be absent."
        )
        for col in (
            "alignment_event_type", "alignment_timestamp",
            "time_from_event_seconds", "time_from_event_hours", "experimental_day"
        ):
            df[col] = None

    # 5. Baseline (requires an alignment anchor)
    if event_type or fallback_ts:
        bsl_cfg = st.session_state.baseline_cfg
        df, bsl_summary, bsl_warns = baseline.compute_baseline(
            df,
            start_hours=float(bsl_cfg.get("start_hours", -72)),
            end_hours=float(bsl_cfg.get("end_hours", -24)),
            method=bsl_cfg.get("method", "mean"),
            exclude_excluded=bool(bsl_cfg.get("exclude_excluded", True)),
            min_coverage=float(bsl_cfg.get("min_coverage", 0.7)),
            impute_from_group_mean=bool(bsl_cfg.get("impute_from_group_mean", False)),
            baseline_overrides=st.session_state.baseline_overrides,
        )
        warns.extend(bsl_warns)
        st.session_state.baseline_summary = bsl_summary
    else:
        _add_empty_baseline_columns(df)
        st.session_state.baseline_summary = pd.DataFrame()

    # 6. (optional) aggregation
    bin_s = st.session_state.aggregation_bin
    if bin_s is not None:
        df, agg_warns = aggregation.aggregate(df, bin_s)
        warns.extend(agg_warns)

    st.session_state.processed_df = df
    try:
        st.session_state.quality_report = quality.build_quality_report(df)
    except Exception as exc:
        st.session_state.quality_report = None
        warns.append(f"Quality report could not be built: {exc}")
    if st.session_state.schema_validation_enabled:
        warns.extend(schemas.validate_processed_df(df))
    st.session_state.all_warnings = warns
    st.session_state.pipeline_run_ts = pd.Timestamp.now()
    st.session_state.analysis_tables = {}
    st.session_state.analysis_figures = {}
    st.session_state.export_zip_ready = False


# ---------------------------------------------------------------------------
# UX scaffolding helpers
# ---------------------------------------------------------------------------

def _normalise_metadata_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Ensure edited/re-uploaded metadata retains expected columns, then keep extras."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    ordered_cols = columns + [c for c in out.columns if c not in columns]
    return out[ordered_cols].fillna("")


def _workflow_steps() -> list[dict[str, str]]:
    long_df = st.session_state.long_df
    event_df = st.session_state.event_df
    proc = st.session_state.processed_df
    subject_meta = st.session_state.subject_meta
    group_meta = st.session_state.group_meta
    aln_cfg = st.session_state.alignment_cfg

    has_metric = long_df is not None and not long_df.empty
    has_events = event_df is not None and not event_df.empty
    has_processed = proc is not None and not proc.empty
    has_alignment_cfg = bool(aln_cfg.get("event_type") or aln_cfg.get("fallback_timestamp"))

    meta_errors: list[str] = []
    meta_warns: list[str] = []
    if has_metric and subject_meta is not None:
        meta_errors, meta_warns = meta_mod.validate_subject_metadata(subject_meta, long_df)

    metadata_ready = has_metric
    metadata_done = (
        metadata_ready
        and subject_meta is not None
        and group_meta is not None
        and not meta_errors
    )

    baseline_summary = st.session_state.baseline_summary
    has_baseline = baseline_summary is not None and not baseline_summary.empty

    def _status(done: bool, ready: bool) -> str:
        if done:
            return "done"
        return "ready" if ready else "locked"

    return [
        {
            "label": "1. Import",
            "status": _status(has_metric, True),
            "detail": f"{len(long_df):,} metric rows" if has_metric else "Load metric CSVs or examples.",
        },
        {
            "label": "2. Validate",
            "status": _status(has_metric, has_metric),
            "detail": "Review detected subjects, groups, metrics, and warnings.",
        },
        {
            "label": "3. Metadata",
            "status": _status(metadata_done, metadata_ready),
            "detail": (
                "Metadata has validation errors."
                if meta_errors
                else (
                    f"{len(meta_warns)} warning(s) remain."
                    if meta_warns
                    else "Study, subject, and group tables are ready."
                    if metadata_done
                    else "Complete subject and group metadata."
                )
            ),
        },
        {
            "label": "4. Events & Alignment",
            "status": _status(has_alignment_cfg, has_metric),
            "detail": (
                f"{len(event_df):,} events loaded; alignment configured."
                if has_events and has_alignment_cfg
                else "Manual alignment is available when no event file is loaded."
                if has_metric and not has_events
                else "Choose event or manual timestamp."
            ),
        },
        {
            "label": "5. Baseline, Aggregation & Pipeline",
            "status": _status(has_processed, has_metric),
            "detail": (
                "Pipeline complete; baseline summary available."
                if has_processed and has_baseline
                else "Pipeline complete; baseline skipped because alignment is disabled."
                if has_processed and not has_alignment_cfg
                else "Run preprocessing after baseline settings are reviewed."
            ),
        },
        {
            "label": "6. QC Plots",
            "status": _status(has_processed, has_metric),
            "detail": "Processed plots are available." if has_processed else "Raw preview is available after import.",
        },
        {
            "label": "7. Export",
            "status": _status(bool(st.session_state.export_zip_ready), has_metric),
            "detail": (
                "ZIP has been built in this session."
                if st.session_state.export_zip_ready
                else "Build the ZIP after reviewing the report."
            ),
        },
        {
            "label": "8. Analysis",
            "status": _status(bool(st.session_state.analysis_tables), has_processed),
            "detail": (
                "Exploratory analysis tables are ready."
                if st.session_state.analysis_tables
                else "Run the preprocessing pipeline first."
            ),
        },
    ]


def _render_sidebar() -> str:
    with st.sidebar:
        st.subheader("Workflow Progress")
        steps = _workflow_steps()
        render_progress_stepper(steps)
        st.divider()
        step_by_label = {step["label"]: step for step in steps}
        step_labels = list(step_by_label)
        if st.session_state.get("workflow_step_label_choice") not in step_by_label:
            st.session_state.workflow_step_label_choice = step_labels[0]
        choice = st.radio(
            "Go to step",
            step_labels,
            key="workflow_step_label_choice",
            format_func=lambda label: _workflow_step_radio_label(step_by_label[label]),
        )
        selected = step_by_label[choice]
        if selected["status"] == "locked":
            st.warning("This step is locked until the earlier required workflow state exists.")
        return selected["label"]


def _workflow_step_radio_label(step: dict[str, str]) -> str:
    icon = "✓" if step["status"] == "done" else "!" if step["status"] == "ready" else "🔒"
    return f"{icon} {step['label']}"


def _go_to_workflow_step(step_label: str) -> None:
    st.session_state.workflow_step_label_choice = step_label
    st.session_state.scroll_to_top = True


def _scroll_to_top_once() -> None:
    if not st.session_state.get("scroll_to_top", False):
        return
    st.session_state.scroll_to_top = False
    components.html(
        """
        <script>
        const streamlitDoc = window.parent.document;
        const scrollTarget = streamlitDoc.querySelector('section.main');
        if (scrollTarget) {
            scrollTarget.scrollTo({ top: 0, behavior: 'auto' });
        }
        window.parent.scrollTo({ top: 0, behavior: 'auto' });
        </script>
        """,
        height=0,
    )


def _render_next_step_button(current_step_label: str) -> None:
    steps = _workflow_steps()
    current_idx = next(
        (idx for idx, step in enumerate(steps) if step["label"] == current_step_label),
        None,
    )
    if current_idx is None or current_idx >= len(steps) - 1:
        return

    next_step = steps[current_idx + 1]

    st.divider()
    st.button(
        f"Go to Next Step: {next_step['label']}",
        type="primary",
        disabled=next_step["status"] == "locked",
        on_click=_go_to_workflow_step,
        args=(next_step["label"],),
    )


# ---------------------------------------------------------------------------
# Load-status helper
# ---------------------------------------------------------------------------

def _file_status_card(label: str, name: str, n_bytes: int, extra: str = "") -> None:
    """Render a single loaded-file status row."""
    kb = n_bytes / 1024
    size_str = f"{kb:,.0f} KB" if kb < 1024 else f"{kb/1024:,.1f} MB"
    icon = {"Metric": "📈", "Events": "🗓️"}.get(label, "📄")
    st.markdown(
        f"&nbsp;&nbsp;{icon} **{label}** &nbsp;`{name}`&nbsp; "
        f"<span style='color:grey;font-size:0.85em'>{size_str}{' — ' + extra if extra else ''}</span>",
        unsafe_allow_html=True,
    )


def _classify_file_role(name: str, selected_role: str) -> str:
    """Classify an uploaded CSV as metric or event from filename hints."""
    lower_name = name.lower()
    if "events" in lower_name:
        return "event"
    if "animal_loc" in lower_name:
        return "metric"
    return selected_role


def _split_uploaded_files(
    uploaded_metrics: list[Any] | None,
    uploaded_events: list[Any] | None,
) -> tuple[list[tuple[str, bytes]], list[tuple[str, bytes]], list[str]]:
    """Read uploaded files and auto-correct obvious metric/event mix-ups."""
    metric_files: list[tuple[str, bytes]] = []
    event_files: list[tuple[str, bytes]] = []
    corrections: list[str] = []

    for selected_role, uploads in (
        ("metric", uploaded_metrics or []),
        ("event", uploaded_events or []),
    ):
        for uploaded in uploads:
            name = uploaded.name
            inferred_role = _classify_file_role(name, selected_role)
            if inferred_role != selected_role:
                corrections.append(
                    f"`{name}` was uploaded as {selected_role}, but its filename looks like "
                    f"{inferred_role}; it was moved to {inferred_role} files."
                )
            target = event_files if inferred_role == "event" else metric_files
            target.append((name, uploaded.read()))

    return metric_files, event_files, corrections


def _normalise_match_name(name: str) -> str:
    """Return a compact filename token for metric/event pairing guesses."""
    stem = Path(name).stem.lower()
    for token in (
        "events",
        "metrics",
        "event",
        "activity",
        "metric",
        "animal",
        "loc",
        "index",
        "smoothed",
        "csv",
    ):
        stem = stem.replace(token, " ")
    return " ".join(stem.replace("-", " ").replace("_", " ").split())


def _guess_event_file_mapping(metric_names: list[str], event_names: list[str]) -> dict[str, str]:
    """Guess metric-file to event-file mapping from filename similarity."""
    if not event_names:
        return {name: _NO_EVENT_FILE for name in metric_names}

    normalised_events = {name: _normalise_match_name(name) for name in event_names}
    mapping: dict[str, str] = {}
    for metric_name in metric_names:
        metric_key = _normalise_match_name(metric_name)
        best_name = max(
            event_names,
            key=lambda event_name: SequenceMatcher(
                None,
                metric_key,
                normalised_events[event_name],
            ).ratio(),
        )
        mapping[metric_name] = best_name
    return mapping


def _sync_raw_event_file_mapping() -> None:
    """Keep stored raw preview file mappings aligned with the loaded files."""
    metric_names = [name for name, _ in st.session_state.metric_files]
    event_names = [name for name, _ in st.session_state.event_files]
    valid_event_choices = set(event_names) | {_NO_EVENT_FILE}
    guessed = _guess_event_file_mapping(metric_names, event_names)
    current = st.session_state.raw_event_file_mapping or {}

    st.session_state.raw_event_file_mapping = {
        metric_name: (
            current[metric_name]
            if metric_name in current and current[metric_name] in valid_event_choices
            else guessed[metric_name]
        )
        for metric_name in metric_names
    }


def _add_event_overlay(
    fig: Any,
    event_df: pd.DataFrame,
    *,
    x_col: str,
    y_value: float,
) -> None:
    """Overlay event timestamps on a raw Plotly time-series figure."""
    if event_df.empty or x_col not in event_df.columns or "event_type" not in event_df.columns:
        return

    events_for_plot = event_df.dropna(subset=[x_col]).sort_values(x_col)
    if events_for_plot.empty:
        return

    for event_type, type_df in events_for_plot.groupby("event_type", dropna=False):
        fig.add_scatter(
            x=type_df[x_col],
            y=[y_value] * len(type_df),
            mode="markers",
            marker=dict(size=9, symbol="triangle-down", line=dict(width=1)),
            name=f"Event: {event_type}",
            text=type_df["event_type"],
            hovertemplate="Event: %{text}<br>Time: %{x}<extra></extra>",
        )
    for ts in events_for_plot[x_col]:
        fig.add_vline(x=ts, line_dash="dot", line_color="rgba(80,80,80,0.35)", line_width=1)


def _alignment_preview_table(
    long_df: pd.DataFrame | None,
    event_df: pd.DataFrame | None,
    *,
    event_type: str | None,
    scope: str,
    fallback_timestamp: str | None,
) -> pd.DataFrame:
    """Build a compact preview of per-subject alignment anchors."""
    if long_df is None or long_df.empty or "subject_id" not in long_df.columns:
        return pd.DataFrame()

    pairs = long_df[["subject_id", "group_id"]].drop_duplicates().copy()
    rows: list[dict[str, Any]] = []
    fallback_ts = pd.to_datetime(fallback_timestamp, utc=True, errors="coerce") if fallback_timestamp else pd.NaT
    events = event_df if event_df is not None else pd.DataFrame()

    for _, pair in pairs.iterrows():
        subject_id = str(pair["subject_id"])
        group_id = str(pair.get("group_id", ""))
        source = "missing"
        anchor = pd.NaT

        if event_type and not events.empty and "event_type" in events.columns:
            event_mask = events["event_type"] == event_type
            if scope == "subject" and "subject_id" in events.columns:
                candidates = events.loc[
                    event_mask & (events["subject_id"].astype(str) == subject_id),
                    "timestamp_utc",
                ]
                source = "subject event"
                if candidates.empty and "group_id" in events.columns:
                    candidates = events.loc[
                        event_mask & (events["group_id"].astype(str) == group_id),
                        "timestamp_utc",
                    ]
                    source = "group event fallback"
            elif "group_id" in events.columns:
                candidates = events.loc[
                    event_mask & (events["group_id"].astype(str) == group_id),
                    "timestamp_utc",
                ]
                source = "group event"
            else:
                candidates = pd.Series(dtype="datetime64[ns, UTC]")

            candidates = pd.to_datetime(candidates, utc=True, errors="coerce").dropna()
            if not candidates.empty:
                anchor = candidates.iloc[0]

        if pd.isna(anchor) and not pd.isna(fallback_ts):
            anchor = fallback_ts
            source = "manual fallback"

        rows.append(
            {
                "subject_id": subject_id,
                "group_id": group_id,
                "alignment_timestamp": anchor if not pd.isna(anchor) else "",
                "source": source,
            }
        )

    return pd.DataFrame(rows)


def _pipeline_run_summary() -> list[str]:
    """Describe what the current pipeline configuration will do."""
    summary: list[str] = []

    event_df = st.session_state.event_df
    if event_df is not None and not event_df.empty:
        event_counts = event_df["event_type"].value_counts().head(6)
        event_text = ", ".join(f"{event_type}: {count}" for event_type, count in event_counts.items())
        summary.append(
            f"Events: {len(event_df):,} event rows are loaded. The most frequent event types are {event_text}."
        )
    else:
        summary.append(
            "Events: no event file is loaded. Event-based exclusions and event-based alignment will be skipped unless a manual alignment timestamp is set."
        )

    active_rules = [
        (
            event_type,
            rule,
        )
        for event_type, rule in st.session_state.exclusion_rules.items()
        if rule.get("exclude") or rule.get("flag")
    ]
    if active_rules:
        rule_text = "; ".join(
            (
                f"{event_type}: {rule.get('before_hours', 0):g} h before, "
                f"{rule.get('after_hours', 0):g} h after"
                f"{' excluded' if rule.get('exclude') else ''}"
                f"{' flagged' if rule.get('flag') else ''}"
            )
            for event_type, rule in active_rules
        )
        summary.append(f"Exclusions: matching rows around configured events will be marked using these rules: {rule_text}.")
    else:
        summary.append("Exclusions: no event exclusion or flagging rules are active.")

    aln = st.session_state.alignment_cfg
    event_type = aln.get("event_type")
    fallback = aln.get("fallback_timestamp")
    if event_type:
        summary.append(
            f"Alignment: rows will be aligned to the first `{event_type}` event using `{aln.get('scope', 'subject')}` scope. "
            f"The app will create `time_from_event_hours`, where 0 is `{aln.get('label', 'J0')}`."
            + (f" Missing anchors will use fallback `{fallback}`." if fallback else "")
        )
    elif fallback:
        summary.append(
            f"Alignment: all rows will be aligned to the manual global timestamp `{fallback}`. "
            f"The app will create `time_from_event_hours`, where 0 is `{aln.get('label', 'J0')}`."
        )
    else:
        summary.append(
            "Alignment: disabled. No `time_from_event_hours` anchor will be created, so baseline and aligned plots may be unavailable."
        )

    bsl = st.session_state.baseline_cfg
    if event_type or fallback:
        summary.append(
            f"Baseline: for each subject and metric, baseline will be computed from {float(bsl.get('start_hours', -72)):g} h "
            f"to {float(bsl.get('end_hours', -24)):g} h relative to alignment. "
            f"At least {float(bsl.get('min_coverage', 0.7)):.0%} coverage is required. "
            f"Excluded rows {'will' if bsl.get('exclude_excluded', True) else 'will not'} be removed from baseline calculations. "
            f"Invalid baselines {'will' if bsl.get('impute_from_group_mean', False) else 'will not'} be imputed from group means."
        )
    else:
        summary.append("Baseline: skipped unless alignment is configured, because the baseline window is relative to time zero.")

    aggregation_bin = st.session_state.aggregation_bin
    if aggregation_bin is None:
        summary.append("Aggregation: disabled. The processed output will keep the native time resolution.")
    else:
        summary.append(
            f"Aggregation: after preprocessing, rows will be combined into {aggregation_bin:g}-second bins to reduce noise and output size."
        )

    return summary


def _decimal_time(time_text: str, default: float) -> float:
    try:
        parts = str(time_text).split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        return hour + minute / 60.0
    except Exception:
        return default


def _dark_onset_zt() -> float:
    light_on_h = _decimal_time(st.session_state.light_on, 7.0)
    light_off_h = _decimal_time(st.session_state.light_off, 19.0)
    return (light_off_h - light_on_h) % 24.0


def _circadian_profile(
    df: pd.DataFrame,
    *,
    value_col: str,
    max_days: int,
    zt_bin_hours: float,
    normalize_mode: str,
) -> pd.DataFrame:
    required = {"zeitgeber_time_hours", "timestamp_local", "subject_id", value_col}
    if df.empty or not required <= set(df.columns):
        return pd.DataFrame()

    data = df.copy()
    data["timestamp_local"] = pd.to_datetime(data["timestamp_local"], errors="coerce")
    data["zeitgeber_time_hours"] = pd.to_numeric(data["zeitgeber_time_hours"], errors="coerce") % 24.0
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=["timestamp_local", "zeitgeber_time_hours", "subject_id", value_col])
    if data.empty:
        return pd.DataFrame()

    first_day = data["timestamp_local"].dt.normalize().min()
    data["_analysis_day"] = (data["timestamp_local"].dt.normalize() - first_day).dt.days + 1
    data = data[(data["_analysis_day"] >= 1) & (data["_analysis_day"] <= max_days)].copy()
    if data.empty:
        return pd.DataFrame()

    label_col = "group_label" if "group_label" in data.columns else "group_id"
    if label_col not in data.columns:
        data[label_col] = "All"
    data["_zt_bin"] = (data["zeitgeber_time_hours"] // zt_bin_hours) * zt_bin_hours

    if normalize_mode != "Raw values":
        dark_onset = _dark_onset_zt()
        norm_keys = [label_col, "subject_id", "_analysis_day"]
        if normalize_mode == "Normalize to 24 h max":
            denom = data.groupby(norm_keys, dropna=False)[value_col].transform("max")
        else:
            window_end = (dark_onset + 3.0) % 24.0
            if window_end > dark_onset:
                in_window = data["zeitgeber_time_hours"].between(dark_onset, window_end, inclusive="left")
            else:
                in_window = (data["zeitgeber_time_hours"] >= dark_onset) | (
                    data["zeitgeber_time_hours"] < window_end
                )
            max_lookup = (
                data.loc[in_window]
                .groupby(norm_keys, dropna=False)[value_col]
                .max()
                .rename("_norm_denom")
            )
            data = data.merge(max_lookup, on=norm_keys, how="left")
            denom = data["_norm_denom"]
        data[value_col] = data[value_col] / denom.where(denom != 0)

    subject_profile = (
        data.groupby([label_col, "subject_id", "_analysis_day", "_zt_bin"], dropna=False)[value_col]
        .mean()
        .reset_index(name="subject_mean")
    )
    summary = (
        subject_profile.groupby([label_col, "_zt_bin"], dropna=False)["subject_mean"]
        .agg(mean="mean", sd=lambda s: s.std(ddof=1), n="count")
        .reset_index()
        .rename(columns={label_col: "group_label", "_zt_bin": "zeitgeber_time_hours"})
    )
    summary["sem"] = summary.apply(
        lambda row: row["sd"] / (row["n"] ** 0.5) if row["n"] > 1 else 0.0,
        axis=1,
    )
    summary["max_days"] = max_days
    summary["normalization"] = normalize_mode
    return summary


def _plot_circadian_profile(profile: pd.DataFrame, *, value_label: str) -> go.Figure:
    fig = go.Figure()
    dark_onset = _dark_onset_zt()
    fig.add_vrect(
        x0=dark_onset,
        x1=24,
        fillcolor="rgba(25, 25, 35, 0.18)",
        layer="below",
        line_width=0,
        annotation_text="Dark phase",
        annotation_position="top left",
    )

    if profile.empty:
        fig.update_layout(title="No circadian profile data available.", template="plotly_white")
        return fig

    for group_label, group_df in profile.groupby("group_label", dropna=False):
        group_df = group_df.sort_values("zeitgeber_time_hours")
        x = group_df["zeitgeber_time_hours"]
        mean = group_df["mean"]
        sem = group_df["sem"].fillna(0)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean,
                mode="lines+markers",
                name=str(group_label),
                line=dict(width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pd.concat([x, x.iloc[::-1]]),
                y=pd.concat([mean + sem, (mean - sem).iloc[::-1]]),
                fill="toself",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
                name=f"{group_label} SEM",
            )
        )

    fig.update_layout(
        title="Circadian rhythm profile",
        xaxis_title="Zeitgeber time (hours from lights on)",
        yaxis_title=value_label,
        xaxis=dict(range=[0, 24], dtick=3),
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def _render_raw_import_preview() -> None:
    """Render metric/event file mapping controls and raw preview plot."""
    long_df = st.session_state.long_df
    event_df = st.session_state.event_df
    if long_df is None or long_df.empty or "source_file" not in long_df.columns:
        return

    st.divider()
    st.subheader("Raw data preview with events")
    st.caption(
        "Confirm which event file belongs with each activity/metric file before plotting."
    )

    _sync_raw_event_file_mapping()
    metric_names = [name for name, _ in st.session_state.metric_files]
    event_names = [name for name, _ in st.session_state.event_files]
    event_choices = [_NO_EVENT_FILE] + event_names

    mapping_rows = pd.DataFrame(
        {
            "metric_file": metric_names,
            "event_file": [
                st.session_state.raw_event_file_mapping.get(name, _NO_EVENT_FILE)
                for name in metric_names
            ],
        }
    )
    edited_mapping = st.data_editor(
        mapping_rows,
        width="stretch",
        hide_index=True,
        disabled=["metric_file"],
        column_config={
            "metric_file": st.column_config.TextColumn(
                "Activity/metric file",
                help="Loaded metric CSV file.",
            ),
            "event_file": st.column_config.SelectboxColumn(
                "Matching event file",
                help="Confirm or correct the event CSV to overlay with this metric file.",
                options=event_choices,
                required=True,
            ),
        },
        key="raw_event_mapping_editor",
    )
    st.session_state.raw_event_file_mapping = {
        str(row["metric_file"]): str(row["event_file"])
        for _, row in edited_mapping.iterrows()
    }

    source_options = [
        name
        for name in metric_names
        if name in set(long_df["source_file"].dropna().astype(str))
    ]
    if not source_options:
        return

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        selected_source = st.selectbox(
            "Activity file",
            source_options,
            key="raw_preview_metric_source",
        )

    source_df = long_df[long_df["source_file"] == selected_source].copy()
    metric_options = (
        sorted(source_df["metric_name"].dropna().unique().tolist())
        if "metric_name" in source_df.columns
        else []
    )
    subject_options = (
        sorted(source_df["subject_id"].dropna().unique().tolist())
        if "subject_id" in source_df.columns
        else []
    )
    with c2:
        selected_metric = st.selectbox(
            "Metric",
            metric_options,
            key="raw_preview_metric_name",
        ) if metric_options else None
    with c3:
        selected_subjects = st.multiselect(
            "Subjects",
            subject_options,
            default=subject_options[:6],
            key="raw_preview_subjects",
        )

    if not selected_metric:
        st.info("No metric is available for the selected activity file.")
        return

    selected_subjects_key = tuple(selected_subjects)
    plot_settings = {
        "source": selected_source,
        "metric": selected_metric,
        "subjects": selected_subjects_key,
        "event_file": st.session_state.raw_event_file_mapping.get(selected_source, _NO_EVENT_FILE),
    }
    if st.button("Plot raw data with events", type="primary", key="raw_preview_plot_button"):
        st.session_state.raw_preview_plot_requested = True
        st.session_state.raw_preview_plot_settings = plot_settings

    if (
        not st.session_state.raw_preview_plot_requested
        or st.session_state.raw_preview_plot_settings != plot_settings
    ):
        st.caption("Click Plot raw data with events to generate this preview.")
        return

    x_col = "timestamp_local" if "timestamp_local" in source_df.columns else "timestamp_utc"
    with _working_status(
        "Drawing raw preview",
        "The app is plotting the selected activity file and preparing mapped event markers.",
    ):
        fig = qc.plot_raw_timeseries(
            source_df,
            selected_metric,
            subjects=selected_subjects if selected_subjects else None,
            x_col=x_col,
            show_exclusions=False,
        )

    mapped_event_file = st.session_state.raw_event_file_mapping.get(
        selected_source,
        _NO_EVENT_FILE,
    )
    if (
        mapped_event_file != _NO_EVENT_FILE
        and event_df is not None
        and not event_df.empty
        and "source_file" in event_df.columns
    ):
        mapped_events = event_df[event_df["source_file"] == mapped_event_file].copy()
        event_x_col = x_col if x_col in mapped_events.columns else "timestamp_utc"
        y_max = pd.to_numeric(
            source_df.loc[source_df["metric_name"] == selected_metric, "value"],
            errors="coerce",
        ).max()
        if pd.notna(y_max):
            _add_event_overlay(fig, mapped_events, x_col=event_x_col, y_value=float(y_max))
        n_events = int(mapped_events[event_x_col].notna().sum())
        st.caption(f"Overlaying {n_events:,} events from `{mapped_event_file}`.")
    else:
        st.caption("No event file is mapped for this activity file.")

    st.plotly_chart(fig, width="stretch")


def _render_load_status() -> None:
    long_df = st.session_state.long_df
    event_df = st.session_state.event_df

    # ---- Per-file cards ----
    st.subheader("Loaded files")

    if st.session_state.metric_files:
        for name, data in st.session_state.metric_files:
            # Pull per-file stats from long_df
            extra = ""
            if long_df is not None and "source_file" in long_df.columns:
                sub = long_df[long_df["source_file"] == name]
                if not sub.empty:
                    groups = ", ".join(sorted(sub["group_id"].dropna().unique()))
                    n_subj = sub["subject_id"].nunique()
                    n_rows = len(sub)
                    nb = sub["native_bin_seconds"].dropna()
                    bin_str = f"{nb.mode().iloc[0]:.0f}s bins" if not nb.empty else ""
                    extra = (
                        f"{n_rows:,} rows · {n_subj} subjects · groups: {groups}"
                        + (f" · {bin_str}" if bin_str else "")
                    )
            _file_status_card("Metric", name, len(data), extra)
    else:
        st.info("No metric files loaded.")

    if st.session_state.event_files:
        for name, data in st.session_state.event_files:
            extra = ""
            if event_df is not None and "source_file" in event_df.columns:
                sub = event_df[event_df["source_file"] == name]
                if not sub.empty:
                    counts = sub["event_type"].value_counts()
                    extra = " · ".join(f"{v}×{k}" for k, v in counts.items())
            _file_status_card("Events", name, len(data), extra)
    else:
        st.caption("No event files loaded — exclusion and event-based alignment will be skipped.")

    _render_raw_import_preview()

    # ---- Pipeline stage tracker ----
    st.divider()
    st.subheader("Pipeline readiness")

    has_metric = long_df is not None and not long_df.empty
    has_events = event_df is not None and not event_df.empty
    has_subj_meta = st.session_state.subject_meta is not None
    has_processed = st.session_state.processed_df is not None
    has_baseline = (
        st.session_state.baseline_summary is not None
        and not st.session_state.baseline_summary.empty
    )

    aln_cfg = st.session_state.alignment_cfg
    has_alignment_cfg = bool(
        aln_cfg.get("event_type") or aln_cfg.get("fallback_timestamp")
    )

    def _stage(icon: str, label: str, detail: str = "") -> None:
        suffix = f"  <span style='color:grey;font-size:0.85em'>{detail}</span>" if detail else ""
        st.markdown(f"{icon} {label}{suffix}", unsafe_allow_html=True)

    _stage(
        "✅" if has_metric else "⬜",
        "**Metric data loaded**",
        f"{len(long_df):,} rows" if has_metric else "upload or load examples above",
    )
    _stage(
        "✅" if has_events else "➖",
        "**Event data loaded**",
        f"{len(event_df):,} events" if has_events else "optional",
    )
    _stage(
        "✅" if has_subj_meta else ("⬜" if has_metric else "🔒"),
        "**Subject metadata initialised**",
        "edit in Tab 3" if has_subj_meta else ("auto-built — go to Tab 3" if has_metric else "load data first"),
    )
    _stage(
        "✅" if has_alignment_cfg else ("⬜" if has_metric else "🔒"),
        "**Alignment configured**",
        f"event: {aln_cfg.get('event_type') or 'manual'}" if has_alignment_cfg else "configure in Tab 4",
    )
    _stage(
        "✅" if has_processed else ("⬜" if has_metric else "🔒"),
        "**Pipeline run**",
        "processed — go to Tab 6 for plots" if has_processed else "run in Tab 5",
    )
    _stage(
        "✅" if has_baseline else ("⬜" if has_processed else "🔒"),
        "**Baseline computed**",
        f"{len(st.session_state.baseline_summary)} subject/metric pairs" if has_baseline else "",
    )

    # Warnings inline
    warns = st.session_state.all_warnings
    if warns:
        with st.expander(f"⚠️ {len(warns)} warning(s)", expanded=False):
            for w in warns:
                st.warning(w)


# ---------------------------------------------------------------------------
# Tab 1: Import
# ---------------------------------------------------------------------------

def _tab_import() -> None:
    st.header("Data Import")
    render_contextual_help("import")

    with st.expander("Data-flow diagram", expanded=False):
        render_data_flow_diagram()

    col_metric, col_event = st.columns(2)

    with col_metric:
        st.subheader("DVC Metric CSV files")
        uploaded_metrics = st.file_uploader(
            "Upload one or more metric CSV files",
            type=["csv"],
            accept_multiple_files=True,
            key="upload_metric",
        )

    with col_event:
        st.subheader("DVC Event CSV files (optional)")
        uploaded_events = st.file_uploader(
            "Upload one or more event CSV files",
            type=["csv"],
            accept_multiple_files=True,
            key="upload_event",
        )

    # ---- Build file list ----
    if uploaded_metrics or uploaded_events:
        metric_files, event_files, corrections = _split_uploaded_files(
            uploaded_metrics,
            uploaded_events,
        )
        # Compare by corrected name lists only; file bytes are already read above.
        new_metric_names = sorted(name for name, _ in metric_files)
        old_metric_names = sorted(name for name, _ in st.session_state.metric_files)
        new_event_names = sorted(name for name, _ in event_files)
        old_event_names = sorted(name for name, _ in st.session_state.event_files)

        changed = new_metric_names != old_metric_names or new_event_names != old_event_names
        if changed:
            st.session_state.metric_files = metric_files
            st.session_state.event_files = event_files
            st.session_state.raw_event_file_mapping = {}
            st.session_state.raw_preview_plot_requested = False
            st.session_state.raw_preview_plot_settings = {}
            st.session_state.subject_meta = None
            st.session_state.group_meta = None
            with _working_status(
                "Reading uploaded files",
                (
                    "The app is classifying activity and event CSVs, parsing timestamps, "
                    "combining metric rows, and preparing metadata templates."
                ),
            ):
                _parse_all_files()
        if corrections:
            st.warning(
                "File type auto-correction applied. Please confirm the loaded files and "
                "activity/event mapping below."
            )
            for correction in corrections:
                st.caption(correction)

    # ---- Loading status panel ----
    st.divider()
    _render_load_status()


# ---------------------------------------------------------------------------
# Tab 2: Validate
# ---------------------------------------------------------------------------

def _tab_validate() -> None:
    st.header("Data Validation")
    render_contextual_help("validate")

    long_df = st.session_state.long_df
    event_df = st.session_state.event_df

    if long_df is None:
        st.info("No data loaded. Go to **Import** to upload or load example files.")
        return

    # Metric data overview
    st.subheader("Metric data overview")

    groups = sorted(long_df["group_id"].dropna().unique().tolist()) if "group_id" in long_df.columns else []
    subjects = sorted(long_df["subject_id"].dropna().unique().tolist()) if "subject_id" in long_df.columns else []
    metrics = sorted(long_df["metric_name"].dropna().unique().tolist()) if "metric_name" in long_df.columns else []
    sources = sorted(long_df["source_file"].dropna().unique().tolist()) if "source_file" in long_df.columns else []

    ts_range = "N/A"
    if "timestamp_utc" in long_df.columns:
        t = long_df["timestamp_utc"].dropna()
        if not t.empty:
            ts_range = f"{t.min()} → {t.max()}"

    native_bin = "N/A"
    if "native_bin_seconds" in long_df.columns:
        nb = long_df["native_bin_seconds"].dropna().mode()
        if not nb.empty:
            native_bin = f"{nb.iloc[0]:.0f} s"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Source files", len(sources))
        st.metric("Detected groups", len(groups))
    with col2:
        st.metric("Detected subjects", len(subjects))
        st.metric("Total rows (long)", f"{len(long_df):,}")
    with col3:
        st.metric("Metrics", len(metrics))
        st.metric("Native bin size", native_bin)

    st.write(f"**Timestamp range:** {ts_range}")
    st.write(f"**Groups:** {', '.join(groups)}")
    st.write(f"**Metrics:** {', '.join(metrics)}")

    # Missing values
    n_missing = int(long_df["value"].isna().sum())
    if n_missing:
        st.warning(f"{n_missing:,} rows have missing `value`.")

    # Warnings
    if st.session_state.all_warnings:
        with st.expander(f"Parse warnings ({len(st.session_state.all_warnings)})"):
            for w in st.session_state.all_warnings:
                st.warning(w)

    # Event data overview
    st.subheader("Event data overview")
    if event_df is not None and not event_df.empty:
        st.write(f"Total events: **{len(event_df):,}**")
        if "event_type" in event_df.columns:
            vc = event_df["event_type"].value_counts().reset_index()
            vc.columns = ["event_type", "count"]
            st.dataframe(vc, width='stretch', hide_index=True)
    else:
        st.info("No event data loaded.")

    # Data preview
    st.subheader("Long-format data preview (first 200 rows)")
    preview_cols = [
        c for c in [
            "source_file", "metric_name", "group_id", "subject_id",
            "timestamp", "relative_time_seconds", "value",
            "group_avg", "group_sem", "samples", "native_bin_seconds"
        ] if c in long_df.columns
    ]
    st.dataframe(long_df[preview_cols].head(200), width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3: Metadata & Study Design
# ---------------------------------------------------------------------------

def _tab_metadata() -> None:
    st.header("Experiment Metadata & Study Design")
    render_contextual_help("metadata")

    long_df = st.session_state.long_df
    if long_df is None:
        st.info("Load data first (Tab 1).")
        return

    # ---- 1. Study-level metadata ----
    st.subheader("1. Study-level metadata")
    sm = st.session_state.study_meta

    with st.expander("Edit study metadata", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            sm["study_id"] = st.text_input("Study ID", value=sm.get("study_id", ""))
            sm["study_name"] = st.text_input("Study name", value=sm.get("study_name", ""))
            sm["project_name"] = st.text_input("Project name", value=sm.get("project_name", ""))
            sm["partner_name"] = st.text_input("Partner name", value=sm.get("partner_name", ""))
            sm["operator_name"] = st.text_input("Operator name", value=sm.get("operator_name", ""))
            sm["species"] = st.text_input("Species", value=sm.get("species", "mouse"))
            sm["strain"] = st.text_input("Strain", value=sm.get("strain", ""))
        with c2:
            sm["experiment_start_date"] = st.text_input(
                "Experiment start date (YYYY-MM-DD)", value=sm.get("experiment_start_date", "")
            )
            sm["experiment_end_date"] = st.text_input(
                "Experiment end date (YYYY-MM-DD)", value=sm.get("experiment_end_date", "")
            )
            sm["timezone"] = st.text_input(
                "Timezone", value=sm.get("timezone", cfg.DEFAULT_TIMEZONE)
            )
            sm["light_on_time"] = st.text_input(
                "Lights ON (HH:MM)", value=sm.get("light_on_time", "07:00")
            )
            sm["light_off_time"] = st.text_input(
                "Lights OFF (HH:MM)", value=sm.get("light_off_time", "19:00")
            )
            sm["main_experimental_event_name"] = st.text_input(
                "Main event name (e.g. surgery, injection)",
                value=sm.get("main_experimental_event_name", ""),
            )
        sm["experiment_description"] = st.text_area(
            "Experiment description", value=sm.get("experiment_description", "")
        )
        sm["notes"] = st.text_area("Notes", value=sm.get("notes", ""))

        # Sync timezone/light cycle into processing config
        st.session_state.timezone = sm["timezone"]
        st.session_state.light_on = sm["light_on_time"]
        st.session_state.light_off = sm["light_off_time"]

    st.session_state.study_meta = sm

    # ---- 2. Subject metadata ----
    st.subheader("2. Subject-level metadata")
    if st.session_state.subject_meta is None:
        st.session_state.subject_meta = meta_mod.build_subject_metadata_template(long_df)
    st.session_state.subject_meta = _normalise_metadata_table(
        st.session_state.subject_meta, cfg.SUBJECT_METADATA_COLUMNS
    )

    st.write(
        "Edit the table below. Required columns are marked in the editor; add rows for subjects "
        "that need manual metadata entries."
    )
    edited_subject = st.data_editor(
        st.session_state.subject_meta,
        width='stretch',
        num_rows="dynamic",
        column_config=subject_column_config(),
        key="subject_meta_editor",
    )
    st.session_state.subject_meta = _normalise_metadata_table(
        edited_subject, cfg.SUBJECT_METADATA_COLUMNS
    )

    # Download blank template
    template_csv = edited_subject.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download subject metadata template",
        data=template_csv,
        file_name="subject_metadata_template.csv",
        mime="text/csv",
    )

    # Re-upload filled template
    uploaded_meta = st.file_uploader(
        "Re-upload filled subject metadata CSV", type=["csv"], key="reupload_subject_meta"
    )
    if uploaded_meta is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_meta)
            st.session_state.subject_meta = _normalise_metadata_table(
                uploaded_df, cfg.SUBJECT_METADATA_COLUMNS
            )
            st.success(f"Loaded subject metadata: {len(uploaded_df)} rows.")
        except Exception as exc:
            st.error(f"Could not read metadata file: {exc}")

    # Validation
    errors, meta_warns = meta_mod.validate_subject_metadata(
        st.session_state.subject_meta, long_df
    )
    if errors:
        for e in errors:
            st.error(e)
    if meta_warns:
        for w in meta_warns:
            st.warning(w)
    if not errors and not meta_warns:
        st.success("Subject metadata validated.")

    # ---- 3. Group metadata ----
    st.subheader("3. Group-level metadata")
    if st.session_state.group_meta is None:
        st.session_state.group_meta = meta_mod.build_group_metadata_template(long_df)
    st.session_state.group_meta = _normalise_metadata_table(
        st.session_state.group_meta, cfg.GROUP_METADATA_COLUMNS
    )

    st.write(
        "Map DVC group names to scientific labels. "
        "The `group_label` column will be used in plots and exports."
    )
    with st.expander("Guided group builder", expanded=False):
        detected_subjects = sorted(long_df["subject_id"].dropna().astype(str).unique().tolist())
        n_groups = st.number_input(
            "How many groups do you have?",
            min_value=1,
            max_value=max(1, len(detected_subjects)),
            value=max(1, len(st.session_state.group_meta)),
            step=1,
        )
        current = st.session_state.group_meta.copy().reset_index(drop=True)
        while len(current) < n_groups:
            idx = len(current) + 1
            current.loc[len(current), :] = {
                "group_id": f"group_{idx}",
                "group_label": f"Group {idx}",
                "group_color": "#1f77b4",
            }
        st.session_state.group_meta = _normalise_metadata_table(
            current.head(int(n_groups)), cfg.GROUP_METADATA_COLUMNS
        )
        st.caption("Use the group table below for labels, colors, and treatment descriptions.")

    edited_group = st.data_editor(
        st.session_state.group_meta,
        width='stretch',
        num_rows="fixed",
        column_config=group_column_config(),
        key="group_meta_editor",
    )
    st.session_state.group_meta = _normalise_metadata_table(
        edited_group, cfg.GROUP_METADATA_COLUMNS
    )

    with st.expander("Assign animals to groups", expanded=False):
        if st.session_state.subject_meta is not None and not st.session_state.subject_meta.empty:
            subjects = sorted(st.session_state.subject_meta["subject_id"].dropna().astype(str).unique().tolist())
            assigned_updates: dict[str, str] = {}
            for _, group_row in st.session_state.group_meta.iterrows():
                label = str(group_row.get("group_label") or group_row.get("group_id") or "").strip()
                if not label:
                    continue
                current_subjects = st.session_state.subject_meta[
                    st.session_state.subject_meta["treatment_group"].astype(str) == label
                ]["subject_id"].astype(str).tolist()
                selected_subjects = st.multiselect(
                    label,
                    subjects,
                    default=[s for s in current_subjects if s in subjects],
                    key=f"group_assign_{label}",
                )
                for sid in selected_subjects:
                    assigned_updates[sid] = label
            if assigned_updates:
                for sid, label in assigned_updates.items():
                    mask = st.session_state.subject_meta["subject_id"].astype(str) == sid
                    st.session_state.subject_meta.loc[mask, "treatment_group"] = label

    # Download
    grp_csv = edited_group.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download group metadata",
        data=grp_csv,
        file_name="group_metadata.csv",
        mime="text/csv",
    )

    # ---- 4. Treatment schedule ----
    st.subheader("4. Treatment schedule")
    st.write(
        "Add repeated treatments, injections, surgeries, or dosing events. These rows are "
        "exported and can be used to document alignment choices."
    )
    treatment_cols = ["animal_id", "event_type", "timestamp", "dose", "unit", "route", "notes"]
    st.session_state.treatment_schedule = _normalise_metadata_table(
        st.session_state.treatment_schedule, treatment_cols
    )
    st.session_state.treatment_schedule = st.data_editor(
        st.session_state.treatment_schedule,
        width="stretch",
        num_rows="dynamic",
        column_config={
            "animal_id": st.column_config.TextColumn(
                "Animal ID", help="Animal identifier matching the subject metadata table.", required=True
            ),
            "event_type": st.column_config.TextColumn(
                "Event type", help="Example: injection, surgery, gavage, drug_on.", required=True
            ),
            "timestamp": st.column_config.TextColumn(
                "Timestamp", help="ISO timestamp for the treatment event.", required=True
            ),
            "dose": st.column_config.NumberColumn("Dose", help="Numeric dose when relevant."),
            "unit": st.column_config.TextColumn("Unit", help="Example: mg/kg, uL, mL."),
            "route": st.column_config.TextColumn("Route", help="Example: i.p., s.c., oral."),
            "notes": st.column_config.TextColumn("Notes"),
        },
        key="treatment_schedule_editor",
    )


# ---------------------------------------------------------------------------
# Tab 4: Events, Alignment & Exclusions
# ---------------------------------------------------------------------------

def _tab_events() -> None:
    st.header("Events, Alignment & Exclusions")
    render_contextual_help("events")

    event_df = st.session_state.event_df

    # ---- Event table preview ----
    st.subheader("Event table")
    if event_df is not None and not event_df.empty:
        cage_pairs = exclusions.detect_cage_change_pairs(
            event_df,
            max_gap_hours=float(st.session_state.exclusion_rules.get("CAGE_CHANGE", {}).get("max_gap_hours", 6.0)),
        )
        if not cage_pairs.empty:
            st.info(
                f"Detected {len(cage_pairs)} REMOVED/INSERTED cage-change pair(s). "
                "Use the CAGE_CHANGE exclusion rule for asymmetric handling."
            )
            with st.expander("Detected cage-change pairs", expanded=False):
                st.dataframe(cage_pairs, width="stretch", hide_index=True)

        # Filter controls
        c1, c2, c3 = st.columns(3)
        with c1:
            etypes = ["All"] + sorted(event_df["event_type"].dropna().unique().tolist())
            sel_type = st.selectbox("Filter by event type", etypes)
        with c2:
            groups = ["All"] + sorted(event_df["group_id"].dropna().unique().tolist())
            sel_grp = st.selectbox("Filter by group", groups)
        with c3:
            subjects = ["All"] + sorted(event_df["subject_id"].dropna().unique().tolist())
            sel_sub = st.selectbox("Filter by subject", subjects)

        display_ev = event_df.copy()
        if sel_type != "All":
            display_ev = display_ev[display_ev["event_type"] == sel_type]
        if sel_grp != "All":
            display_ev = display_ev[display_ev["group_id"] == sel_grp]
        if sel_sub != "All":
            display_ev = display_ev[display_ev["subject_id"] == sel_sub]

        st.dataframe(display_ev.head(500), width='stretch', hide_index=True)
    else:
        st.info("No event file loaded. Alignment and event-based exclusion will be skipped.")

    st.subheader("Facility event calendar")
    st.write("Add room-level or facility-level events that affect every animal or selected groups.")
    facility_cols = ["timestamp_start", "timestamp_end", "event_type", "affected_groups", "notes"]
    st.session_state.facility_events = _normalise_metadata_table(
        st.session_state.facility_events, facility_cols
    )
    st.session_state.facility_events = st.data_editor(
        st.session_state.facility_events,
        width="stretch",
        num_rows="dynamic",
        column_config={
            "timestamp_start": st.column_config.TextColumn(
                "Start timestamp", help="Required ISO timestamp for event start.", required=True
            ),
            "timestamp_end": st.column_config.TextColumn(
                "End timestamp", help="Optional ISO timestamp; start is used if empty."
            ),
            "event_type": st.column_config.TextColumn(
                "Event type", help="Example: room_alarm, power_outage, caretaker_change."
            ),
            "affected_groups": st.column_config.TextColumn(
                "Affected groups", help="Leave blank for all groups, or enter one DVC group ID."
            ),
            "notes": st.column_config.TextColumn("Notes"),
        },
        key="facility_events_editor",
    )

    # ---- Alignment configuration ----
    st.subheader("Alignment")
    st.write(
        "Alignment defines the time-zero anchor used later for baseline windows, "
        "event-centered plots, and exported `time_from_event` columns. After alignment, "
        "each raw timestamp is converted into hours relative to the selected anchor: "
        "negative values are before the anchor, 0 is the anchor, and positive values are after."
    )
    with st.expander("How to fill this section", expanded=True):
        st.markdown(
            """
            - Use **Event-based** when the event file contains the experimental anchor, such as treatment, surgery, REMOVED, INSERTED, or another recorded event.
            - Use **Manual global timestamp** when every subject should share the same anchor timestamp, or when no event file is available.
            - Use **Disabled** only if you do not need baseline windows or aligned plots.
            - Choose **subject scope** when each cage/animal has its own event timestamp. Choose **group scope** when one event timestamp applies to all subjects in a group.
            - Fill the fallback timestamp when some subjects may not have the selected event; the fallback is used only for missing anchors.
            """
        )
    aln = st.session_state.alignment_cfg
    event_types_avail = (
        ev_mod.get_unique_event_types(event_df)
        if event_df is not None and not event_df.empty
        else []
    )

    alignment_modes = ["Disabled", "Event-based", "Manual global timestamp"]
    if aln.get("event_type"):
        default_alignment_mode = "Event-based"
    elif aln.get("fallback_timestamp"):
        default_alignment_mode = "Manual global timestamp"
    else:
        default_alignment_mode = "Disabled"
    alignment_mode = st.radio(
        "Alignment mode",
        alignment_modes,
        index=alignment_modes.index(default_alignment_mode),
        help=(
            "Select how time zero should be chosen. Event-based uses timestamps from "
            "the event table; manual uses one timestamp for all subjects."
        ),
    )
    aln["label"] = st.text_input(
        "Alignment label",
        value=aln.get("label", "J0"),
        help="Short name stored in exports for this anchor. Example: J0, surgery, treatment_start.",
    )
    aln["scope"] = st.selectbox(
        "Who gets their own anchor timestamp?",
        ["subject", "group"],
        index=0,
        help=(
            "Subject: find one anchor per subject/cage, with group fallback if needed. "
            "Group: all subjects in the same group use the group event timestamp."
        ),
    )

    if alignment_mode == "Event-based":
        if event_types_avail:
            sel_etype = st.selectbox(
                "Event type to use as time zero",
                event_types_avail,
                index=0,
                help=(
                    "Pick the event that should become time zero. The pipeline uses the "
                    "first matching timestamp for each subject or group."
                ),
            )
            aln["event_type"] = sel_etype
        else:
            st.info("No events loaded. Use Manual global timestamp if every subject shares one anchor.")
            aln["event_type"] = None
        aln["fallback_timestamp"] = st.text_input(
            "Fallback timestamp for missing event anchors",
            value=aln.get("fallback_timestamp") or "",
            help=(
                "Optional ISO timestamp. Used only when no matching event is found for a "
                "subject or group. Example: 2025-03-18T09:00:00Z."
            ),
        ) or None

    elif alignment_mode == "Manual global timestamp":
        aln["event_type"] = None
        aln["fallback_timestamp"] = st.text_input(
            "Global time-zero timestamp",
            value=aln.get("fallback_timestamp") or "",
            help=(
                "Required for manual alignment. Every row is aligned to this same ISO "
                "timestamp. Example: 2025-03-18T09:00:00Z."
            ),
        ) or None

    else:  # Disabled
        aln["event_type"] = None
        aln["fallback_timestamp"] = None
        st.info("Alignment is disabled. Baseline windows and aligned plots will not be available.")

    st.session_state.alignment_cfg = aln

    preview = _alignment_preview_table(
        st.session_state.long_df,
        event_df,
        event_type=aln.get("event_type"),
        scope=aln.get("scope", "subject"),
        fallback_timestamp=aln.get("fallback_timestamp"),
    )
    if alignment_mode != "Disabled" and not preview.empty:
        missing = int((preview["alignment_timestamp"] == "").sum())
        with st.expander("Alignment preview", expanded=True):
            if missing:
                st.warning(
                    f"{missing} subject(s) do not have an alignment timestamp yet. "
                    "Choose a different event type, switch scope, or add a fallback timestamp."
                )
            else:
                st.success("Every detected subject has an alignment timestamp.")
            st.dataframe(preview, width="stretch", hide_index=True)

    # ---- Exclusion rules ----
    st.subheader("Exclusion rules")
    st.write(
        "Configure exclusion windows around each event type. "
        "Excluded rows are retained but flagged in the output."
    )

    rules = st.session_state.exclusion_rules
    all_event_types = sorted(
        set(rules.keys())
        | (set(event_df["event_type"].dropna().unique()) if event_df is not None and not event_df.empty else set())
    )

    for etype in all_event_types:
        rule = rules.get(etype, {"before_hours": 0.0, "after_hours": 0.0, "exclude": False, "flag": False})
        with st.expander(f"Rule: {etype}", expanded=etype in cfg.DEFAULT_EXCLUSION_RULES):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                before = st.number_input(
                    "Before REMOVED / event (h)", value=float(rule.get("before_hours", 0.0)),
                    min_value=0.0, step=1.0, key=f"excl_before_{etype}"
                )
            with c2:
                after = st.number_input(
                    "After INSERTED / event (h)", value=float(rule.get("after_hours", 0.0)),
                    min_value=0.0, step=1.0, key=f"excl_after_{etype}"
                )
            with c3:
                do_excl = st.checkbox(
                    "Exclude rows", value=bool(rule.get("exclude", False)), key=f"excl_excl_{etype}"
                )
            with c4:
                do_flag = st.checkbox(
                    "Flag rows", value=bool(rule.get("flag", False)), key=f"excl_flag_{etype}"
                )
            max_gap = rule.get("max_gap_hours")
            if etype == "CAGE_CHANGE":
                max_gap = st.number_input(
                    "Maximum REMOVED→INSERTED pairing gap (h)",
                    value=float(rule.get("max_gap_hours", 6.0)),
                    min_value=0.0,
                    step=0.5,
                    key=f"excl_max_gap_{etype}",
                )
            rules[etype] = {
                "before_hours": before,
                "after_hours": after,
                "exclude": do_excl,
                "flag": do_flag,
            }
            if max_gap is not None:
                rules[etype]["max_gap_hours"] = max_gap

    st.session_state.exclusion_rules = rules


# ---------------------------------------------------------------------------
# Tab 5: Baseline, Aggregation & Pipeline
# ---------------------------------------------------------------------------

def _tab_baseline() -> None:
    st.header("Baseline, Aggregation & Pipeline")
    render_contextual_help("baseline")

    long_df = st.session_state.long_df
    if long_df is None:
        st.info("Load data first (Tab 1).")
        return

    alignment_ready = _alignment_is_configured()

    # ---- Baseline configuration ----
    st.subheader("Baseline configuration")
    bsl = st.session_state.baseline_cfg
    if not alignment_ready:
        st.info(
            "Baseline is disabled because Alignment is disabled. This is expected: baseline "
            "windows are defined relative to time zero, so the app needs an event-based or "
            "manual alignment anchor first. Aggregation and the rest of the pipeline can still run."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            bsl["start_hours"] = st.number_input(
                "Window start (h relative to event)",
                value=float(bsl.get("start_hours", -72)),
                step=1.0,
                key="bsl_start",
            )
        with c2:
            bsl["end_hours"] = st.number_input(
                "Window end (h relative to event)",
                value=float(bsl.get("end_hours", -24)),
                step=1.0,
                key="bsl_end",
            )
        with c3:
            bsl["min_coverage"] = st.slider(
                "Min coverage (fraction)",
                min_value=0.0,
                max_value=1.0,
                value=float(bsl.get("min_coverage", 0.7)),
                step=0.05,
                key="bsl_cov",
            )
        with c4:
            bsl["exclude_excluded"] = st.checkbox(
                "Exclude flagged rows from baseline",
                value=bool(bsl.get("exclude_excluded", True)),
                key="bsl_excl",
            )
        bsl["impute_from_group_mean"] = st.checkbox(
            "Impute invalid baselines from group mean",
            value=bool(bsl.get("impute_from_group_mean", False)),
            key="bsl_impute_group",
        )
        st.session_state.baseline_cfg = bsl

        with st.expander("Manual baseline overrides", expanded=False):
            st.session_state.baseline_overrides = st.data_editor(
                st.session_state.baseline_overrides,
                width="stretch",
                num_rows="dynamic",
                column_config={
                    "subject_id": st.column_config.TextColumn("Subject ID", required=True),
                    "metric_name": st.column_config.TextColumn("Metric", required=True),
                    "baseline_value": st.column_config.NumberColumn("Baseline value", required=True),
                    "notes": st.column_config.TextColumn("Notes"),
                },
                key="baseline_overrides_editor",
            )

    # ---- Aggregation ----
    st.subheader("Aggregation")
    st.write(
        "Aggregation is independent from alignment. It can be used with or without baseline "
        "to reduce data density and smooth downstream plots/exports."
    )
    agg_keys = list(cfg.AGGREGATION_OPTIONS.keys())
    sel_agg = st.selectbox("Aggregation bin size", agg_keys, index=0)
    st.session_state.aggregation_bin = cfg.AGGREGATION_OPTIONS[sel_agg]

    # ---- Run pipeline ----
    st.divider()
    st.subheader("Run full preprocessing pipeline")
    st.write("Before running, review what the current settings will do to the data:")
    for item in _pipeline_run_summary():
        st.markdown(f"- {item}")

    if st.button("Run pipeline", type="primary"):
        with _working_status(
            "Running preprocessing pipeline",
            (
                "The app is merging metadata, annotating light/dark phases, applying event "
                "exclusions, "
                + (
                    "aligning rows to time zero, computing baselines, and aggregating if requested."
                    if _alignment_is_configured()
                    else "skipping alignment and baseline, then aggregating if requested."
                )
            ),
        ):
            _run_pipeline()
        ts = st.session_state.pipeline_run_ts
        if ts:
            st.success(f"Pipeline complete. {ts.strftime('%H:%M:%S')}")

    # ---- Results preview ----
    if st.session_state.processed_df is not None:
        proc = st.session_state.processed_df
        st.subheader("Processed data summary")

        col1, col2, col3 = st.columns(3)
        with col1:
            n_excl = int(proc["is_excluded"].sum()) if "is_excluded" in proc.columns else 0
            st.metric("Excluded rows", f"{n_excl:,}")
        with col2:
            if "baseline_valid" in proc.columns:
                n_valid_bsl = int(proc["baseline_valid"].sum())
                st.metric("Rows with valid baseline", f"{n_valid_bsl:,}")
        with col3:
            if "time_from_event_hours" in proc.columns:
                n_aligned = int(proc["time_from_event_hours"].notna().sum())
                st.metric("Aligned rows", f"{n_aligned:,}")

        # Baseline summary table
        if st.session_state.baseline_summary is not None and not st.session_state.baseline_summary.empty:
            st.subheader("Baseline validity per subject/metric")
            st.dataframe(st.session_state.baseline_summary, width='stretch', hide_index=True)

        # Warnings
        warns = st.session_state.all_warnings
        if warns:
            with st.expander(f"Warnings ({len(warns)})"):
                for w in warns:
                    st.warning(w)


# ---------------------------------------------------------------------------
# Tab 6: QC Plots
# ---------------------------------------------------------------------------

def _tab_qc() -> None:
    st.header("QC Visualisations")
    render_contextual_help("qc")

    proc = st.session_state.processed_df
    long_df = st.session_state.long_df

    display_df = proc if proc is not None else long_df

    if display_df is None:
        st.info("No data available. Load and process data first.")
        return

    with st.expander("Data quality report", expanded=False):
        quality_report = st.session_state.quality_report
        if quality_report is None:
            try:
                quality_report = quality.build_quality_report(display_df)
            except Exception:
                quality_report = pd.DataFrame()
        if quality_report.empty:
            st.caption("No data-quality report available.")
        else:
            flagged = quality_report[
                quality_report["irregular_interval_flag"]
                | quality_report["missing_value_count"].gt(0)
                | quality_report["duplicate_timestamp_count"].gt(0)
                | quality_report["long_gap_count"].gt(0)
                | quality_report["negative_value_count"].gt(0)
                | quality_report["zero_variance_flag"]
            ]
            if flagged.empty:
                st.success("No subject/metric quality flags detected.")
            else:
                st.warning(f"{len(flagged)} subject/metric stream(s) have quality flags.")
            st.dataframe(quality_report, width="stretch", hide_index=True)

        if st.session_state.baseline_summary is not None and not st.session_state.baseline_summary.empty:
            with _working_status(
                "Drawing baseline quality heatmap",
                "The app is visualizing baseline coverage and validity across subjects and metrics.",
            ):
                baseline_fig = qc.plot_baseline_quality_heatmap(st.session_state.baseline_summary)
            st.plotly_chart(baseline_fig, width="stretch")

    metrics = sorted(display_df["metric_name"].dropna().unique().tolist()) if "metric_name" in display_df.columns else []
    subjects = sorted(display_df["subject_id"].dropna().unique().tolist()) if "subject_id" in display_df.columns else []
    groups = sorted(display_df["group_id"].dropna().unique().tolist()) if "group_id" in display_df.columns else []

    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        sel_metric = st.selectbox("Metric", metrics) if metrics else None
    with c2:
        sel_subjects = st.multiselect("Subjects (leave empty = all)", subjects, default=subjects[:6])
    with c3:
        sel_groups = st.multiselect("Groups", groups, default=groups)

    if not sel_metric:
        st.info("No metric selected.")
        return

    y_col_options = ["value", "baseline_corrected_value", "baseline_percent_change"]
    y_col_options = [c for c in y_col_options if c in display_df.columns]
    sel_y = st.selectbox("Y axis", y_col_options)

    # Plot 1: Raw timeseries
    st.subheader("Raw timeseries")
    x_col = "timestamp_local" if "timestamp_local" in display_df.columns else "timestamp_utc"
    with _working_status(
        "Drawing raw time-series plot",
        "The app is plotting subject-level raw traces and marking excluded periods when available.",
    ):
        fig1 = qc.plot_raw_timeseries(
            display_df, sel_metric,
            subjects=sel_subjects if sel_subjects else None,
            x_col=x_col,
            show_exclusions=True,
        )
    st.plotly_chart(fig1, width='stretch')

    # Plot 2: Aligned timeseries
    if "time_from_event_hours" in display_df.columns and not display_df["time_from_event_hours"].isna().all():
        st.subheader("Aligned timeseries")
        bsl_cfg = st.session_state.baseline_cfg
        with _working_status(
            "Drawing aligned time-series plot",
            "The app is plotting traces relative to the alignment anchor and shading the baseline window.",
        ):
            fig2 = qc.plot_aligned_timeseries(
                display_df, sel_metric,
                subjects=sel_subjects if sel_subjects else None,
                y_col=sel_y,
                show_exclusions=True,
                show_baseline_window=True,
                baseline_start_h=float(bsl_cfg.get("start_hours", -72)),
                baseline_end_h=float(bsl_cfg.get("end_hours", -24)),
            )
        st.plotly_chart(fig2, width='stretch')

        # Plot 3: Group mean
        st.subheader("Group mean ± SEM")
        with _working_status(
            "Drawing group mean plot",
            "The app is summarizing selected subjects into group mean traces with SEM bands.",
        ):
            fig3 = qc.plot_group_mean_timeseries(
                display_df, sel_metric,
                y_col=sel_y,
                groups=sel_groups if sel_groups else None,
            )
        st.plotly_chart(fig3, width='stretch')
    else:
        st.info("Run alignment (Tab 4) and the pipeline (Tab 5) to see aligned plots.")


# ---------------------------------------------------------------------------
# Tab 7: Export
# ---------------------------------------------------------------------------

def _tab_export() -> None:
    st.header("Export")
    render_contextual_help("export")

    long_df = st.session_state.long_df
    proc = st.session_state.processed_df

    with st.expander("Output-column glossary", expanded=False):
        render_output_glossary(proc.columns if proc is not None else None)

    if long_df is None:
        st.info("Load data first.")
        return

    # Collect all warnings accumulated so far
    warns = st.session_state.all_warnings

    # Build analysis config
    tz = st.session_state.timezone
    aln_cfg = st.session_state.alignment_cfg
    bsl_cfg = st.session_state.baseline_cfg

    uploaded_names = [name for name, _ in st.session_state.metric_files + st.session_state.event_files]

    analysis_config = exp_mod.build_analysis_config(
        uploaded_files=uploaded_names,
        study_metadata=st.session_state.study_meta,
        timezone=tz,
        light_on=st.session_state.light_on,
        light_off=st.session_state.light_off,
        alignment_cfg=aln_cfg,
        exclusion_rules=st.session_state.exclusion_rules,
        baseline_cfg=bsl_cfg,
        aggregation_bin=st.session_state.aggregation_bin,
    )
    quality_report = st.session_state.quality_report
    if quality_report is None and proc is not None:
        try:
            quality_report = quality.build_quality_report(proc)
        except Exception:
            quality_report = pd.DataFrame()
    analysis_tables_for_export = dict(st.session_state.analysis_tables)
    if quality_report is not None and not quality_report.empty:
        analysis_tables_for_export["quality_report.csv"] = quality_report

    manifest = provenance.build_provenance_manifest(
        input_files=st.session_state.metric_files + st.session_state.event_files,
        selected_config=analysis_config,
        tables={
            "processed_timeseries": proc,
            "baseline_summary": st.session_state.baseline_summary,
            "exclusion_log": st.session_state.exclusion_log,
            "event_table_clean": st.session_state.event_df,
            "subject_metadata": st.session_state.subject_meta,
            "group_metadata": st.session_state.group_meta,
            "quality_report": quality_report,
        },
        app_version=cfg.APP_VERSION,
    )

    # Generate reports
    proc_report = reporting.generate_processing_report(
        long_df=long_df,
        processed_df=proc,
        event_df=st.session_state.event_df,
        exclusion_log=st.session_state.exclusion_log,
        baseline_summary=st.session_state.baseline_summary,
        warnings=warns,
        analysis_config=analysis_config,
    )

    _subject_meta_for_validation = (
        st.session_state.subject_meta
        if st.session_state.subject_meta is not None
        else pd.DataFrame()
    )
    meta_errors, meta_warns = meta_mod.validate_subject_metadata(
        _subject_meta_for_validation, long_df
    )
    meta_report = reporting.generate_metadata_validation_report(
        meta_errors, meta_warns, st.session_state.subject_meta
    )

    # Preview processing report
    st.subheader("Provenance summary")
    st.write(
        f"{len(manifest['input_files'])} input file(s), "
        f"{sum(item['size_bytes'] for item in manifest['input_files']):,} input bytes, "
        f"{len(manifest['row_counts'])} tracked output table(s)."
    )
    with st.expander("Input file hashes", expanded=False):
        st.dataframe(pd.DataFrame(manifest["input_files"]), width="stretch", hide_index=True)

    st.subheader("Processing report preview")
    st.markdown(proc_report)

    # Build ZIP
    if st.button("Build export ZIP", type="primary"):
        with _working_status(
            "Building export ZIP",
            (
                "The app is collecting processed tables, metadata, reports, analysis outputs, "
                "figures, and provenance hashes into one downloadable archive."
            ),
        ):
            estimated_bytes = int(proc.memory_usage(deep=True).sum()) if proc is not None else 0
            export_kwargs = dict(
                processed_df=proc,
                baseline_summary=st.session_state.baseline_summary,
                exclusion_log=st.session_state.exclusion_log,
                event_table=st.session_state.event_df,
                subject_metadata=st.session_state.subject_meta,
                group_metadata=st.session_state.group_meta,
                study_metadata=st.session_state.study_meta,
                analysis_config=analysis_config,
                processing_report=proc_report,
                metadata_validation_report=meta_report,
                treatment_schedule=st.session_state.treatment_schedule,
                facility_events=st.session_state.facility_events,
                analysis_tables=analysis_tables_for_export,
                figures=st.session_state.analysis_figures,
                manifest=manifest,
            )
            if estimated_bytes > 500_000_000:
                tmp_dir = Path(tempfile.mkdtemp(prefix="dvc_export_"))
                zip_path = exp_mod.create_export_zip_file(tmp_dir / "DVC_Workbench_Export.zip", **export_kwargs)
                zip_data = open(zip_path, "rb")
            else:
                zip_data = exp_mod.create_export_zip(**export_kwargs)

        st.download_button(
            label="Download DVC_Workbench_Export.zip",
            data=zip_data,
            file_name="DVC_Workbench_Export.zip",
            mime="application/zip",
        )
        st.session_state.export_zip_ready = True
        st.success("ZIP ready.")

    # Show list of files in ZIP
    st.subheader("ZIP contents")
    for fname in [
        "processed_timeseries.csv",
        "baseline_summary.csv",
        "exclusion_log.csv",
        "event_table_clean.csv",
        "subject_metadata.csv",
        "group_metadata.csv",
        "treatment_schedule.csv",
        "facility_events.csv",
        "daily_means.csv",
        "circadian_summary.csv",
        "circadian_profile.csv",
        "light_dark_summary.csv",
        "quality_report.csv",
        "auc_summary.csv",
        "stats_summary.csv",
        "figures/",
        "study_metadata.yaml",
        "event_metadata.csv",
        "analysis_config.yaml",
        "manifest.yaml",
        "processing_report.md",
        "metadata_validation_report.md",
    ]:
        included = (
            (fname == "processed_timeseries.csv" and proc is not None)
            or (fname == "baseline_summary.csv" and st.session_state.baseline_summary is not None)
            or (fname == "exclusion_log.csv" and st.session_state.exclusion_log is not None)
            or (fname == "event_table_clean.csv" and st.session_state.event_df is not None)
            or (fname == "subject_metadata.csv" and st.session_state.subject_meta is not None)
            or (fname == "group_metadata.csv" and st.session_state.group_meta is not None)
            or (
                fname == "treatment_schedule.csv"
                and st.session_state.treatment_schedule is not None
                and not st.session_state.treatment_schedule.empty
            )
            or (
                fname == "facility_events.csv"
                and st.session_state.facility_events is not None
                and not st.session_state.facility_events.empty
            )
            or fname in analysis_tables_for_export
            or (fname == "figures/" and bool(st.session_state.analysis_figures))
            or fname in ("study_metadata.yaml", "analysis_config.yaml", "processing_report.md",
                         "metadata_validation_report.md", "event_metadata.csv", "manifest.yaml")
        )
        icon = "✓" if included else "–"
        st.write(f"{icon} `{fname}`")


# ---------------------------------------------------------------------------
# Tab 8: Analysis
# ---------------------------------------------------------------------------

def _tab_analysis() -> None:
    st.header("Exploratory Analysis")
    render_contextual_help("analysis")
    st.info(
        "These results are for orientation only. Consult a statistician for confirmatory analysis."
    )

    proc = st.session_state.processed_df
    if proc is None or proc.empty:
        st.info("Run the preprocessing pipeline before using the analysis page.")
        return

    metrics = sorted(proc["metric_name"].dropna().unique().tolist()) if "metric_name" in proc.columns else []
    if not metrics:
        st.info("No metric column available.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        sel_metric = st.selectbox("Metric", metrics, key="analysis_metric")
    with c2:
        value_options = [
            c for c in ["baseline_percent_change", "baseline_corrected_value", "value"] if c in proc.columns
        ]
        sel_value = st.selectbox("Value", value_options, key="analysis_value")
    with c3:
        bin_size = st.selectbox("Summary bin", ["1D", "2D", "1W"], key="analysis_bin")

    metric_df = proc[proc["metric_name"] == sel_metric].copy()
    subjects = sorted(metric_df["subject_id"].dropna().astype(str).unique().tolist()) if "subject_id" in metric_df.columns else []

    raw_c1, raw_c2 = st.columns([3, 2])
    with raw_c1:
        raw_subjects = st.multiselect(
            "Raw trace subjects",
            subjects,
            default=subjects[:6],
            key="analysis_raw_subjects",
        )
    with raw_c2:
        show_events = st.checkbox("Show events on raw trace", value=True, key="analysis_show_events")

    max_available_days = 1
    if "timestamp_local" in metric_df.columns:
        ts_for_days = pd.to_datetime(metric_df["timestamp_local"], errors="coerce").dropna()
        if not ts_for_days.empty:
            max_available_days = max(1, int((ts_for_days.dt.normalize().max() - ts_for_days.dt.normalize().min()).days) + 1)

    circ_c1, circ_c2, circ_c3 = st.columns(3)
    with circ_c1:
        circ_days = st.number_input(
            "Circadian days to include",
            min_value=1,
            max_value=max_available_days,
            value=min(1, max_available_days),
            step=1,
            key="analysis_circ_days",
        )
    with circ_c2:
        circ_bin = st.selectbox("Circadian ZT bin", [0.5, 1.0, 2.0, 3.0], index=1, key="analysis_circ_bin")
    with circ_c3:
        circ_norm = st.selectbox(
            "Circadian normalization",
            ["Raw values", "Normalize to 24 h max", "Normalize to max in 3 h after dark onset"],
            key="analysis_circ_norm",
        )

    auc_c1, auc_c2 = st.columns(2)
    with auc_c1:
        auc_start = st.number_input("AUC start (h)", value=0.0, step=24.0)
    with auc_c2:
        auc_end = st.number_input("AUC end (h)", value=168.0, step=24.0)

    st.subheader("Raw trace with events")
    x_col = "timestamp_local" if "timestamp_local" in metric_df.columns else "timestamp_utc"
    with _working_status(
        "Drawing raw analysis trace",
        "The app is plotting raw subject traces and overlaying event timestamps for visual inspection.",
    ):
        raw_fig = qc.plot_raw_timeseries(
            metric_df,
            sel_metric,
            subjects=raw_subjects if raw_subjects else None,
            x_col=x_col,
            show_exclusions=True,
        )
        event_df = st.session_state.event_df
        if show_events and event_df is not None and not event_df.empty:
            event_x_col = x_col if x_col in event_df.columns else "timestamp_utc"
            raw_values = pd.to_numeric(metric_df[sel_value if sel_value in metric_df.columns else "value"], errors="coerce")
            if raw_values.notna().any():
                _add_event_overlay(raw_fig, event_df, x_col=event_x_col, y_value=float(raw_values.max()))
    st.plotly_chart(raw_fig, width="stretch")

    with _working_status(
        "Computing exploratory analysis",
        (
            "The app is calculating circadian rhythm summaries, light/dark summaries, "
            "time-bin means, per-animal AUC, quick exploratory statistics, and the "
            "circadian profile table."
        ),
    ):
        circadian, circ_warns = analysis.summarize_circadian_cosinor(metric_df, value_col="value")
        phase, phase_warns = analysis.summarize_light_dark(metric_df, value_col="value")
        time_bins, bin_warns = analysis.summarize_time_bins(
            metric_df,
            bin_size=bin_size,
            relative_to="alignment",
            value_col=sel_value,
        )
        auc, auc_warns = analysis.compute_auc_per_animal(
            metric_df,
            start=auc_start,
            end=auc_end,
            value_col=sel_value,
        )
        stats, stats_warns = (
            analysis.quick_exploratory_stats(auc, value_col="auc") if not auc.empty else (pd.DataFrame(), [])
        )
        analysis_warns = circ_warns + phase_warns + bin_warns + auc_warns + stats_warns
        circadian_profile = _circadian_profile(
            metric_df,
            value_col=sel_value,
            max_days=int(circ_days),
            zt_bin_hours=float(circ_bin),
            normalize_mode=circ_norm,
        )

    st.session_state.analysis_tables = {
        "daily_means.csv": time_bins,
        "circadian_summary.csv": circadian,
        "circadian_profile.csv": circadian_profile,
        "light_dark_summary.csv": phase,
        "auc_summary.csv": auc,
        "stats_summary.csv": stats,
    }
    st.session_state.analysis_warning = "\n".join(analysis_warns)

    if analysis_warns:
        for warning in analysis_warns:
            st.warning(warning)

    st.subheader("Analysis summary")
    st.markdown(
        "\n".join(
            [
                f"- Metric: `{sel_metric}`.",
                f"- Value column: `{sel_value}`.",
                f"- Raw trace: {len(raw_subjects) if raw_subjects else 'all'} subject(s) shown; events {'shown' if show_events else 'hidden'}.",
                f"- Circadian rhythm: first {int(circ_days)} day(s), {float(circ_bin):g} h ZT bins, `{circ_norm}`.",
                f"- Dark phase starts at ZT{_dark_onset_zt():g}; the circadian plot shades ZT{_dark_onset_zt():g}-24.",
                f"- Time-bin summary: `{bin_size}` bins.",
                f"- AUC window: {auc_start:g} h to {auc_end:g} h.",
            ]
        )
    )

    st.subheader("Circadian rhythm profile")
    with _working_status(
        "Drawing circadian rhythm profile",
        "The app is plotting group mean circadian traces with SEM and shaded dark phase.",
    ):
        circ_fig = _plot_circadian_profile(circadian_profile, value_label=sel_value)
    st.plotly_chart(circ_fig, width="stretch")

    st.subheader("Baseline-corrected group plot")
    with _working_status(
        "Drawing baseline-corrected group plot",
        "The app is plotting group mean traces with SEM for the selected value column.",
    ):
        fig = qc.plot_group_mean_timeseries(metric_df, sel_metric, y_col=sel_value)
    st.plotly_chart(fig, width="stretch")
    st.session_state.analysis_figures = {
        "analysis_raw_trace.png": raw_fig,
        "analysis_circadian_profile.png": circ_fig,
        "analysis_group_mean.png": fig,
    }

    st.subheader("Time-bin group summary")
    st.dataframe(time_bins, width="stretch", hide_index=True)

    st.subheader("AUC per animal")
    st.dataframe(auc, width="stretch", hide_index=True)

    st.subheader("Circadian cosinor summary")
    st.dataframe(circadian, width="stretch", hide_index=True)

    with st.expander("Light/dark summary", expanded=False):
        st.dataframe(phase, width="stretch", hide_index=True)

    st.subheader("Exploratory statistics")
    if stats.empty:
        st.caption("Statistics table is empty. AUC needs at least one plottable animal per group.")
    else:
        st.dataframe(stats, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title=cfg.APP_NAME,
        page_icon="DVC",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_state()

    selected_step = _render_sidebar()

    st.title(cfg.APP_NAME)
    st.caption(f"v{cfg.APP_VERSION}  |  Digital Ventilated Cage behavioral data preprocessing")
    _scroll_to_top_once()

    if selected_step.startswith("1."):
        _tab_import()
    elif selected_step.startswith("2."):
        _tab_validate()
    elif selected_step.startswith("3."):
        _tab_metadata()
    elif selected_step.startswith("4."):
        _tab_events()
    elif selected_step.startswith("5."):
        _tab_baseline()
    elif selected_step.startswith("6."):
        _tab_qc()
    elif selected_step.startswith("7."):
        _tab_export()
    elif selected_step.startswith("8."):
        _tab_analysis()

    _render_next_step_button(selected_step)


if __name__ == "__main__":
    main()
