"""Workflow scaffolding components for the Streamlit app."""

from __future__ import annotations

from html import escape
from textwrap import dedent
from typing import Iterable

import pandas as pd
import streamlit as st


WORKFLOW_HELP: dict[str, dict[str, object]] = {
    "import": {
        "title": "Import",
        "body": (
            "Load DVC metric CSV files first. Event CSV files are optional, but they enable "
            "event-based alignment and exclusion windows later in the workflow."
        ),
        "checks": [
            "Metric files parse into long-format rows.",
            "Event files expose event types, subjects, groups, and timestamps when available.",
            "Detected subjects and groups seed the metadata editors.",
        ],
    },
    "validate": {
        "title": "Validate",
        "body": (
            "Confirm that the parser detected the expected subjects, groups, metrics, native "
            "bin size, timestamp range, and event counts before editing metadata."
        ),
        "checks": [
            "Unexpected missing values or parse warnings should be reviewed.",
            "The first 200 long-format rows should look like the source data.",
        ],
    },
    "metadata": {
        "title": "Metadata & Study Design",
        "body": (
            "Complete study, subject, and group metadata that will be merged onto the processed "
            "time series and included in exports."
        ),
        "checks": [
            "Required subject columns identify each animal and key experimental grouping fields.",
            "Group labels are used in plots and exported summaries.",
            "Timezone and light-cycle fields feed the processing configuration.",
        ],
    },
    "events": {
        "title": "Events, Alignment & Exclusions",
        "body": (
            "Alignment creates a common time-zero anchor so animals can be compared relative "
            "to the same experimental moment. After alignment, each row gets "
            "`time_from_event_hours`: negative values are before the anchor, 0 is the anchor, "
            "and positive values are after. Event-based alignment uses timestamps from the "
            "event table; manual alignment uses one global timestamp for all subjects. "
            "Exclusion rules then flag periods around events that should be ignored or reviewed."
        ),
        "checks": [
            "Use Event-based when the event file contains the biological or experimental anchor, such as treatment, surgery, REMOVED, or INSERTED.",
            "Use Manual global timestamp when all subjects share one anchor or when no event file is available.",
            "Use subject scope when each subject/cage has its own event timestamp; use group scope when one event applies to the group.",
            "Use a fallback timestamp when some subjects lack the selected event; it is only used for missing anchors.",
            "Excluded rows remain in the output with exclusion flags.",
        ],
    },
    "baseline": {
        "title": "Baseline, Aggregation & Pipeline",
        "body": (
            "Baseline defines a pre-event reference window for each subject and metric, such "
            "as -72 h to -24 h before alignment. The pipeline computes a baseline value from "
            "that window, then can export raw-minus-baseline and percent-change-from-baseline "
            "columns. Aggregation optionally combines smaller time bins into larger bins, "
            "for example 1-minute data into 1-hour data, to reduce noise and file size. "
            "Baseline requires alignment; aggregation does not."
        ),
        "checks": [
            "Baseline windows require alignment because the window is defined relative to time zero.",
            "If alignment is disabled, baseline controls are hidden and baseline columns stay empty.",
            "Choose a baseline window before the intervention or event, where behavior should represent the subject's normal pre-event level.",
            "Minimum coverage controls how much data must exist in the window before a baseline is considered valid.",
            "Flagged or excluded rows can be left out of baseline statistics.",
            "Aggregation smooths plots and reduces export size, but larger bins remove short-term detail.",
        ],
    },
    "qc": {
        "title": "QC Plots",
        "body": (
            "Inspect raw, aligned, and group-level traces. After the pipeline runs, plots can "
            "include exclusions, baseline windows, and baseline-corrected values."
        ),
        "checks": [
            "Compare selected subjects before reviewing group means.",
            "Use the Y-axis selector to switch between raw and baseline-normalized outputs.",
        ],
    },
    "export": {
        "title": "Export",
        "body": (
            "Review the processing report, build the ZIP bundle, and confirm which tables and "
            "metadata files are included."
        ),
        "checks": [
            "Run the pipeline first to include processed_timeseries.csv.",
            "Open the output-column glossary when reviewing processed fields.",
        ],
    },
    "analysis": {
        "title": "Exploratory Analysis",
        "body": (
            "Review exploratory summaries after preprocessing. These tables and figures are "
            "orientation outputs, not confirmatory statistics."
        ),
        "checks": [
            "Run the preprocessing pipeline before opening analysis outputs.",
            "Choose the metric and value column that match the QC question.",
            "Export can include generated analysis tables and figures.",
        ],
    },
}


OUTPUT_COLUMN_GLOSSARY = [
    ("source_file", "Input metric file that produced the row.", "Import"),
    ("metric_name", "DVC metric represented by the row.", "Import"),
    ("group_id", "Detected DVC group identifier from the source file.", "Import"),
    ("subject_id", "Detected animal, cage, or subject identifier.", "Import"),
    ("timestamp", "Original timestamp value as parsed from the input.", "Import"),
    ("timestamp_utc", "Timestamp normalized to UTC.", "Import"),
    ("timestamp_local", "Timestamp converted to the configured study timezone.", "Light/dark"),
    ("relative_time_seconds", "Source-relative elapsed time in seconds.", "Import"),
    ("value", "Observed metric value for the subject and time bin.", "Import"),
    ("group_avg", "Group average reported by DVC, when present in the source.", "Import"),
    ("group_sem", "Group standard error reported by DVC, when present.", "Import"),
    ("samples", "Sample count reported for the source bin.", "Import"),
    ("native_bin_seconds", "Native time-bin size detected from the source.", "Import"),
    ("light_phase", "Light or dark phase from local timestamp and configured cycle.", "Light/dark"),
    ("is_light", "Boolean marker for rows inside the light phase.", "Light/dark"),
    ("is_dark", "Boolean marker for rows inside the dark phase.", "Light/dark"),
    ("is_excluded", "Boolean marker for rows excluded by event windows.", "Exclusions"),
    ("exclusion_reason", "Event/window rule that excluded the row.", "Exclusions"),
    ("flag_reason", "Event/window rule that flagged the row without necessarily excluding it.", "Exclusions"),
    ("alignment_event_type", "Event type used as the relative-time anchor.", "Alignment"),
    ("alignment_timestamp", "Timestamp of the alignment anchor used for the row.", "Alignment"),
    ("time_from_event_seconds", "Seconds from the configured alignment anchor.", "Alignment"),
    ("time_from_event_hours", "Hours from the configured alignment anchor.", "Alignment"),
    ("experimental_day", "Integer day relative to the configured alignment label.", "Alignment"),
    ("baseline_value", "Subject/metric baseline value computed from the baseline window.", "Baseline"),
    ("baseline_valid", "Whether the baseline met the configured coverage requirement.", "Baseline"),
    ("baseline_coverage", "Fraction of expected baseline bins present for subject/metric.", "Baseline"),
    ("baseline_corrected_value", "Raw value minus baseline value.", "Baseline"),
    ("baseline_percent_change", "Percent change from baseline value.", "Baseline"),
    ("baseline_percent_change_unstable", "True when the baseline magnitude is below the near-zero floor, so percent change is left blank to avoid blow-up.", "Baseline"),
    ("metadata_complete", "Whether key subject metadata fields are sufficiently complete.", "Metadata"),
    ("metadata_warning", "Metadata completeness warning for the subject row.", "Metadata"),
    ("metadata_quality_score", "Fraction of key subject metadata fields filled.", "Metadata"),
    ("group_label", "Human-readable group label from group metadata.", "Metadata"),
]


def render_contextual_help(step_id: str) -> None:
    """Render a compact contextual help panel for a workflow step."""
    help_cfg = WORKFLOW_HELP[step_id]
    with st.expander(f"Workflow help: {help_cfg['title']}", expanded=False):
        st.write(help_cfg["body"])
        checks = help_cfg.get("checks", [])
        if checks:
            st.markdown("\n".join(f"- {item}" for item in checks))


def render_progress_stepper(steps: Iterable[dict[str, str]]) -> None:
    """Render a vertical progress stepper with status badges."""
    st.markdown(
        dedent(
            """
        <style>
        .dvc-stepper { display: grid; gap: 0.45rem; margin-top: 0.35rem; }
        .dvc-step {
            border-left: 3px solid #d7dde8;
            padding: 0.2rem 0 0.45rem 0.7rem;
        }
        .dvc-step.done { border-left-color: #237a57; }
        .dvc-step.ready { border-left-color: #b7791f; }
        .dvc-step.locked { border-left-color: #8a94a6; opacity: 0.78; }
        .dvc-step-title {
            display: flex;
            justify-content: space-between;
            gap: 0.45rem;
            align-items: baseline;
            font-weight: 650;
            line-height: 1.25;
        }
        .dvc-step-detail {
            color: #687385;
            font-size: 0.82rem;
            line-height: 1.25;
            margin-top: 0.15rem;
        }
        .dvc-badge {
            border: 1px solid currentColor;
            border-radius: 8px;
            font-size: 0.68rem;
            font-weight: 700;
            padding: 0.03rem 0.32rem;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .dvc-badge.done { color: #237a57; }
        .dvc-badge.ready { color: #9c5f10; }
        .dvc-badge.locked { color: #687385; }
        </style>
        """
        ).strip(),
        unsafe_allow_html=True,
    )

    items = []
    for step in steps:
        status = step["status"]
        label = escape(step["label"])
        detail = escape(step.get("detail", ""))
        badge = escape(status)
        items.append(
            dedent(
                f"""
            <div class="dvc-step {status}">
                <div class="dvc-step-title">
                    <span>{label}</span>
                    <span class="dvc-badge {status}">{badge}</span>
                </div>
                <div class="dvc-step-detail">{detail}</div>
            </div>
            """
            ).strip()
        )

    st.markdown(f"<div class='dvc-stepper'>{''.join(items)}</div>", unsafe_allow_html=True)


def render_data_flow_diagram() -> None:
    """Render a compact data-flow diagram for the preprocessing workflow."""
    st.graphviz_chart(
        """
        digraph dvc_flow {
            graph [rankdir=LR, bgcolor="transparent", margin=0.04, nodesep=0.38, ranksep=0.52]
            node [shape=box, style="rounded,filled", fontname="Arial", fontsize=11, margin="0.10,0.06", color="#c9d3df", fillcolor="#f7fafc"]
            edge [color="#7a8798", arrowsize=0.7]

            metrics [label="Metric CSVs"]
            events [label="Event CSVs"]
            import [label="Parse + combine"]
            metadata [label="Metadata merge"]
            lightdark [label="Timezone + light/dark"]
            exclusions [label="Exclusions + flags"]
            alignment [label="Event alignment"]
            baseline [label="Baseline correction"]
            aggregation [label="Optional aggregation"]
            qc [label="QC plots"]
            export [label="Export ZIP"]

            metrics -> import
            events -> exclusions
            events -> alignment
            import -> metadata -> lightdark -> exclusions -> alignment -> baseline -> aggregation
            aggregation -> qc
            aggregation -> export
            metadata -> export
        }
        """
    )


def render_output_glossary(active_columns: Iterable[str] | None = None) -> None:
    """Render the output-column glossary with optional present/absent markers."""
    active = set(active_columns) if active_columns is not None else set()
    rows = []
    for name, description, stage in OUTPUT_COLUMN_GLOSSARY:
        rows.append(
            {
                "column": name,
                "stage": stage,
                "present": "Yes" if name in active else ("Pending" if active else "Depends on run"),
                "meaning": description,
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "column": st.column_config.TextColumn("Output column", help="Column name in processed exports."),
            "stage": st.column_config.TextColumn("Stage", help="Workflow stage that creates or uses the field."),
            "present": st.column_config.TextColumn("Current output", help="Whether the current processed table includes the field."),
            "meaning": st.column_config.TextColumn("Meaning", help="Plain-language definition."),
        },
    )
