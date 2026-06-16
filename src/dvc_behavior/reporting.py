"""Generate human-readable processing reports in Markdown."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


def generate_processing_report(
    long_df: pd.DataFrame,
    processed_df: pd.DataFrame | None,
    event_df: pd.DataFrame | None,
    exclusion_log: pd.DataFrame | None,
    baseline_summary: pd.DataFrame | None,
    warnings: list[str],
    analysis_config: dict[str, Any],
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = [
        "# DVC Behavioral Preprocessing – Processing Report",
        "",
        f"**Generated:** {ts}",
        f"**App version:** {analysis_config.get('app_version', 'unknown')}",
        "",
        "---",
        "",
        "## 1. Input files",
        "",
    ]

    for f in analysis_config.get("uploaded_files", []):
        lines.append(f"- `{f}`")
    lines.append("")

    # Data overview
    if not long_df.empty:
        groups = long_df["group_id"].unique().tolist() if "group_id" in long_df.columns else []
        subjects = long_df["subject_id"].unique().tolist() if "subject_id" in long_df.columns else []
        metrics = long_df["metric_name"].unique().tolist() if "metric_name" in long_df.columns else []
        ts_range = ""
        if "timestamp_utc" in long_df.columns:
            t = long_df["timestamp_utc"].dropna()
            if not t.empty:
                ts_range = f"{t.min()} → {t.max()}"

        native_bin = ""
        if "native_bin_seconds" in long_df.columns:
            nb = long_df["native_bin_seconds"].dropna().mode()
            if not nb.empty:
                native_bin = f"{nb.iloc[0]:.0f} s"

        lines += [
            "## 2. Detected structure",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| Groups | {', '.join(str(g) for g in groups)} |",
            f"| Subjects | {len(subjects)} unique |",
            f"| Metrics | {', '.join(str(m) for m in metrics)} |",
            f"| Total rows (long format) | {len(long_df):,} |",
            f"| Timestamp range (UTC) | {ts_range} |",
            f"| Native bin size | {native_bin} |",
            "",
        ]
    else:
        lines += ["## 2. Detected structure", "", "_No data loaded._", ""]

    # Event summary
    lines.append("## 3. Events")
    lines.append("")
    if event_df is not None and not event_df.empty:
        etype_counts = event_df["event_type"].value_counts()
        for etype, cnt in etype_counts.items():
            lines.append(f"- {etype}: {cnt} events")
    else:
        lines.append("- No event file loaded.")
    lines.append("")

    # Exclusion summary
    lines.append("## 4. Exclusions applied")
    lines.append("")
    if processed_df is not None and "is_excluded" in processed_df.columns:
        n_excl = int(processed_df["is_excluded"].sum())
        n_total = len(processed_df)
        pct = 100.0 * n_excl / n_total if n_total > 0 else 0.0
        lines.append(f"- Excluded rows: {n_excl:,} / {n_total:,} ({pct:.1f} %)")
    else:
        lines.append("- Exclusion not applied or no processed data.")
    lines.append("")

    if exclusion_log is not None and not exclusion_log.empty:
        lines.append("### Exclusion log summary")
        lines.append("")
        lines.append("| subject_id | event_type | n_rows_excluded |")
        lines.append("|-----------|------------|-----------------|")
        for _, row in exclusion_log.iterrows():
            lines.append(
                f"| {row.get('subject_id','')} | {row.get('event_type','')} "
                f"| {row.get('n_rows_excluded','')} |"
            )
        lines.append("")

    # Baseline summary
    lines.append("## 5. Baseline validity")
    lines.append("")
    if baseline_summary is not None and not baseline_summary.empty:
        if "baseline_valid" in baseline_summary.columns:
            n_valid = int(baseline_summary["baseline_valid"].sum())
            n_total = len(baseline_summary)
            lines.append(f"- Valid baselines: {n_valid} / {n_total}")
            invalid = baseline_summary[~baseline_summary["baseline_valid"]]
            if not invalid.empty:
                for _, row in invalid.iterrows():
                    lines.append(
                        f"  - Subject '{row.get('subject_id','')}', "
                        f"metric '{row.get('metric_name','')}': "
                        f"coverage={row.get('baseline_coverage','?')}"
                    )
    else:
        lines.append("- Baseline not computed.")
    lines.append("")

    # Warnings
    lines.append("## 6. Warnings")
    lines.append("")
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- No warnings.")
    lines.append("")

    # Config summary
    lines.append("## 7. Analysis configuration")
    lines.append("")
    ld = analysis_config.get("light_dark_cycle", {})
    aln = analysis_config.get("alignment", {})
    bsl = analysis_config.get("baseline", {})
    lines += [
        f"- Timezone: `{analysis_config.get('timezone', 'unknown')}`",
        f"- Light cycle: {ld.get('light_on','?')} → {ld.get('light_off','?')}",
        f"- Alignment event: `{aln.get('event_type','none')}`  scope=`{aln.get('scope','?')}`",
        f"- Baseline window: {bsl.get('start_hours','?')}h → {bsl.get('end_hours','?')}h",
        f"- Aggregation bin: {analysis_config.get('aggregation_bin_seconds', 'native')} s",
        "",
    ]

    return "\n".join(lines)


def generate_metadata_validation_report(
    errors: list[str],
    warnings: list[str],
    subject_meta: pd.DataFrame | None,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Metadata Validation Report",
        "",
        f"**Generated:** {ts}",
        "",
    ]

    if errors:
        lines += ["## Errors (must fix)", ""]
        for e in errors:
            lines.append(f"- **{e}**")
        lines.append("")

    if warnings:
        lines += ["## Warnings", ""]
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    if not errors and not warnings:
        lines += ["## Status", "", "All metadata checks passed.", ""]

    if subject_meta is not None and not subject_meta.empty:
        lines += [
            "## Subject metadata quality",
            "",
            "| subject_id | metadata_quality_score | metadata_complete |",
            "|-----------|----------------------|-------------------|",
        ]
        for col in ("metadata_quality_score", "metadata_complete"):
            if col not in subject_meta.columns:
                subject_meta = subject_meta.copy()
                subject_meta[col] = "N/A"
        for _, row in subject_meta.iterrows():
            lines.append(
                f"| {row.get('subject_id','')} "
                f"| {row.get('metadata_quality_score','?')} "
                f"| {row.get('metadata_complete','?')} |"
            )
        lines.append("")

    return "\n".join(lines)
