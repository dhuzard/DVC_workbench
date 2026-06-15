"""
ZIP export: collect all processed artefacts into a downloadable archive.
"""

from __future__ import annotations

import io
from pathlib import Path
import zipfile
from datetime import date
from datetime import datetime
from datetime import timezone as _dt_timezone
from typing import Any

import pandas as pd
import yaml


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _text_to_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def build_analysis_config(
    uploaded_files: list[str],
    study_metadata: dict[str, Any],
    timezone: str,
    light_on: str,
    light_off: str,
    alignment_cfg: dict[str, Any],
    exclusion_rules: dict[str, Any],
    baseline_cfg: dict[str, Any],
    aggregation_bin: int | None,
    app_version: str = "0.1.0",
) -> dict[str, Any]:
    return {
        "app_version": app_version,
        "processing_timestamp": datetime.now(_dt_timezone.utc).isoformat(),
        "uploaded_files": uploaded_files,
        "study_metadata": study_metadata,
        "timezone": timezone,
        "light_dark_cycle": {"light_on": light_on, "light_off": light_off},
        "alignment": alignment_cfg,
        "exclusion_rules": exclusion_rules,
        "baseline": baseline_cfg,
        "aggregation_bin_seconds": aggregation_bin,
    }


def create_export_zip(
    processed_df: pd.DataFrame | None,
    baseline_summary: pd.DataFrame | None,
    exclusion_log: pd.DataFrame | None,
    event_table: pd.DataFrame | None,
    subject_metadata: pd.DataFrame | None,
    group_metadata: pd.DataFrame | None,
    study_metadata: dict[str, Any] | None,
    analysis_config: dict[str, Any] | None,
    processing_report: str | None,
    metadata_validation_report: str | None,
    treatment_schedule: pd.DataFrame | None = None,
    facility_events: pd.DataFrame | None = None,
    analysis_tables: dict[str, pd.DataFrame] | None = None,
    figures: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    text_artifacts: dict[str, str] | None = None,
) -> bytes:
    """
    Assemble all artefacts into an in-memory ZIP and return the bytes.

    ``text_artifacts`` maps an in-archive path (e.g. ``insights/narrative.md``)
    to its text content and is written verbatim. It is used for the optional
    LLM-insights bundle (narrative + payload JSON).
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Timeseries
        if processed_df is not None and not processed_df.empty:
            zf.writestr("processed_timeseries.csv", _df_to_csv_bytes(processed_df))

        # Baseline
        if baseline_summary is not None and not baseline_summary.empty:
            zf.writestr("baseline_summary.csv", _df_to_csv_bytes(baseline_summary))

        # Exclusion log
        if exclusion_log is not None and not exclusion_log.empty:
            zf.writestr("exclusion_log.csv", _df_to_csv_bytes(exclusion_log))

        # Event table
        if event_table is not None and not event_table.empty:
            zf.writestr("event_table_clean.csv", _df_to_csv_bytes(event_table))

        # Subject metadata
        if subject_metadata is not None and not subject_metadata.empty:
            zf.writestr("subject_metadata.csv", _df_to_csv_bytes(subject_metadata))

        # Group metadata
        if group_metadata is not None and not group_metadata.empty:
            zf.writestr("group_metadata.csv", _df_to_csv_bytes(group_metadata))

        if treatment_schedule is not None and not treatment_schedule.empty:
            zf.writestr("treatment_schedule.csv", _df_to_csv_bytes(treatment_schedule))

        if facility_events is not None and not facility_events.empty:
            zf.writestr("facility_events.csv", _df_to_csv_bytes(facility_events))

        if analysis_tables:
            for name, table in analysis_tables.items():
                if table is not None and not table.empty:
                    safe_name = name if name.endswith(".csv") else f"{name}.csv"
                    zf.writestr(safe_name, _df_to_csv_bytes(table))

        if figures:
            for name, fig in figures.items():
                try:
                    image_bytes = fig.to_image(format="png")
                except Exception:
                    continue
                safe_name = name if name.endswith(".png") else f"{name}.png"
                zf.writestr(f"figures/{safe_name}", image_bytes)

        # Study metadata → YAML
        if study_metadata:
            zf.writestr(
                "study_metadata.yaml",
                _text_to_bytes(yaml.dump(study_metadata, allow_unicode=True, sort_keys=False)),
            )

        # Analysis config → YAML
        if analysis_config:
            # Convert any non-serialisable items
            safe_cfg = _make_yaml_safe(analysis_config)
            zf.writestr(
                "analysis_config.yaml",
                _text_to_bytes(yaml.dump(safe_cfg, allow_unicode=True, sort_keys=False)),
            )

        # Processing report
        if processing_report:
            zf.writestr("processing_report.md", _text_to_bytes(processing_report))

        # Metadata validation report
        if metadata_validation_report:
            zf.writestr(
                "metadata_validation_report.md", _text_to_bytes(metadata_validation_report)
            )

        if manifest is not None:
            safe_manifest = _make_yaml_safe(manifest)
            zf.writestr(
                "manifest.yaml",
                _text_to_bytes(yaml.dump(safe_manifest, allow_unicode=True, sort_keys=False)),
            )

        if text_artifacts:
            for name, text in text_artifacts.items():
                if text:
                    zf.writestr(name, _text_to_bytes(text))

        # Placeholder event metadata (user-defined events)
        zf.writestr(
            "event_metadata.csv",
            _text_to_bytes(
                "event_id,event_type,event_label,subject_id,group_id,timestamp,"
                "timezone,event_scope,used_for_alignment,used_for_exclusion,description,notes\n"
            ),
        )

    buf.seek(0)
    return buf.read()


def create_export_zip_file(
    output_path: str | Path,
    processed_df: pd.DataFrame | None,
    baseline_summary: pd.DataFrame | None,
    exclusion_log: pd.DataFrame | None,
    event_table: pd.DataFrame | None,
    subject_metadata: pd.DataFrame | None,
    group_metadata: pd.DataFrame | None,
    study_metadata: dict[str, Any] | None,
    analysis_config: dict[str, Any] | None,
    processing_report: str | None,
    metadata_validation_report: str | None,
    treatment_schedule: pd.DataFrame | None = None,
    facility_events: pd.DataFrame | None = None,
    analysis_tables: dict[str, pd.DataFrame] | None = None,
    figures: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    text_artifacts: dict[str, str] | None = None,
) -> Path:
    """Build the export ZIP directly on disk and return its path.

    This avoids holding the final ZIP archive in memory for large datasets.
    Individual CSV members are still serialized one at a time.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if processed_df is not None and not processed_df.empty:
            zf.writestr("processed_timeseries.csv", _df_to_csv_bytes(processed_df))
        if baseline_summary is not None and not baseline_summary.empty:
            zf.writestr("baseline_summary.csv", _df_to_csv_bytes(baseline_summary))
        if exclusion_log is not None and not exclusion_log.empty:
            zf.writestr("exclusion_log.csv", _df_to_csv_bytes(exclusion_log))
        if event_table is not None and not event_table.empty:
            zf.writestr("event_table_clean.csv", _df_to_csv_bytes(event_table))
        if subject_metadata is not None and not subject_metadata.empty:
            zf.writestr("subject_metadata.csv", _df_to_csv_bytes(subject_metadata))
        if group_metadata is not None and not group_metadata.empty:
            zf.writestr("group_metadata.csv", _df_to_csv_bytes(group_metadata))
        if treatment_schedule is not None and not treatment_schedule.empty:
            zf.writestr("treatment_schedule.csv", _df_to_csv_bytes(treatment_schedule))
        if facility_events is not None and not facility_events.empty:
            zf.writestr("facility_events.csv", _df_to_csv_bytes(facility_events))
        if analysis_tables:
            for name, table in analysis_tables.items():
                if table is not None and not table.empty:
                    safe_name = name if name.endswith(".csv") else f"{name}.csv"
                    zf.writestr(safe_name, _df_to_csv_bytes(table))
        if figures:
            for name, fig in figures.items():
                try:
                    image_bytes = fig.to_image(format="png")
                except Exception:
                    continue
                safe_name = name if name.endswith(".png") else f"{name}.png"
                zf.writestr(f"figures/{safe_name}", image_bytes)
        if study_metadata:
            zf.writestr(
                "study_metadata.yaml",
                _text_to_bytes(yaml.dump(study_metadata, allow_unicode=True, sort_keys=False)),
            )
        if analysis_config:
            safe_cfg = _make_yaml_safe(analysis_config)
            zf.writestr(
                "analysis_config.yaml",
                _text_to_bytes(yaml.dump(safe_cfg, allow_unicode=True, sort_keys=False)),
            )
        if processing_report:
            zf.writestr("processing_report.md", _text_to_bytes(processing_report))
        if metadata_validation_report:
            zf.writestr("metadata_validation_report.md", _text_to_bytes(metadata_validation_report))
        if manifest is not None:
            safe_manifest = _make_yaml_safe(manifest)
            zf.writestr(
                "manifest.yaml",
                _text_to_bytes(yaml.dump(safe_manifest, allow_unicode=True, sort_keys=False)),
            )
        if text_artifacts:
            for name, text in text_artifacts.items():
                if text:
                    zf.writestr(name, _text_to_bytes(text))
        zf.writestr(
            "event_metadata.csv",
            _text_to_bytes(
                "event_id,event_type,event_label,subject_id,group_id,timestamp,"
                "timezone,event_scope,used_for_alignment,used_for_exclusion,description,notes\n"
            ),
        )
    return path


def _make_yaml_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_yaml_safe(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    return obj
