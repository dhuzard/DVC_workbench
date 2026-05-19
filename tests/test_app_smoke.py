"""Smoke tests for Streamlit app component configuration."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from app.components.metadata_tables import group_column_config, subject_column_config
from app.components import workflow
from dvc_behavior import config as cfg


def test_metadata_column_config_covers_expected_columns():
    subject_cfg = subject_column_config()
    group_cfg = group_column_config()

    assert set(cfg.SUBJECT_METADATA_COLUMNS) <= set(subject_cfg)
    assert set(cfg.GROUP_METADATA_COLUMNS) <= set(group_cfg)


def test_workflow_stepper_markup_is_not_rendered_as_code(monkeypatch):
    rendered: list[str] = []

    def capture_markdown(body: str, **_: object) -> None:
        rendered.append(body)

    monkeypatch.setattr(workflow.st, "markdown", capture_markdown)

    workflow.render_progress_stepper(
        [
            {
                "label": "1. Import",
                "status": "ready",
                "detail": "Load files.",
            }
        ]
    )

    assert rendered[0].startswith("<style>")
    assert rendered[1].startswith("<div class='dvc-stepper'>")
    assert "\n            <div" not in rendered[1]


def test_import_event_mapping_guess_uses_filename_similarity():
    from app import streamlit_app

    mapping = streamlit_app._guess_event_file_mapping(
        [
            "cohort1_animal_loc__index_smoothed.csv",
            "cohort2_animal_loc__index_smoothed.csv",
        ],
        [
            "cohort2_events.csv",
            "cohort1_events.csv",
        ],
    )

    assert mapping == {
        "cohort1_animal_loc__index_smoothed.csv": "cohort1_events.csv",
        "cohort2_animal_loc__index_smoothed.csv": "cohort2_events.csv",
    }

    wt_mapping = streamlit_app._guess_event_file_mapping(
        ["WT-Group_animal_loc__index_smoothed.csv"],
        ["WT-Group_events.csv"],
    )

    assert wt_mapping == {
        "WT-Group_animal_loc__index_smoothed.csv": "WT-Group_events.csv",
    }


def test_upload_split_auto_corrects_obvious_file_type_mixups():
    from app import streamlit_app

    metric_upload = BytesIO(b"event-bytes")
    metric_upload.name = "WT-Group_events.csv"
    event_upload = BytesIO(b"metric-bytes")
    event_upload.name = "WT-Group_animal_loc__index_smoothed.csv"

    metric_files, event_files, corrections = streamlit_app._split_uploaded_files(
        [metric_upload],
        [event_upload],
    )

    assert metric_files == [("WT-Group_animal_loc__index_smoothed.csv", b"metric-bytes")]
    assert event_files == [("WT-Group_events.csv", b"event-bytes")]
    assert len(corrections) == 2


def test_event_overlay_keeps_late_events_after_first_200():
    from app import streamlit_app
    import plotly.graph_objects as go

    event_times = pd.date_range("2025-05-01", periods=250, freq="h", tz="UTC")
    event_df = pd.DataFrame(
        {
            "timestamp_utc": event_times,
            "event_type": ["REMOVED"] * len(event_times),
        }
    )
    fig = go.Figure()

    streamlit_app._add_event_overlay(
        fig,
        event_df,
        x_col="timestamp_utc",
        y_value=1.0,
    )

    vlines = [
        shape
        for shape in fig.layout.shapes
        if getattr(shape, "type", None) == "line"
    ]
    assert len(vlines) == 250
    assert vlines[-1].x0 == event_times[-1]


def test_analysis_value_options_ignore_empty_baseline_columns():
    from app import streamlit_app

    df = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0],
            "baseline_percent_change": [pd.NA, pd.NA, pd.NA],
            "baseline_corrected_value": [pd.NA, pd.NA, pd.NA],
        }
    )

    options, default, notice = streamlit_app._analysis_value_options(df)

    assert options == ["value"]
    assert default == "value"
    assert notice is not None


def test_analysis_value_options_prefer_valid_baseline_columns():
    from app import streamlit_app

    df = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0],
            "baseline_percent_change": [10.0, 20.0, 30.0],
            "baseline_corrected_value": [0.1, 0.2, 0.3],
        }
    )

    options, default, notice = streamlit_app._analysis_value_options(df)

    assert options == ["baseline_percent_change", "baseline_corrected_value", "value"]
    assert default == "baseline_percent_change"
    assert notice is None


def test_circadian_profile_works_with_raw_processed_values():
    from app import streamlit_app

    df = pd.DataFrame(
        {
            "group_id": ["A", "A", "A", "A"],
            "subject_id": ["A1", "A1", "A2", "A2"],
            "timestamp_local": pd.date_range("2025-01-01 07:00", periods=4, freq="6h", tz="Europe/Paris"),
            "zeitgeber_time_hours": [0.0, 6.0, 0.0, 6.0],
            "value": [1.0, 3.0, 2.0, 4.0],
        }
    )

    profile = streamlit_app._circadian_profile(
        df,
        value_col="value",
        max_days=1,
        zt_bin_hours=6.0,
        normalize_mode="Raw values",
    )

    assert not profile.empty
    assert set(profile["zeitgeber_time_hours"]) == {0.0, 6.0}
    assert streamlit_app._circadian_profile_issue(df, value_col="value", max_days=1) is None


def test_circadian_profile_issue_names_missing_requirements():
    from app import streamlit_app

    issue = streamlit_app._circadian_profile_issue(
        pd.DataFrame({"value": [1.0]}),
        value_col="value",
        max_days=1,
    )

    assert issue is not None
    assert "timestamp_local" in issue
    assert "zeitgeber_time_hours" in issue
    assert "subject_id" in issue


def test_analysis_time_and_auc_fallback_to_timestamps_without_alignment():
    from app import streamlit_app
    from dvc_behavior import analysis

    df = pd.DataFrame(
        {
            "group_id": ["A", "A", "A"],
            "metric_name": ["activity"] * 3,
            "subject_id": ["A1"] * 3,
            "timestamp_local": pd.date_range("2025-01-01 07:00", periods=3, freq="1h", tz="Europe/Paris"),
            "time_from_event_hours": [pd.NA, pd.NA, pd.NA],
            "value": [0.0, 1.0, 2.0],
            "is_excluded": [False, False, False],
        }
    )

    relative_to, timestamp_col, _ = streamlit_app._time_bin_settings(df)
    auc_x_col, auc_start, auc_end, _ = streamlit_app._auc_settings(df)

    assert relative_to == "absolute"
    assert timestamp_col == "timestamp_local"
    assert auc_x_col == "timestamp_local"
    assert auc_start is None
    assert auc_end is None

    time_bins, bin_warns = analysis.summarize_time_bins(
        df,
        bin_size="1D",
        relative_to=relative_to,
        timestamp_col=timestamp_col,
        value_col="value",
    )
    auc, auc_warns = analysis.compute_auc_per_animal(
        df,
        x_col=auc_x_col,
        start=auc_start,
        end=auc_end,
        value_col="value",
    )

    assert bin_warns == []
    assert not time_bins.empty
    assert auc_warns == []
    assert not auc.empty
    assert auc["auc"].iloc[0] == 2.0
