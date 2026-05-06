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
