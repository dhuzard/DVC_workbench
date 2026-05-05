"""Tests for src/dvc_behavior/qc.py"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("plotly")
from dvc_behavior.qc import detect_irregular_bins, plot_group_mean_timeseries


def test_group_mean_empty_after_exclusion_returns_empty_figure():
    df = pd.DataFrame(
        {
            "metric_name": ["activity", "activity"],
            "subject_id": ["C57_1", "C57_2"],
            "group_id": ["C57", "C57"],
            "time_from_event_hours": [0.0, 0.0],
            "value": [1.0, 2.0],
            "is_excluded": [True, True],
        }
    )

    fig = plot_group_mean_timeseries(df, "activity")

    assert fig.layout.title.text == "No data after exclusion filter."
    assert len(fig.data) == 0


def test_detect_irregular_bins_flags_dropout():
    ts = [
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T00:01:00Z"),
        pd.Timestamp("2024-01-01T00:02:00Z"),
        pd.Timestamp("2024-01-01T00:20:00Z"),
    ]
    df = pd.DataFrame(
        {
            "metric_name": ["activity"] * len(ts),
            "subject_id": ["C57_1"] * len(ts),
            "group_id": ["C57"] * len(ts),
            "timestamp_utc": ts,
        }
    )

    report = detect_irregular_bins(df)

    assert bool(report["irregular_bins"].iloc[0]) is True
