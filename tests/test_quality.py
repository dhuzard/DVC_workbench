"""Tests for src/dvc_behavior/quality.py"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from dvc_behavior.quality import build_quality_report


def test_build_quality_report_computes_subject_metric_diagnostics():
    df = pd.DataFrame(
        {
            "subject_id": ["S1"] * 5 + ["S2"] * 3,
            "metric_name": ["activity"] * 8,
            "group_id": ["G1"] * 8,
            "timestamp_utc": [
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:01:00Z",
                "2024-01-01T00:02:00Z",
                "2024-01-01T00:20:00Z",
                "2024-01-01T00:20:00Z",
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:01:00Z",
                "2024-01-01T00:02:00Z",
            ],
            "value": [1.0, None, -2.0, 4.0, 4.0, 7.0, 7.0, 7.0],
        }
    )

    report = build_quality_report(df)

    assert len(report) == 2

    s1 = report[report["subject_id"] == "S1"].iloc[0]
    assert s1["metric_name"] == "activity"
    assert s1["group_id"] == "G1"
    assert s1["n_rows"] == 5
    assert s1["missing_value_count"] == 1
    assert math.isclose(s1["missing_value_rate"], 0.2)
    assert s1["duplicate_timestamp_count"] == 1
    assert s1["negative_value_count"] == 1
    assert bool(s1["irregular_interval_flag"]) is True
    assert s1["long_gap_count"] == 1
    assert bool(s1["zero_variance_flag"]) is False

    s2 = report[report["subject_id"] == "S2"].iloc[0]
    assert s2["missing_value_rate"] == 0.0
    assert s2["duplicate_timestamp_count"] == 0
    assert bool(s2["irregular_interval_flag"]) is False
    assert s2["long_gap_count"] == 0
    assert bool(s2["zero_variance_flag"]) is True


def test_build_quality_report_respects_explicit_long_gap_threshold():
    df = pd.DataFrame(
        {
            "subject_id": ["S1"] * 3,
            "metric_name": ["distance"] * 3,
            "timestamp_utc": [
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:01:00Z",
                "2024-01-01T00:03:00Z",
            ],
            "value": [1.0, 2.0, 3.0],
        }
    )

    report = build_quality_report(df, long_gap_threshold_seconds=90)

    row = report.iloc[0]
    assert row["long_gap_threshold_seconds"] == 90.0
    assert row["long_gap_count"] == 1


def test_build_quality_report_empty_input_returns_empty_schema():
    df = pd.DataFrame(columns=["subject_id", "metric_name", "timestamp_utc", "value"])

    report = build_quality_report(df)

    assert report.empty
    assert {"missing_value_rate", "duplicate_timestamp_count", "zero_variance_flag"} <= set(
        report.columns
    )


def test_build_quality_report_missing_required_column_raises():
    df = pd.DataFrame({"subject_id": ["S1"], "metric_name": ["activity"], "value": [1.0]})

    with pytest.raises(ValueError, match="timestamp_utc"):
        build_quality_report(df)
