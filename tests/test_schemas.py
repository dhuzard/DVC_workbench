"""Tests for warn-only DataFrame schema validation helpers."""

from __future__ import annotations

import pandas as pd

from dvc_behavior import schemas


def test_validate_dataframe_disabled_returns_no_warnings():
    warnings = schemas.validate_dataframe(pd.DataFrame(), "long_df", enabled=False)

    assert warnings == []


def test_validate_dataframe_dispatches_to_named_validator():
    df = pd.DataFrame(
        {
            "source_file": ["metrics.csv"],
            "metric_name": ["activity"],
            "group_id": ["G1"],
            "subject_id": ["S1"],
            "timestamp_utc": [pd.Timestamp("2024-01-01T00:00:00Z")],
            "value": [1.0],
        }
    )

    warnings = schemas.validate_dataframe(df, "long_df", enabled=True)

    assert not any("missing expected columns" in warning for warning in warnings)


def test_validate_dataframe_unknown_name_warns():
    warnings = schemas.validate_dataframe(pd.DataFrame(), "unknown_df", enabled=True)

    assert warnings == ["unknown_df: no schema validator is registered."]
