"""Tests for src/dvc_behavior/qc.py"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("plotly")
import math

import numpy as np

from dvc_behavior.qc import (
    confidence_halfwidth,
    detect_irregular_bins,
    plot_group_mean_timeseries,
)


def _two_subject_group_df():
    rows = []
    for group in ("A", "B"):
        for subj in (f"{group}1", f"{group}2"):
            for t in (0.0, 0.0, 1.0, 1.0):  # two raw bins per (subject, t)
                rows.append(
                    {
                        "metric_name": "activity",
                        "subject_id": subj,
                        "group_id": group,
                        "time_from_event_hours": t,
                        "value": 1.0 if group == "A" else 3.0,
                        "is_excluded": False,
                    }
                )
    return pd.DataFrame(rows)


class TestConfidenceHalfwidth:
    def test_nan_when_n_too_small(self):
        assert math.isnan(confidence_halfwidth(1.0, 1))
        assert math.isnan(confidence_halfwidth(float("nan"), 5))

    def test_t_interval_wider_than_sem(self):
        sd, n = 2.0, 4
        sem = sd / math.sqrt(n)
        half = confidence_halfwidth(sd, n)
        # t(0.975, 3) ≈ 3.182 > 1.96, so the CI half-width exceeds ~2*SEM.
        assert half > sem
        assert half == pytest.approx(3.182446 * sem, rel=1e-3)


class TestGroupMeanBands:
    def test_band_variants_change_traces_and_title(self):
        df = _two_subject_group_df()
        ci = plot_group_mean_timeseries(df, "activity", band="ci95")
        sem = plot_group_mean_timeseries(df, "activity", band="sem")
        none = plot_group_mean_timeseries(df, "activity", band="none")

        assert "95% CI" in ci.layout.title.text
        assert "SEM" in sem.layout.title.text
        # ci/sem add a band trace per group (2 groups → 4 traces); none → 2 traces.
        assert len(ci.data) == 4
        assert len(none.data) == 2

    def test_band_is_computed_over_subjects_not_raw_rows(self):
        # 2 subjects per group; identical values → zero variance → zero half-width,
        # and the hover n must report 2 subjects, not 4 raw rows.
        df = _two_subject_group_df()
        fig = plot_group_mean_timeseries(df, "activity", band="ci95")
        mean_traces = [tr for tr in fig.data if tr.mode == "lines+markers"]
        assert mean_traces
        for tr in mean_traces:
            assert set(np.asarray(tr.customdata).ravel()) == {2}


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
