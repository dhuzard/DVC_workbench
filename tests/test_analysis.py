"""Tests for src/dvc_behavior/analysis.py"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dvc_behavior.analysis import (
    compute_auc_per_animal,
    quick_exploratory_stats,
    summarize_circadian_cosinor,
    summarize_daily,
    summarize_light_dark,
    summarize_time_bins,
    summarize_weekly,
)


def _analysis_df() -> pd.DataFrame:
    rows = []
    for group, offset in [("A", 0.0), ("B", 10.0)]:
        for subject_idx in range(2):
            subject_id = f"{group}{subject_idx + 1}"
            for hour in range(0, 48, 6):
                zt = hour % 24
                rows.append(
                    {
                        "group_id": group,
                        "metric_name": "activity",
                        "subject_id": subject_id,
                        "time_from_event_hours": float(hour),
                        "timestamp_utc": pd.Timestamp("2024-01-01T00:00:00Z")
                        + pd.Timedelta(hours=hour),
                        "zeitgeber_time_hours": float(zt),
                        "light_dark_phase": "light" if zt < 12 else "dark",
                        "value": offset + subject_idx + 5.0 + 2.0 * np.cos(2.0 * np.pi * zt / 24.0),
                        "is_excluded": False,
                    }
                )
    return pd.DataFrame(rows)


def test_circadian_cosinor_recovers_simple_cosine():
    df = _analysis_df()

    out, warns = summarize_circadian_cosinor(df[df["group_id"] == "A"], min_points=4)

    assert warns == []
    assert len(out) == 1
    row = out.iloc[0]
    assert row["group_id"] == "A"
    assert abs(row["MESOR"] - 5.5) < 0.15
    assert abs(row["amplitude"] - 2.0) < 0.15
    assert row["acrophase_ZT"] < 0.25 or row["acrophase_ZT"] > 23.75
    assert row["R2"] > 0.99


def test_light_dark_summary_uses_subject_means_and_ratio():
    df = pd.DataFrame(
        {
            "group_id": ["A", "A", "A", "A"],
            "metric_name": ["activity"] * 4,
            "subject_id": ["A1", "A1", "A2", "A2"],
            "light_dark_phase": ["light", "dark", "light", "dark"],
            "value": [2.0, 6.0, 4.0, 12.0],
        }
    )

    out, warns = summarize_light_dark(df)

    assert warns == []
    light = out[out["phase"] == "light"].iloc[0]
    dark = out[out["phase"] == "dark"].iloc[0]
    assert light["mean"] == 3.0
    assert dark["mean"] == 9.0
    assert light["n_subjects"] == 2
    assert dark["dark_light_ratio"] == 3.0


def test_daily_summary_bins_relative_to_alignment():
    df = _analysis_df()

    out, warns = summarize_daily(df)

    assert warns == []
    assert set(out["time_bin_start"]) == {0.0, 24.0}
    first_bin = out[(out["group_id"] == "A") & (out["time_bin_start"] == 0.0)].iloc[0]
    assert first_bin["n_subjects"] == 2
    assert first_bin["n_observations"] == 8
    assert first_bin["relative_to"] == "alignment"


def test_weekly_summary_wrapper_uses_168_hour_bins():
    df = _analysis_df()

    out, warns = summarize_weekly(df)

    assert warns == []
    assert set(out["time_bin_start"]) == {0.0}
    assert set(out["time_bin_end"]) == {168.0}


def test_custom_time_bin_summary_absolute_dates():
    df = _analysis_df()

    out, warns = summarize_time_bins(df, bin_size="1D", relative_to="absolute")

    assert warns == []
    assert len(out) == 4
    assert out["time_bin_start"].min() == pd.Timestamp("2024-01-01T00:00:00Z")


def test_auc_per_animal_uses_trapezoidal_rule_in_window():
    df = pd.DataFrame(
        {
            "group_id": ["A", "A", "A", "A"],
            "metric_name": ["activity"] * 4,
            "subject_id": ["A1"] * 4,
            "time_from_event_hours": [0.0, 1.0, 2.0, 3.0],
            "value": [0.0, 1.0, 1.0, 0.0],
        }
    )

    out, warns = compute_auc_per_animal(df, start=0.0, end=3.0)

    assert warns == []
    assert len(out) == 1
    assert out["auc"].iloc[0] == 2.0
    assert out["n_points"].iloc[0] == 4


def test_auc_per_animal_ignores_excluded_rows_by_default():
    df = pd.DataFrame(
        {
            "group_id": ["A", "A", "A"],
            "metric_name": ["activity"] * 3,
            "subject_id": ["A1"] * 3,
            "time_from_event_hours": [0.0, 1.0, 2.0],
            "value": [0.0, 100.0, 2.0],
            "is_excluded": [False, True, False],
        }
    )

    out, _ = compute_auc_per_animal(df)

    assert out["auc"].iloc[0] == 2.0
    assert out["n_points"].iloc[0] == 2


def test_quick_exploratory_stats_runs_nonparametric_comparison(monkeypatch):
    import dvc_behavior.analysis as analysis

    class FakeStats:
        @staticmethod
        def shapiro(values):
            return 0.95, 0.5

        @staticmethod
        def mannwhitneyu(left, right, alternative="two-sided"):
            assert alternative == "two-sided"
            assert list(left) == [1.0, 2.0, 3.0]
            assert list(right) == [10.0, 11.0, 12.0]
            return 0.0, 0.05

    monkeypatch.setattr(analysis, "_load_scipy_stats", lambda: FakeStats)
    df = pd.DataFrame(
        {
            "group_id": ["A", "A", "A", "B", "B", "B"],
            "metric_name": ["activity"] * 6,
            "subject_id": ["A1", "A2", "A3", "B1", "B2", "B3"],
            "value": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
        }
    )

    out, warns = quick_exploratory_stats(df)

    assert warns == []
    comparison = out[out["test"] == "Mann-Whitney U"].iloc[0]
    assert comparison["comparison"] == "A vs B"
    assert bool(comparison["scipy_available"]) is True
    assert comparison["p_value"] == 0.05
    assert comparison["q_value"] == 0.05
    assert comparison["effect_size_name"] == "rank-biserial"
    assert comparison["effect_size"] == -1.0


def test_quick_exploratory_stats_adds_kruskal_effect_size_and_fdr(monkeypatch):
    import dvc_behavior.analysis as analysis

    class FakeStats:
        @staticmethod
        def shapiro(values):
            return 0.95, 0.5

        @staticmethod
        def kruskal(*groups):
            first = list(groups[0])
            if first == [1.0, 2.0, 3.0]:
                return 7.0, 0.01
            return 4.0, 0.04

    monkeypatch.setattr(analysis, "_load_scipy_stats", lambda: FakeStats)
    df = pd.DataFrame(
        {
            "group_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"] * 2,
            "metric_name": ["activity"] * 9 + ["distance"] * 9,
            "subject_id": ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"] * 2,
            "value": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
            ],
        }
    )

    out, warns = quick_exploratory_stats(df)

    assert warns == []
    comparisons = out[out["test"] == "Kruskal-Wallis"].sort_values("p_value")
    activity = comparisons[comparisons["p_value"] == 0.01].iloc[0]
    distance = comparisons[comparisons["p_value"] == 0.04].iloc[0]
    assert activity["effect_size_name"] == "epsilon-squared"
    assert np.isclose(activity["effect_size"], 5.0 / 6.0)
    assert activity["q_value"] == 0.02
    assert np.isclose(distance["effect_size"], 2.0 / 6.0)
    assert distance["q_value"] == 0.04
    assert out[out["test"] == "Shapiro-Wilk"]["q_value"].isna().all()


def test_quick_exploratory_stats_graceful_without_scipy(monkeypatch):
    import dvc_behavior.analysis as analysis

    monkeypatch.setattr(analysis, "_load_scipy_stats", lambda: None)
    df = pd.DataFrame(
        {
            "group_id": ["A", "B"],
            "metric_name": ["activity", "activity"],
            "subject_id": ["A1", "B1"],
            "value": [1.0, 2.0],
        }
    )

    out, warns = quick_exploratory_stats(df)

    assert warns
    assert out["test"].iloc[0] == "scipy.stats unavailable"
    assert out["p_value"].isna().all()
    assert out["q_value"].isna().all()
