"""Tests for the advanced scientific-analytics helpers in analysis.py.

These cover the additive enhancements (cosinor inference, configurable
photoperiod, estimation-first stats) and the new public functions
(non-parametric circadian metrics, activity bouts, window-summary contrast,
period estimation).  Several are property-style recovery tests: a signal with
known structure is synthesized and the recovered descriptors are asserted to
fall within tolerance.  All randomness is seeded for determinism.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dvc_behavior.analysis import (
    compare_window_summaries,
    estimate_period,
    quick_exploratory_stats,
    summarize_activity_bouts,
    summarize_circadian_cosinor,
    summarize_nonparametric_circadian,
)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

def _cosine_df(
    amplitude: float = 3.0,
    mesor: float = 10.0,
    acrophase_zt: float = 14.0,
    noise_sd: float = 0.2,
    n_subjects: int = 4,
    n_days: int = 4,
    seed: int = 0,
) -> pd.DataFrame:
    """Multi-day hourly cosine signal with a known amplitude/acrophase."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01T00:00:00")
    rows = []
    for s in range(n_subjects):
        for h in range(24 * n_days):
            zt = h % 24
            ts = start + pd.Timedelta(hours=h)
            val = (
                mesor
                + amplitude * np.cos(2.0 * np.pi * (zt - acrophase_zt) / 24.0)
                + rng.normal(0.0, noise_sd)
            )
            rows.append(
                {
                    "group_id": "A",
                    "metric_name": "activity",
                    "subject_id": f"s{s}",
                    "zeitgeber_time_hours": float(zt),
                    "time_from_event_hours": float(h),
                    "timestamp_local": ts,
                    "timestamp_utc": ts,
                    "value": val,
                    "is_excluded": False,
                }
            )
    return pd.DataFrame(rows)


def _flat_noise_df(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01T00:00:00")
    rows = []
    for s in range(4):
        for h in range(24 * 4):
            zt = h % 24
            ts = start + pd.Timedelta(hours=h)
            rows.append(
                {
                    "group_id": "A",
                    "metric_name": "activity",
                    "subject_id": f"s{s}",
                    "zeitgeber_time_hours": float(zt),
                    "timestamp_local": ts,
                    "value": rng.normal(10.0, 1.0),
                    "is_excluded": False,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Cosinor rhythmicity test + CIs
# ---------------------------------------------------------------------------

def test_cosinor_recovers_known_params_and_detects_rhythm():
    df = _cosine_df(amplitude=3.0, mesor=10.0, acrophase_zt=14.0, noise_sd=0.1)

    out, _ = summarize_circadian_cosinor(df, min_points=6)

    assert len(out) == 1
    row = out.iloc[0]
    assert abs(row["MESOR"] - 10.0) < 0.3
    assert abs(row["amplitude"] - 3.0) < 0.3
    assert abs(row["acrophase_ZT"] - 14.0) < 0.5
    # Strong rhythm -> tiny p, finite CIs that bracket the truth.
    assert row["p_rhythm"] < 0.001
    assert row["amplitude_ci_low"] <= 3.0 <= row["amplitude_ci_high"]
    assert np.isfinite(row["acrophase_ci_low"])
    assert row["fit_level"] == "subject"


def test_cosinor_flat_noise_is_not_significant():
    df = _flat_noise_df()

    out, _ = summarize_circadian_cosinor(df, min_points=6)

    row = out.iloc[0]
    # No real rhythm: p should be large (or NaN if amplitude collapses).
    assert (not np.isfinite(row["p_rhythm"])) or row["p_rhythm"] > 0.05


def test_cosinor_binned_matches_legacy_point_estimates():
    df = _cosine_df(noise_sd=0.0)

    subject_out, _ = summarize_circadian_cosinor(df, fit_level="subject", min_points=6)
    binned_out, _ = summarize_circadian_cosinor(df, fit_level="binned", min_points=6)

    # Legacy point estimates are identical regardless of fit_level.
    for col in ("MESOR", "amplitude", "acrophase_ZT", "R2"):
        assert np.isclose(subject_out.iloc[0][col], binned_out.iloc[0][col])
    assert binned_out.iloc[0]["fit_level"] == "binned"


# ---------------------------------------------------------------------------
# 2. Configurable photoperiod
# ---------------------------------------------------------------------------

def test_photoperiod_labels_flip_for_8_vs_16():
    # Acrophase around ZT10: light under 16:8, dark under 8:16.
    df = _cosine_df(acrophase_zt=10.0, noise_sd=0.05)

    out16, _ = summarize_circadian_cosinor(df, photoperiod_hours=16.0, min_points=6)
    out8, _ = summarize_circadian_cosinor(df, photoperiod_hours=8.0, min_points=6)

    assert out16.iloc[0]["acrophase_ZT"] == pytest.approx(10.0, abs=0.5)
    assert out16.iloc[0]["phase"] == "light"
    assert out8.iloc[0]["phase"] == "dark"


# ---------------------------------------------------------------------------
# 3. Estimation-first stats
# ---------------------------------------------------------------------------

def _two_group_df(shift: float = 5.0) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for g, base in (("A", 10.0), ("B", 10.0 + shift)):
        for i in range(6):
            rows.append(
                {
                    "group_id": g,
                    "metric_name": "activity",
                    "subject_id": f"{g}{i}",
                    "value": base + rng.normal(0.0, 0.5),
                }
            )
    return pd.DataFrame(rows)


def test_estimation_stats_min_p_and_small_n_warning():
    df = pd.DataFrame(
        {
            "group_id": ["A", "A", "A", "B", "B", "B"],
            "metric_name": ["activity"] * 6,
            "subject_id": ["A1", "A2", "A3", "B1", "B2", "B3"],
            "value": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
        }
    )

    out, _ = quick_exploratory_stats(df)
    comp = out[out["test"] == "Mann-Whitney U"].iloc[0]

    # n1=n2=3 -> 2 / C(6,3) = 2/20 = 0.1 > 0.05.
    assert comp["min_possible_p"] == pytest.approx(0.1)
    assert bool(comp["small_n_warning"]) is True


def test_estimation_stats_diff_ci_brackets_shift_and_is_deterministic():
    df = _two_group_df(shift=5.0)

    out1, _ = quick_exploratory_stats(df)
    out2, _ = quick_exploratory_stats(df)

    comp1 = out1[out1["test"] == "Mann-Whitney U"].iloc[0]
    comp2 = out2[out2["test"] == "Mann-Whitney U"].iloc[0]

    # group2 (B) median minus group1 (A) median ~ +5.
    assert comp1["median_difference"] == pytest.approx(5.0, abs=1.0)
    assert comp1["diff_ci_low"] <= 5.0 <= comp1["diff_ci_high"]
    assert np.isfinite(comp1["effect_size_ci_low"])
    assert np.isfinite(comp1["effect_size_ci_high"])

    # Determinism: two runs identical.
    assert comp1["diff_ci_low"] == comp2["diff_ci_low"]
    assert comp1["diff_ci_high"] == comp2["diff_ci_high"]
    assert comp1["effect_size_ci_low"] == comp2["effect_size_ci_low"]


def test_estimation_large_n_not_flagged():
    rng = np.random.default_rng(3)
    rows = []
    for g, base in (("A", 10.0), ("B", 12.0)):
        for i in range(10):
            rows.append(
                {
                    "group_id": g,
                    "metric_name": "activity",
                    "subject_id": f"{g}{i}",
                    "value": base + rng.normal(0.0, 0.5),
                }
            )
    out, _ = quick_exploratory_stats(pd.DataFrame(rows))
    comp = out[out["test"] == "Mann-Whitney U"].iloc[0]
    assert comp["min_possible_p"] < 0.05
    assert bool(comp["small_n_warning"]) is False


# ---------------------------------------------------------------------------
# 4. Non-parametric circadian metrics
# ---------------------------------------------------------------------------

def test_nonparametric_circadian_clean_rhythm_high_is_and_ra():
    df = _cosine_df(amplitude=5.0, mesor=10.0, acrophase_zt=14.0, noise_sd=0.05, n_days=5)

    out, warns = summarize_nonparametric_circadian(df)

    assert not out.empty
    row = out.iloc[0]
    assert row["IS"] > 0.9  # strongly day-to-day stable
    assert row["RA"] > 0.3  # clear amplitude
    assert row["M10"] > row["L5"]
    assert row["n_days"] == pytest.approx(5.0, abs=0.5)
    assert row["fit_method"] == "van_someren"


def test_nonparametric_circadian_flat_signal_low_ra():
    start = pd.Timestamp("2024-01-01T00:00:00")
    rows = []
    for h in range(24 * 5):
        ts = start + pd.Timedelta(hours=h)
        rows.append(
            {
                "group_id": "A",
                "metric_name": "activity",
                "subject_id": "s0",
                "timestamp_local": ts,
                "value": 10.0,  # perfectly flat
                "is_excluded": False,
            }
        )
    out, _ = summarize_nonparametric_circadian(pd.DataFrame(rows))
    row = out.iloc[0]
    assert row["RA"] == pytest.approx(0.0, abs=1e-9)


def test_nonparametric_circadian_warns_without_timestamp():
    df = pd.DataFrame(
        {
            "group_id": ["A"],
            "metric_name": ["activity"],
            "subject_id": ["s0"],
            "value": [1.0],
        }
    )
    out, warns = summarize_nonparametric_circadian(df)
    assert out.empty
    assert warns


# ---------------------------------------------------------------------------
# 5. Activity bouts / fragmentation
# ---------------------------------------------------------------------------

def test_activity_bouts_known_on_off_pattern():
    # 4h on, 4h off, repeated -> with median threshold, each "on" block is a bout.
    start = pd.Timestamp("2024-01-01T00:00:00")
    rows = []
    pattern = ([5.0] * 4 + [0.0] * 4) * 3  # 24 hourly bins, 3 active blocks
    for h, val in enumerate(pattern):
        ts = start + pd.Timedelta(hours=h)
        rows.append(
            {
                "group_id": "A",
                "metric_name": "activity",
                "subject_id": "s0",
                "timestamp_local": ts,
                "value": val,
                "is_excluded": False,
            }
        )
    out, warns = summarize_activity_bouts(pd.DataFrame(rows), threshold=1.0)

    row = out.iloc[0]
    assert row["bin_minutes"] == pytest.approx(60.0)
    assert row["n_active_bouts"] == 3
    assert row["mean_active_bout_minutes"] == pytest.approx(240.0)  # 4h
    assert row["max_active_bout_minutes"] == pytest.approx(240.0)
    assert row["fraction_time_active"] == pytest.approx(0.5)


def test_activity_bouts_warns_without_timestamp():
    df = pd.DataFrame(
        {
            "group_id": ["A", "A"],
            "metric_name": ["activity"] * 2,
            "subject_id": ["s0", "s0"],
            "value": [1.0, 2.0],
        }
    )
    out, warns = summarize_activity_bouts(df)
    assert out.empty
    assert warns


# ---------------------------------------------------------------------------
# 6. Window-summary contrast
# ---------------------------------------------------------------------------

def _window_df() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for s in range(8):
        grp = "A" if s < 4 else "B"
        for h in range(48):
            rows.append(
                {
                    "group_id": grp,
                    "metric_name": "activity",
                    "subject_id": f"{grp}{s}",
                    "time_from_event_hours": float(h),
                    "value": (12.0 if grp == "B" else 10.0) + rng.normal(0.0, 0.5),
                    "is_excluded": False,
                }
            )
    return pd.DataFrame(rows)


def test_window_contrast_one_row_per_window_with_fdr():
    df = _window_df()
    windows = [("early", 0.0, 24.0), ("late", 24.0, 48.0)]

    out, _ = compare_window_summaries(df, windows)

    assert len(out) == len(windows)
    assert set(out["window_label"]) == {"early", "late"}
    assert "q_value" in out.columns
    assert out["q_value"].notna().all()
    assert (out["test"] == "Mann-Whitney U").all()
    assert bool(out["exploratory"].iloc[0]) is True
    assert "disclaimer" in out.columns


def test_window_contrast_auc_statistic_runs():
    df = _window_df()
    out, _ = compare_window_summaries(df, [("w", 0.0, 48.0)], statistic="auc")
    assert len(out) == 1
    assert out.iloc[0]["window_statistic"] == "auc"
    assert np.isfinite(out.iloc[0]["p_value"])


# ---------------------------------------------------------------------------
# 7. Period estimation
# ---------------------------------------------------------------------------

def test_estimate_period_recovers_24h():
    df = _cosine_df(amplitude=4.0, acrophase_zt=12.0, noise_sd=0.1, n_subjects=2, n_days=6)

    out, _ = estimate_period(df)

    assert not out.empty
    for _, row in out.iterrows():
        assert row["method"] == "lomb_scargle"
        assert row["estimated_period_hours"] == pytest.approx(24.0, abs=0.6)
        assert np.isfinite(row["peak_power"])


def test_estimate_period_warns_without_timestamp():
    df = pd.DataFrame(
        {
            "group_id": ["A"],
            "metric_name": ["activity"],
            "subject_id": ["s0"],
            "value": [1.0],
        }
    )
    out, warns = estimate_period(df)
    assert out.empty
    assert warns
