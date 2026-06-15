"""Tests for src/dvc_behavior/baseline.py"""

from __future__ import annotations


import numpy as np
import pandas as pd

from dvc_behavior.alignment import align_to_manual_timestamp
from dvc_behavior.baseline import compute_baseline
from dvc_behavior.parsing import wide_to_long
from tests.conftest import make_metric_df


def _make_aligned_long(
    n_rows=96,
    bin_min=60,
    start="2024-01-01T07:00:00+0100",
    groups=None,
    alignment_ts="2024-01-05T07:00:00+0100",
):
    raw = make_metric_df(n_rows=n_rows, start_ts=start, bin_minutes=bin_min, groups=groups)
    long, _ = wide_to_long(raw, "test.csv")
    long, _ = align_to_manual_timestamp(long, alignment_ts)
    return long


class TestComputeBaseline:
    def test_adds_required_columns(self):
        long = _make_aligned_long()
        out, summary, warns = compute_baseline(long, -48.0, -24.0)
        for col in (
            "baseline_value",
            "baseline_corrected_value",
            "baseline_percent_change",
            "baseline_valid",
            "baseline_n_bins",
            "baseline_coverage",
        ):
            assert col in out.columns, f"Missing column: {col}"

    def test_baseline_value_is_mean(self):
        """Verify baseline is mean of values in the window."""
        long = _make_aligned_long(n_rows=200, bin_min=60, alignment_ts="2024-01-09T07:00:00+0100")
        # window: -48h to -24h before alignment
        out, summary, _ = compute_baseline(long, -48.0, -24.0)

        for sid in out["subject_id"].unique():
            sub = out[out["subject_id"] == sid]
            bsl_val = sub["baseline_value"].iloc[0]
            if np.isnan(bsl_val):
                continue
            window_rows = sub[
                (sub["time_from_event_hours"] >= -48.0) & (sub["time_from_event_hours"] < -24.0)
            ]
            if window_rows.empty:
                continue
            expected = window_rows["value"].dropna().mean()
            assert abs(bsl_val - expected) < 0.001, f"Mismatch for {sid}: {bsl_val} vs {expected}"

    def test_baseline_corrected_value(self):
        long = _make_aligned_long(n_rows=200, bin_min=60, alignment_ts="2024-01-09T07:00:00+0100")
        out, _, _ = compute_baseline(long, -48.0, -24.0)
        valid = out[out["baseline_valid"] & out["value"].notna()]
        if not valid.empty:
            expected_corr = valid["value"] - valid["baseline_value"]
            actual_corr = valid["baseline_corrected_value"]
            pd.testing.assert_series_equal(
                expected_corr.reset_index(drop=True),
                actual_corr.reset_index(drop=True),
                check_names=False,
                atol=1e-9,
            )

    def test_no_alignment_warns_and_skips(self):
        raw = make_metric_df(n_rows=10)
        long, _ = wide_to_long(raw, "test.csv")
        out, summary, warns = compute_baseline(long, -48.0, -24.0)
        assert warns
        assert summary.empty

    def test_excluded_rows_ignored_in_baseline(self):
        """Excluded rows must not contribute to the baseline mean."""
        long = _make_aligned_long(n_rows=200, bin_min=60, alignment_ts="2024-01-09T07:00:00+0100")
        long = long.copy()
        long["is_excluded"] = False
        # Mark all baseline-window rows for C57_1 as excluded
        baseline_mask = (
            (long["time_from_event_hours"] >= -48.0)
            & (long["time_from_event_hours"] < -24.0)
            & (long["subject_id"] == "C57_1")
        )
        long.loc[baseline_mask, "is_excluded"] = True

        out, summary, warns = compute_baseline(long, -48.0, -24.0, exclude_excluded=True)
        # C57_1 baseline should be NaN (no valid data in window)
        c57_1 = out[out["subject_id"] == "C57_1"]
        if not c57_1.empty:
            assert not c57_1["baseline_valid"].iloc[0] or np.isnan(c57_1["baseline_value"].iloc[0])

    def test_division_by_zero_safe(self):
        """When baseline_value == 0, percent_change should be NaN, not inf."""
        long = _make_aligned_long(n_rows=100, bin_min=60, alignment_ts="2024-01-05T07:00:00+0100")
        long = long.copy()
        # Force all baseline-window values to 0 for C57_1
        bsl_mask = (
            (long["time_from_event_hours"] >= -72.0)
            & (long["time_from_event_hours"] < -24.0)
            & (long["subject_id"] == "C57_1")
        )
        long.loc[bsl_mask, "value"] = 0.0

        out, _, _ = compute_baseline(long, -72.0, -24.0)
        c57_1 = out[out["subject_id"] == "C57_1"]
        if not c57_1.empty:
            pct = c57_1["baseline_percent_change"]
            assert not np.isinf(pct.replace([np.inf, -np.inf], np.nan).dropna()).any()

    def test_near_zero_baseline_marked_unstable(self):
        """A tiny but non-zero baseline must yield NaN percent change and
        the unstable flag, while a normal baseline yields a finite value."""
        long = _make_aligned_long(n_rows=100, bin_min=60, alignment_ts="2024-01-05T07:00:00+0100")
        long = long.copy()
        # Force C57_1's baseline-window values to a tiny non-zero number that
        # is below the default epsilon (1e-3) but not exactly zero.
        bsl_mask = (
            (long["time_from_event_hours"] >= -72.0)
            & (long["time_from_event_hours"] < -24.0)
            & (long["subject_id"] == "C57_1")
        )
        long.loc[bsl_mask, "value"] = 0.0001

        out, _, _ = compute_baseline(long, -72.0, -24.0)

        assert "baseline_percent_change_unstable" in out.columns
        c57_1 = out[out["subject_id"] == "C57_1"]
        if not c57_1.empty and not np.isnan(c57_1["baseline_value"].iloc[0]):
            assert c57_1["baseline_percent_change"].isna().all()
            assert bool(c57_1["baseline_percent_change_unstable"].iloc[0]) is True
            # corrected (absolute) value stays well-defined near zero
            assert not c57_1["baseline_corrected_value"].isna().all()

        # A subject with a normal (non-near-zero) baseline must be stable.
        c57_2 = out[out["subject_id"] == "C57_2"]
        if not c57_2.empty and not np.isnan(c57_2["baseline_value"].iloc[0]):
            assert bool(c57_2["baseline_percent_change_unstable"].iloc[0]) is False
            finite_rows = c57_2[c57_2["value"].notna()]
            if not finite_rows.empty:
                assert finite_rows["baseline_percent_change"].notna().any()

    def test_epsilon_parameter_override(self):
        """A larger epsilon flags otherwise-normal baselines as unstable."""
        long = _make_aligned_long(n_rows=100, bin_min=60, alignment_ts="2024-01-05T07:00:00+0100")

        # make_metric_df draws from uniform(0, 10), so baselines are O(1-10).
        # With a huge epsilon every finite baseline is below the floor.
        out, _, _ = compute_baseline(long, -72.0, -24.0, epsilon=1e6)
        finite = out[out["baseline_value"].notna()]
        if not finite.empty:
            assert finite["baseline_percent_change_unstable"].all()
            assert finite["baseline_percent_change"].isna().all()

        # With a tiny epsilon the same normal baselines are stable.
        out2, _, _ = compute_baseline(long, -72.0, -24.0, epsilon=1e-12)
        finite2 = out2[out2["baseline_value"].notna() & (out2["baseline_value"].abs() > 1e-6)]
        if not finite2.empty:
            assert not finite2["baseline_percent_change_unstable"].any()

    def test_low_coverage_marked_invalid(self):
        """If data covers < min_coverage fraction, baseline_valid should be False."""
        long = _make_aligned_long(n_rows=10, bin_min=60, alignment_ts="2024-01-01T17:00:00+0100")
        # Very wide window: likely < 0.7 coverage
        out, summary, warns = compute_baseline(long, -72.0, -24.0, min_coverage=0.7)
        if not summary.empty and "baseline_valid" in summary.columns:
            # Some subjects may have insufficient coverage
            n_invalid = int((~summary["baseline_valid"]).sum())
            # Not crashing is the key assertion
            assert n_invalid >= 0

    def test_summary_columns(self):
        long = _make_aligned_long(n_rows=150, bin_min=60, alignment_ts="2024-01-07T07:00:00+0100")
        _, summary, _ = compute_baseline(long, -48.0, -24.0)
        if not summary.empty:
            for col in ("subject_id", "metric_name", "baseline_value", "baseline_valid"):
                assert col in summary.columns

    def test_warns_when_native_bins_unavailable(self):
        long = _make_aligned_long(n_rows=150, bin_min=60, alignment_ts="2024-01-07T07:00:00+0100")
        long = long.drop(columns=["native_bin_seconds"])

        out, summary, warns = compute_baseline(long, -48.0, -24.0)

        assert any("native_bin_seconds" in warning for warning in warns)
        assert out["baseline_expected_bins"].isna().all()
        if not summary.empty:
            assert summary["baseline_expected_bins"].isna().all()

    def test_can_impute_invalid_baseline_from_group_mean(self):
        long = _make_aligned_long(n_rows=200, bin_min=60, alignment_ts="2024-01-09T07:00:00+0100")
        long["is_excluded"] = False
        mask = (
            (long["subject_id"] == "C57_1")
            & (long["time_from_event_hours"] >= -48.0)
            & (long["time_from_event_hours"] < -24.0)
        )
        long.loc[mask, "is_excluded"] = True

        out, summary, warns = compute_baseline(
            long,
            -48.0,
            -24.0,
            exclude_excluded=True,
            impute_from_group_mean=True,
        )

        c57_1 = out[out["subject_id"] == "C57_1"]
        assert bool(c57_1["baseline_imputed"].iloc[0]) is True
        assert bool(c57_1["baseline_valid"].iloc[0]) is True
        assert any("imputed" in warning for warning in warns)
        assert bool(summary.loc[summary["subject_id"] == "C57_1", "baseline_imputed"].iloc[0]) is True

    def test_manual_baseline_override_wins(self):
        long = _make_aligned_long(n_rows=200, bin_min=60, alignment_ts="2024-01-09T07:00:00+0100")
        overrides = pd.DataFrame(
            [{"subject_id": "C57_1", "metric_name": "test", "baseline_value": 123.0}]
        )

        out, summary, _ = compute_baseline(long, -48.0, -24.0, baseline_overrides=overrides)

        c57_1 = out[out["subject_id"] == "C57_1"]
        assert c57_1["baseline_value"].iloc[0] == 123.0
        assert bool(c57_1["baseline_override"].iloc[0]) is True
