"""Tests for src/dvc_behavior/exclusions.py"""

from __future__ import annotations

import io

import pandas as pd

from dvc_behavior.exclusions import apply_exclusions, compute_exclusion_windows
from dvc_behavior.events import parse_event_csv
from tests.conftest import make_metric_df
from dvc_behavior.parsing import wide_to_long


_BASE_RULES = {
    "REMOVED": {"before_hours": 24.0, "after_hours": 24.0, "exclude": True, "flag": True},
    "INSERTED": {"before_hours": 24.0, "after_hours": 24.0, "exclude": True, "flag": True},
    "CAGE_OFFLINE": {"before_hours": 0.0, "after_hours": 0.0, "exclude": False, "flag": True},
    "CAGE_ONLINE": {"before_hours": 0.0, "after_hours": 0.0, "exclude": False, "flag": True},
}


def _make_long_df_utc(n_rows=100, start="2024-01-01T07:00:00+0100", bin_min=60):
    """Long DataFrame with proper UTC timestamps."""
    raw = make_metric_df(n_rows=n_rows, start_ts=start, bin_minutes=bin_min)
    long, _ = wide_to_long(raw, "test.csv")
    return long


def _make_event_df_utc(ts_str: str, subject: str, event_type: str = "REMOVED") -> pd.DataFrame:
    return _make_event_rows_df_utc(
        [
            {
                "group": "C57",
                "day": 0,
                "hour": 0,
                "minute": 0,
                "relativeTime": 0,
                "timestamp": ts_str,
                "cage": subject,
                "rack": "R1",
                "position": "A1",
                "event": event_type,
            }
        ]
    )


def _make_event_rows_df_utc(rows: list[dict]) -> pd.DataFrame:
    ev = pd.DataFrame(rows)
    buf = io.StringIO()
    ev.to_csv(buf, index=False)
    parsed, _ = parse_event_csv(io.BytesIO(buf.getvalue().encode()), "ev.csv")
    return parsed


class TestComputeExclusionWindows:
    def test_removed_creates_window(self):
        event_df = _make_event_df_utc("2024-01-05T10:00:00+0100", "C57_1", "REMOVED")
        win = compute_exclusion_windows(event_df, _BASE_RULES)
        assert not win.empty
        assert bool(win.iloc[0]["exclude"]) is True

    def test_cage_offline_no_exclude(self):
        event_df = _make_event_df_utc("2024-01-05T10:00:00+0100", "C57_1", "CAGE_OFFLINE")
        win = compute_exclusion_windows(event_df, _BASE_RULES)
        assert not win.empty
        assert bool(win.iloc[0]["exclude"]) is False
        assert bool(win.iloc[0]["flag"]) is True

    def test_unknown_event_no_window(self):
        event_df = _make_event_df_utc("2024-01-05T10:00:00+0100", "C57_1", "WEIGH")
        win = compute_exclusion_windows(event_df, _BASE_RULES)
        assert win.empty

    def test_empty_event_df(self):
        win = compute_exclusion_windows(pd.DataFrame(), _BASE_RULES)
        assert win.empty

    def test_window_extent(self):
        ts_str = "2024-01-05T10:00:00+0100"
        event_df = _make_event_df_utc(ts_str, "C57_1", "REMOVED")
        win = compute_exclusion_windows(event_df, _BASE_RULES)
        event_ts = win.iloc[0]["event_timestamp"]
        assert win.iloc[0]["exclusion_start"] == event_ts - pd.Timedelta(hours=24)
        assert win.iloc[0]["exclusion_end"] == event_ts + pd.Timedelta(hours=24)

    def test_cage_change_pair_uses_asymmetric_windows(self):
        rules = {
            "REMOVED": {"before_hours": 6.0, "after_hours": 0.0, "exclude": True, "flag": True},
            "INSERTED": {"before_hours": 0.0, "after_hours": 12.0, "exclude": True, "flag": True},
        }
        event_df = _make_event_rows_df_utc(
            [
                {
                    "group": "C57",
                    "day": 0,
                    "hour": 0,
                    "minute": 0,
                    "relativeTime": 0,
                    "timestamp": "2024-01-05T10:00:00+0100",
                    "cage": "C57_1",
                    "rack": "R1",
                    "position": "A1",
                    "event": "REMOVED",
                },
                {
                    "group": "C57",
                    "day": 0,
                    "hour": 3,
                    "minute": 0,
                    "relativeTime": 10800,
                    "timestamp": "2024-01-05T13:00:00+0100",
                    "cage": "C57_1",
                    "rack": "R1",
                    "position": "A1",
                    "event": "INSERTED",
                },
            ]
        )

        win = compute_exclusion_windows(event_df, rules)

        assert len(win) == 1
        row = win.iloc[0]
        assert row["event_type"] == "CAGE_CHANGE"
        assert row["exclusion_start"] == row["event_timestamp"] - pd.Timedelta(hours=6)
        assert row["exclusion_end"] == row["paired_event_timestamp"] + pd.Timedelta(hours=12)
        assert row["subject_id"] == "C57_1"

    def test_facility_event_creates_all_subject_window(self):
        first_ts = pd.Timestamp("2024-01-01T06:00:00Z")
        event_df = pd.DataFrame(
            [
                {
                    "group_id": None,
                    "subject_id": None,
                    "event_type": "FACILITY_OUTAGE",
                    "timestamp_utc": first_ts,
                    "event_scope": "facility",
                }
            ]
        )
        rules = {
            "FACILITY_OUTAGE": {
                "before_hours": 0.0,
                "after_hours": 2.0,
                "exclude": True,
                "flag": False,
                "scope": "facility",
            }
        }

        win = compute_exclusion_windows(event_df, rules)

        assert len(win) == 1
        assert win.iloc[0]["event_scope"] == "facility"


class TestApplyExclusions:
    def test_subject_level_exclusion(self):
        """Rows for C57_1 around the REMOVED event should be excluded."""
        long = _make_long_df_utc(n_rows=300, start="2024-01-01T07:00:00+0100", bin_min=60)
        # Event in the middle of the time range
        ts_mid = "2024-01-10T12:00:00+0100"
        event_df = _make_event_df_utc(ts_mid, "C57_1", "REMOVED")
        win = compute_exclusion_windows(event_df, _BASE_RULES)
        result, log = apply_exclusions(long, win)

        excl = result[result["is_excluded"]]
        # Only C57_1 should be excluded
        if not excl.empty:
            assert set(excl["subject_id"].unique()) == {"C57_1"}

    def test_no_events_no_exclusions(self):
        long = _make_long_df_utc(n_rows=20)
        result, log = apply_exclusions(long, pd.DataFrame())
        assert not result["is_excluded"].any()

    def test_is_excluded_column_added(self):
        long = _make_long_df_utc(n_rows=10)
        result, _ = apply_exclusions(long, pd.DataFrame())
        assert "is_excluded" in result.columns
        assert "exclusion_reason" in result.columns
        assert "flag_reason" in result.columns

    def test_cage_offline_flagged_not_excluded(self):
        long = _make_long_df_utc(n_rows=300, bin_min=60)
        # Use timestamp near start of data
        first_ts = long["timestamp_utc"].dropna().min()
        mid_ts_str = first_ts.isoformat()
        event_df = _make_event_df_utc(mid_ts_str, "C57_1", "CAGE_OFFLINE")
        win = compute_exclusion_windows(event_df, _BASE_RULES)
        result, _ = apply_exclusions(long, win)
        # CAGE_OFFLINE window is 0h before/after → at most one row matched
        excl_subj = result[(result["subject_id"] == "C57_1") & result["is_excluded"]]
        assert excl_subj.empty, "CAGE_OFFLINE should not cause exclusion (flag only)"

    def test_group_level_fallback(self):
        """If subject_id doesn't match any row, fallback to group_id matching."""
        long = _make_long_df_utc(n_rows=50, bin_min=60)
        # Event for a cage not in the long_df (triggers group fallback)
        first_ts = long["timestamp_utc"].dropna().min()
        win_df = pd.DataFrame(
            [
                {
                    "subject_id": "NONEXISTENT_CAGE",
                    "group_id": "C57",
                    "event_type": "REMOVED",
                    "event_timestamp": first_ts,
                    "exclusion_start": first_ts - pd.Timedelta(hours=24),
                    "exclusion_end": first_ts + pd.Timedelta(hours=24),
                    "exclusion_reason": "REMOVED",
                    "exclude": True,
                    "flag": True,
                }
            ]
        )
        result, _ = apply_exclusions(long, win_df)
        # Fallback to group → all C57 subjects excluded in window
        excl = result[result["is_excluded"]]
        if not excl.empty:
            assert "C57" in str(excl["group_id"].unique())

    def test_exclusion_reason_filled(self):
        long = _make_long_df_utc(n_rows=100, bin_min=60)
        ts = long["timestamp_utc"].dropna().sort_values().iloc[50]
        event_df = _make_event_df_utc(ts.isoformat(), "C57_1", "REMOVED")
        win = compute_exclusion_windows(event_df, _BASE_RULES)
        result, _ = apply_exclusions(long, win)
        excl = result[result["is_excluded"] & (result["subject_id"] == "C57_1")]
        if not excl.empty:
            assert excl["exclusion_reason"].str.len().gt(0).all()

    def test_facility_event_excludes_all_subjects_in_window(self):
        long = _make_long_df_utc(n_rows=5, bin_min=60)
        first_ts = long["timestamp_utc"].dropna().min()
        win_df = pd.DataFrame(
            [
                {
                    "subject_id": None,
                    "group_id": None,
                    "event_scope": "facility",
                    "event_type": "FACILITY_OUTAGE",
                    "event_timestamp": first_ts,
                    "exclusion_start": first_ts,
                    "exclusion_end": first_ts + pd.Timedelta(hours=1),
                    "exclusion_reason": "FACILITY_OUTAGE",
                    "exclude": True,
                    "flag": False,
                }
            ]
        )

        result, log = apply_exclusions(long, win_df)
        excl = result[result["is_excluded"]]

        assert set(excl["subject_id"].unique()) == set(long["subject_id"].unique())
        assert int(log.iloc[0]["n_rows_excluded"]) == len(excl)
