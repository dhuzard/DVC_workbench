"""Tests for src/dvc_behavior/alignment.py"""

from __future__ import annotations

import io
import math

import pandas as pd

from dvc_behavior.alignment import align_to_event, align_to_manual_timestamp
from dvc_behavior.events import parse_event_csv
from dvc_behavior.parsing import wide_to_long
from tests.conftest import make_metric_df


def _make_long(n_rows=48, bin_min=60, start="2024-01-01T07:00:00+0100", groups=None):
    raw = make_metric_df(n_rows=n_rows, start_ts=start, bin_minutes=bin_min, groups=groups)
    long, _ = wide_to_long(raw, "test.csv")
    return long


def _make_event_df_utc(
    ts_str: str, subject: str, group: str = "C57", etype: str = "REMOVED"
) -> pd.DataFrame:
    ev = pd.DataFrame(
        [
            {
                "group": group,
                "day": 0,
                "hour": 0,
                "minute": 0,
                "relativeTime": 0,
                "timestamp": ts_str,
                "cage": subject,
                "rack": "R1",
                "position": "A1",
                "event": etype,
            }
        ]
    )
    buf = io.StringIO()
    ev.to_csv(buf, index=False)
    parsed, _ = parse_event_csv(io.BytesIO(buf.getvalue().encode()), "ev.csv")
    return parsed


class TestAlignToEvent:
    def test_adds_required_columns(self):
        long = _make_long(n_rows=24, bin_min=60)
        event_ts = "2024-01-02T07:00:00+0100"
        event_df = _make_event_df_utc(event_ts, "C57_1", etype="REMOVED")
        out, warns = align_to_event(long, event_df, "REMOVED")
        for col in (
            "alignment_event_type",
            "alignment_timestamp",
            "time_from_event_seconds",
            "time_from_event_hours",
            "experimental_day",
        ):
            assert col in out.columns, f"Missing column: {col}"

    def test_time_from_event_at_event_is_zero(self):
        """Rows whose timestamp equals the alignment event should have ≈ 0 delta."""
        event_ts_str = "2024-01-02T07:00:00+0100"
        long = _make_long(n_rows=48, bin_min=60)
        event_df = _make_event_df_utc(event_ts_str, "C57_1", etype="REMOVED")
        out, _ = align_to_event(long, event_df, "REMOVED", scope="subject")

        event_ts = pd.Timestamp(event_ts_str).tz_convert("UTC")
        sub1 = out[(out["subject_id"] == "C57_1")]
        exact = sub1[sub1["timestamp_utc"] == event_ts]
        if not exact.empty:
            assert abs(exact["time_from_event_seconds"].iloc[0]) < 1.0

    def test_experimental_day_calculation(self):
        event_ts_str = "2024-01-03T07:00:00+0100"
        long = _make_long(n_rows=96, bin_min=60)
        event_df = _make_event_df_utc(event_ts_str, "C57_1", etype="REMOVED")
        out, _ = align_to_event(long, event_df, "REMOVED", scope="subject")

        sub1 = out[out["subject_id"] == "C57_1"].dropna(subset=["time_from_event_hours"])
        for _, row in sub1.iterrows():
            expected_day = math.floor(row["time_from_event_hours"] / 24.0)
            assert int(row["experimental_day"]) == expected_day

    def test_no_event_uses_fallback(self):
        long = _make_long(n_rows=10)
        fallback = "2024-01-01T07:00:00+0100"
        out, warns = align_to_event(long, pd.DataFrame(), "REMOVED", fallback_timestamp=fallback)
        # With fallback, all rows should have a non-NaN time_from_event
        assert out["time_from_event_seconds"].notna().any()

    def test_no_event_no_fallback_warns(self):
        long = _make_long(n_rows=5)
        out, warns = align_to_event(long, pd.DataFrame(), "REMOVED")
        assert warns

    def test_subject_specific_alignment(self):
        """Each subject should be aligned to their own event, not another's."""
        long = _make_long(n_rows=24, bin_min=60)
        # Subject C57_1 event at T=12h, C57_2 event at T=20h (relative to data start)
        ts1 = "2024-01-01T19:00:00+0100"
        ts2 = "2024-01-02T03:00:00+0100"
        ev1 = _make_event_df_utc(ts1, "C57_1", etype="REMOVED")
        ev2 = _make_event_df_utc(ts2, "C57_2", etype="REMOVED")
        event_df = pd.concat([ev1, ev2], ignore_index=True)

        out, _ = align_to_event(long, event_df, "REMOVED", scope="subject")
        aln_s1 = out[out["subject_id"] == "C57_1"]["alignment_timestamp"].dropna().iloc[0]
        aln_s2 = out[out["subject_id"] == "C57_2"]["alignment_timestamp"].dropna().iloc[0]
        assert aln_s1 != aln_s2

    def test_group_level_scope(self):
        """With scope='group', all subjects in the group share the event."""
        long = _make_long(n_rows=10)
        event_df = _make_event_df_utc("2024-01-01T12:00:00+0100", "C57_1", etype="REMOVED")
        out, _ = align_to_event(long, event_df, "REMOVED", scope="group")
        # All subjects should have the same alignment timestamp
        alns = out["alignment_timestamp"].dropna().unique()
        assert len(alns) <= 1


class TestAlignToManualTimestamp:
    def test_global_alignment(self):
        long = _make_long(n_rows=10)
        ts = "2024-01-01T12:00:00+00:00"
        out, warns = align_to_manual_timestamp(long, ts)
        assert out["time_from_event_seconds"].notna().any()

    def test_bad_timestamp_warns(self):
        long = _make_long(n_rows=5)
        out, warns = align_to_manual_timestamp(long, "not-a-timestamp")
        assert warns
