"""Tests for src/dvc_behavior/events.py"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from dvc_behavior.events import parse_event_csv, get_unique_event_types
from tests.conftest import make_event_csv_bytes

_EXAMPLES = Path(__file__).parent.parent / "data" / "examples"


class TestParseEventCsv:
    def test_basic(self):
        data = make_event_csv_bytes()
        ev, warns = parse_event_csv(io.BytesIO(data), "test_events.csv")
        assert not ev.empty

    def test_required_columns_output(self):
        data = make_event_csv_bytes()
        ev, _ = parse_event_csv(io.BytesIO(data), "test_events.csv")
        required = {
            "source_file", "group_id", "subject_id", "event_type",
            "timestamp", "timestamp_utc", "timestamp_local",
            "raw_event_label", "event_category",
        }
        assert required <= set(ev.columns)

    def test_event_type_upper(self):
        data = make_event_csv_bytes()
        ev, _ = parse_event_csv(io.BytesIO(data), "test_events.csv")
        assert all(ev["event_type"] == ev["event_type"].str.upper())

    def test_event_category_mapping(self):
        data = make_event_csv_bytes()
        ev, _ = parse_event_csv(io.BytesIO(data), "test_events.csv")
        removed = ev[ev["event_type"] == "REMOVED"]
        assert (removed["event_category"] == "cage_handling").all()
        inserted = ev[ev["event_type"] == "INSERTED"]
        assert (inserted["event_category"] == "cage_handling").all()

    def test_cage_becomes_subject_id(self):
        data = make_event_csv_bytes()
        ev, _ = parse_event_csv(io.BytesIO(data), "test_events.csv")
        assert "C57_1" in ev["subject_id"].values

    def test_cage_offline_online_category(self):
        events = [
            {
                "group": "C57",
                "day": 1,
                "hour": 0,
                "minute": 0,
                "relativeTime": 86400,
                "timestamp": "2024-01-02T00:00:00+0100",
                "cage": "C57_1",
                "rack": "R1",
                "position": "A1",
                "event": "CAGE_OFFLINE",
            },
            {
                "group": "C57",
                "day": 1,
                "hour": 1,
                "minute": 0,
                "relativeTime": 90000,
                "timestamp": "2024-01-02T01:00:00+0100",
                "cage": "C57_1",
                "rack": "R1",
                "position": "A1",
                "event": "CAGE_ONLINE",
            },
        ]
        data = make_event_csv_bytes(events)
        ev, _ = parse_event_csv(io.BytesIO(data), "t.csv")
        assert (ev[ev["event_type"] == "CAGE_OFFLINE"]["event_category"] == "cage_status").all()
        assert (ev[ev["event_type"] == "CAGE_ONLINE"]["event_category"] == "cage_status").all()

    def test_unknown_event_category_other(self):
        events = [{
            "group": "C57", "day": 0, "hour": 0, "minute": 0,
            "relativeTime": 0,
            "timestamp": "2024-01-01T07:00:00+0100",
            "cage": "C57_1", "rack": "R1", "position": "A1",
            "event": "WEIGH",
        }]
        data = make_event_csv_bytes(events)
        ev, _ = parse_event_csv(io.BytesIO(data), "t.csv")
        assert ev["event_category"].iloc[0] == "other"

    def test_missing_required_column(self):
        bad_df = pd.DataFrame({"group": ["C57"], "cage": ["C57_1"]})
        buf = io.StringIO()
        bad_df.to_csv(buf, index=False)
        ev, warns = parse_event_csv(io.BytesIO(buf.getvalue().encode()), "bad.csv")
        assert ev.empty
        assert warns

    def test_timestamp_parsed(self):
        data = make_event_csv_bytes()
        ev, _ = parse_event_csv(io.BytesIO(data), "t.csv")
        assert pd.api.types.is_datetime64_any_dtype(ev["timestamp"])

    def test_example_E_events(self):
        p = _EXAMPLES / "E_events.csv"
        if not p.exists():
            pytest.skip("example file not present")
        ev, warns = parse_event_csv(p, p.name)
        assert not ev.empty
        assert "event_type" in ev.columns

    def test_example_cohort2_events(self):
        p = _EXAMPLES / "cohort2_events.csv"
        if not p.exists():
            pytest.skip("example file not present")
        ev, warns = parse_event_csv(p, p.name)
        # cohort2_events.csv may be empty
        if not ev.empty:
            assert "group_id" in ev.columns


class TestGetUniqueEventTypes:
    def test_returns_sorted_list(self):
        data = make_event_csv_bytes()
        ev, _ = parse_event_csv(io.BytesIO(data), "t.csv")
        etypes = get_unique_event_types(ev)
        assert isinstance(etypes, list)
        assert sorted(etypes) == etypes

    def test_empty_df(self):
        assert get_unique_event_types(pd.DataFrame()) == []
