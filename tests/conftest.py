"""Shared fixtures for the DVC Behavioral Preprocessing test suite."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

_EXAMPLES = Path(__file__).parent.parent / "data" / "examples"


# ---------------------------------------------------------------------------
# Synthetic metric data
# ---------------------------------------------------------------------------

def make_metric_df(
    n_rows: int = 10,
    groups: list[str] | None = None,
    subjects_per_group: int = 3,
    start_ts: str = "2024-01-01T07:00:00+0100",
    bin_minutes: int = 1,
) -> pd.DataFrame:
    """
    Build a synthetic wide-format DVC metric DataFrame.
    """
    if groups is None:
        groups = ["C57"]

    import numpy as np
    from datetime import timedelta

    ts0 = pd.Timestamp(start_ts)
    rows = []
    for i in range(n_rows):
        ts = ts0 + timedelta(minutes=i * bin_minutes)
        row: dict = {
            "day": ts.day,
            "hour": ts.hour,
            "minute": ts.minute,
            "relativeTime": i * bin_minutes * 60,
        }
        for grp in groups:
            subj_ids = [f"{grp}_{k+1}" for k in range(subjects_per_group)]
            values = np.random.default_rng(i).uniform(0, 10, subjects_per_group)
            row[f"{grp}_TIMESTAMP"] = ts.isoformat()
            row[f"{grp}_AVG"] = float(values.mean())
            row[f"{grp}_SEM"] = float(values.std() / (subjects_per_group ** 0.5))
            row[f"{grp}_QRT"] = str(list(values.tolist()))
            row[f"{grp}_SAMPLES"] = subjects_per_group * 100
            for sid, val in zip(subj_ids, values):
                row[f"{grp}_{sid}"] = float(val)
        rows.append(row)

    return pd.DataFrame(rows)


def make_metric_csv_bytes(
    n_rows: int = 10,
    groups: list[str] | None = None,
    subjects_per_group: int = 3,
    start_ts: str = "2024-01-01T07:00:00+0100",
    bin_minutes: int = 1,
) -> bytes:
    df = make_metric_df(n_rows, groups, subjects_per_group, start_ts, bin_minutes)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Synthetic event data
# ---------------------------------------------------------------------------

def make_event_df(
    events: list[dict] | None = None,
) -> pd.DataFrame:
    """Build a synthetic event DataFrame."""
    if events is None:
        events = [
            {
                "group": "C57",
                "day": 2,
                "hour": 10,
                "minute": 0,
                "relativeTime": 2 * 24 * 3600,
                "timestamp": "2024-01-03T10:00:00+0100",
                "cage": "C57_1",
                "rack": "Rack1",
                "position": "A1",
                "event": "REMOVED",
            },
            {
                "group": "C57",
                "day": 2,
                "hour": 10,
                "minute": 30,
                "relativeTime": 2 * 24 * 3600 + 1800,
                "timestamp": "2024-01-03T10:30:00+0100",
                "cage": "C57_1",
                "rack": "Rack1",
                "position": "A1",
                "event": "INSERTED",
            },
        ]
    return pd.DataFrame(events)


def make_event_csv_bytes(events: list[dict] | None = None) -> bytes:
    df = make_event_df(events)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def metric_df():
    return make_metric_df()


@pytest.fixture
def metric_df_two_groups():
    return make_metric_df(groups=["C57", "3_S_C"], subjects_per_group=3)


@pytest.fixture
def event_df():
    return make_event_df()


@pytest.fixture
def examples_dir():
    return _EXAMPLES
