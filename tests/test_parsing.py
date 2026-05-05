"""Tests for src/dvc_behavior/parsing.py"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dvc_behavior.parsing import (
    combine_long_dfs,
    detect_group_prefixes,
    detect_native_bin_seconds,
    extract_subject_id,
    load_metric_csv,
    wide_to_long,
)
from tests.conftest import make_metric_df, make_metric_csv_bytes

_EXAMPLES = Path(__file__).parent.parent / "data" / "examples"


# ---------------------------------------------------------------------------
# Group prefix detection
# ---------------------------------------------------------------------------

class TestDetectGroupPrefixes:
    def test_single_group(self):
        cols = ["day", "hour", "C57_TIMESTAMP", "C57_AVG", "C57_SEM"]
        assert detect_group_prefixes(cols) == ["C57"]

    def test_multi_group(self):
        cols = [
            "day",
            "C57_TIMESTAMP",
            "C57_AVG",
            "3_S_C_TIMESTAMP",
            "3_S_C_AVG",
            "70Q_WT_TIMESTAMP",
        ]
        prefixes = detect_group_prefixes(cols)
        assert "C57" in prefixes
        assert "3_S_C" in prefixes
        assert "70Q_WT" in prefixes

    def test_no_timestamp_cols(self):
        cols = ["day", "hour", "minute"]
        assert detect_group_prefixes(cols) == []

    def test_prefix_with_underscores(self):
        cols = ["3_S_C_TIMESTAMP"]
        assert detect_group_prefixes(cols) == ["3_S_C"]


# ---------------------------------------------------------------------------
# Subject ID extraction
# ---------------------------------------------------------------------------

class TestExtractSubjectId:
    def test_simple(self):
        assert extract_subject_id("C57_C57_2", "C57") == "C57_2"

    def test_triple_underscore_group(self):
        assert extract_subject_id("3_S_C_3_S_C_9", "3_S_C") == "3_S_C_9"

    def test_compound_group_no_separator(self):
        assert extract_subject_id("70Q_WT_70Q_WT8", "70Q_WT") == "70Q_WT8"

    def test_no_prefix_match(self):
        # Falls back to original column name
        assert extract_subject_id("UNKNOWN_COL", "C57") == "UNKNOWN_COL"


# ---------------------------------------------------------------------------
# Native bin detection
# ---------------------------------------------------------------------------

class TestDetectNativeBin:
    def test_regular_1min(self):
        ts = pd.to_datetime(
            pd.date_range("2024-01-01 07:00", periods=20, freq="1min", tz="Europe/Paris")
        )
        median, irregular = detect_native_bin_seconds(pd.Series(ts))
        assert abs(median - 60.0) < 1.0
        assert not irregular

    def test_regular_5min(self):
        ts = pd.to_datetime(
            pd.date_range("2024-01-01 07:00", periods=20, freq="5min", tz="Europe/Paris")
        )
        median, irregular = detect_native_bin_seconds(pd.Series(ts))
        assert abs(median - 300.0) < 1.0
        assert not irregular

    def test_irregular(self):
        rng = np.random.default_rng(42)
        base = pd.Timestamp("2024-01-01 07:00:00", tz="UTC")
        deltas = rng.integers(30, 300, 30)  # wildly irregular
        ts = pd.Series([base + pd.Timedelta(seconds=int(d)) for d in np.cumsum(deltas)])
        _, irregular = detect_native_bin_seconds(ts)
        assert irregular

    def test_single_row(self):
        ts = pd.Series([pd.Timestamp("2024-01-01 07:00:00", tz="UTC")])
        median, irregular = detect_native_bin_seconds(ts)
        assert median is None
        assert not irregular


# ---------------------------------------------------------------------------
# wide_to_long
# ---------------------------------------------------------------------------

class TestWideToLong:
    def test_basic_conversion(self):
        df = make_metric_df(n_rows=5, groups=["C57"], subjects_per_group=3)
        long, warns = wide_to_long(df, "test.csv", "test_metric")
        assert not long.empty
        # 5 rows × 3 subjects = 15
        assert len(long) == 15

    def test_columns_present(self):
        df = make_metric_df(n_rows=3)
        long, _ = wide_to_long(df, "f.csv")
        required = {
            "source_file", "metric_name", "group_id", "subject_id",
            "source_column", "timestamp", "timestamp_utc",
            "value", "group_avg", "group_sem", "samples",
            "native_bin_seconds", "is_group_average",
        }
        assert required <= set(long.columns)

    def test_two_groups(self):
        df = make_metric_df(groups=["C57", "3_S_C"], subjects_per_group=2, n_rows=4)
        long, warns = wide_to_long(df, "f.csv")
        assert set(long["group_id"].unique()) == {"C57", "3_S_C"}
        # 2 groups × 2 subjects × 4 rows = 16
        assert len(long) == 16

    def test_subject_ids_stripped(self):
        df = make_metric_df(n_rows=2, groups=["C57"], subjects_per_group=2)
        long, _ = wide_to_long(df, "f.csv")
        assert "C57_1" in long["subject_id"].values
        assert "C57_2" in long["subject_id"].values
        # Raw column names should not appear as subject_id
        assert "C57_C57_1" not in long["subject_id"].values

    def test_no_timestamp_column_returns_empty(self):
        df = pd.DataFrame({"day": [0], "value": [1.0]})
        long, warns = wide_to_long(df, "bad.csv")
        assert long.empty
        assert warns

    def test_is_group_average_false(self):
        df = make_metric_df(n_rows=3)
        long, _ = wide_to_long(df, "f.csv")
        assert (long["is_group_average"] == False).all()  # noqa: E712

    def test_group_avg_attached(self):
        df = make_metric_df(n_rows=5, groups=["C57"], subjects_per_group=3)
        long, _ = wide_to_long(df, "f.csv")
        # group_avg should be the same for all subjects at the same timestamp
        for _, grp in long.groupby(["group_id", "timestamp"]):
            assert grp["group_avg"].nunique() == 1

    def test_metric_name_inferred(self):
        df = make_metric_df(n_rows=2)
        long, _ = wide_to_long(df, "MyExperiment_animal_loc_index_smoothed.csv")
        # _loc_index_smoothed → MyExperiment_animal
        assert long["metric_name"].iloc[0] == "MyExperiment_animal"


# ---------------------------------------------------------------------------
# load_metric_csv
# ---------------------------------------------------------------------------

class TestLoadMetricCsv:
    def test_from_bytes(self):
        data = make_metric_csv_bytes(n_rows=5)
        long, warns = load_metric_csv(io.BytesIO(data), source_file="synth.csv")
        assert not long.empty

    def test_bad_data_returns_empty(self):
        bad = b"not,a,dvc,file\n1,2,3,4\n"
        long, warns = load_metric_csv(io.BytesIO(bad), source_file="bad.csv")
        assert long.empty
        assert warns

    def test_example_cohort2(self, examples_dir):
        p = examples_dir / "cohort2_animal_loc__index_smoothed.csv"
        if not p.exists():
            pytest.skip("example file not present")
        long, warns = load_metric_csv(p, source_file=p.name)
        assert not long.empty
        assert "C57" in long["group_id"].values

    def test_example_E_animal(self, examples_dir):
        p = examples_dir / "E_animal_loc__index_smoothed.csv"
        if not p.exists():
            pytest.skip("example file not present")
        long, warns = load_metric_csv(p, source_file=p.name)
        assert not long.empty

    def test_example_partnerC(self, examples_dir):
        p = examples_dir / "PartnerC_cohort1_animal_loc_index_smoothed.csv"
        if not p.exists():
            pytest.skip("example file not present")
        long, warns = load_metric_csv(p, source_file=p.name)
        assert not long.empty
        # Should detect 70Q_WT group
        assert "70Q_WT" in long["group_id"].values


# ---------------------------------------------------------------------------
# combine_long_dfs
# ---------------------------------------------------------------------------

class TestCombineLongDfs:
    def test_empty_list(self):
        result = combine_long_dfs([])
        assert result.empty

    def test_single(self):
        df = make_metric_df(n_rows=3)
        long, _ = wide_to_long(df, "f.csv")
        result = combine_long_dfs([long])
        assert len(result) == len(long)

    def test_multiple(self):
        df1 = make_metric_df(n_rows=3)
        df2 = make_metric_df(n_rows=5)
        l1, _ = wide_to_long(df1, "f1.csv")
        l2, _ = wide_to_long(df2, "f2.csv")
        result = combine_long_dfs([l1, l2])
        assert len(result) == len(l1) + len(l2)
