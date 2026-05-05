"""Tests for src/dvc_behavior/light_dark.py"""

from __future__ import annotations

import pandas as pd
from zoneinfo import ZoneInfo

from dvc_behavior.light_dark import annotate_light_dark, add_light_dark_columns


_PARIS = ZoneInfo("Europe/Paris")


def _local(ts_str: str, tz: str = "Europe/Paris") -> pd.Timestamp:
    tz_obj = ZoneInfo(tz)
    return pd.Timestamp(ts_str).tz_localize(tz_obj)


class TestAnnotateLightDark:
    """Default cycle: lights ON 07:00, lights OFF 19:00 (Paris)."""

    def _annotate(self, local_times):
        series = pd.Series(local_times)
        phases, zts = annotate_light_dark(series, "07:00", "19:00")
        return phases.tolist(), zts.tolist()

    def test_midday_is_light(self):
        ts = _local("2024-01-01 12:00:00")
        phases, _ = self._annotate([ts])
        assert phases[0] == "light"

    def test_midnight_is_dark(self):
        ts = _local("2024-01-01 00:00:00")
        phases, _ = self._annotate([ts])
        assert phases[0] == "dark"

    def test_early_morning_before_lights_on(self):
        ts = _local("2024-01-01 06:59:00")
        phases, _ = self._annotate([ts])
        assert phases[0] == "dark"

    def test_at_lights_on_is_light(self):
        ts = _local("2024-01-01 07:00:00")
        phases, _ = self._annotate([ts])
        assert phases[0] == "light"

    def test_at_lights_off_is_dark(self):
        ts = _local("2024-01-01 19:00:00")
        phases, _ = self._annotate([ts])
        assert phases[0] == "dark"

    def test_late_evening_is_dark(self):
        ts = _local("2024-01-01 22:00:00")
        phases, _ = self._annotate([ts])
        assert phases[0] == "dark"

    def test_dark_phase_crosses_midnight(self):
        """19:00 tonight → 07:00 tomorrow must all be dark."""
        dark_times = [
            _local("2024-01-01 19:30:00"),
            _local("2024-01-01 23:00:00"),
            _local("2024-01-02 00:00:00"),
            _local("2024-01-02 03:00:00"),
            _local("2024-01-02 06:30:00"),
        ]
        phases, _ = self._annotate(dark_times)
        assert all(p == "dark" for p in phases), f"Expected all dark, got: {phases}"

    def test_zt0_at_lights_on(self):
        ts = _local("2024-01-01 07:00:00")
        _, zts = self._annotate([ts])
        assert abs(zts[0] - 0.0) < 0.01

    def test_zt12_at_lights_off(self):
        ts = _local("2024-01-01 19:00:00")
        _, zts = self._annotate([ts])
        assert abs(zts[0] - 12.0) < 0.01

    def test_nat_returns_none(self):
        phases, zts = self._annotate([pd.NaT])
        assert phases[0] is None
        assert zts[0] is None

    def test_zt_wraps_at_24(self):
        """ZT for a time just before lights on should be close to 24, not negative."""
        ts = _local("2024-01-01 06:59:00")
        _, zts = self._annotate([ts])
        assert 23.0 <= zts[0] < 24.0

    def test_mixed_phases(self):
        times = [
            _local("2024-01-01 08:00:00"),   # light
            _local("2024-01-01 20:00:00"),   # dark
            _local("2024-01-01 12:00:00"),   # light
            _local("2024-01-02 02:00:00"),   # dark
        ]
        phases, _ = self._annotate(times)
        assert phases == ["light", "dark", "light", "dark"]


class TestAddLightDarkColumns:
    def _make_df_with_utc(self, utc_timestamps):
        ts = pd.to_datetime(utc_timestamps, utc=True)
        return pd.DataFrame({"timestamp_utc": ts, "value": range(len(ts))})

    def test_adds_columns(self):
        df = self._make_df_with_utc(["2024-01-01 11:00:00+00:00", "2024-01-01 23:00:00+00:00"])
        out, warns = add_light_dark_columns(df, "Europe/Paris", "07:00", "19:00")
        assert "timestamp_local" in out.columns
        assert "light_dark_phase" in out.columns
        assert "zeitgeber_time_hours" in out.columns

    def test_no_utc_column(self):
        df = pd.DataFrame({"value": [1, 2]})
        out, warns = add_light_dark_columns(df)
        assert warns
        assert "light_dark_phase" in out.columns

    def test_paris_noon_utc_is_light(self):
        # Paris in January is UTC+1; 11:00 UTC = 12:00 Paris → light
        df = self._make_df_with_utc(["2024-01-01 11:00:00+00:00"])
        out, _ = add_light_dark_columns(df, "Europe/Paris", "07:00", "19:00")
        assert out["light_dark_phase"].iloc[0] == "light"

    def test_paris_midnight_utc_is_dark(self):
        # 23:00 UTC = 00:00 Paris → dark
        df = self._make_df_with_utc(["2024-01-01 23:00:00+00:00"])
        out, _ = add_light_dark_columns(df, "Europe/Paris", "07:00", "19:00")
        assert out["light_dark_phase"].iloc[0] == "dark"
