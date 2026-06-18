"""
Light/dark cycle annotation.

ZT0 = lights-on time.  ZT12 = lights-off time (for a 12 h cycle).
Dark phase crosses midnight in the default 07:00–19:00 schedule.
"""

from __future__ import annotations

import pandas as pd
from zoneinfo import ZoneInfo

__all__ = [
    "localise_timestamps",
    "annotate_light_dark",
    "add_light_dark_columns",
]


def _parse_time_str(t: str) -> tuple[int, int]:
    """Parse 'HH:MM' → (hour, minute)."""
    h, m = t.strip().split(":")
    return int(h), int(m)


def _decimal_hour(hour: int, minute: int, second: int = 0) -> float:
    return hour + minute / 60.0 + second / 3600.0


def localise_timestamps(
    ts_utc_series: pd.Series,
    timezone: str = "Europe/Paris",
) -> pd.Series:
    """Convert a UTC-aware timestamp series to the given timezone."""
    tz = ZoneInfo(timezone)
    return ts_utc_series.dt.tz_convert(tz)


def annotate_light_dark(
    local_times: pd.Series,
    light_on: str = "07:00",
    light_off: str = "19:00",
) -> tuple[pd.Series, pd.Series]:
    """
    Annotate a series of local datetimes with light/dark phase and ZT hour.

    Returns
    -------
    (light_dark_phase, zeitgeber_time_hours)
        phase is 'light' or 'dark'; ZT is float hours since lights-on.
    """
    on_h, on_m = _parse_time_str(light_on)
    off_h, off_m = _parse_time_str(light_off)
    on_dec = _decimal_hour(on_h, on_m)
    off_dec = _decimal_hour(off_h, off_m)

    phases: list[str | None] = []
    zts: list[float | None] = []

    for dt in local_times:
        if dt is pd.NaT or dt is None:
            phases.append(None)
            zts.append(None)
            continue

        t = _decimal_hour(dt.hour, dt.minute, dt.second)

        # Light period: on_dec → off_dec (normal; doesn't cross midnight)
        if on_dec < off_dec:
            is_light = on_dec <= t < off_dec
        else:
            # Light period crosses midnight (unusual reverse schedule)
            is_light = t >= on_dec or t < off_dec

        phases.append("light" if is_light else "dark")
        # ZT = hours elapsed since lights on, wrapping at 24 h
        zt = (t - on_dec) % 24.0
        zts.append(round(zt, 4))

    return pd.Series(phases, index=local_times.index), pd.Series(zts, index=local_times.index)


def add_light_dark_columns(
    df: pd.DataFrame,
    timezone: str = "Europe/Paris",
    light_on: str = "07:00",
    light_off: str = "19:00",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Add timestamp_local, light_dark_phase, and zeitgeber_time_hours to df.
    Requires df["timestamp_utc"] to be a tz-aware datetime series.

    Returns (modified_df, warnings).
    """
    warns: list[str] = []
    df = df.copy()

    if "timestamp_utc" not in df.columns or df["timestamp_utc"].isna().all():
        warns.append("No valid UTC timestamps found; skipping light/dark annotation.")
        df["timestamp_local"] = pd.NaT
        df["light_dark_phase"] = None
        df["zeitgeber_time_hours"] = None
        return df, warns

    try:
        local = localise_timestamps(df["timestamp_utc"], timezone)
        df["timestamp_local"] = local
    except Exception as exc:
        warns.append(f"Could not localise timestamps to '{timezone}': {exc}")
        df["timestamp_local"] = df["timestamp_utc"]
        local = df["timestamp_local"]

    phases, zts = annotate_light_dark(local, light_on, light_off)
    df["light_dark_phase"] = phases.values
    df["zeitgeber_time_hours"] = zts.values

    return df, warns
