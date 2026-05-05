"""
Temporal alignment to an event or manual timestamp.

For each subject, find the alignment event and compute:
  - alignment_event_type
  - alignment_timestamp
  - time_from_event_seconds
  - time_from_event_hours
  - experimental_day  (floor(time_from_event_hours / 24))
"""

from __future__ import annotations

import math

import pandas as pd


def _find_alignment_timestamp(
    subject_id: str,
    group_id: str,
    event_df: pd.DataFrame,
    event_type: str,
    scope: str,  # "subject" | "group"
) -> pd.Timestamp | None:
    """
    Find the alignment event timestamp for a given subject.
    Returns the first matching event timestamp, or None.
    """
    if event_df.empty:
        return None

    mask = event_df["event_type"] == event_type

    if scope == "subject":
        subj_mask = event_df["subject_id"] == subject_id
        candidate = event_df.loc[mask & subj_mask, "timestamp_utc"]
        if candidate.empty:
            # fallback: group-level
            grp_mask = event_df["group_id"] == group_id
            candidate = event_df.loc[mask & grp_mask, "timestamp_utc"]
    else:
        grp_mask = event_df["group_id"] == group_id
        candidate = event_df.loc[mask & grp_mask, "timestamp_utc"]

    if candidate.empty or candidate.isna().all():
        return None

    ts = candidate.dropna().iloc[0]
    return ts if ts is not pd.NaT else None


def align_to_event(
    df: pd.DataFrame,
    event_df: pd.DataFrame,
    event_type: str,
    scope: str = "subject",
    fallback_timestamp: pd.Timestamp | str | None = None,
    alignment_label: str = "J0",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Add alignment columns to df.

    Parameters
    ----------
    df            : long_df (must have subject_id, group_id, timestamp_utc)
    event_df      : clean event table
    event_type    : event label used for alignment (e.g. "REMOVED", "SURGERY")
    scope         : "subject" → per-subject lookup; "group" → per-group
    fallback_timestamp : used when no event is found for a subject
    alignment_label : label stored in alignment_event_type column

    Returns
    -------
    (annotated_df, warnings)
    """
    warns: list[str] = []
    df = df.copy()

    if "timestamp_utc" not in df.columns:
        warns.append("timestamp_utc column missing; alignment skipped.")
        _add_empty_alignment_cols(df)
        return df, warns

    # Parse fallback timestamp if provided as string
    fallback_ts: pd.Timestamp | None = None
    if fallback_timestamp is not None:
        try:
            fallback_ts = pd.to_datetime(fallback_timestamp, utc=True)
        except Exception:
            warns.append(f"Could not parse fallback timestamp: {fallback_timestamp!r}")

    # Build per-(subject, group) alignment map
    pairs = df[["subject_id", "group_id"]].drop_duplicates()
    align_map: dict[tuple[str, str], pd.Timestamp | None] = {}

    for _, row in pairs.iterrows():
        sid = str(row["subject_id"])
        gid = str(row["group_id"])
        ts = _find_alignment_timestamp(sid, gid, event_df, event_type, scope)
        if ts is None:
            ts = fallback_ts
            if ts is None:
                warns.append(
                    f"No alignment event '{event_type}' found for subject '{sid}' "
                    f"(group '{gid}') and no fallback provided."
                )
        align_map[(sid, gid)] = ts

    # Apply alignment
    alignment_timestamps = df.apply(
        lambda r: align_map.get((str(r["subject_id"]), str(r["group_id"]))), axis=1
    )
    df["alignment_event_type"] = alignment_label
    df["alignment_timestamp"] = alignment_timestamps

    n_no_align = alignment_timestamps.isna().sum()
    if n_no_align:
        warns.append(
            f"{n_no_align} rows have no alignment timestamp; "
            "time_from_event columns will be NaN."
        )

    # Compute time deltas
    ref_utc = pd.to_datetime(alignment_timestamps, utc=True, errors="coerce")
    row_utc = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    time_from_event_s = (row_utc - ref_utc).dt.total_seconds()
    df["time_from_event_seconds"] = time_from_event_s
    df["time_from_event_hours"] = time_from_event_s / 3600.0
    df["experimental_day"] = (time_from_event_s / 86400.0).apply(
        lambda x: math.floor(x) if x == x else None
    )

    return df, warns


def _add_empty_alignment_cols(df: pd.DataFrame) -> None:
    for col in (
        "alignment_event_type",
        "alignment_timestamp",
        "time_from_event_seconds",
        "time_from_event_hours",
        "experimental_day",
    ):
        df[col] = None


def align_to_manual_timestamp(
    df: pd.DataFrame,
    timestamp: pd.Timestamp | str,
    alignment_label: str = "J0",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Align all rows to a single global manual timestamp.
    """
    warns: list[str] = []
    try:
        ts = pd.to_datetime(timestamp, utc=True)
    except Exception as exc:
        warns.append(f"Cannot parse manual alignment timestamp: {exc}")
        _add_empty_alignment_cols(df)
        return df, warns

    df = df.copy()
    dummy_event_df = pd.DataFrame()
    return align_to_event(
        df,
        dummy_event_df,
        event_type="__manual__",
        scope="subject",
        fallback_timestamp=ts,
        alignment_label=alignment_label,
    )
