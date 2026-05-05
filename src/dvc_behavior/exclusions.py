"""
Exclusion window logic.

Rules are applied around events: a window before and/or after each event
marks rows in the long_df as excluded or flagged.

Matching priority:
  1. Facility-level: explicit facility/global/all events match every subject
  2. Subject-level: event.subject_id matches row.subject_id
  3. Group-level fallback: event.group_id matches row.group_id

Overlapping windows per subject are merged before marking.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Window computation
# ---------------------------------------------------------------------------

def compute_exclusion_windows(
    event_df: pd.DataFrame,
    exclusion_rules: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """
    Generate a table of exclusion windows from events and rules.

    Returns a DataFrame with columns:
        subject_id, group_id, event_scope, event_type, event_timestamp,
        paired_event_timestamp, exclusion_start, exclusion_end, exclusion_reason,
        exclude, flag
    """
    rows: list[dict[str, Any]] = []

    if event_df.empty:
        return _empty_window_df()

    paired_indices = _find_cage_change_pairs(event_df, exclusion_rules)
    paired_event_indices: set[Any] = set()

    for removed_idx, inserted_idx in paired_indices:
        removed = event_df.loc[removed_idx]
        inserted = event_df.loc[inserted_idx]
        cage_rule = exclusion_rules.get("CAGE_CHANGE", {})
        removed_rule = exclusion_rules.get("REMOVED", {})
        inserted_rule = exclusion_rules.get("INSERTED", {})

        removed_ts = _event_timestamp(removed)
        inserted_ts = _event_timestamp(inserted)
        if removed_ts is None or inserted_ts is None:
            continue

        before_h = float(cage_rule.get("before_hours", removed_rule.get("before_hours", 0.0)))
        after_h = float(cage_rule.get("after_hours", inserted_rule.get("after_hours", 0.0)))
        do_excl = bool(
            cage_rule.get("exclude", False)
            or removed_rule.get("exclude", False)
            or inserted_rule.get("exclude", False)
        )
        do_flag = bool(
            cage_rule.get("flag", False)
            or removed_rule.get("flag", False)
            or inserted_rule.get("flag", False)
        )
        if not do_excl and not do_flag:
            continue

        paired_event_indices.update({removed_idx, inserted_idx})
        scope = _infer_event_scope(removed, removed_rule)
        rows.append(
            {
                "subject_id": _clean_identifier(removed.get("subject_id")),
                "group_id": _clean_identifier(removed.get("group_id")),
                "event_scope": scope,
                "event_type": "CAGE_CHANGE",
                "event_timestamp": removed_ts,
                "paired_event_timestamp": inserted_ts,
                "exclusion_start": removed_ts - pd.Timedelta(hours=before_h),
                "exclusion_end": inserted_ts + pd.Timedelta(hours=after_h),
                "exclusion_reason": (
                    f"CAGE_CHANGE REMOVED->INSERTED "
                    f"(before REMOVED {before_h}h/after INSERTED {after_h}h)"
                ),
                "exclude": do_excl,
                "flag": do_flag,
            }
        )

    for _, ev in event_df.iterrows():
        if ev.name in paired_event_indices:
            continue

        etype = str(ev.get("event_type", "")).strip().upper()
        rule = exclusion_rules.get(etype)
        if rule is None:
            continue
        if not rule.get("exclude") and not rule.get("flag"):
            continue

        ts = _event_timestamp(ev)
        if ts is None:
            continue

        before_h = float(rule.get("before_hours", 0.0))
        after_h = float(rule.get("after_hours", 0.0))
        scope = _infer_event_scope(ev, rule)
        end_ts = ev.get("timestamp_end_utc") if "timestamp_end_utc" in ev.index else None
        if end_ts is None or end_ts is pd.NaT or pd.isna(end_ts):
            end_ts = ev.get("timestamp_end") if "timestamp_end" in ev.index else None
        if end_ts is not None and end_ts is not pd.NaT and not pd.isna(end_ts):
            window_end = pd.Timestamp(end_ts) + pd.Timedelta(hours=after_h)
        else:
            window_end = ts + pd.Timedelta(hours=after_h)

        rows.append(
            {
                "subject_id": _clean_identifier(ev.get("subject_id")),
                "group_id": _clean_identifier(ev.get("group_id")),
                "event_scope": scope,
                "event_type": etype,
                "event_timestamp": ts,
                "exclusion_start": ts - pd.Timedelta(hours=before_h),
                "exclusion_end": window_end,
                "exclusion_reason": f"{etype} (before {before_h}h/after {after_h}h)",
                "exclude": bool(rule.get("exclude", False)),
                "flag": bool(rule.get("flag", False)),
            }
        )

    if not rows:
        return _empty_window_df()

    return pd.DataFrame(rows)


def _empty_window_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "subject_id", "group_id", "event_scope", "event_type", "event_timestamp",
            "paired_event_timestamp", "exclusion_start", "exclusion_end", "exclusion_reason",
            "exclude", "flag",
        ]
    )


def _clean_identifier(value: Any) -> str | None:
    if value is None or value is pd.NA or pd.isna(value):
        return None
    cleaned = str(value).strip()
    if not cleaned or cleaned.lower() in {"nan", "none", "nat", "<na>"}:
        return None
    return cleaned


def _event_timestamp(ev: pd.Series) -> pd.Timestamp | None:
    ts = ev.get("timestamp_utc")
    if ts is None or ts is pd.NaT or pd.isna(ts):
        ts = ev.get("timestamp")
    if ts is None or ts is pd.NaT or pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _infer_event_scope(ev: pd.Series, rule: dict[str, Any]) -> str:
    raw_scope = rule.get("event_scope", rule.get("scope", ev.get("event_scope", None)))
    if raw_scope is not None and not pd.isna(raw_scope):
        scope = str(raw_scope).strip().lower()
        if scope in {"facility", "global", "all"}:
            return "facility"
        if scope in {"group", "subject"}:
            return scope

    if _clean_identifier(ev.get("subject_id")):
        return "subject"
    if _clean_identifier(ev.get("group_id")):
        return "group"
    return "facility"


def _find_cage_change_pairs(
    event_df: pd.DataFrame,
    exclusion_rules: dict[str, dict[str, Any]],
) -> list[tuple[Any, Any]]:
    if (
        "CAGE_CHANGE" not in exclusion_rules
        and ("REMOVED" not in exclusion_rules or "INSERTED" not in exclusion_rules)
    ):
        return []
    if "event_type" not in event_df.columns:
        return []

    events = event_df.copy()
    events["_event_ts"] = events.apply(_event_timestamp, axis=1)
    events = events[events["_event_ts"].notna()].sort_values("_event_ts")

    removed = events[events["event_type"].astype(str).str.upper() == "REMOVED"]
    inserted = events[events["event_type"].astype(str).str.upper() == "INSERTED"]
    if removed.empty or inserted.empty:
        return []

    pairs: list[tuple[Any, Any]] = []
    used_inserted: set[Any] = set()

    cage_rule = exclusion_rules.get("CAGE_CHANGE", {})
    max_gap_hours = float(cage_rule.get("max_gap_hours", 6.0))

    for rem_idx, rem in removed.iterrows():
        candidates = inserted[
            (inserted["_event_ts"] >= rem["_event_ts"])
            & ~inserted.index.isin(used_inserted)
            & inserted.apply(lambda ins: _same_cage_change_target(rem, ins), axis=1)
        ]
        candidates = candidates[
            (candidates["_event_ts"] - rem["_event_ts"]).dt.total_seconds()
            <= max_gap_hours * 3600.0
        ]
        if candidates.empty:
            continue
        ins_idx = candidates.index[0]
        used_inserted.add(ins_idx)
        pairs.append((rem_idx, ins_idx))

    return pairs


def detect_cage_change_pairs(
    event_df: pd.DataFrame,
    max_gap_hours: float = 6.0,
) -> pd.DataFrame:
    """Public cage-change detector returning REMOVED/INSERTED pair details."""
    rules = {"CAGE_CHANGE": {"max_gap_hours": max_gap_hours}}
    pairs = _find_cage_change_pairs(event_df, rules)
    rows: list[dict[str, Any]] = []
    for removed_idx, inserted_idx in pairs:
        removed = event_df.loc[removed_idx]
        inserted = event_df.loc[inserted_idx]
        removed_ts = _event_timestamp(removed)
        inserted_ts = _event_timestamp(inserted)
        if removed_ts is None or inserted_ts is None:
            continue
        rows.append(
            {
                "subject_id": _clean_identifier(removed.get("subject_id")),
                "group_id": _clean_identifier(removed.get("group_id")),
                "removed_timestamp": removed_ts,
                "inserted_timestamp": inserted_ts,
                "gap_hours": (inserted_ts - removed_ts).total_seconds() / 3600.0,
                "removed_event_index": removed_idx,
                "inserted_event_index": inserted_idx,
            }
        )
    return pd.DataFrame(rows)


def _same_cage_change_target(removed: pd.Series, inserted: pd.Series) -> bool:
    rem_subject = _clean_identifier(removed.get("subject_id"))
    ins_subject = _clean_identifier(inserted.get("subject_id"))
    if rem_subject and ins_subject:
        return rem_subject == ins_subject

    rem_group = _clean_identifier(removed.get("group_id"))
    ins_group = _clean_identifier(inserted.get("group_id"))
    if rem_group and ins_group and rem_group != ins_group:
        return False

    for col in ("rack", "position"):
        rem_value = _clean_identifier(removed.get(col))
        ins_value = _clean_identifier(inserted.get(col))
        if rem_value and ins_value and rem_value != ins_value:
            return False

    return bool(rem_group or ins_group or rem_subject or ins_subject)


# ---------------------------------------------------------------------------
# Window merging (per subject)
# ---------------------------------------------------------------------------

def _merge_intervals(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp, str, bool, bool]]
) -> list[tuple[pd.Timestamp, pd.Timestamp, str, bool, bool]]:
    """
    Merge overlapping/adjacent intervals.
    Each item: (start, end, reason, exclude, flag)
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[pd.Timestamp, pd.Timestamp, str, bool, bool]] = [intervals[0]]

    for start, end, reason, excl, flag in intervals[1:]:
        prev_start, prev_end, prev_reason, prev_excl, prev_flag = merged[-1]
        if start <= prev_end:
            new_end = max(prev_end, end)
            new_reason = prev_reason if prev_reason == reason else f"{prev_reason}; {reason}"
            merged[-1] = (prev_start, new_end, new_reason, prev_excl or excl, prev_flag or flag)
        else:
            merged.append((start, end, reason, excl, flag))

    return merged


# ---------------------------------------------------------------------------
# Apply exclusions to long_df
# ---------------------------------------------------------------------------

def apply_exclusions(
    long_df: pd.DataFrame,
    window_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mark rows in long_df as excluded or flagged.

    Returns
    -------
    (long_df_annotated, exclusion_log)
    """
    df = long_df.copy()
    df["is_excluded"] = False
    df["exclusion_reason"] = ""
    df["flag_reason"] = ""

    if window_df.empty or "timestamp_utc" not in df.columns:
        return df, pd.DataFrame()

    log_rows: list[dict[str, Any]] = []

    for _, win in window_df.iterrows():
        subj = _clean_identifier(win.get("subject_id"))
        grp = _clean_identifier(win.get("group_id"))
        scope = str(win.get("event_scope", "") or "").strip().lower()
        w_start = win["exclusion_start"]
        w_end = win["exclusion_end"]
        reason = win["exclusion_reason"]
        do_excl = win["exclude"]
        do_flag = win["flag"]

        # Build subject mask: prefer exact subject_id match; fall back to group_id.
        # Facility-scoped events intentionally apply to all rows in the time window.
        if scope in {"facility", "global", "all"}:
            subj_mask = pd.Series(True, index=df.index)
        elif subj:
            subj_mask = df["subject_id"] == str(subj)
            if subj_mask.sum() == 0 and grp:
                subj_mask = df["group_id"] == str(grp)
        elif grp:
            subj_mask = df["group_id"] == str(grp)
        else:
            subj_mask = pd.Series(True, index=df.index)

        # Time mask
        time_mask = (df["timestamp_utc"] >= w_start) & (df["timestamp_utc"] <= w_end)
        row_mask = subj_mask & time_mask

        if do_excl:
            df.loc[row_mask, "is_excluded"] = True
            df.loc[row_mask, "exclusion_reason"] = df.loc[row_mask, "exclusion_reason"].apply(
                lambda x: (x + "; " + reason) if x else reason
            )

        if do_flag and not do_excl:
            df.loc[row_mask, "flag_reason"] = df.loc[row_mask, "flag_reason"].apply(
                lambda x: (x + "; " + reason) if x else reason
            )

        log_rows.append(
            {
                "subject_id": subj,
                "group_id": grp,
                "event_type": win["event_type"],
                "event_timestamp": win["event_timestamp"],
                "exclusion_start": w_start,
                "exclusion_end": w_end,
                "exclusion_reason": reason,
                "n_rows_excluded": int(row_mask.sum()) if do_excl else 0,
                "n_rows_flagged": int(row_mask.sum()) if do_flag else 0,
            }
        )

    log_df = pd.DataFrame(log_rows)
    return df, log_df
