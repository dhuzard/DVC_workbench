"""
Metadata construction and validation.

Handles study-level, subject-level, and group-level metadata tables.
"""

from __future__ import annotations


import pandas as pd

from .config import (
    GROUP_METADATA_COLUMNS,
    STUDY_METADATA_DEFAULTS,
    SUBJECT_METADATA_COLUMNS,
)

__all__ = [
    "build_subject_metadata_template",
    "build_group_metadata_template",
    "build_study_metadata",
    "validate_subject_metadata",
    "compute_metadata_quality",
    "merge_subject_metadata",
    "merge_group_metadata",
]

# Proportion of important subject fields that must be filled for "complete" metadata
_IMPORTANT_SUBJECT_FIELDS = [
    "animal_id",
    "cage_id",
    "sex",
    "treatment_group",
    "genotype",
    "cohort",
]


# ---------------------------------------------------------------------------
# Subject metadata
# ---------------------------------------------------------------------------


def build_subject_metadata_template(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a blank subject metadata table from a parsed long_df.
    """
    if long_df.empty:
        return pd.DataFrame(columns=SUBJECT_METADATA_COLUMNS)

    pairs = (
        long_df[["group_id", "subject_id"]]
        .drop_duplicates()
        .sort_values(["group_id", "subject_id"])
        .reset_index(drop=True)
    )

    meta = pd.DataFrame(index=pairs.index, columns=SUBJECT_METADATA_COLUMNS)
    meta["subject_id"] = pairs["subject_id"].values
    meta["group_id_detected"] = pairs["group_id"].values

    # Fill remaining with empty string so data_editor shows blank cells, not NaN
    for col in SUBJECT_METADATA_COLUMNS:
        if col not in ("subject_id", "group_id_detected"):
            meta[col] = ""

    return meta


def build_group_metadata_template(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=GROUP_METADATA_COLUMNS)

    groups = sorted(long_df["group_id"].dropna().unique().tolist())
    meta = pd.DataFrame({"group_id": groups})
    for col in GROUP_METADATA_COLUMNS:
        if col != "group_id":
            meta[col] = ""
    # Default group_label = group_id
    meta["group_label"] = meta["group_id"]
    if "group_color" in meta.columns:
        palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
        meta["group_color"] = [palette[i % len(palette)] for i in range(len(meta))]
    return meta


def build_study_metadata(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Return study metadata dict with defaults, optionally merged with user overrides."""
    meta = STUDY_METADATA_DEFAULTS.copy()
    if overrides:
        meta.update(overrides)
    return meta


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------


def validate_subject_metadata(
    subject_meta: pd.DataFrame,
    long_df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Validate subject metadata against detected subjects.
    Returns (errors, warnings).
    """
    errors: list[str] = []
    warnings: list[str] = []

    if subject_meta.empty:
        warnings.append("No subject metadata defined.")
        return errors, warnings

    if long_df.empty:
        return errors, warnings

    detected = set(long_df["subject_id"].dropna().unique())
    meta_ids = set(subject_meta["subject_id"].dropna().astype(str))

    missing_from_meta = detected - meta_ids
    if missing_from_meta:
        warnings.append(f"These subjects have no metadata row: {sorted(missing_from_meta)}")

    extra_in_meta = meta_ids - detected
    if extra_in_meta:
        warnings.append(f"Metadata contains subjects not found in data: {sorted(extra_in_meta)}")

    # Duplicate IDs
    dupes = subject_meta["subject_id"][subject_meta["subject_id"].duplicated()]
    if not dupes.empty:
        errors.append(f"Duplicate subject_id in metadata: {dupes.tolist()}")

    return errors, warnings


def compute_metadata_quality(row: pd.Series) -> tuple[bool, str, float]:
    """
    Returns (metadata_complete, metadata_warning, metadata_quality_score).
    """
    filled = sum(
        1
        for f in _IMPORTANT_SUBJECT_FIELDS
        if f in row.index and str(row[f]).strip() not in ("", "nan", "NaN", "None")
    )
    score = filled / len(_IMPORTANT_SUBJECT_FIELDS)
    complete = score >= 0.8
    warning = (
        "" if complete else f"Only {filled}/{len(_IMPORTANT_SUBJECT_FIELDS)} key fields filled"
    )
    return complete, warning, round(score, 2)


# ---------------------------------------------------------------------------
# Metadata merging into long_df
# ---------------------------------------------------------------------------


def merge_subject_metadata(
    long_df: pd.DataFrame,
    subject_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join subject metadata into long_df on subject_id."""
    if subject_meta.empty or long_df.empty:
        return long_df

    cols_to_merge = [c for c in subject_meta.columns if c != "group_id_detected"]
    meta_slim = subject_meta[cols_to_merge].copy()

    # Compute quality columns
    quality_rows = meta_slim.apply(compute_metadata_quality, axis=1)
    meta_slim["metadata_complete"] = [r[0] for r in quality_rows]
    meta_slim["metadata_warning"] = [r[1] for r in quality_rows]
    meta_slim["metadata_quality_score"] = [r[2] for r in quality_rows]

    # Avoid column collision — drop columns already in long_df except subject_id
    drop_cols = [c for c in meta_slim.columns if c in long_df.columns and c != "subject_id"]
    meta_slim = meta_slim.drop(columns=drop_cols, errors="ignore")

    merged = long_df.merge(meta_slim, on="subject_id", how="left")
    return merged


def merge_group_metadata(
    long_df: pd.DataFrame,
    group_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join group metadata into long_df on group_id."""
    if group_meta.empty or long_df.empty:
        return long_df

    drop_cols = [c for c in group_meta.columns if c in long_df.columns and c != "group_id"]
    meta_slim = group_meta.drop(columns=drop_cols, errors="ignore")
    return long_df.merge(meta_slim, on="group_id", how="left")
