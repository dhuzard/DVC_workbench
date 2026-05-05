"""Optional warn-only DataFrame validation schemas.

Pandera is intentionally optional. Importing this module never requires it;
validation helpers return warning strings instead of raising.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


_LONG_REQUIRED = {
    "source_file",
    "metric_name",
    "group_id",
    "subject_id",
    "timestamp_utc",
    "value",
}
_EVENT_REQUIRED = {
    "source_file",
    "group_id",
    "subject_id",
    "event_type",
    "timestamp_utc",
}
_PROCESSED_REQUIRED = _LONG_REQUIRED | {
    "is_excluded",
    "exclusion_reason",
}


def validate_long_df(df: pd.DataFrame) -> list[str]:
    """Validate parsed metric output and return warnings only."""
    return _validate(df, "long_df", _LONG_REQUIRED, _long_schema)


def validate_event_df(df: pd.DataFrame) -> list[str]:
    """Validate parsed event output and return warnings only."""
    return _validate(df, "event_df", _EVENT_REQUIRED, _event_schema)


def validate_processed_df(df: pd.DataFrame) -> list[str]:
    """Validate pipeline output and return warnings only."""
    return _validate(df, "processed_df", _PROCESSED_REQUIRED, _processed_schema)


def _validate(
    df: pd.DataFrame,
    name: str,
    required_columns: set[str],
    schema_factory: Any,
) -> list[str]:
    warnings = _required_column_warnings(df, name, required_columns)

    pa = _pandera()
    if pa is None:
        warnings.append(f"{name}: Pandera is not installed; schema validation skipped.")
        return warnings

    try:
        schema = schema_factory(pa)
        schema.validate(df, lazy=True)
    except Exception as exc:
        warnings.append(f"{name}: schema validation warning: {exc}")

    return warnings


def _required_column_warnings(
    df: pd.DataFrame,
    name: str,
    required_columns: set[str],
) -> list[str]:
    missing = sorted(required_columns - set(df.columns))
    if not missing:
        return []
    return [f"{name}: missing expected columns: {', '.join(missing)}."]


def _pandera() -> Any | None:
    try:
        try:
            import pandera.pandas as pa
        except ImportError:
            import pandera as pa
    except ImportError:
        return None
    return pa


def _long_schema(pa: Any) -> Any:
    return pa.DataFrameSchema(
        {
            "source_file": pa.Column(str, nullable=True, required=True),
            "metric_name": pa.Column(str, nullable=True, required=True),
            "group_id": pa.Column(str, nullable=True, required=True),
            "subject_id": pa.Column(str, nullable=True, required=True),
            "timestamp_utc": pa.Column(nullable=True, required=True),
            "value": pa.Column(nullable=True, required=True),
            "native_bin_seconds": pa.Column(float, nullable=True, required=False),
        },
        coerce=False,
        strict=False,
    )


def _event_schema(pa: Any) -> Any:
    return pa.DataFrameSchema(
        {
            "source_file": pa.Column(str, nullable=True, required=True),
            "group_id": pa.Column(str, nullable=True, required=True),
            "subject_id": pa.Column(str, nullable=True, required=True),
            "event_type": pa.Column(str, nullable=True, required=True),
            "timestamp_utc": pa.Column(nullable=True, required=True),
            "event_category": pa.Column(str, nullable=True, required=False),
        },
        coerce=False,
        strict=False,
    )


def _processed_schema(pa: Any) -> Any:
    columns = _long_schema(pa).columns
    columns.update(
        {
            "is_excluded": pa.Column(bool, nullable=True, required=True),
            "exclusion_reason": pa.Column(str, nullable=True, required=True),
            "flag_reason": pa.Column(str, nullable=True, required=False),
            "baseline_value": pa.Column(nullable=True, required=False),
            "baseline_valid": pa.Column(nullable=True, required=False),
        }
    )
    return pa.DataFrameSchema(columns, coerce=False, strict=False)
