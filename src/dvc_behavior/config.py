"""Application-wide constants and default configuration values."""

from __future__ import annotations

from typing import Any

APP_VERSION = "0.1.0"
APP_NAME = "DVC Behavioral Preprocessing Workbench"

DEFAULT_TIMEZONE = "Europe/Paris"
DEFAULT_LIGHT_ON = "07:00"
DEFAULT_LIGHT_OFF = "19:00"

BASE_COLUMNS = ["day", "hour", "minute", "relativeTime"]
GROUP_META_SUFFIXES = {"_TIMESTAMP", "_AVG", "_SEM", "_QRT", "_SAMPLES"}

DEFAULT_EXCLUSION_RULES: dict[str, dict[str, Any]] = {
    "REMOVED": {"before_hours": 24.0, "after_hours": 24.0, "exclude": True, "flag": True},
    "INSERTED": {"before_hours": 24.0, "after_hours": 24.0, "exclude": True, "flag": True},
    "CAGE_CHANGE": {
        "before_hours": 6.0,
        "after_hours": 48.0,
        "max_gap_hours": 6.0,
        "exclude": True,
        "flag": True,
    },
    "CAGE_OFFLINE": {"before_hours": 0.0, "after_hours": 0.0, "exclude": False, "flag": True},
    "CAGE_ONLINE": {"before_hours": 0.0, "after_hours": 0.0, "exclude": False, "flag": True},
    "FACILITY_EVENT": {"before_hours": 0.0, "after_hours": 0.0, "exclude": True, "flag": True},
}

DEFAULT_BASELINE: dict[str, Any] = {
    "start_hours": -72.0,
    "end_hours": -24.0,
    "method": "mean",
    "exclude_excluded": True,
    "min_coverage": 0.7,
}

DEFAULT_ALIGNMENT: dict[str, Any] = {
    "label": "J0",
    "event_type": None,
    "fallback_timestamp": None,
    "scope": "subject",
}

EVENT_CATEGORY_MAP: dict[str, str] = {
    "REMOVED": "cage_handling",
    "INSERTED": "cage_handling",
    "CAGE_OFFLINE": "cage_status",
    "CAGE_ONLINE": "cage_status",
}

AGGREGATION_OPTIONS: dict[str, int | None] = {
    "Native (no aggregation)": None,
    "1 minute": 60,
    "5 minutes": 300,
    "1 hour": 3600,
    "12 hours": 43200,
    "24 hours": 86400,
}

SUBJECT_METADATA_COLUMNS = [
    "subject_id",
    "group_id_detected",
    "animal_id",
    "cage_id",
    "cage_uuid",
    "rack",
    "position",
    "sex",
    "strain",
    "genotype",
    "treatment_group",
    "cohort",
    "batch",
    "date_of_birth",
    "age_at_start",
    "body_weight",
    "surgery_date",
    "treatment_date",
    "inclusion_status",
    "exclusion_status",
    "notes",
]

GROUP_METADATA_COLUMNS = [
    "group_id",
    "group_label",
    "group_color",
    "treatment",
    "genotype",
    "cohort",
    "experimental_condition",
    "n_expected",
    "description",
    "notes",
]

EVENT_METADATA_COLUMNS = [
    "event_id",
    "event_type",
    "event_label",
    "subject_id",
    "group_id",
    "timestamp",
    "timezone",
    "event_scope",
    "used_for_alignment",
    "used_for_exclusion",
    "description",
    "notes",
]

STUDY_METADATA_FIELDS = [
    "study_id",
    "study_name",
    "project_name",
    "partner_name",
    "operator_name",
    "experiment_description",
    "species",
    "strain",
    "experiment_start_date",
    "experiment_end_date",
    "timezone",
    "light_on_time",
    "light_off_time",
    "main_experimental_event_name",
    "notes",
]

STUDY_METADATA_DEFAULTS: dict[str, str] = {
    "study_id": "",
    "study_name": "",
    "project_name": "",
    "partner_name": "",
    "operator_name": "",
    "experiment_description": "",
    "species": "mouse",
    "strain": "",
    "experiment_start_date": "",
    "experiment_end_date": "",
    "timezone": DEFAULT_TIMEZONE,
    "light_on_time": DEFAULT_LIGHT_ON,
    "light_off_time": DEFAULT_LIGHT_OFF,
    "main_experimental_event_name": "",
    "notes": "",
}
