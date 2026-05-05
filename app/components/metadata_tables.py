"""Metadata editor column configuration for the Streamlit app."""

from __future__ import annotations

from typing import Any

import streamlit as st


REQUIRED_SUBJECT_COLUMNS = {
    "subject_id",
    "animal_id",
    "cage_id",
    "sex",
    "treatment_group",
    "genotype",
    "cohort",
}

REQUIRED_GROUP_COLUMNS = {"group_id", "group_label"}


SUBJECT_COLUMN_HELP = {
    "subject_id": "Required. Detected DVC subject identifier used to merge metadata onto time-series rows.",
    "group_id_detected": "Detected source group for reference. This is not used as the merge key.",
    "animal_id": "Required. Scientific or colony animal identifier.",
    "cage_id": "Required. Cage identifier used during the study.",
    "cage_uuid": "Optional stable DVC cage UUID when available.",
    "rack": "Optional rack identifier.",
    "position": "Optional rack or cage position.",
    "sex": "Required. Biological sex used for stratified review.",
    "strain": "Optional strain label.",
    "genotype": "Required. Genotype or wild-type/control label.",
    "treatment_group": "Required. Treatment arm or experimental group assigned by the study.",
    "cohort": "Required. Cohort, wave, or batch label used for design tracking.",
    "batch": "Optional processing, shipment, or acquisition batch.",
    "date_of_birth": "Optional date of birth, preferably YYYY-MM-DD.",
    "age_at_start": "Optional age at experiment start.",
    "body_weight": "Optional body weight and unit.",
    "surgery_date": "Optional surgery date, preferably YYYY-MM-DD.",
    "treatment_date": "Optional treatment date, preferably YYYY-MM-DD.",
    "inclusion_status": "Optional inclusion status for downstream interpretation.",
    "exclusion_status": "Optional subject-level exclusion status.",
    "notes": "Optional free-text notes.",
}

GROUP_COLUMN_HELP = {
    "group_id": "Required. Detected DVC group identifier used to merge group metadata.",
    "group_label": "Required. Human-readable label used in plots and exports.",
    "group_color": "Optional plot color for the group.",
    "treatment": "Optional treatment label for the group.",
    "genotype": "Optional genotype label for the group.",
    "cohort": "Optional cohort label for the group.",
    "experimental_condition": "Optional condition label for study design summaries.",
    "n_expected": "Optional expected number of subjects in the group.",
    "description": "Optional group description.",
    "notes": "Optional free-text notes.",
}


def subject_column_config() -> dict[str, Any]:
    """Return Streamlit column_config entries for the subject metadata editor."""
    return {
        "subject_id": st.column_config.TextColumn(
            "Subject ID", help=SUBJECT_COLUMN_HELP["subject_id"], required=True
        ),
        "group_id_detected": st.column_config.TextColumn(
            "Detected group", help=SUBJECT_COLUMN_HELP["group_id_detected"], disabled=True
        ),
        "animal_id": st.column_config.TextColumn(
            "Animal ID", help=SUBJECT_COLUMN_HELP["animal_id"], required=True
        ),
        "cage_id": st.column_config.TextColumn(
            "Cage ID", help=SUBJECT_COLUMN_HELP["cage_id"], required=True
        ),
        "cage_uuid": st.column_config.TextColumn("Cage UUID", help=SUBJECT_COLUMN_HELP["cage_uuid"]),
        "rack": st.column_config.TextColumn("Rack", help=SUBJECT_COLUMN_HELP["rack"]),
        "position": st.column_config.TextColumn("Position", help=SUBJECT_COLUMN_HELP["position"]),
        "sex": st.column_config.SelectboxColumn(
            "Sex",
            help=SUBJECT_COLUMN_HELP["sex"],
            options=["", "female", "male", "unknown"],
            required=True,
        ),
        "strain": st.column_config.TextColumn("Strain", help=SUBJECT_COLUMN_HELP["strain"]),
        "genotype": st.column_config.TextColumn(
            "Genotype", help=SUBJECT_COLUMN_HELP["genotype"], required=True
        ),
        "treatment_group": st.column_config.TextColumn(
            "Treatment group", help=SUBJECT_COLUMN_HELP["treatment_group"], required=True
        ),
        "cohort": st.column_config.TextColumn(
            "Cohort", help=SUBJECT_COLUMN_HELP["cohort"], required=True
        ),
        "batch": st.column_config.TextColumn("Batch", help=SUBJECT_COLUMN_HELP["batch"]),
        "date_of_birth": st.column_config.TextColumn(
            "Date of birth", help=SUBJECT_COLUMN_HELP["date_of_birth"]
        ),
        "age_at_start": st.column_config.TextColumn(
            "Age at start", help=SUBJECT_COLUMN_HELP["age_at_start"]
        ),
        "body_weight": st.column_config.TextColumn(
            "Body weight", help=SUBJECT_COLUMN_HELP["body_weight"]
        ),
        "surgery_date": st.column_config.TextColumn(
            "Surgery date", help=SUBJECT_COLUMN_HELP["surgery_date"]
        ),
        "treatment_date": st.column_config.TextColumn(
            "Treatment date", help=SUBJECT_COLUMN_HELP["treatment_date"]
        ),
        "inclusion_status": st.column_config.SelectboxColumn(
            "Inclusion status",
            help=SUBJECT_COLUMN_HELP["inclusion_status"],
            options=["", "included", "excluded", "pending"],
        ),
        "exclusion_status": st.column_config.SelectboxColumn(
            "Exclusion status",
            help=SUBJECT_COLUMN_HELP["exclusion_status"],
            options=["", "none", "excluded", "partial", "pending"],
        ),
        "notes": st.column_config.TextColumn("Notes", help=SUBJECT_COLUMN_HELP["notes"]),
    }


def group_column_config() -> dict[str, Any]:
    """Return Streamlit column_config entries for the group metadata editor."""
    return {
        "group_id": st.column_config.TextColumn(
            "Group ID", help=GROUP_COLUMN_HELP["group_id"], required=True
        ),
        "group_label": st.column_config.TextColumn(
            "Group label", help=GROUP_COLUMN_HELP["group_label"], required=True
        ),
        "group_color": st.column_config.TextColumn(
            "Color", help=GROUP_COLUMN_HELP["group_color"]
        ),
        "treatment": st.column_config.TextColumn("Treatment", help=GROUP_COLUMN_HELP["treatment"]),
        "genotype": st.column_config.TextColumn("Genotype", help=GROUP_COLUMN_HELP["genotype"]),
        "cohort": st.column_config.TextColumn("Cohort", help=GROUP_COLUMN_HELP["cohort"]),
        "experimental_condition": st.column_config.TextColumn(
            "Experimental condition", help=GROUP_COLUMN_HELP["experimental_condition"]
        ),
        "n_expected": st.column_config.TextColumn("Expected N", help=GROUP_COLUMN_HELP["n_expected"]),
        "description": st.column_config.TextColumn(
            "Description", help=GROUP_COLUMN_HELP["description"]
        ),
        "notes": st.column_config.TextColumn("Notes", help=GROUP_COLUMN_HELP["notes"]),
    }
