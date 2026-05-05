"""Smoke tests for Streamlit app component configuration."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from app.components.metadata_tables import (
    REQUIRED_GROUP_COLUMNS,
    REQUIRED_SUBJECT_COLUMNS,
    group_column_config,
    subject_column_config,
)
from app.components.workflow import OUTPUT_COLUMN_GLOSSARY, WORKFLOW_HELP
from dvc_behavior import config as cfg


def test_metadata_column_config_covers_expected_columns():
    subject_cfg = subject_column_config()
    group_cfg = group_column_config()

    assert set(cfg.SUBJECT_METADATA_COLUMNS) <= set(subject_cfg)
    assert set(cfg.GROUP_METADATA_COLUMNS) <= set(group_cfg)


def test_workflow_help_covers_all_expected_steps():
    assert set(WORKFLOW_HELP.keys()) == {"import", "validate", "metadata", "events", "baseline", "qc", "export", "analysis"}
    for step, cfg_dict in WORKFLOW_HELP.items():
        assert "title" in cfg_dict
        assert "body" in cfg_dict
        assert "checks" in cfg_dict


def test_output_column_glossary_has_expected_fields():
    assert len(OUTPUT_COLUMN_GLOSSARY) > 0
    for entry in OUTPUT_COLUMN_GLOSSARY:
        assert isinstance(entry, tuple) and len(entry) == 3
        assert all(isinstance(field, str) for field in entry)
    names = {entry[0] for entry in OUTPUT_COLUMN_GLOSSARY}
    assert "subject_id" in names
    assert "value" in names
    assert "is_excluded" in names
    assert "time_from_event_hours" in names


def test_required_subject_columns_are_subset_of_column_config():
    assert REQUIRED_SUBJECT_COLUMNS <= set(subject_column_config())
    assert REQUIRED_GROUP_COLUMNS <= set(group_column_config())
