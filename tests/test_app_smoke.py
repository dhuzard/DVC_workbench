"""Smoke tests for Streamlit app component configuration."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from app.components.metadata_tables import group_column_config, subject_column_config
from dvc_behavior import config as cfg


def test_metadata_column_config_covers_expected_columns():
    subject_cfg = subject_column_config()
    group_cfg = group_column_config()

    assert set(cfg.SUBJECT_METADATA_COLUMNS) <= set(subject_cfg)
    assert set(cfg.GROUP_METADATA_COLUMNS) <= set(group_cfg)
