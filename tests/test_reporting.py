"""Tests for src/dvc_behavior/reporting.py"""

from __future__ import annotations

import re

import pandas as pd

from dvc_behavior.reporting import generate_processing_report


def test_generate_processing_report_runs_with_empty_inputs():
    report = generate_processing_report(
        long_df=pd.DataFrame(),
        processed_df=None,
        event_df=None,
        exclusion_log=None,
        baseline_summary=None,
        warnings=[],
        analysis_config={"app_version": "test"},
    )
    assert isinstance(report, str)
    assert "Processing Report" in report


def test_report_timestamp_has_no_human_artifact():
    """The leftover '%Human' artifact must be gone and the timestamp valid."""
    report = generate_processing_report(
        long_df=pd.DataFrame(),
        processed_df=None,
        event_df=None,
        exclusion_log=None,
        baseline_summary=None,
        warnings=[],
        analysis_config={},
    )
    assert "Human" not in report
    assert "%H" not in report

    # The Generated line must contain a well-formed UTC timestamp.
    match = re.search(r"\*\*Generated:\*\* (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC)", report)
    assert match, f"No valid timestamp found in report:\n{report}"
