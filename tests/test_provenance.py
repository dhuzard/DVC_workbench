"""Tests for src/dvc_behavior/provenance.py."""

from __future__ import annotations

from datetime import datetime
import hashlib
import io

import pandas as pd

from dvc_behavior.provenance import build_provenance_manifest


def test_manifest_records_path_file_details(tmp_path):
    content = b"day,value\n1,2\n"
    path = tmp_path / "metric.csv"
    path.write_bytes(content)

    manifest = build_provenance_manifest(
        input_files=[path],
        selected_config={"timezone": "UTC"},
        row_counts={"processed_timeseries": 12},
        app_version="test-version",
        processing_timestamp="2024-01-01T00:00:00+00:00",
    )

    assert manifest["manifest_version"] == 1
    assert manifest["app_version"] == "test-version"
    assert datetime.fromisoformat(manifest["processing_timestamp"]).tzinfo is not None
    assert manifest["selected_config"] == {"timezone": "UTC"}
    assert manifest["row_counts"] == {"processed_timeseries": 12}
    assert manifest["input_files"] == [
        {
            "name": "metric.csv",
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    ]


def test_manifest_records_uploaded_bytes_config_and_table_counts(tmp_path):
    metric_content = b"metric"
    event_stream = io.BytesIO(b"event")
    event_stream.name = "events.csv"
    event_stream.seek(2)
    table = pd.DataFrame({"subject_id": ["S1", "S2", "S3"]})

    manifest = build_provenance_manifest(
        input_files=[
            ("metric.csv", metric_content),
            event_stream,
            {"name": "metadata.csv", "content": "subject_id\nS1\n"},
        ],
        selected_config={
            "timestamp": pd.Timestamp("2024-02-03T04:05:06Z"),
            "path": tmp_path / "config.yaml",
            "missing": float("nan"),
        },
        row_counts={"event_table": 5},
        tables={"processed_timeseries": table, "baseline_summary": None},
        processing_timestamp="2024-01-01T00:00:00+00:00",
    )

    assert [record["name"] for record in manifest["input_files"]] == [
        "metric.csv",
        "events.csv",
        "metadata.csv",
    ]
    assert manifest["input_files"][0]["sha256"] == hashlib.sha256(metric_content).hexdigest()
    assert manifest["input_files"][1]["sha256"] == hashlib.sha256(b"event").hexdigest()
    assert event_stream.tell() == 2
    assert manifest["selected_config"]["timestamp"] == "2024-02-03T04:05:06+00:00"
    assert manifest["selected_config"]["path"].endswith("config.yaml")
    assert manifest["selected_config"]["missing"] is None
    assert manifest["row_counts"] == {
        "event_table": 5,
        "processed_timeseries": 3,
        "baseline_summary": None,
    }
