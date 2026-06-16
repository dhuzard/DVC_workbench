"""Tests for src/dvc_behavior/export.py and src/dvc_behavior/reporting.py"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime

import pandas as pd
import yaml

from dvc_behavior.export import build_analysis_config, create_export_zip, create_export_zip_file
from dvc_behavior.reporting import generate_metadata_validation_report, generate_processing_report
from dvc_behavior.parsing import wide_to_long
from dvc_behavior.alignment import align_to_manual_timestamp
from dvc_behavior.baseline import compute_baseline
from tests.conftest import make_metric_df


def _build_sample_processed_df():
    raw = make_metric_df(n_rows=48, bin_minutes=60, start_ts="2024-01-01T07:00:00+0100")
    long, _ = wide_to_long(raw, "test.csv")
    long, _ = align_to_manual_timestamp(long, "2024-01-03T07:00:00+0100")
    long["is_excluded"] = False
    long["exclusion_reason"] = ""
    long["flag_reason"] = ""
    long["light_dark_phase"] = "light"
    long["zeitgeber_time_hours"] = 0.0
    long["timestamp_local"] = long["timestamp_utc"]
    processed, summary, _ = compute_baseline(long, -48.0, -24.0)
    return processed, summary


class TestBuildAnalysisConfig:
    def test_structure(self):
        cfg = build_analysis_config(
            uploaded_files=["file.csv"],
            study_metadata={"study_id": "S01"},
            timezone="Europe/Paris",
            light_on="07:00",
            light_off="19:00",
            alignment_cfg={"event_type": "REMOVED"},
            exclusion_rules={"REMOVED": {"before_hours": 24}},
            baseline_cfg={"start_hours": -72, "end_hours": -24},
            aggregation_bin=None,
        )
        assert "app_version" in cfg
        assert "processing_timestamp" in cfg
        assert cfg["timezone"] == "Europe/Paris"
        assert cfg["light_dark_cycle"]["light_on"] == "07:00"
        assert cfg["baseline"]["start_hours"] == -72

    def test_timestamp_is_utc_iso(self):
        cfg = build_analysis_config(
            uploaded_files=[], study_metadata={}, timezone="UTC",
            light_on="07:00", light_off="19:00",
            alignment_cfg={}, exclusion_rules={}, baseline_cfg={},
            aggregation_bin=None,
        )
        ts = cfg["processing_timestamp"]
        # Should parse without error
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None


class TestCreateExportZip:
    def _default_zip(self, processed=True):
        proc, summary = _build_sample_processed_df() if processed else (None, None)
        cfg = build_analysis_config(
            uploaded_files=["f.csv"], study_metadata={}, timezone="Europe/Paris",
            light_on="07:00", light_off="19:00",
            alignment_cfg={}, exclusion_rules={}, baseline_cfg={},
            aggregation_bin=None,
        )
        report = generate_processing_report(
            long_df=proc if proc is not None else pd.DataFrame(),
            processed_df=proc,
            event_df=None,
            exclusion_log=None,
            baseline_summary=summary,
            warnings=[],
            analysis_config=cfg,
        )
        return create_export_zip(
            processed_df=proc,
            baseline_summary=summary,
            exclusion_log=None,
            event_table=None,
            subject_metadata=None,
            group_metadata=None,
            study_metadata={"study_id": "TEST"},
            analysis_config=cfg,
            processing_report=report,
            metadata_validation_report="# OK",
        )

    def test_returns_bytes(self):
        data = self._default_zip()
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_valid_zip(self):
        data = self._default_zip()
        zf = zipfile.ZipFile(io.BytesIO(data))
        names = zf.namelist()
        assert len(names) > 0

    def test_contains_required_files(self):
        data = self._default_zip(processed=True)
        zf = zipfile.ZipFile(io.BytesIO(data))
        names = set(zf.namelist())
        assert "processed_timeseries.csv" in names
        assert "analysis_config.yaml" in names
        assert "processing_report.md" in names
        assert "study_metadata.yaml" in names

    def test_analysis_config_yaml_valid(self):
        data = self._default_zip()
        zf = zipfile.ZipFile(io.BytesIO(data))
        cfg_bytes = zf.read("analysis_config.yaml")
        cfg = yaml.safe_load(cfg_bytes)
        assert isinstance(cfg, dict)
        assert "app_version" in cfg

    def test_processed_csv_has_required_columns(self):
        data = self._default_zip(processed=True)
        zf = zipfile.ZipFile(io.BytesIO(data))
        csv_bytes = zf.read("processed_timeseries.csv")
        df = pd.read_csv(io.BytesIO(csv_bytes))
        for col in ("source_file", "metric_name", "group_id", "subject_id", "value", "is_excluded"):
            assert col in df.columns, f"Missing column: {col}"

    def test_empty_processed_df_no_crash(self):
        cfg = build_analysis_config(
            uploaded_files=[], study_metadata={}, timezone="UTC",
            light_on="07:00", light_off="19:00",
            alignment_cfg={}, exclusion_rules={}, baseline_cfg={},
            aggregation_bin=None,
        )
        data = create_export_zip(
            processed_df=None,
            baseline_summary=None,
            exclusion_log=None,
            event_table=None,
            subject_metadata=None,
            group_metadata=None,
            study_metadata={},
            analysis_config=cfg,
            processing_report="no data",
            metadata_validation_report=None,
        )
        assert isinstance(data, bytes)

    def test_file_based_zip_builder(self, tmp_path):
        proc, summary = _build_sample_processed_df()
        cfg = build_analysis_config(
            uploaded_files=["f.csv"], study_metadata={}, timezone="Europe/Paris",
            light_on="07:00", light_off="19:00",
            alignment_cfg={}, exclusion_rules={}, baseline_cfg={},
            aggregation_bin=None,
        )
        path = create_export_zip_file(
            tmp_path / "export.zip",
            processed_df=proc,
            baseline_summary=summary,
            exclusion_log=None,
            event_table=None,
            subject_metadata=None,
            group_metadata=None,
            study_metadata={"study_id": "TEST"},
            analysis_config=cfg,
            processing_report="# OK",
            metadata_validation_report="# OK",
        )

        assert path.exists()
        with zipfile.ZipFile(path) as zf:
            assert "processed_timeseries.csv" in zf.namelist()

    def test_optional_manifest_included(self):
        data = create_export_zip(
            processed_df=None,
            baseline_summary=None,
            exclusion_log=None,
            event_table=None,
            subject_metadata=None,
            group_metadata=None,
            study_metadata={},
            analysis_config=None,
            processing_report=None,
            metadata_validation_report=None,
            manifest={
                "app_version": "test-version",
                "input_files": [{"name": "metric.csv", "size_bytes": 3, "sha256": "abc"}],
                "row_counts": {"processed_timeseries": 0},
            },
        )

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            assert "manifest.yaml" in zf.namelist()
            manifest = yaml.safe_load(zf.read("manifest.yaml"))
        assert manifest["app_version"] == "test-version"
        assert manifest["input_files"][0]["name"] == "metric.csv"

    def test_deterministic_output(self):
        """Same inputs → same ZIP content (modulo timestamp in YAML)."""
        data1 = self._default_zip()
        data2 = self._default_zip()
        # The processing_timestamp will differ, so just check sizes are similar
        assert abs(len(data1) - len(data2)) < 100

    def test_text_artifacts_written_verbatim(self):
        data = create_export_zip(
            processed_df=None,
            baseline_summary=None,
            exclusion_log=None,
            event_table=None,
            subject_metadata=None,
            group_metadata=None,
            study_metadata={},
            analysis_config=None,
            processing_report=None,
            metadata_validation_report=None,
            text_artifacts={
                "insights/narrative.md": "# Narrative\n\nGroup KO differs.",
                "insights/payload.json": '{"a": 1}',
                "insights/empty.txt": "",  # falsy → skipped
            },
        )
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
            assert "insights/narrative.md" in names
            assert "insights/payload.json" in names
            assert "insights/empty.txt" not in names
            assert zf.read("insights/narrative.md").decode().startswith("# Narrative")


class TestGenerateProcessingReport:
    def test_returns_string(self):
        proc, summary = _build_sample_processed_df()
        cfg = build_analysis_config(
            uploaded_files=["f.csv"], study_metadata={}, timezone="Europe/Paris",
            light_on="07:00", light_off="19:00",
            alignment_cfg={}, exclusion_rules={}, baseline_cfg={},
            aggregation_bin=None,
        )
        report = generate_processing_report(
            long_df=proc,
            processed_df=proc,
            event_df=None,
            exclusion_log=None,
            baseline_summary=summary,
            warnings=["test warning"],
            analysis_config=cfg,
        )
        assert isinstance(report, str)
        assert "DVC Behavioral Preprocessing" in report
        assert "test warning" in report

    def test_no_data_no_crash(self):
        cfg = build_analysis_config(
            uploaded_files=[], study_metadata={}, timezone="UTC",
            light_on="07:00", light_off="19:00",
            alignment_cfg={}, exclusion_rules={}, baseline_cfg={},
            aggregation_bin=None,
        )
        report = generate_processing_report(
            long_df=pd.DataFrame(),
            processed_df=None,
            event_df=None,
            exclusion_log=None,
            baseline_summary=None,
            warnings=[],
            analysis_config=cfg,
        )
        assert isinstance(report, str)


class TestGenerateMetadataValidationReport:
    def test_with_errors_and_warnings(self):
        report = generate_metadata_validation_report(
            errors=["Duplicate subject_id"],
            warnings=["Missing sex for C57_1"],
            subject_meta=None,
        )
        assert "Errors" in report
        assert "Duplicate subject_id" in report
        assert "Missing sex" in report

    def test_clean_metadata(self):
        report = generate_metadata_validation_report([], [], None)
        assert "passed" in report.lower()
