"""Offline tests for the Circadiem client + schema mirror.

The network is never touched: the ``_http_*`` seam is monkeypatched, mirroring
the literature-module test pattern. PNGs are crafted from signature + IHDR bytes
so no image library or kaleido engine is needed.
"""

from __future__ import annotations

import struct

import numpy as np
import pandas as pd
import pytest

from dvc_behavior import circadiem, qc


def make_png(width: int = 120, height: int = 90, size: int = 256) -> bytes:
    """Build a minimal byte blob with a valid PNG signature + IHDR dimensions."""
    head = (
        circadiem.PNG_SIGNATURE
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
    )
    return head + b"\x00" * max(0, size - len(head))


def good_result_row(label: str = "Cage 3 — Day 7", run_id: str = "run-1") -> dict:
    return {
        "label": label,
        "baseline_light": 1,
        "dark_onset_burst": 3,
        "dark_irregularity": 2,
        "midnight_fragmentation": 0,
        "pre_light_decline": 1,
        "pre_dark_anticipation": 2,
        "confidence": "high",
        "flags": ["low_contrast"],
        "notes": "Clear dark-onset burst.",
        "meta": {"run_id": run_id, "model": "gpt-4o-mini", "created_at": "2026-06-16T08:00:00Z"},
    }


# --------------------------------------------------------------------------- #
# Schema parsing (§4)
# --------------------------------------------------------------------------- #
class TestParseRow:
    def test_parses_result_row(self):
        row = circadiem.parse_row(good_result_row())
        assert isinstance(row, circadiem.ResultRow)
        assert not row.is_error
        assert row.run_id == "run-1"
        assert row.markers.dark_onset_burst == 3
        assert row.markers.as_dict()["baseline_light"] == 1
        assert row.flags == ["low_contrast"]

    def test_error_row_detected_by_error_key(self):
        row = circadiem.parse_row({"label": "bad", "error": "boom", "meta": {"run_id": "r"}})
        assert isinstance(row, circadiem.ErrorRow)
        assert row.is_error
        assert row.error == "boom"
        assert row.run_id == "r"

    def test_missing_markers_downgrade_to_error(self):
        payload = good_result_row()
        del payload["dark_irregularity"]
        row = circadiem.parse_row(payload)
        assert isinstance(row, circadiem.ErrorRow)
        assert "dark_irregularity" in row.error

    def test_out_of_range_marker_downgrades_to_error(self):
        payload = good_result_row()
        payload["baseline_light"] = 9
        row = circadiem.parse_row(payload)
        assert isinstance(row, circadiem.ErrorRow)

    def test_non_dict_row_is_error(self):
        assert isinstance(circadiem.parse_row(["nope"]), circadiem.ErrorRow)


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
class TestValidation:
    def test_validate_openai_key_accepts_well_formed(self):
        circadiem.validate_openai_key("sk-" + "a" * 30)

    @pytest.mark.parametrize("bad", [None, "", "pk-short", "sk-tooshort"])
    def test_validate_openai_key_rejects_bad(self, bad):
        with pytest.raises(circadiem.CircadiemError):
            circadiem.validate_openai_key(bad)

    def test_validate_png_returns_dimensions(self):
        assert circadiem.validate_png(make_png(120, 90)) == (120, 90)

    def test_validate_png_rejects_bad_signature(self):
        with pytest.raises(circadiem.CircadiemError):
            circadiem.validate_png(b"not a png at all..............")

    def test_validate_png_rejects_oversize_dimensions(self):
        with pytest.raises(circadiem.CircadiemError):
            circadiem.validate_png(make_png(circadiem.MAX_DIMENSION + 1, 10))

    def test_resolve_base_url_prefers_explicit_then_env(self, monkeypatch):
        monkeypatch.setenv(circadiem.BASE_URL_ENV, "https://env.example/")
        assert circadiem.resolve_base_url("https://explicit/") == "https://explicit"
        assert circadiem.resolve_base_url() == "https://env.example"

    def test_resolve_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv(circadiem.OPENAI_KEY_ENV, "sk-" + "z" * 30)
        assert circadiem.resolve_api_key().startswith("sk-")
        assert (
            circadiem.resolve_api_key("sk-explicitexplicitexplicit")
            == "sk-explicitexplicitexplicit"
        )


class TestConfig:
    def test_url_building(self):
        cfg = circadiem.CircadiemConfig(base_url="https://host/")
        assert cfg.url("/api/analyze") == "https://host/api/analyze"

    def test_bad_band_rejected(self):
        with pytest.raises(circadiem.CircadiemError):
            circadiem.CircadiemConfig(base_url="https://host", vcg_band="+-9SD")

    def test_missing_base_url_raises_on_use(self):
        cfg = circadiem.CircadiemConfig(base_url="")
        with pytest.raises(circadiem.CircadiemError):
            cfg.url("/health")


# --------------------------------------------------------------------------- #
# analyze() over the monkeypatched HTTP seam
# --------------------------------------------------------------------------- #
class TestAnalyze:
    def _config(self):
        return circadiem.CircadiemConfig(base_url="https://circadiem.test")

    def test_happy_path_builds_form_and_parses(self, monkeypatch):
        captured = {}

        def fake_post(url, *, headers, files, data, timeout):
            captured.update(url=url, headers=headers, files=files, data=data, timeout=timeout)
            return {
                "results": [good_result_row("A"), {"label": "B", "error": "bad image", "meta": {}}]
            }

        monkeypatch.setattr(circadiem, "_http_post_multipart", fake_post)
        rows = circadiem.analyze(
            [("a.png", make_png()), ("b.png", make_png())],
            config=self._config(),
            api_key="sk-" + "k" * 30,
            labels=["A", "B"],
        )
        assert captured["url"] == "https://circadiem.test/api/analyze"
        assert captured["headers"]["Authorization"].startswith("Bearer sk-")
        assert captured["data"]["aligned_to_dark"] == "true"
        assert captured["data"]["vcg_band"] == "+-2SD"
        assert captured["data"]["labels"] == '["A", "B"]'
        assert len(captured["files"]) == 2
        assert isinstance(rows[0], circadiem.ResultRow)
        assert isinstance(rows[1], circadiem.ErrorRow)

    def test_rejects_bad_key_before_calling(self, monkeypatch):
        monkeypatch.setattr(
            circadiem, "_http_post_multipart", lambda *a, **k: pytest.fail("called")
        )
        with pytest.raises(circadiem.CircadiemError):
            circadiem.analyze([("a.png", make_png())], config=self._config(), api_key="nope")

    def test_rejects_empty_and_oversized_batches(self):
        cfg = self._config()
        with pytest.raises(circadiem.CircadiemError):
            circadiem.analyze([], config=cfg, api_key="sk-" + "k" * 30)
        too_many = [("x.png", make_png())] * (circadiem.MAX_FILES + 1)
        with pytest.raises(circadiem.CircadiemError):
            circadiem.analyze(too_many, config=cfg, api_key="sk-" + "k" * 30)

    def test_label_count_mismatch_raises(self):
        with pytest.raises(circadiem.CircadiemError):
            circadiem.analyze(
                [("a.png", make_png())],
                config=self._config(),
                api_key="sk-" + "k" * 30,
                labels=["A", "B"],
            )

    def test_transport_error_normalized(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("connection refused")

        monkeypatch.setattr(circadiem, "_http_post_multipart", boom)
        with pytest.raises(circadiem.CircadiemError, match="Could not reach Circadiem"):
            circadiem.analyze(
                [("a.png", make_png())], config=self._config(), api_key="sk-" + "k" * 30
            )

    def test_missing_results_array_raises(self, monkeypatch):
        monkeypatch.setattr(circadiem, "_http_post_multipart", lambda *a, **k: {"oops": 1})
        with pytest.raises(circadiem.CircadiemError, match="results"):
            circadiem.analyze(
                [("a.png", make_png())], config=self._config(), api_key="sk-" + "k" * 30
            )


class TestHealthAndPrompt:
    def test_health_true(self, monkeypatch):
        monkeypatch.setattr(circadiem, "_http_get_json", lambda url, **k: {"ok": True})
        assert circadiem.health(circadiem.CircadiemConfig(base_url="https://h")) is True

    def test_health_false_on_error(self, monkeypatch):
        def boom(url, **k):
            raise OSError("down")

        monkeypatch.setattr(circadiem, "_http_get_json", boom)
        assert circadiem.health(circadiem.CircadiemConfig(base_url="https://h")) is False

    def test_get_prompt(self, monkeypatch):
        monkeypatch.setattr(circadiem, "_http_get_json", lambda url, **k: {"prompt": "rubric text"})
        assert (
            circadiem.get_prompt(circadiem.CircadiemConfig(base_url="https://h")) == "rubric text"
        )


# --------------------------------------------------------------------------- #
# results_to_frame
# --------------------------------------------------------------------------- #
class TestResultsToFrame:
    def test_keeps_error_rows_and_columns(self):
        rows = [
            circadiem.parse_row(good_result_row("A", "run-9")),
            circadiem.parse_row({"label": "B", "error": "bad", "meta": {"run_id": "run-9"}}),
        ]
        frame = circadiem.results_to_frame(rows)
        assert list(frame["status"]) == ["ok", "error"]
        assert set(circadiem.MARKER_FIELDS) <= set(frame.columns)
        assert frame.loc[0, "dark_onset_burst"] == 3
        assert pd.isna(frame.loc[1, "dark_onset_burst"])
        assert frame.loc[1, "error"] == "bad"
        assert frame.loc[0, "run_id"] == "run-9"

    def test_empty_input_returns_typed_empty_frame(self):
        frame = circadiem.results_to_frame([])
        assert frame.empty
        assert "run_id" in frame.columns


# --------------------------------------------------------------------------- #
# qc.plot_circadiem_vcg / figure_to_png_bytes
# --------------------------------------------------------------------------- #
class TestVcgPlot:
    def _df(self):
        rng = np.random.default_rng(0)
        rows = []
        for subject in range(4):
            for zt in np.arange(0, 24, 1.0):
                rows.append(
                    {
                        "subject_id": f"s{subject}",
                        "zeitgeber_time_hours": zt,
                        "value": 10 + 5 * np.sin(zt / 24 * 2 * np.pi) + rng.normal(0, 0.5),
                    }
                )
        return pd.DataFrame(rows)

    def test_returns_figure_with_mean_and_band(self):
        fig = qc.plot_circadiem_vcg(self._df(), dark_onset_zt=12.0, photoperiod_dark_hours=12.0)
        names = [tr.name for tr in fig.data]
        assert "Global mean (VCG)" in names
        # the mean curve is drawn in black per the rubric convention
        mean_trace = next(tr for tr in fig.data if tr.name == "Global mean (VCG)")
        assert mean_trace.line.color == "black"

    def test_empty_input_degrades_gracefully(self):
        fig = qc.plot_circadiem_vcg(pd.DataFrame(), dark_onset_zt=12.0)
        assert fig.layout.title.text  # has an explanatory title, did not raise

    def test_png_helper_raises_without_engine(self, monkeypatch):
        class FakeFig:
            def to_image(self, **kwargs):
                raise RuntimeError("kaleido missing")

        with pytest.raises(RuntimeError, match="kaleido"):
            qc.figure_to_png_bytes(FakeFig())
