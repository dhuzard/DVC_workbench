"""Offline tests for the grounded LLM-insights module.

These tests run fully offline: no network, no API keys, and no optional
dependencies (anthropic / ollama / requests). The default path is the
deterministic NullProvider / templated narrative.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dvc_behavior.insights import (
    ANALYSIS_TOOL_REGISTRY,
    INSIGHT_DISCLAIMER,
    AnthropicProvider,
    InsightResult,
    NullProvider,
    OllamaProvider,
    build_insight_payload,
    build_system_prompt,
    draft_methods_section,
    generate_narrative,
    payload_hash,
    triage_quality,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def stats_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric_name": "locomotion",
                "comparison": "KO vs WT",
                "group": None,
                "n_groups": 2,
                "n_total": 12,
                "statistic": 30.0,
                "effect_size": 0.62,
                "effect_size_name": "rank-biserial",
                "p_value": 0.031,
                "q_value": 0.062,
            }
        ]
    )


@pytest.fixture
def circadian_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group_id": "KO",
                "metric_name": "locomotion",
                "MESOR": np.float64(10.2),
                "amplitude": np.float64(4.1),
                "acrophase_ZT": np.float64(14.5),
                "R2": np.float64(0.78),
                "p_rhythm": np.float64(0.004),
                "phase": "dark",
            }
        ]
    )


@pytest.fixture
def light_dark_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group_id": "KO",
                "metric_name": "locomotion",
                "phase": "dark",
                "mean": 12.0,
                "dark_light_ratio": 2.3,
            },
            {
                "group_id": "KO",
                "metric_name": "locomotion",
                "phase": "light",
                "mean": 5.2,
                "dark_light_ratio": 2.3,
            },
        ]
    )


@pytest.fixture
def analysis_tables(stats_summary, circadian_summary, light_dark_summary) -> dict:
    return {
        "stats_summary": stats_summary,
        "circadian_summary": circadian_summary,
        "light_dark_summary": light_dark_summary,
        "empty_table": pd.DataFrame(),
    }


@pytest.fixture
def analysis_config() -> dict:
    return {
        "timezone": "Europe/Rome",
        "light_dark_cycle": {"light_on": "07:00", "light_off": "19:00"},
        "alignment": {"event_type": "treatment", "scope": "per_subject"},
        "baseline": {"start_hours": -48, "end_hours": 0},
        "aggregation_bin_seconds": 3600,
        "cosinor_period_hours": 24,
        "fdr_method": "Benjamini-Hochberg",
    }


@pytest.fixture
def manifest() -> dict:
    return {
        "app_version": "0.1.0",
        "inputs": [
            {"name": "a.csv", "sha256": "abc123"},
            {"name": "b.csv", "sha256": "def456"},
        ],
    }


@pytest.fixture
def quality_report() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": "M01",
                "metric_name": "locomotion",
                "missing_value_count": 0,
                "duplicate_timestamp_count": 0,
                "long_gap_count": 0,
                "negative_value_count": 0,
                "irregular_interval_flag": False,
                "zero_variance_flag": False,
            },
            {
                "subject_id": "M07",
                "metric_name": "locomotion",
                "missing_value_count": 120,
                "duplicate_timestamp_count": 0,
                "long_gap_count": 2,
                "negative_value_count": 0,
                "irregular_interval_flag": True,
                "zero_variance_flag": False,
            },
        ]
    )


# --------------------------------------------------------------------------- #
# build_insight_payload
# --------------------------------------------------------------------------- #
def test_payload_is_json_serializable(analysis_tables, analysis_config, manifest, quality_report):
    payload = build_insight_payload(
        analysis_tables,
        analysis_config=analysis_config,
        manifest=manifest,
        quality_report=quality_report,
    )
    # Must not raise — numpy types converted.
    text = json.dumps(payload)
    assert isinstance(text, str)
    assert payload["kind"] == "dvc_insight_payload"
    assert "stats_summary" in payload["tables"]
    assert payload["config"]["timezone"] == "Europe/Rome"
    assert payload["manifest"]["input_file_count"] == 2
    assert payload["qc"]["n_flagged_subjects"] == 1


def test_payload_respects_max_rows():
    big = pd.DataFrame({"metric_name": ["m"] * 100, "value": list(range(100))})
    payload = build_insight_payload({"daily_means": big}, max_rows_per_table=10)
    table = payload["tables"]["daily_means"]
    assert table["n_rows"] == 100
    assert table["truncated"] is True
    assert len(table["records"]) == 10
    assert table["shape"] == [100, 2]


def test_payload_handles_empty_and_missing_inputs():
    # Completely empty.
    payload = build_insight_payload({})
    assert json.dumps(payload)
    assert payload["tables"] == {}
    assert "config" not in payload
    assert "manifest" not in payload
    assert "qc" not in payload

    # None values and empty frames.
    payload2 = build_insight_payload(
        {"a": None, "b": pd.DataFrame()},
        analysis_config=None,
        manifest=None,
        quality_report=None,
    )
    assert payload2["tables"]["a"]["n_rows"] == 0
    assert payload2["tables"]["b"]["n_rows"] == 0


def test_payload_highlights_extract_groups_and_metrics(analysis_tables):
    payload = build_insight_payload(analysis_tables)
    hi = payload["highlights"]
    assert "stats_summary" in hi
    assert hi["stats_summary"]["comparisons"][0]["comparison"] == "KO vs WT"
    assert "circadian_summary" in hi


# --------------------------------------------------------------------------- #
# payload_hash
# --------------------------------------------------------------------------- #
def test_payload_hash_deterministic_and_sensitive(analysis_tables):
    payload = build_insight_payload(analysis_tables)
    h1 = payload_hash(payload)
    h2 = payload_hash(payload)
    assert h1 == h2
    assert len(h1) == 64

    changed = dict(payload)
    changed["extra"] = "something"
    assert payload_hash(changed) != h1


# --------------------------------------------------------------------------- #
# NullProvider via generate_narrative
# --------------------------------------------------------------------------- #
def test_null_provider_narrative_grounded(analysis_tables, analysis_config):
    payload = build_insight_payload(analysis_tables, analysis_config=analysis_config)
    result = generate_narrative(payload)

    assert isinstance(result, InsightResult)
    assert result.model_id == "offline-template"
    assert result.provider == "null"
    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert result.payload_sha256 == payload_hash(payload)

    text = result.text
    assert text
    # Mentions a specific group comparison and numbers from the payload.
    assert "KO vs WT" in text
    assert "0.62" in text  # effect size
    assert "0.031" in text  # p-value
    # Circadian + light/dark numbers present.
    assert "14.5" in text or "4.1" in text
    assert "2.3" in text  # dark/light ratio
    # Disclaimer appended.
    assert INSIGHT_DISCLAIMER in text

    # to_dict round-trips.
    d = result.to_dict()
    assert d["model_id"] == "offline-template"
    assert d["payload_sha256"] == result.payload_sha256


def test_null_provider_direct_complete(analysis_tables):
    payload = build_insight_payload(analysis_tables)
    text, p_tok, c_tok = NullProvider().complete(build_system_prompt(), payload)
    assert "KO vs WT" in text
    assert p_tok is None and c_tok is None


def test_generate_narrative_appends_disclaimer_when_missing():
    class StubProvider:
        name = "stub"
        model_id = "stub-1"

        def complete(self, system_prompt, payload):
            return "Short narrative with no caveat.", 5, 9

    payload = build_insight_payload({})
    result = generate_narrative(payload, provider=StubProvider())
    assert INSIGHT_DISCLAIMER in result.text
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 9
    assert result.model_id == "stub-1"


# --------------------------------------------------------------------------- #
# draft_methods_section
# --------------------------------------------------------------------------- #
def test_draft_methods_section_includes_config(analysis_config, manifest):
    text = draft_methods_section(analysis_config, manifest)
    assert "Europe/Rome" in text
    assert "07:00" in text and "19:00" in text
    assert "treatment" in text
    assert "-48" in text and "0 h" in text
    assert "Benjamini-Hochberg" in text
    assert "3600 s" in text
    assert "0.1.0" in text


def test_draft_methods_section_safe_on_empty():
    text = draft_methods_section({})
    assert isinstance(text, str)
    assert text  # produces something sensible


# --------------------------------------------------------------------------- #
# triage_quality
# --------------------------------------------------------------------------- #
def test_triage_quality_flags_bad_subject(quality_report):
    summary, issues = triage_quality(quality_report)
    assert "M07" in summary
    assert issues  # non-empty
    subjects = {i["subject_id"] for i in issues}
    assert "M07" in subjects
    assert "M01" not in subjects  # clean subject not flagged
    for issue in issues:
        assert issue["suggested_action"]
    # Missing values and irregular interval both detected for M07.
    kinds = {i["issue"] for i in issues}
    assert "missing values" in kinds
    assert "irregular sampling interval" in kinds
    assert "long gaps" in kinds


def test_triage_quality_clean_report():
    clean = pd.DataFrame(
        [
            {
                "subject_id": "M01",
                "metric_name": "locomotion",
                "missing_value_count": 0,
                "duplicate_timestamp_count": 0,
                "long_gap_count": 0,
                "negative_value_count": 0,
                "irregular_interval_flag": False,
                "zero_variance_flag": False,
            }
        ]
    )
    summary, issues = triage_quality(clean)
    assert issues == []
    assert "passed" in summary.lower()


def test_triage_quality_empty_input():
    summary, issues = triage_quality(pd.DataFrame())
    assert issues == []
    assert isinstance(summary, str)


def test_triage_quality_uses_exclusion_log(quality_report):
    exclusion_log = pd.DataFrame([{"subject_id": "M07", "event_type": "manual"}])
    summary, _ = triage_quality(quality_report, exclusion_log=exclusion_log)
    assert "exclusion log" in summary.lower()


# --------------------------------------------------------------------------- #
# Optional providers raise clear errors without their deps/host/key
# --------------------------------------------------------------------------- #
def test_anthropic_requires_model():
    with pytest.raises(ValueError):
        AnthropicProvider(model="")


def test_anthropic_missing_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = AnthropicProvider(model="claude-opus-4-8", api_key=None)
    payload = build_insight_payload({})
    with pytest.raises(RuntimeError) as exc:
        provider.complete(build_system_prompt(), payload)
    assert "key" in str(exc.value).lower()


def test_ollama_unreachable_or_missing_dep_raises():
    # With no Ollama server running (and possibly no requests), this must raise
    # a clear RuntimeError rather than hang or crash obscurely.
    provider = OllamaProvider(model="llama3", host="http://127.0.0.1:1")
    payload = build_insight_payload({})
    with pytest.raises(RuntimeError) as exc:
        provider.complete(build_system_prompt(), payload)
    msg = str(exc.value).lower()
    assert "ollama" in msg or "requests" in msg


# --------------------------------------------------------------------------- #
# Tool registry scaffold
# --------------------------------------------------------------------------- #
def test_tool_registry_is_scaffold():
    assert isinstance(ANALYSIS_TOOL_REGISTRY, dict)
    assert "summarize_circadian_cosinor" in ANALYSIS_TOOL_REGISTRY
    assert "quick_exploratory_stats" in ANALYSIS_TOOL_REGISTRY
    for name, desc in ANALYSIS_TOOL_REGISTRY.items():
        assert isinstance(name, str) and isinstance(desc, str) and desc
