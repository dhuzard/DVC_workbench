"""Offline tests for the grounded tool-calling (agentic) Q&A loop.

These tests run fully offline: no network, no API keys, and no optional
dependencies (the ``anthropic`` SDK is never imported). The agent loop is driven
by a scripted, in-memory provider; the executed tools are the REAL
``analysis.py`` functions running on a small synthetic DataFrame.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dvc_behavior.insights import (
    INSIGHT_DISCLAIMER,
    AnthropicToolProvider,
    AssistantTurn,
    QAResult,
    ScriptedToolProvider,
    answer_question,
    build_qa_system_prompt,
    build_tool_specs,
    execute_analysis_tool,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def processed_df() -> pd.DataFrame:
    """A small but realistic processed-timeseries frame (two groups, hourly)."""
    rng = np.random.default_rng(7)
    rows: list[dict] = []
    base = pd.Timestamp("2024-01-01 00:00:00")
    for group in ("KO", "WT"):
        for subject in range(3):
            sid = f"{group}-{subject}"
            for hour in range(48):
                zt = hour % 24
                phase = "light" if 6 <= zt < 18 else "dark"
                level = 10.0 if group == "KO" else 6.0
                value = max(0.0, level + 4.0 * np.cos((zt - 14) / 24 * 2 * np.pi)
                            + rng.normal(0, 0.5))
                ts = base + pd.Timedelta(hours=hour)
                rows.append(
                    {
                        "group_id": group,
                        "metric_name": "locomotion",
                        "subject_id": sid,
                        "value": value,
                        "light_dark_phase": phase,
                        "zeitgeber_time_hours": float(zt),
                        "timestamp_local": ts,
                        "timestamp_utc": ts,
                        "time_from_event_hours": float(hour),
                        "is_excluded": False,
                    }
                )
    return pd.DataFrame(rows)


def _tool_call_turn(call_id: str, name: str, tool_input: dict | None = None) -> AssistantTurn:
    return AssistantTurn(
        text="",
        tool_calls=[{"id": call_id, "name": name, "input": tool_input or {}}],
        stop_reason="tool_use",
    )


# --------------------------------------------------------------------------- #
# build_tool_specs
# --------------------------------------------------------------------------- #
def test_build_tool_specs_shape_and_json() -> None:
    specs = build_tool_specs()
    assert len(specs) == 9
    for spec in specs:
        assert set(spec) >= {"name", "description", "input_schema"}
        assert spec["name"]
        assert spec["description"]
        schema = spec["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert schema["required"] == []
        # df is never exposed.
        assert "df" not in schema["properties"]
    # Fully JSON-serializable.
    assert json.loads(json.dumps(specs))


def test_build_tool_specs_custom_registry() -> None:
    specs = build_tool_specs({"summarize_light_dark": "desc"})
    assert len(specs) == 1
    assert specs[0]["name"] == "summarize_light_dark"


# --------------------------------------------------------------------------- #
# execute_analysis_tool
# --------------------------------------------------------------------------- #
def test_execute_summarize_light_dark(processed_df: pd.DataFrame) -> None:
    result = execute_analysis_tool("summarize_light_dark", {}, processed_df, max_rows=30)
    assert result["tool"] == "summarize_light_dark"
    assert isinstance(result["columns"], list)
    assert result["n_rows"] >= 1
    assert len(result["records"]) <= 30
    assert isinstance(result["warnings"], list)
    # JSON-serializable.
    assert json.loads(json.dumps(result))


def test_execute_quick_exploratory_stats(processed_df: pd.DataFrame) -> None:
    result = execute_analysis_tool("quick_exploratory_stats", {}, processed_df)
    assert result["tool"] == "quick_exploratory_stats"
    assert "records" in result and len(result["records"]) <= 30
    assert json.loads(json.dumps(result))


def test_execute_respects_max_rows(processed_df: pd.DataFrame) -> None:
    result = execute_analysis_tool(
        "summarize_circadian_cosinor", {}, processed_df, max_rows=1
    )
    assert len(result["records"]) <= 1


def test_execute_unknown_tool(processed_df: pd.DataFrame) -> None:
    result = execute_analysis_tool("not_a_real_tool", {}, processed_df)
    assert "error" in result
    assert "unknown" in result["error"].lower()


def test_execute_filters_bogus_arguments(processed_df: pd.DataFrame) -> None:
    # Bogus keys must be dropped (no crash); the call still succeeds.
    result = execute_analysis_tool(
        "summarize_light_dark",
        {"value_col": "value", "totally_unknown_kwarg": 123, "df": "ignored"},
        processed_df,
    )
    assert "error" not in result
    assert "totally_unknown_kwarg" not in result["arguments"]
    assert "df" not in result["arguments"]


def test_execute_internal_error_returns_dict() -> None:
    # When the underlying analysis function GENUINELY raises (here: ``df`` is not
    # a DataFrame, so ``.copy()`` raises ``AttributeError`` inside the function),
    # ``execute_analysis_tool`` must catch it and return an ``{"error": ...}``
    # dict rather than propagating the exception.
    result = execute_analysis_tool("summarize_light_dark", {}, df=None)
    assert "error" in result
    assert result["tool"] == "summarize_light_dark"
    # An int is equally not a DataFrame -> same defensive behaviour, no raise.
    result_int = execute_analysis_tool("summarize_light_dark", {}, df=123)
    assert "error" in result_int
    assert result_int["tool"] == "summarize_light_dark"


def test_execute_bad_value_col_is_graceful(processed_df: pd.DataFrame) -> None:
    # The analysis functions are *defensive*: given a bad ``value_col`` they
    # return an empty result with warnings rather than raising. The executor
    # therefore returns a normal (empty) result dict -- NOT an ``{"error": ...}``
    # dict -- and must not raise.
    result = execute_analysis_tool(
        "summarize_light_dark",
        {"value_col": "definitely_not_a_real_column"},
        processed_df,
    )
    assert "error" not in result
    assert result["tool"] == "summarize_light_dark"
    assert result["n_rows"] == 0
    assert result["records"] == []
    assert isinstance(result["warnings"], list)
    assert len(result["warnings"]) >= 1
    # Still a clean, JSON-serializable result.
    assert json.loads(json.dumps(result))


# --------------------------------------------------------------------------- #
# answer_question -- happy path
# --------------------------------------------------------------------------- #
def test_answer_question_happy_path(processed_df: pd.DataFrame) -> None:
    provider = ScriptedToolProvider(
        [
            _tool_call_turn("call-1", "summarize_light_dark"),
            AssistantTurn(
                text="Across n=6 subjects, the dark/light ratio suggests higher "
                "dark-phase activity.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ]
    )
    result = answer_question("Is activity higher in the dark phase?", processed_df, provider)
    assert isinstance(result, QAResult)
    assert result.answer
    assert INSIGHT_DISCLAIMER in result.answer
    assert result.steps == 2
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call["name"] == "summarize_light_dark"
    assert call["result"]["n_rows"] >= 1
    assert "error" not in call["result"]
    assert result.provider == "scripted"
    # Round-trips to a JSON-able dict.
    assert json.loads(json.dumps(result.to_dict()))


# --------------------------------------------------------------------------- #
# answer_question -- max_steps guard (no infinite loop)
# --------------------------------------------------------------------------- #
def test_answer_question_max_steps_guard(processed_df: pd.DataFrame) -> None:
    # Provider ALWAYS returns a tool call; supply enough turns to cover max_steps.
    turns = [_tool_call_turn(f"call-{i}", "summarize_light_dark") for i in range(10)]
    provider = ScriptedToolProvider(turns)
    result = answer_question(
        "loop forever?", processed_df, provider, max_steps=3
    )
    assert result.steps == 3
    assert INSIGHT_DISCLAIMER in result.answer
    # Each step executed one tool call.
    assert len(result.tool_calls) == 3


# --------------------------------------------------------------------------- #
# AnthropicToolProvider -- key check fires before SDK import
# --------------------------------------------------------------------------- #
def test_anthropic_tool_provider_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = AnthropicToolProvider("some-model")
    with pytest.raises(RuntimeError, match="API key"):
        provider.run("system", [{"role": "user", "content": "hi"}], build_tool_specs())


def test_anthropic_tool_provider_requires_model() -> None:
    with pytest.raises(ValueError, match="model"):
        AnthropicToolProvider("")


# --------------------------------------------------------------------------- #
# system prompt
# --------------------------------------------------------------------------- #
def test_qa_system_prompt_mentions_disclaimer_and_grounding() -> None:
    prompt = build_qa_system_prompt()
    assert INSIGHT_DISCLAIMER in prompt
    assert "tool" in prompt.lower()
