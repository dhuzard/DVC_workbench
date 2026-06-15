"""
Grounded, privacy-preserving LLM-insights layer for DVC behavioral analyses.

This module turns the *already-computed* analysis tables into a small, derived,
JSON-serializable **summary payload** and then into a plain-language narrative.

Core design principle (privacy + hallucination defense):

    The LLM never sees raw per-animal time series.  It only ever receives a
    small, derived, aggregated summary payload, and it *interprets* -- it does
    *not* compute.  The default path is fully OFFLINE and deterministic (no
    network, no API key, no optional dependencies).  Cloud / local model
    providers are opt-in enhancements.  Every output is traceable: it records
    the payload hash and the model id.

The offline baseline (:class:`NullProvider`) produces a deterministic templated
narrative grounded entirely in the payload numbers, so the feature is complete,
testable, and network-free without any LLM.  :class:`OllamaProvider` (fully
local) and :class:`AnthropicProvider` (BYO-key cloud, summary-level egress only)
are optional enhancements layered on top.

Only the standard scientific stack (numpy / pandas / scipy) is required.  The
``anthropic``, ``ollama`` and ``requests`` packages are OPTIONAL and are imported
lazily inside the providers that need them; they are never required for the
default path or for the tests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


INSIGHT_DISCLAIMER = (
    "These insights are exploratory orientation only -- they interpret summary "
    "statistics, not raw data, and are not confirmatory. They may be incomplete "
    "or wrong; verify every number against the analysis tables and consult a "
    "statistician before drawing conclusions."
)


# Canonical table names this module understands.  Unknown keys are still
# serialized generically, but these drive the "highlights" extraction.
KNOWN_TABLE_NAMES = (
    "circadian_summary",
    "light_dark_summary",
    "stats_summary",
    "auc_summary",
    "daily_means",
    "time_bins",
    "nonparametric_circadian",
    "activity_bouts",
)


# --------------------------------------------------------------------------- #
# JSON / numpy helpers
# --------------------------------------------------------------------------- #
def _to_jsonable(value: Any, *, float_round: int = 4) -> Any:
    """Recursively convert numpy/pandas scalars and containers to JSON types."""
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if not math.isfinite(f):
            return None
        return round(f, float_round)
    if isinstance(value, (np.ndarray,)):
        return [_to_jsonable(v, float_round=float_round) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v, float_round=float_round) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v, float_round=float_round) for v in value]
    if isinstance(value, pd.Series):
        return [_to_jsonable(v, float_round=float_round) for v in value.tolist()]
    # pandas NA / NaT and anything else -> string fallback, NA -> None.
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _table_summary(
    df: pd.DataFrame | None,
    *,
    max_rows: int,
    float_round: int = 4,
) -> dict[str, Any]:
    """Compact, JSON-serializable summary of one analysis table."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"shape": [0, 0], "columns": [], "n_rows": 0, "truncated": False, "records": []}

    columns = [str(c) for c in df.columns]
    n_rows = int(len(df))
    truncated = n_rows > max_rows
    head = df.head(max_rows)
    records = [
        {str(k): _to_jsonable(v, float_round=float_round) for k, v in row.items()}
        for row in head.to_dict(orient="records")
    ]
    return {
        "shape": [n_rows, int(df.shape[1])],
        "columns": columns,
        "n_rows": n_rows,
        "truncated": bool(truncated),
        "records": records,
    }


def _safe_records(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    return [
        {str(k): _to_jsonable(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


# --------------------------------------------------------------------------- #
# Highlights extraction (deterministic, payload-driven)
# --------------------------------------------------------------------------- #
def _first_present(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _table_highlights(name: str, df: pd.DataFrame | None) -> dict[str, Any]:
    """Extract a few headline facts from a table without re-computing anything."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    cols = [str(c) for c in df.columns]
    hi: dict[str, Any] = {"metrics": []}

    if "metric_name" in cols:
        hi["metrics"] = sorted(str(m) for m in df["metric_name"].dropna().unique())
    group_col = _first_present(cols, ("group_id", "group", "comparison"))
    if group_col and group_col != "comparison":
        hi["groups"] = sorted(str(g) for g in df[group_col].dropna().unique())

    n_col = _first_present(cols, ("n_subjects", "n_total", "n_points"))
    if n_col:
        hi["n_per_row"] = [_to_jsonable(v) for v in df[n_col].tolist()]

    if name == "stats_summary":
        comp = df
        if "n_groups" in cols:
            comp = df[pd.to_numeric(df["n_groups"], errors="coerce").fillna(0) >= 2]
        if not comp.empty:
            es_col = _first_present(cols, ("effect_size",))
            p_col = _first_present(cols, ("p_value",))
            hi["comparisons"] = [
                {
                    "comparison": str(r.get("comparison", "")),
                    "metric_name": _to_jsonable(r.get("metric_name")),
                    "effect_size": _to_jsonable(r.get(es_col)) if es_col else None,
                    "effect_size_name": _to_jsonable(r.get("effect_size_name")),
                    "p_value": _to_jsonable(r.get(p_col)) if p_col else None,
                    "q_value": _to_jsonable(r.get("q_value")),
                    "n_total": _to_jsonable(r.get("n_total")),
                }
                for r in comp.to_dict(orient="records")
            ]
    elif name == "circadian_summary":
        keep = [
            c
            for c in (
                "metric_name",
                "group_id",
                "amplitude",
                "acrophase_ZT",
                "MESOR",
                "R2",
                "p_rhythm",
                "phase",
            )
            if c in cols
        ]
        if keep:
            hi["rhythms"] = [
                {k: _to_jsonable(r.get(k)) for k in keep}
                for r in df[keep].to_dict(orient="records")
            ]
    elif name == "light_dark_summary":
        ratio_col = _first_present(cols, ("dark_light_ratio",))
        if ratio_col:
            keep = [c for c in ("metric_name", "group_id", "phase", "mean", ratio_col) if c in cols]
            hi["light_dark"] = [
                {k: _to_jsonable(r.get(k)) for k in keep}
                for r in df[keep].to_dict(orient="records")
            ]
    elif name == "nonparametric_circadian":
        keep = [
            c
            for c in ("metric_name", "group_id", "IS", "IV", "RA", "M10", "L5")
            if c in cols
        ]
        if keep:
            hi["nonparametric"] = [
                {k: _to_jsonable(r.get(k)) for k in keep}
                for r in df[keep].to_dict(orient="records")
            ]

    return {k: v for k, v in hi.items() if v}


# --------------------------------------------------------------------------- #
# Config / manifest / QC summaries
# --------------------------------------------------------------------------- #
def _config_summary(analysis_config: dict | None) -> dict[str, Any]:
    if not analysis_config:
        return {}
    cfg = analysis_config
    ld = cfg.get("light_dark_cycle", {}) or {}
    aln = cfg.get("alignment", {}) or {}
    bsl = cfg.get("baseline", {}) or {}
    summary = {
        "timezone": cfg.get("timezone"),
        "light_on": ld.get("light_on"),
        "light_off": ld.get("light_off"),
        "alignment_event": aln.get("event_type"),
        "alignment_scope": aln.get("scope"),
        "baseline_start_hours": bsl.get("start_hours"),
        "baseline_end_hours": bsl.get("end_hours"),
        "aggregation_bin_seconds": cfg.get("aggregation_bin_seconds"),
        "cosinor_period_hours": cfg.get("cosinor_period_hours"),
        "fdr_method": cfg.get("fdr_method"),
    }
    return _to_jsonable({k: v for k, v in summary.items() if v is not None})


def _manifest_summary(manifest: dict | None) -> dict[str, Any]:
    if not manifest:
        return {}
    inputs = manifest.get("inputs") or manifest.get("input_files") or manifest.get("uploaded_files") or []
    hashes: list[str] = []
    if isinstance(inputs, (list, tuple)):
        n_inputs = len(inputs)
        for item in inputs:
            if isinstance(item, dict):
                h = item.get("sha256") or item.get("hash")
                if h:
                    hashes.append(str(h))
    else:
        n_inputs = 0
    summary = {
        "app_version": manifest.get("app_version") or manifest.get("version"),
        "generated_at": manifest.get("generated_at") or manifest.get("created_at"),
        "input_file_count": n_inputs,
        "input_hashes": hashes,
    }
    return _to_jsonable({k: v for k, v in summary.items() if v not in (None, [], 0) or k == "input_file_count"})


_QC_FLAGS = (
    ("irregular_interval_flag", "irregular sampling interval"),
    ("zero_variance_flag", "zero variance (flat signal)"),
)
_QC_COUNT_FLAGS = (
    ("missing_value_count", "missing values"),
    ("duplicate_timestamp_count", "duplicate timestamps"),
    ("long_gap_count", "long gaps"),
    ("negative_value_count", "negative values"),
    ("missing_timestamp_count", "missing timestamps"),
)


def _qc_summary(quality_report: pd.DataFrame | None) -> dict[str, Any]:
    if quality_report is None or not isinstance(quality_report, pd.DataFrame) or quality_report.empty:
        return {}
    df = quality_report
    cols = set(df.columns)
    counts: dict[str, int] = {"n_streams": int(len(df))}
    flagged_mask = pd.Series(False, index=df.index)

    for col, _ in _QC_FLAGS:
        if col in cols:
            mask = df[col].fillna(False).astype(bool)
            counts[f"{col}_subjects"] = int(mask.sum())
            flagged_mask |= mask
    for col, _ in _QC_COUNT_FLAGS:
        if col in cols:
            mask = pd.to_numeric(df[col], errors="coerce").fillna(0) > 0
            counts[f"{col}_subjects"] = int(mask.sum())
            flagged_mask |= mask

    counts["n_flagged_streams"] = int(flagged_mask.sum())
    if "subject_id" in cols:
        counts["n_flagged_subjects"] = int(df.loc[flagged_mask, "subject_id"].nunique())
    return _to_jsonable(counts)


# --------------------------------------------------------------------------- #
# 1. build_insight_payload
# --------------------------------------------------------------------------- #
def build_insight_payload(
    analysis_tables: dict[str, pd.DataFrame],
    analysis_config: dict | None = None,
    manifest: dict | None = None,
    quality_report: pd.DataFrame | None = None,
    *,
    max_rows_per_table: int = 50,
) -> dict:
    """Build a compact, JSON-serializable summary payload for the LLM layer.

    PURE and deterministic.  The returned dict captures only *headline,
    aggregated* numbers -- never raw per-animal time series.  For each table it
    records the shape, columns and a truncated, float-rounded records list
    (capped at ``max_rows_per_table`` rows), plus a small derived ``highlights``
    section (groups compared, metrics present, per-row n where available).

    Safe with missing/empty inputs: any ``None``/empty table is summarized as an
    empty entry and optional sections are simply omitted.  All numbers are
    converted from numpy/pandas types so :func:`json.dumps` always succeeds.

    Parameters
    ----------
    analysis_tables:
        Mapping of csv-ish table name (e.g. ``"circadian_summary"``,
        ``"light_dark_summary"``, ``"stats_summary"``, ``"auc_summary"``,
        ``"daily_means"``, ``"nonparametric_circadian"``, ``"activity_bouts"``)
        to its DataFrame.
    analysis_config, manifest, quality_report:
        Optional context; summarized into ``config``, ``manifest`` and ``qc``
        sections when provided.
    max_rows_per_table:
        Maximum number of rows kept per table in the payload.
    """
    tables: dict[str, Any] = {}
    highlights: dict[str, Any] = {}
    analysis_tables = analysis_tables or {}

    for name in sorted(analysis_tables.keys(), key=str):
        df = analysis_tables[name]
        key = str(name)
        tables[key] = _table_summary(df, max_rows=max_rows_per_table)
        hi = _table_highlights(key, df)
        if hi:
            highlights[key] = hi

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "dvc_insight_payload",
        "tables": tables,
        "highlights": highlights,
        "disclaimer": INSIGHT_DISCLAIMER,
    }

    config = _config_summary(analysis_config)
    if config:
        payload["config"] = config
    manifest_summary = _manifest_summary(manifest)
    if manifest_summary:
        payload["manifest"] = manifest_summary
    qc = _qc_summary(quality_report)
    if qc:
        payload["qc"] = qc

    # Guarantee JSON-serializability defensively.
    return _to_jsonable(payload)


# --------------------------------------------------------------------------- #
# 2. payload_hash
# --------------------------------------------------------------------------- #
def payload_hash(payload: dict) -> str:
    """Deterministic SHA256 over the canonical (sorted-key) JSON of ``payload``."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# 3. InsightResult
# --------------------------------------------------------------------------- #
@dataclass
class InsightResult:
    """A generated narrative plus full traceability metadata."""

    text: str
    provider: str
    model_id: str
    payload_sha256: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    disclaimer: str = INSIGHT_DISCLAIMER
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 4. Provider abstraction
# --------------------------------------------------------------------------- #
@runtime_checkable
class LLMProvider(Protocol):
    """Protocol every insight provider must satisfy.

    ``complete`` receives the system prompt and the (already-derived) summary
    payload and returns ``(text, prompt_tokens, completion_tokens)`` where the
    token counts may be ``None`` when the provider does not report them.
    """

    name: str
    model_id: str

    def complete(
        self, system_prompt: str, payload: dict
    ) -> tuple[str, int | None, int | None]:
        ...


# --------- narrative formatting helpers shared by NullProvider -------------- #
def _fmt_num(value: Any, digits: int = 2) -> str | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f"{f:.{digits}f}"


def _sentence_groups(highlights: dict) -> list[str]:
    out: list[str] = []
    seen_groups: set[str] = set()
    for tbl in highlights.values():
        for g in tbl.get("groups", []):
            seen_groups.add(str(g))
    if seen_groups:
        out.append(
            "Groups compared: " + ", ".join(sorted(seen_groups)) + "."
        )
    return out


def _sentence_stats(highlights: dict) -> list[str]:
    out: list[str] = []
    stats = highlights.get("stats_summary", {})
    for comp in stats.get("comparisons", []):
        comparison = comp.get("comparison") or "groups"
        metric = comp.get("metric_name")
        es = _fmt_num(comp.get("effect_size"))
        es_name = comp.get("effect_size_name") or "effect size"
        p = _fmt_num(comp.get("p_value"), digits=3)
        q = _fmt_num(comp.get("q_value"), digits=3)
        n = comp.get("n_total")
        parts = [f"For {comparison}"]
        if metric:
            parts[0] += f" ({metric})"
        detail: list[str] = []
        if es is not None:
            detail.append(f"{es_name} = {es}")
        if p is not None:
            detail.append(f"exploratory p = {p}")
        if q is not None:
            detail.append(f"FDR q = {q}")
        if n is not None:
            detail.append(f"n = {n}")
        if detail:
            out.append(parts[0] + ": " + ", ".join(detail) + ".")
    return out


def _sentence_circadian(highlights: dict) -> list[str]:
    out: list[str] = []
    for rhythm in highlights.get("circadian_summary", {}).get("rhythms", []):
        group = rhythm.get("group_id")
        metric = rhythm.get("metric_name")
        amp = _fmt_num(rhythm.get("amplitude"))
        acro = _fmt_num(rhythm.get("acrophase_ZT"))
        p = _fmt_num(rhythm.get("p_rhythm"), digits=3)
        if amp is None and acro is None:
            continue
        label = " / ".join(str(x) for x in (group, metric) if x)
        prefix = f"Circadian rhythm for {label}" if label else "Circadian rhythm"
        detail = []
        if amp is not None:
            detail.append(f"amplitude {amp}")
        if acro is not None:
            detail.append(f"acrophase ZT {acro} h")
        if p is not None:
            detail.append(f"rhythmicity p = {p}")
        out.append(prefix + ": " + ", ".join(detail) + ".")
    return out


def _sentence_light_dark(highlights: dict) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in highlights.get("light_dark_summary", {}).get("light_dark", []):
        ratio = _fmt_num(row.get("dark_light_ratio"))
        if ratio is None:
            continue
        group = row.get("group_id")
        metric = row.get("metric_name")
        key = f"{group}|{metric}"
        if key in seen:
            continue
        seen.add(key)
        label = " / ".join(str(x) for x in (group, metric) if x)
        prefix = f"Dark/light activity ratio for {label}" if label else "Dark/light activity ratio"
        out.append(f"{prefix}: {ratio}.")
    return out


def _sentence_nonparametric(highlights: dict) -> list[str]:
    out: list[str] = []
    for row in highlights.get("nonparametric_circadian", {}).get("nonparametric", []):
        group = row.get("group_id")
        metric = row.get("metric_name")
        parts = []
        for key in ("IS", "IV", "RA"):
            val = _fmt_num(row.get(key))
            if val is not None:
                parts.append(f"{key} = {val}")
        if not parts:
            continue
        label = " / ".join(str(x) for x in (group, metric) if x)
        prefix = f"Non-parametric circadian metrics for {label}" if label else "Non-parametric circadian metrics"
        out.append(prefix + ": " + ", ".join(parts) + ".")
    return out


def render_offline_narrative(payload: dict) -> str:
    """Deterministic, grounded plain-language narrative built from the payload.

    Each sentence is derived directly from numbers present in ``payload`` -- no
    model, no network, no invention.  Used by :class:`NullProvider`.
    """
    highlights = payload.get("highlights", {}) or {}
    config = payload.get("config", {}) or {}
    qc = payload.get("qc", {}) or {}

    sentences: list[str] = []
    sentences.append(
        "Exploratory summary of the DVC behavioral analysis (interprets summary "
        "tables only; computes nothing new)."
    )
    if config.get("timezone") or config.get("light_on") is not None:
        ld = ""
        if config.get("light_on") is not None and config.get("light_off") is not None:
            ld = f", light cycle {config.get('light_on')}-{config.get('light_off')}"
        sentences.append(
            f"Analysis timezone {config.get('timezone', 'unknown')}{ld}."
        )

    sentences += _sentence_groups(highlights)
    sentences += _sentence_stats(highlights)
    sentences += _sentence_circadian(highlights)
    sentences += _sentence_light_dark(highlights)
    sentences += _sentence_nonparametric(highlights)

    if qc:
        flagged = qc.get("n_flagged_streams", qc.get("n_flagged_subjects"))
        if flagged is not None:
            sentences.append(
                f"Quality control: {flagged} of {qc.get('n_streams', '?')} "
                "subject/metric streams were flagged for review."
            )

    if len(sentences) <= 2:
        sentences.append(
            "No group comparisons, circadian parameters or light/dark ratios were "
            "present in the supplied tables, so no quantitative narrative could be "
            "produced."
        )

    return " ".join(sentences)


class NullProvider:
    """Default OFFLINE provider: deterministic templated narrative, no LLM.

    Produces a genuinely useful plain-language summary built directly from the
    payload numbers (group comparisons with effect sizes / p-values / n,
    circadian amplitude / acrophase / rhythmicity p, dark/light ratios, and
    IS / IV / RA when present).  Requires no network, no API key and no optional
    dependencies, which makes it the testable baseline for the whole feature.
    """

    name = "null"
    model_id = "offline-template"

    def complete(
        self, system_prompt: str, payload: dict
    ) -> tuple[str, int | None, int | None]:
        return render_offline_narrative(payload), None, None


class OllamaProvider:
    """Fully local provider that calls an Ollama server (no external egress).

    Lazy-imports ``requests``.  Raises a clear :class:`RuntimeError` with
    guidance when ``requests`` is missing or the Ollama host is unreachable.
    """

    def __init__(self, model: str, host: str = "http://localhost:11434") -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.name = "ollama"
        self.model_id = model

    def complete(
        self, system_prompt: str, payload: dict
    ) -> tuple[str, int | None, int | None]:
        try:
            import requests  # type: ignore  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "OllamaProvider requires the optional 'requests' package. "
                "Install it with `pip install requests`, or use the default "
                "offline NullProvider."
            ) from exc

        user_content = (
            "Interpret ONLY the following analysis summary payload (JSON). "
            "Do not invent numbers.\n\n" + json.dumps(payload, sort_keys=True)
        )
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "options": {"temperature": 0.1},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                },
                timeout=120,
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - normalize to a guidance error
            raise RuntimeError(
                f"Could not reach Ollama at {self.host} for model '{self.model}'. "
                "Ensure Ollama is installed and running (`ollama serve`) and the "
                f"model is pulled (`ollama pull {self.model}`). Original error: {exc}"
            ) from exc

        data = response.json()
        text = (data.get("message", {}) or {}).get("content", "")
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        return text, prompt_tokens, completion_tokens


class AnthropicProvider:
    """BYO-key cloud provider (Anthropic Claude); egress = summary tables only.

    Lazy-imports the ``anthropic`` SDK and reads the key from ``api_key`` or the
    ``ANTHROPIC_API_KEY`` environment variable.  Uses a low temperature and a
    system prompt that forbids inventing numbers and requires citing the
    payload.  Raises a clear :class:`RuntimeError` if the SDK or key is missing.

    ``model`` is a REQUIRED argument: pass the latest Claude model id from the
    caller (e.g. configuration) rather than hardcoding a value that will rot.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        max_tokens: int = 1024,
    ) -> None:
        if not model:
            raise ValueError("AnthropicProvider requires an explicit 'model' id.")
        self.model = model
        self.name = "anthropic"
        self.model_id = model
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def complete(
        self, system_prompt: str, payload: dict
    ) -> tuple[str, int | None, int | None]:
        if not self._api_key:
            raise RuntimeError(
                "AnthropicProvider requires an API key. Pass api_key=... or set "
                "the ANTHROPIC_API_KEY environment variable. No data is sent "
                "until a key is configured; the offline NullProvider needs no key."
            )
        try:
            import anthropic  # type: ignore  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "AnthropicProvider requires the optional 'anthropic' SDK. "
                "Install it with `pip install anthropic`, or use the default "
                "offline NullProvider."
            ) from exc

        client = anthropic.Anthropic(api_key=self._api_key)
        user_content = (
            "Interpret ONLY the following analysis summary payload (JSON). Cite "
            "its numbers and do not invent any statistic.\n\n"
            + json.dumps(payload, sort_keys=True)
        )
        # Low temperature keeps the interpretation deterministic. Newer Claude
        # models (Opus 4.7+/Fable) reject the `temperature` parameter, so fall
        # back to a temperature-free request if the first call is rejected for
        # that reason; grounding is enforced by the system prompt regardless.
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_content}],
        }
        try:
            try:
                message = client.messages.create(temperature=0.0, **create_kwargs)
            except Exception as exc:  # noqa: BLE001 - retry without temperature
                if "temperature" not in str(exc).lower():
                    raise
                message = client.messages.create(**create_kwargs)
        except Exception as exc:  # noqa: BLE001 - normalize to a guidance error
            raise RuntimeError(
                f"Anthropic request failed for model '{self.model}': {exc}"
            ) from exc

        text_parts = [
            getattr(block, "text", "")
            for block in getattr(message, "content", [])
            if getattr(block, "type", None) == "text"
        ]
        text = "".join(text_parts)
        usage = getattr(message, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", None) if usage else None
        completion_tokens = getattr(usage, "output_tokens", None) if usage else None
        return text, prompt_tokens, completion_tokens


# --------------------------------------------------------------------------- #
# 6. build_system_prompt
# --------------------------------------------------------------------------- #
def build_system_prompt() -> str:
    """System prompt constraining the model to grounded, cautious interpretation."""
    return (
        "You are a cautious behavioral-neuroscience analyst helping interpret "
        "summary statistics from a Digital Ventilated Cage (DVC) rodent "
        "home-cage experiment.\n"
        "Rules you must follow:\n"
        "1. Interpret ONLY the numbers provided in the summary payload. Never "
        "compute, estimate, or invent any statistic, p-value, effect size or "
        "sample size that is not present in the payload.\n"
        "2. Quote the payload's numbers when you make a claim, and always state "
        "the sample size (n) alongside any group comparison.\n"
        "3. Keep all framing EXPLORATORY. Never describe a result as "
        "'significant', 'confirmed' or 'proven'; prefer 'suggests', "
        "'consistent with', 'warrants follow-up'.\n"
        "4. If a number is missing or NaN, say so rather than guessing.\n"
        "5. Be concise, plain-language and accessible to a non-statistician.\n"
        "End your answer with the following disclaimer verbatim:\n"
        f"{INSIGHT_DISCLAIMER}"
    )


# --------------------------------------------------------------------------- #
# 7. generate_narrative
# --------------------------------------------------------------------------- #
def generate_narrative(
    payload: dict,
    provider: LLMProvider | None = None,
    *,
    disclaimer: str = INSIGHT_DISCLAIMER,
) -> InsightResult:
    """Generate a traceable narrative from a payload.

    Defaults to the fully offline :class:`NullProvider`.  Computes the payload
    hash, calls ``provider.complete``, ensures ``disclaimer`` is present in the
    text (appending it when absent) and returns a populated
    :class:`InsightResult`.
    """
    provider = provider or NullProvider()
    sha = payload_hash(payload)
    system_prompt = build_system_prompt()
    text, prompt_tokens, completion_tokens = provider.complete(system_prompt, payload)
    text = text or ""
    if disclaimer and disclaimer not in text:
        text = (text.rstrip() + "\n\n" + disclaimer).strip()
    return InsightResult(
        text=text,
        provider=getattr(provider, "name", provider.__class__.__name__),
        model_id=getattr(provider, "model_id", "unknown"),
        payload_sha256=sha,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        disclaimer=disclaimer,
    )


# --------------------------------------------------------------------------- #
# 8. draft_methods_section
# --------------------------------------------------------------------------- #
def draft_methods_section(analysis_config: dict, manifest: dict | None = None) -> str:
    """Deterministic, template-based reproducible Methods paragraph from config.

    Transcribes the configuration (binning, timezone / light cycle, alignment
    event, baseline window, statistical tests + FDR) into prose.  No LLM is
    involved, so hallucination risk is essentially zero.
    """
    cfg = analysis_config or {}
    ld = cfg.get("light_dark_cycle", {}) or {}
    aln = cfg.get("alignment", {}) or {}
    bsl = cfg.get("baseline", {}) or {}

    tz = cfg.get("timezone", "the configured timezone")
    light_on = ld.get("light_on", "?")
    light_off = ld.get("light_off", "?")
    bin_seconds = cfg.get("aggregation_bin_seconds")
    bin_text = (
        f"{bin_seconds} s bins" if bin_seconds else "the native acquisition cadence"
    )
    period = cfg.get("cosinor_period_hours", 24)
    fdr = cfg.get("fdr_method", "Benjamini-Hochberg")

    sentences: list[str] = []
    sentences.append(
        f"Home-cage activity from the Digital Ventilated Cage system was "
        f"aggregated to {bin_text} and timestamps were expressed in {tz}."
    )
    sentences.append(
        f"A {light_on}:{light_off} light/dark cycle was used to assign each "
        "observation to the light or dark phase."
    )
    if aln.get("event_type"):
        sentences.append(
            f"Time series were aligned to the '{aln.get('event_type')}' event "
            f"(scope: {aln.get('scope', 'per-subject')}), with time expressed "
            "relative to that event."
        )
    if bsl.get("start_hours") is not None or bsl.get("end_hours") is not None:
        sentences.append(
            f"A baseline window from {bsl.get('start_hours', '?')} h to "
            f"{bsl.get('end_hours', '?')} h relative to alignment was used for "
            "per-subject baseline normalization."
        )
    sentences.append(
        f"Circadian rhythmicity was assessed with a cosinor model (period "
        f"{period} h), reporting MESOR, amplitude and acrophase with a "
        "zero-amplitude rhythm-detection test."
    )
    sentences.append(
        "Group differences were assessed at the subject level using "
        "distribution-free tests (Mann-Whitney U for two groups, Kruskal-Wallis "
        "for more), reporting effect sizes; p-values were corrected for multiple "
        f"comparisons with the {fdr} false-discovery-rate procedure. All "
        "statistics are exploratory."
    )
    if manifest:
        ms = _manifest_summary(manifest)
        version = ms.get("app_version")
        n_files = ms.get("input_file_count")
        bits = []
        if version:
            bits.append(f"DVC Behavioral Workbench {version}")
        if n_files:
            bits.append(f"{n_files} input file(s)")
        if bits:
            sentences.append(
                "Processing was performed with " + ", ".join(bits) + "; inputs "
                "were SHA256-hashed for provenance."
            )

    return " ".join(sentences)


# --------------------------------------------------------------------------- #
# 9. triage_quality
# --------------------------------------------------------------------------- #
_TRIAGE_RULES: tuple[tuple[str, str, str, str], ...] = (
    # (column, kind, label, suggested_action)
    (
        "irregular_interval_flag",
        "flag",
        "irregular sampling interval",
        "Inspect acquisition logs for dropouts; consider excluding or resampling.",
    ),
    (
        "zero_variance_flag",
        "flag",
        "zero variance (flat signal)",
        "Check the sensor/channel; a flat trace usually indicates a hardware fault.",
    ),
    (
        "missing_value_count",
        "count",
        "missing values",
        "Review imputation/exclusion; high missingness can bias group means.",
    ),
    (
        "duplicate_timestamp_count",
        "count",
        "duplicate timestamps",
        "De-duplicate the stream before analysis to avoid double counting.",
    ),
    (
        "long_gap_count",
        "count",
        "long gaps",
        "Verify coverage of the analysis window; long gaps can break alignment.",
    ),
    (
        "negative_value_count",
        "count",
        "negative values",
        "Negative locomotion is non-physical; check parsing/units.",
    ),
)


def triage_quality(
    quality_report: pd.DataFrame,
    exclusion_log: pd.DataFrame | None = None,
) -> tuple[str, list[dict]]:
    """Deterministic QC triage from the quality report.

    Scans for flagged subjects (``irregular_interval_flag``,
    ``missing_value_count`` > 0, ``duplicate_timestamp_count`` > 0,
    ``long_gap_count`` > 0, ``negative_value_count`` > 0,
    ``zero_variance_flag``) and returns ``(summary_text, issues)`` where
    ``issues`` is a structured list of one dict per detected problem, each with a
    suggested action.  Safe on empty input.
    """
    issues: list[dict] = []
    if (
        quality_report is None
        or not isinstance(quality_report, pd.DataFrame)
        or quality_report.empty
    ):
        return "No quality issues to triage (the quality report was empty).", issues

    df = quality_report
    cols = set(df.columns)
    for row in df.to_dict(orient="records"):
        subject = row.get("subject_id")
        metric = row.get("metric_name")
        for col, kind, label, action in _TRIAGE_RULES:
            if col not in cols:
                continue
            value = row.get(col)
            if kind == "flag":
                if bool(value) is True:
                    issues.append(
                        {
                            "subject_id": _to_jsonable(subject),
                            "metric_name": _to_jsonable(metric),
                            "issue": label,
                            "detail": col,
                            "value": True,
                            "suggested_action": action,
                        }
                    )
            else:  # count
                numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.notna(numeric) and numeric > 0:
                    issues.append(
                        {
                            "subject_id": _to_jsonable(subject),
                            "metric_name": _to_jsonable(metric),
                            "issue": label,
                            "detail": col,
                            "value": int(numeric),
                            "suggested_action": action,
                        }
                    )

    n_streams = int(len(df))
    flagged_subjects = sorted({str(i["subject_id"]) for i in issues if i["subject_id"] is not None})

    if not issues:
        summary = (
            f"All {n_streams} subject/metric streams passed the QC checks; "
            "no triage actions are suggested."
        )
    else:
        already_excluded = ""
        if exclusion_log is not None and isinstance(exclusion_log, pd.DataFrame) and not exclusion_log.empty:
            if "subject_id" in exclusion_log.columns:
                excl = sorted({str(s) for s in exclusion_log["subject_id"].dropna().unique()})
                if excl:
                    already_excluded = (
                        f" Note: {len(excl)} subject(s) already appear in the "
                        "exclusion log."
                    )
        kinds = sorted({i["issue"] for i in issues})
        summary = (
            f"{len(issues)} QC issue(s) across {len(flagged_subjects)} subject(s) "
            f"({', '.join(flagged_subjects)}) out of {n_streams} stream(s). "
            f"Issue types: {', '.join(kinds)}." + already_excluded
        )

    return summary, issues


# --------------------------------------------------------------------------- #
# 10. ANALYSIS_TOOL_REGISTRY (scaffold for grounded tool-calling Q&A)
# --------------------------------------------------------------------------- #
ANALYSIS_TOOL_REGISTRY: dict[str, str] = {
    "summarize_circadian_cosinor": (
        "Fit a cosinor model per group/metric and return MESOR, amplitude, "
        "acrophase, R2 and a rhythm-detection p-value with CIs."
    ),
    "summarize_light_dark": (
        "Summarize mean +/- SEM by light vs dark phase and the dark/light ratio "
        "per group/metric."
    ),
    "summarize_time_bins": (
        "Summarize group means over daily/weekly/custom time bins, relative to "
        "alignment or absolute calendar time."
    ),
    "compute_auc_per_animal": (
        "Compute trapezoidal area-under-curve per animal/metric over an optional "
        "time window."
    ),
    "quick_exploratory_stats": (
        "Run subject-level exploratory group comparisons (Mann-Whitney / "
        "Kruskal-Wallis) with effect sizes and Benjamini-Hochberg FDR q-values."
    ),
    "summarize_nonparametric_circadian": (
        "Compute distribution-free circadian metrics (IS, IV, RA, M10, L5) per "
        "group/metric."
    ),
    "summarize_activity_bouts": (
        "Summarize active/inactive bout structure (count, mean/longest duration, "
        "percent time active) given an activity threshold."
    ),
    "compare_window_summaries": (
        "Compare per-subject window summaries (e.g. baseline vs treatment) for a "
        "single contrast per metric."
    ),
    "estimate_period": (
        "Estimate the dominant rhythmic period (e.g. via periodogram) for "
        "free-running or tau-mutant designs."
    ),
}
"""Scaffolding for a future grounded function-calling Q&A agent.

Maps the public ``analysis.py`` function names to one-line descriptions so a
tool-calling model can answer numeric questions by CALLING these real functions
on the data -- the model never invents numbers, it requests a computation and
interprets the returned table.  Some entries are forward-looking and may be
added to ``analysis.py`` in later phases.
"""


__all__ = [
    "ANALYSIS_TOOL_REGISTRY",
    "AnthropicProvider",
    "INSIGHT_DISCLAIMER",
    "InsightResult",
    "KNOWN_TABLE_NAMES",
    "LLMProvider",
    "NullProvider",
    "OllamaProvider",
    "build_insight_payload",
    "build_system_prompt",
    "draft_methods_section",
    "generate_narrative",
    "payload_hash",
    "render_offline_narrative",
    "triage_quality",
]
