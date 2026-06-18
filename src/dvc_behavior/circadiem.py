"""HTTP client and schema mirror for the **Circadiem** circadian-scoring service.

Circadiem is a standalone, stateless HTTP service that takes circadian activity
PNG plots, scores six circadian markers on a fixed 0--3 rubric using an OpenAI
vision model, and returns structured JSON.  It stores nothing: the OpenAI key is
passed per request as a bearer token and never persisted.  It does **not**
generate plots -- DVC_workbench produces the PNGs; Circadiem only scores them.

This module is the workbench-side client.  Following the library/app split it
must **not** import ``streamlit``; the UI lives in ``app/streamlit_app.py``.  The
``requests`` dependency is OPTIONAL and lazily imported inside the HTTP seam, so
nothing here is required for the default path or for the tests (which monkeypatch
the ``_http_*`` seam and never touch the network).

Design notes:

* **Schema mirror.**  DVC_workbench is Python, not Node/TS, so it cannot depend
  on ``@circadiem/schema``.  :class:`ResultRow` / :class:`ErrorRow` mirror the
  service contract; ``meta`` is injected server-side and trusted, and ``run_id``
  (shared by all rows in a batch) groups a run.
* **Plot convention.**  The rubric assumes dark onset at ``x = 0``
  (``aligned_to_dark=true``), a global mean curve in black, and a ``+-2SD``
  variability band.  Render PNGs with :func:`qc.plot_circadiem_vcg` so the plot
  and the rubric agree, and pass the matching ``aligned_to_dark`` / ``vcg_band``.
* **Failures are per image.**  A single bad image yields an :class:`ErrorRow`,
  not a failed batch -- :func:`results_to_frame` keeps error rows so failures are
  surfaced rather than silently dropped.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

__all__ = [
    "BASE_URL_ENV",
    "OPENAI_KEY_ENV",
    "DEFAULT_MODEL",
    "DEFAULT_ALIGNED_TO_DARK",
    "DEFAULT_VCG_BAND",
    "VCG_BANDS",
    "MARKER_FIELDS",
    "CONFIDENCE_LEVELS",
    "MAX_FILES",
    "MAX_FILE_BYTES",
    "MAX_DIMENSION",
    "CircadiemError",
    "CircadiemConfig",
    "MarkerScores",
    "ResultRow",
    "ErrorRow",
    "resolve_base_url",
    "resolve_api_key",
    "validate_openai_key",
    "validate_png",
    "parse_row",
    "analyze",
    "health",
    "get_prompt",
    "results_to_frame",
]


# --------------------------------------------------------------------------- #
# Constants — the service contract (§3/§4 of the integration spec)
# --------------------------------------------------------------------------- #
BASE_URL_ENV = "CIRCADIEM_BASE_URL"
OPENAI_KEY_ENV = "OPENAI_API_KEY"

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_ALIGNED_TO_DARK = True
DEFAULT_VCG_BAND = "+-2SD"
VCG_BANDS = ("+-1SD", "+-2SD", "+-3SD")

# The six markers, in the contract's order.  Each is an integer in 0|1|2|3.
MARKER_FIELDS = (
    "baseline_light",
    "dark_onset_burst",
    "dark_irregularity",
    "midnight_fragmentation",
    "pre_light_decline",
    "pre_dark_anticipation",
)
CONFIDENCE_LEVELS = ("low", "med", "high")

# Request limits (§7).  Batch larger jobs client-side.
MAX_FILES = 20
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_DIMENSION = 8192
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# A batch caps at 2 concurrent calls and 60 s/call with one JSON-repair retry,
# so a full 20-image batch can run well over a minute; give it generous headroom.
DEFAULT_TIMEOUT = 300.0
_HEALTH_TIMEOUT = 10.0


class CircadiemError(RuntimeError):
    """Raised for client-side validation failures and normalized HTTP errors.

    Carries a human-readable, actionable message (the UI surfaces ``str(exc)``).
    """


# --------------------------------------------------------------------------- #
# Schema mirror (§4)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MarkerScores:
    """The six circadian marker scores, each an integer in ``0|1|2|3``."""

    baseline_light: int
    dark_onset_burst: int
    dark_irregularity: int
    midnight_fragmentation: int
    pre_light_decline: int
    pre_dark_anticipation: int

    def as_dict(self) -> dict[str, int]:
        return {field_name: getattr(self, field_name) for field_name in MARKER_FIELDS}


@dataclass(frozen=True)
class ResultRow:
    """A successfully scored image.

    ``meta`` is injected server-side and trustworthy (never model-controlled);
    ``run_id`` is shared by every row in one batch.  ``raw`` keeps the original
    JSON row for provenance.
    """

    label: str
    markers: MarkerScores
    confidence: str
    flags: list[str] = field(default_factory=list)
    notes: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str | None:
        value = (self.meta or {}).get("run_id")
        return str(value) if value is not None else None

    @property
    def is_error(self) -> bool:
        return False


@dataclass(frozen=True)
class ErrorRow:
    """A single image that failed to score (the batch still succeeds)."""

    label: str
    error: str
    meta: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str | None:
        value = (self.meta or {}).get("run_id")
        return str(value) if value is not None else None

    @property
    def is_error(self) -> bool:
        return True


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class CircadiemConfig:
    """Where Circadiem lives and how to score against it."""

    base_url: str
    model: str = DEFAULT_MODEL
    aligned_to_dark: bool = DEFAULT_ALIGNED_TO_DARK
    vcg_band: str = DEFAULT_VCG_BAND
    timeout: float = DEFAULT_TIMEOUT

    def __post_init__(self) -> None:
        self.base_url = (self.base_url or "").strip().rstrip("/")
        if self.vcg_band not in VCG_BANDS:
            raise CircadiemError(f"vcg_band must be one of {VCG_BANDS}, got {self.vcg_band!r}.")

    def url(self, path: str) -> str:
        if not self.base_url:
            raise CircadiemError(
                "No Circadiem base URL configured. Set the "
                f"{BASE_URL_ENV} environment variable or enter the service URL."
            )
        return f"{self.base_url}/{path.lstrip('/')}"


# --------------------------------------------------------------------------- #
# Resolution / validation helpers
# --------------------------------------------------------------------------- #
def resolve_base_url(explicit: str | None = None) -> str:
    """Resolve the base URL from an explicit value, else the env var, else ``""``."""
    if explicit and explicit.strip():
        return explicit.strip().rstrip("/")
    return (os.environ.get(BASE_URL_ENV) or "").strip().rstrip("/")


def resolve_api_key(explicit: str | None = None) -> str | None:
    """Resolve the OpenAI key from an explicit value, else ``OPENAI_API_KEY``.

    Decision §5.1(b) — server-side proxy key: the workbench holds one org key in
    the environment and forwards it as the bearer token.  The key is never
    persisted by the workbench or by Circadiem.
    """
    if explicit and explicit.strip():
        return explicit.strip()
    value = os.environ.get(OPENAI_KEY_ENV)
    return value.strip() if value and value.strip() else None


def validate_openai_key(key: str | None) -> None:
    """Validate the bearer token shape Circadiem requires (``sk-`` prefix, ≥20 chars)."""
    if not key:
        raise CircadiemError(
            "No OpenAI API key available. Set the "
            f"{OPENAI_KEY_ENV} environment variable on the host running the "
            "workbench (it is forwarded to Circadiem as a bearer token and never stored)."
        )
    if not key.startswith("sk-") or len(key) < 20:
        raise CircadiemError(
            "OpenAI API key looks malformed: it must start with 'sk-' and be at "
            "least 20 characters."
        )


def validate_png(data: bytes, *, name: str = "image") -> tuple[int, int]:
    """Validate PNG signature, dimensions and size; return ``(width, height)``.

    Raises :class:`CircadiemError` with an actionable message on any violation
    so a malformed plot is caught before egress rather than after a wasted call.
    """
    if not isinstance(data, (bytes, bytearray)) or len(data) < 24:
        raise CircadiemError(f"{name}: not a valid PNG (too short).")
    if not bytes(data[:8]) == PNG_SIGNATURE:
        raise CircadiemError(f"{name}: not a PNG (bad signature). Circadiem accepts PNG only.")
    if len(data) > MAX_FILE_BYTES:
        raise CircadiemError(
            f"{name}: {len(data) / 1e6:.1f} MB exceeds the {MAX_FILE_BYTES / 1e6:.0f} MB limit."
        )
    # IHDR is the first chunk: width/height are big-endian uint32 at bytes 16..24.
    width = int.from_bytes(bytes(data[16:20]), "big")
    height = int.from_bytes(bytes(data[20:24]), "big")
    if width <= 0 or height <= 0:
        raise CircadiemError(f"{name}: could not read PNG dimensions.")
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise CircadiemError(
            f"{name}: {width}x{height}px exceeds the {MAX_DIMENSION}x{MAX_DIMENSION}px limit."
        )
    return width, height


def _coerce_marker(value: Any) -> int | None:
    """Coerce a marker score to an int in ``0..3``; return ``None`` if impossible."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    if 0 <= score <= 3:
        return score
    return None


def parse_row(row: dict[str, Any]) -> ResultRow | ErrorRow:
    """Parse one JSON row into a :class:`ResultRow` or :class:`ErrorRow`.

    Per §4, an error is detected via the presence of ``error`` or the absence of
    the marker fields.  Defensive: missing/invalid markers downgrade to an
    :class:`ErrorRow` rather than fabricating a score.
    """
    if not isinstance(row, dict):
        return ErrorRow(label="", error=f"Malformed result row: {row!r}", raw={})

    label = str(row.get("label", "") or "")
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}

    if "error" in row:
        return ErrorRow(label=label, error=str(row.get("error")), meta=meta, raw=row)

    markers = {field_name: _coerce_marker(row.get(field_name)) for field_name in MARKER_FIELDS}
    missing = [name for name, value in markers.items() if value is None]
    if missing:
        return ErrorRow(
            label=label,
            error=f"Missing or invalid marker score(s): {', '.join(missing)}.",
            meta=meta,
            raw=row,
        )

    flags_raw = row.get("flags") or []
    flags = [str(flag) for flag in flags_raw] if isinstance(flags_raw, list) else []
    return ResultRow(
        label=label,
        markers=MarkerScores(**markers),  # type: ignore[arg-type]
        confidence=str(row.get("confidence", "") or ""),
        flags=flags,
        notes=str(row.get("notes", "") or ""),
        meta=meta,
        raw=row,
    )


# --------------------------------------------------------------------------- #
# HTTP seam (lazy ``requests``; monkeypatched in tests, never hit in tests)
# --------------------------------------------------------------------------- #
def _require_requests():  # pragma: no cover - thin import guard
    try:
        import requests  # type: ignore  # noqa: PLC0415

        return requests
    except ImportError as exc:
        raise CircadiemError(
            "Circadiem scoring requires the optional 'requests' package. "
            "Install it with `pip install requests`."
        ) from exc


def _http_post_multipart(
    url: str,
    *,
    headers: dict[str, str],
    files: list[tuple[str, tuple[str, bytes, str]]],
    data: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    """POST multipart/form-data and return parsed JSON (the network seam)."""
    requests = _require_requests()
    response = requests.post(url, headers=headers, files=files, data=data, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _http_get_json(url: str, *, timeout: float = _HEALTH_TIMEOUT) -> dict[str, Any]:
    """GET and return parsed JSON (the network seam)."""
    requests = _require_requests()
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


# --------------------------------------------------------------------------- #
# Public API (§3)
# --------------------------------------------------------------------------- #
def analyze(
    images: list[tuple[str, bytes]],
    *,
    config: CircadiemConfig,
    api_key: str | None,
    labels: list[str] | None = None,
    custom_prompt: str | None = None,
) -> list[ResultRow | ErrorRow]:
    """Batch-score PNG plots via ``POST /api/analyze``.

    ``images`` is a list of ``(filename, png_bytes)``.  Returns one parsed row
    per image, in upload order; a single bad image yields an :class:`ErrorRow`.
    Raises :class:`CircadiemError` for whole-batch problems (bad key, no URL,
    too many/large/invalid images, transport/HTTP failures).
    """
    validate_openai_key(api_key)
    if not images:
        raise CircadiemError("No images to score.")
    if len(images) > MAX_FILES:
        raise CircadiemError(
            f"{len(images)} images exceeds the per-request limit of {MAX_FILES}. "
            "Batch larger jobs into multiple requests."
        )
    if labels is not None and len(labels) != len(images):
        raise CircadiemError(
            f"labels ({len(labels)}) must match the number of images ({len(images)})."
        )

    files: list[tuple[str, tuple[str, bytes, str]]] = []
    for name, data in images:
        validate_png(data, name=name)
        files.append(("images", (name, bytes(data), "image/png")))

    form: dict[str, str] = {
        "model": config.model,
        "aligned_to_dark": "true" if config.aligned_to_dark else "false",
        "vcg_band": config.vcg_band,
    }
    if labels:
        form["labels"] = json.dumps(labels)
    if custom_prompt:
        form["custom_prompt"] = custom_prompt

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        body = _http_post_multipart(
            config.url("/api/analyze"),
            headers=headers,
            files=files,
            data=form,
            timeout=config.timeout,
        )
    except CircadiemError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize transport/HTTP errors
        raise CircadiemError(
            f"Could not reach Circadiem at {config.base_url}. Check the URL, that "
            "the service is running (GET /health), and your network. "
            f"Original error: {exc}"
        ) from exc

    results = body.get("results") if isinstance(body, dict) else None
    if not isinstance(results, list):
        raise CircadiemError(
            "Unexpected Circadiem response: missing 'results' array. "
            f"Got keys: {list(body) if isinstance(body, dict) else type(body).__name__}."
        )
    return [parse_row(row) for row in results]


def health(config: CircadiemConfig) -> bool:
    """Return ``True`` when ``GET /health`` reports ``{"ok": true}``."""
    try:
        body = _http_get_json(config.url("/health"))
    except Exception:  # noqa: BLE001 - health is a best-effort probe
        return False
    return bool(isinstance(body, dict) and body.get("ok"))


def get_prompt(config: CircadiemConfig) -> str:
    """Fetch the default scoring system prompt via ``GET /api/prompt``."""
    try:
        body = _http_get_json(config.url("/api/prompt"))
    except CircadiemError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CircadiemError(
            f"Could not fetch the Circadiem prompt from {config.base_url}. Original error: {exc}"
        ) from exc
    return str(body.get("prompt", "")) if isinstance(body, dict) else ""


# --------------------------------------------------------------------------- #
# Tidy table (persistence / display) — list-in / table-out
# --------------------------------------------------------------------------- #
def results_to_frame(rows: list[ResultRow | ErrorRow]) -> pd.DataFrame:
    """Flatten parsed rows into one tidy DataFrame keyed by ``run_id`` + ``label``.

    Keeps error rows (``status == "error"``) so failures are persisted, not
    dropped.  Columns: ``run_id``, ``label``, ``status``, the six marker scores,
    ``confidence``, ``flags`` (``"; "``-joined), ``notes``, ``error``, ``model``,
    ``created_at`` — the last two pulled from the trusted server ``meta`` when present.
    """
    records: list[dict[str, Any]] = []
    for row in rows:
        meta = row.meta or {}
        record: dict[str, Any] = {
            "run_id": row.run_id,
            "label": row.label,
            "status": "error" if row.is_error else "ok",
        }
        for field_name in MARKER_FIELDS:
            record[field_name] = (
                getattr(row.markers, field_name) if isinstance(row, ResultRow) else pd.NA
            )
        if isinstance(row, ResultRow):
            record["confidence"] = row.confidence
            record["flags"] = "; ".join(row.flags)
            record["notes"] = row.notes
            record["error"] = ""
        else:
            record["confidence"] = ""
            record["flags"] = ""
            record["notes"] = ""
            record["error"] = row.error
        record["model"] = str(meta.get("model", "") or "")
        record["created_at"] = str(meta.get("created_at", "") or "")
        records.append(record)

    columns = [
        "run_id",
        "label",
        "status",
        *MARKER_FIELDS,
        "confidence",
        "flags",
        "notes",
        "error",
        "model",
        "created_at",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(records)[columns]
