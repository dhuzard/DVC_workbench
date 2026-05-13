"""
Container-level smoke test.

Runs a real bundled example file through the full pipeline and asserts the
export ZIP is non-empty and contains the core artefacts. Designed to run
inside the published Docker image to catch packaging regressions
(missing example data, missing system libs, broken paths, write permissions
on /app/outputs) that pytest in a developer venv would not detect.

Exits 0 on success, 1 on failure. Prints a single status line per check.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

# Resolve project paths whether run from /app (container) or repo root (CI).
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd  # noqa: E402

from dvc_behavior import (  # noqa: E402
    aggregation,
    alignment,
    baseline,
    exclusions,
    export as exp_mod,
    light_dark,
    metadata,
    parsing,
)

EXAMPLES = _ROOT / "data" / "examples"
OUTPUTS = _ROOT / "outputs"
EXPECTED_ZIP_MEMBERS = {
    "processed_timeseries.csv",
    "subject_metadata.csv",
    "group_metadata.csv",
}


def _fail(msg: str) -> "Never":  # type: ignore[name-defined]
    print(f"FAIL: {msg}")
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"ok: {msg}")


def pick_example() -> Path:
    candidates = sorted(EXAMPLES.glob("*animal*_index_smoothed.csv"))
    if not candidates:
        _fail(f"no bundled example metric files found under {EXAMPLES}")
    # Prefer the smallest file for a fast smoke test.
    return min(candidates, key=lambda p: p.stat().st_size)


def main() -> None:
    if not EXAMPLES.is_dir():
        _fail(f"examples directory missing in image: {EXAMPLES}")
    _ok(f"examples directory present: {EXAMPLES}")

    example = pick_example()
    _ok(f"selected example: {example.name} ({example.stat().st_size} bytes)")

    long_df, parse_warnings = parsing.load_metric_csv(example, source_file=example.name)
    if long_df.empty:
        _fail(f"parser returned empty long_df for {example.name}; warnings={parse_warnings}")
    _ok(f"parsed {len(long_df)} rows, {long_df['subject_id'].nunique()} subjects")

    subject_meta = metadata.build_subject_metadata_template(long_df)
    group_meta = metadata.build_group_metadata_template(long_df)
    processed = metadata.merge_subject_metadata(long_df, subject_meta)
    processed = metadata.merge_group_metadata(processed, group_meta)
    processed, _ = light_dark.add_light_dark_columns(processed)
    processed, exclusion_log = exclusions.apply_exclusions(processed, pd.DataFrame())

    align_ts = processed["timestamp_utc"].dropna().quantile(0.5)
    processed, _ = alignment.align_to_manual_timestamp(processed, align_ts.isoformat())
    processed, baseline_summary, _ = baseline.compute_baseline(
        processed, start_hours=-24, end_hours=0, min_coverage=0.0
    )
    processed, _ = aggregation.aggregate(processed, 3600)

    for col in ("light_dark_phase", "time_from_event_hours", "baseline_value", "is_excluded"):
        if col not in processed.columns:
            _fail(f"expected column missing from processed_df: {col}")
    _ok(f"pipeline produced {len(processed)} processed rows with all expected columns")

    zip_bytes = exp_mod.create_export_zip(
        processed_df=processed,
        baseline_summary=baseline_summary,
        exclusion_log=exclusion_log,
        event_table=None,
        subject_metadata=subject_meta,
        group_metadata=group_meta,
        study_metadata={"study_name": "container_smoke"},
        analysis_config={"source": "container_smoke"},
        processing_report="container smoke test",
        metadata_validation_report=None,
    )
    if not zip_bytes:
        _fail("create_export_zip returned empty bytes")

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
    missing = EXPECTED_ZIP_MEMBERS - names
    if missing:
        _fail(f"export ZIP missing expected members: {sorted(missing)}; got {sorted(names)}")
    _ok(f"export ZIP contains {len(names)} files including all expected members")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    target = OUTPUTS / "container_smoke_export.zip"
    target.write_bytes(zip_bytes)
    _ok(f"wrote {target} ({len(zip_bytes)} bytes)")

    print("SUCCESS: container smoke test passed")


if __name__ == "__main__":
    main()
