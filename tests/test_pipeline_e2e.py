"""End-to-end pipeline smoke test over bundled example metric files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dvc_behavior import aggregation, alignment, baseline, exclusions, light_dark, metadata, parsing


_EXAMPLES = Path(__file__).parent.parent / "data" / "examples"


def test_pipeline_e2e_all_example_metric_files():
    metric_files = sorted(_EXAMPLES.glob("*animal*_index_smoothed.csv"))
    assert len(metric_files) >= 3

    parsed = []
    warnings = []
    for path in metric_files:
        df, warns = parsing.load_metric_csv(path, source_file=path.name)
        warnings.extend(warns)
        if not df.empty:
            parsed.append(df)

    long_df = parsing.combine_long_dfs(parsed)
    assert not long_df.empty
    assert long_df["source_file"].nunique() == len(metric_files)

    subject_meta = metadata.build_subject_metadata_template(long_df)
    group_meta = metadata.build_group_metadata_template(long_df)
    processed = metadata.merge_subject_metadata(long_df, subject_meta)
    processed = metadata.merge_group_metadata(processed, group_meta)
    processed, ld_warns = light_dark.add_light_dark_columns(processed)
    warnings.extend(ld_warns)

    processed, exclusion_log = exclusions.apply_exclusions(processed, pd.DataFrame())
    align_ts = processed["timestamp_utc"].dropna().quantile(0.5)
    processed, align_warns = alignment.align_to_manual_timestamp(processed, align_ts.isoformat())
    warnings.extend(align_warns)
    processed, baseline_summary, baseline_warns = baseline.compute_baseline(
        processed,
        start_hours=-24,
        end_hours=0,
        min_coverage=0.0,
    )
    warnings.extend(baseline_warns)
    processed, agg_warns = aggregation.aggregate(processed, 3600)
    warnings.extend(agg_warns)

    assert "light_dark_phase" in processed.columns
    assert "time_from_event_hours" in processed.columns
    assert "baseline_value" in processed.columns
    assert "is_excluded" in processed.columns
    assert exclusion_log.empty
    assert baseline_summary is not None
    assert isinstance(warnings, list)
