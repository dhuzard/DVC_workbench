# DVC Behavioral Preprocessing Workbench

A Python + Streamlit MVP for preprocessing binned **Digital Ventilated Cage (DVC)** behavioral data exports.

> **Note:** DVC here means *Digital Ventilated Cage* (Tecniplast/TSE), **not** Data Version Control.

---

## Quick start

```bash
# Install in editable mode with dev extras
pip install -e ".[dev]"

# Run the app
streamlit run app/streamlit_app.py

# Run tests
pytest

# Lint
ruff check .
```

---

## Features

| Tab | What it does |
|-----|-------------|
| 1 · Import | Upload DVC metric & event CSVs, or load bundled examples |
| 2 · Validate | Detect groups, subjects, timestamps, native bin size, show warnings |
| 3 · Metadata & Study Design | Edit study / subject / group metadata; download/re-upload templates |
| 4 · Events, Alignment & Exclusions | Preview events; configure alignment event & exclusion windows |
| 5 · Baseline & Aggregation | Configure baseline window; run full pipeline |
| 6 · QC Plots | Raw, aligned, and group-mean timeseries with exclusion overlays |
| 7 · Export | Download ZIP with all processed files |

---

## Expected input formats

### DVC metric CSV (wide format)

```
day,hour,minute,relativeTime,
{group}_TIMESTAMP,{group}_AVG,{group}_SEM,{group}_QRT,{group}_SAMPLES,
{group}_{subject_1},{group}_{subject_2},...
```

Multiple group blocks can appear in the same file.

**Examples from `data/examples/`:**

| File | Groups |
|------|--------|
| `PartnerC_cohort1_animal_loc_index_smoothed.csv` | `70Q_WT` |
| `cohort2_animal_loc__index_smoothed.csv` | `C57` |
| `E_animal_loc__index_smoothed.csv` | `3_S_C`, `5_S_C` |

### DVC event CSV

```
group,day,hour,minute,relativeTime,timestamp,cage,rack,position,event
```

Known event values: `REMOVED`, `INSERTED`, `CAGE_OFFLINE`, `CAGE_ONLINE`.

---

## Output ZIP contents

| File | Description |
|------|-------------|
| `processed_timeseries.csv` | Full long-format processed data |
| `baseline_summary.csv` | Baseline stats per subject/metric |
| `exclusion_log.csv` | Per-event exclusion windows and row counts |
| `event_table_clean.csv` | Parsed event table |
| `subject_metadata.csv` | Subject-level metadata |
| `group_metadata.csv` | Group-level metadata with scientific labels |
| `study_metadata.yaml` | Study-level metadata |
| `event_metadata.csv` | Manually entered events (header only if unused) |
| `analysis_config.yaml` | All parameters used for this run |
| `processing_report.md` | Human-readable run summary |
| `metadata_validation_report.md` | Metadata quality summary |

---

## Project structure

```
dvc-behavioral-preprocessing-workbench/
├── app/
│   └── streamlit_app.py          # 7-tab Streamlit GUI
├── src/
│   └── dvc_behavior/
│       ├── config.py             # Constants & defaults
│       ├── io.py                 # File I/O utilities
│       ├── parsing.py            # Wide→long DVC CSV parser
│       ├── metadata.py           # Subject/group metadata
│       ├── events.py             # Event CSV parser
│       ├── light_dark.py         # ZT & light/dark annotation
│       ├── exclusions.py         # Exclusion window logic
│       ├── alignment.py          # Temporal alignment
│       ├── baseline.py           # Baseline calculation
│       ├── aggregation.py        # Optional coarser binning
│       ├── qc.py                 # Plotly QC figures
│       ├── export.py             # ZIP export builder
│       └── reporting.py          # Markdown report generator
├── tests/                        # pytest test suite
├── data/
│   └── examples/                 # Bundled example CSVs
├── outputs/                      # Local export directory
└── pyproject.toml
```

---

## Scientific assumptions

- **Binned exports only.** This app processes pre-binned DVC CSV exports. It does not connect to any DVC API or database.
- **Timestamp is source of truth.** The `{group}_TIMESTAMP` column is used for all time-based operations. `relativeTime` is stored but not used for alignment.
- **subject_id may be a cage label,** not necessarily a biological animal ID. Keep both `subject_id` (detected) and `animal_id` (user-defined) separate.
- **Baseline is per subject and per metric.** Group-level baselines are not computed; group means are derived from individual subjects.
- **Exclusions are traceable.** Every excluded row carries its reason in `exclusion_reason`. Excluded rows are retained in the output with `is_excluded=True`.
- **No statistical inference.** This MVP produces tidy data for downstream analysis. No p-values, no group comparisons.
- **Light/dark annotation uses local time.** The user selects a timezone (default: `Europe/Paris`). ZT0 = lights-on time.

---

## Reproducibility

Given the same input files and the same `analysis_config.yaml`, the pipeline produces identical output.

The exported `analysis_config.yaml` captures:
- All uploaded file names
- Timezone and light/dark cycle
- Alignment event type and scope
- Exclusion rules (per event type)
- Baseline window and method
- Aggregation bin size
- App version and processing timestamp

---

## Development

```bash
# Lint
ruff check .

# Format
ruff format .

# Tests
pytest -v

# Tests with coverage
pytest --tb=short
```

---

## Future work (not in MVP)

- DVC API adapter (placeholder module raises `NotImplementedError`)
- Pandera schema validation
- Statistical group comparison
- Interactive exclusion drawing on the QC plot
- Multi-metric dashboard
