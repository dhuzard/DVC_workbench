# DVC Behavioral Preprocessing Workbench

A Python + Streamlit workbench for preprocessing binned **Digital Ventilated Cage (DVC)** behavioral data exports, reviewing quality/confounds, and producing traceable outputs for exploratory analysis.

> **Note:** DVC here means *Digital Ventilated Cage* (Tecniplast/TSE), **not** Data Version Control.

---

## Quick start

### For beta testers (no Python install required)

The app runs in [Docker](https://www.docker.com/products/docker-desktop/), so testers only need Docker Desktop installed. All uploaded data stays on the tester's machine — nothing is sent to a remote server.

1. Install **Docker Desktop** and start it.
2. Download (or clone) this repository as a folder.
3. Double-click the launcher for your OS:
   - **Windows:** `run.bat`
   - **macOS / Linux:** `run.sh` (you may first need `chmod +x run.sh`)
4. The first run builds the image (a few minutes). Subsequent runs start in seconds.
5. Open <http://localhost:8501> in your browser.
6. To stop the app, press **Ctrl+C** in the terminal (or close the window on Windows).

Exported ZIPs and CSVs land in the local `outputs/` folder (mounted into the container).

If you prefer the raw Docker command instead of the launcher:

```bash
docker compose up --build
```

### For developers

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

| Workflow step | What it does |
|---------------|--------------|
| Import | Upload DVC metric & event CSVs, or load bundled examples |
| Validate | Detect groups, subjects, timestamps, native bin size, and parse warnings |
| Metadata & Study Design | Edit study, subject, group, treatment schedule, and group assignment metadata |
| Events, Alignment & Exclusions | Preview events, detect cage-change pairs, configure alignment and confound windows |
| Baseline & Aggregation | Configure baseline window, optional group-mean imputation/overrides, and run the pipeline |
| QC Plots | Review raw/aligned plots, baseline quality heatmap, irregular-bin report, and group means |
| Export | Download a ZIP with processed data, config, metadata, reports, and optional analysis outputs |
| Analysis | Generate exploratory circadian, binned, AUC, and quick-statistics summaries |

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
| `treatment_schedule.csv` | Editable treatment/dosing schedule, when provided |
| `facility_events.csv` | Facility-level confound calendar, when provided |
| `daily_means.csv` | Group mean ± SEM by selected analysis time bin |
| `circadian_summary.csv` | Cosinor MESOR, amplitude, acrophase, R2, and phase |
| `light_dark_summary.csv` | Light/dark group summaries and dark/light ratio |
| `quality_report.csv` | Per-subject/metric quality diagnostics |
| `auc_summary.csv` | Per-animal trapezoidal AUC values for the selected window |
| `stats_summary.csv` | Exploratory p-values, FDR q-values, effect sizes, labels, and statistical notes |
| `analysis_config.yaml` | All parameters used for this run |
| `manifest.yaml` | Input file hashes, row counts, selected config, and app version |
| `processing_report.md` | Human-readable run summary |
| `metadata_validation_report.md` | Metadata quality summary |

---

## Project structure

```
dvc-behavioral-preprocessing-workbench/
├── app/
│   ├── streamlit_app.py          # guided Streamlit GUI
│   └── components/               # workflow and metadata editor components
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
│       ├── analysis.py           # Exploratory analysis helpers
│       ├── api_adapter.py        # Future direct DVC API placeholder
│       ├── qc.py                 # Plotly QC figures
│       ├── export.py             # ZIP export builder
│       ├── schemas.py            # Optional warn-only dataframe validation
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
- **Exploratory statistics only.** P-values and group comparisons in the app are orientation tools, not confirmatory inference.
- **Light/dark annotation uses local time.** The user selects a timezone (default: `Europe/Paris`). ZT0 = lights-on time.
- **Imputed baselines are flagged.** Group-mean baseline imputation is optional and sets `baseline_imputed=True`.

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

The exported `manifest.yaml` records input file names, sizes, SHA256 hashes,
tracked table row counts, selected configuration, app version, and processing
timestamp.

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

## Current roadmap

The v0.2 roadmap in `TODO.md` focuses on:

- Maintaining export provenance and file hashing as the reproducibility anchor
- Expanding reusable data-quality diagnostics as new real-world failure modes appear
- Keeping exploratory statistics clearly labelled with effect sizes and FDR correction
- Strengthening smoke coverage around the app workflow
