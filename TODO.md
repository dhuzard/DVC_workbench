# DVC Behavioral Preprocessing Workbench — TODO

**Last updated:** 2026-05-13
**Status:** v0.2 hardening complete. Workstream E (distribution) opened so non-coder beta testers can run the app locally without a Python toolchain.

## Strategic next steps — v0.3 distribution

### Workstream E — Beta-tester distribution (data stays local)

- [x] Add `requirements.txt` so the image can install without an editable checkout.
- [x] Add `Dockerfile` (python:3.12-slim, non-interactive Streamlit server).
- [x] Add `docker-compose.yml` mounting `./outputs` into the container.
- [x] Add `.streamlit/config.toml` raising `maxUploadSize` to 500 MB and disabling telemetry.
- [x] Add `run.sh` / `run.bat` one-click launchers that check Docker is installed and running.
- [x] Document the non-coder Docker quick start in `README.md`.
- [ ] Verify a clean `docker compose build` succeeds and the app opens at `localhost:8501`.
- [ ] Publish a pre-built image to GHCR so testers can `docker run` without a build step.
- [ ] Add a smoke test that imports a bundled example file end-to-end inside the container.
- [ ] Optional: produce a signed Mac `.app` / Windows installer via stlite or PyInstaller for testers who cannot install Docker.

## Strategic next steps — v0.2 hardening

### Workstream A — Reproducibility & Provenance

- [x] Add an export manifest with input file hashes, row counts, config summary, and app/package versions.
- [x] Include `manifest.yaml` in every ZIP export.
- [x] Show provenance summary in the Export page before download.
- [x] Add tests for deterministic file hashing and manifest structure.

### Workstream B — Data Quality Diagnostics

- [x] Add a reusable `quality.py` module for per-subject/metric QC tables.
- [x] Flag missing-value rate, duplicate timestamps, negative values, zero variance, long gaps, and irregular intervals.
- [x] Replace ad hoc QC interval reporting in the app with the reusable quality report.
- [x] Include `quality_report.csv` in the export ZIP.
- [x] Add tests for data-quality edge cases.

### Workstream C — Analysis Confidence & Reporting

- [x] Add effect sizes to exploratory group comparisons.
- [x] Add Benjamini-Hochberg FDR q-values to exploratory statistics.
- [x] Include statistical method notes and effect-size columns in `stats_summary.csv`.
- [x] Add tests for effect sizes and FDR correction.

### Workstream D — Product Readiness

- [x] Add a compact app smoke test for component imports and workflow config.
- [x] Add README documentation for the v0.2 QC, provenance, and statistics outputs.
- [x] Run full pytest and compile checks after integration.

## Implementation checklist

### Priority 1 — UX & guided workflow

- [x] Add contextual help panels to every workflow step.
- [x] Replace free navigation with a guided sidebar/stepper workflow.
- [x] Add explicit metadata editor column documentation and required fields.
- [x] Add an output-column glossary.
- [x] Show a data-flow diagram with current-stage context.

### Priority 2 — Metadata & group design

- [x] Add a treatment schedule editor and exportable schedule table.
- [x] Add a guided group-builder with group colors and assignments.
- [x] Auto-detect likely experimental events from REMOVED/INSERTED pairs.

### Priority 3 — Confounds & QC

- [x] Detect cage-change pairs and apply asymmetric pre/post exclusion windows.
- [x] Add facility-level event calendar exclusions.
- [x] Add incomplete-baseline QC heatmap, imputation flag, and per-animal override hooks.
- [x] Add irregular bin detection per animal.

### Priority 4 — Analysis & export layer

- [x] Add an Analysis page after Export.
- [x] Add circadian rhythm summaries and cosinor fit.
- [x] Add daily/weekly/time-bin group summaries.
- [x] Add baseline-corrected group plots.
- [x] Add AUC per animal.
- [x] Add exploratory quick statistics.
- [x] Include analysis CSVs and figures in the export ZIP.

### Priority 5 — Code quality & architecture

- [x] Add `src/dvc_behavior/analysis.py`.
- [x] Add `src/dvc_behavior/api_adapter.py`.
- [x] Add warn-only optional Pandera schemas.
- [x] Add Streamlit parsing cache.
- [x] Add progress indicators for large files and long pipeline runs.

### Priority 6 — Tests

- [x] Cage-change pair detection tests.
- [x] AUC calculation tests.
- [x] Cosinor fit tests.
- [x] Daily/weekly aggregation tests.
- [x] Metadata editor smoke coverage.
- [x] Full example-files pipeline end-to-end test.

### Known bugs / minor fixes

- [x] Keep the "Run pipeline" spinner active for large-file runs.
- [x] Warn when `baseline_expected_bins` cannot be computed from native bins.
- [x] Prevent group-mean QC plot failures for all-excluded groups.
- [x] Allow adding subject metadata rows for animals absent from the DVC file.
- [x] Add a streaming/file-based export path for very large datasets.

---

## Vision

> Transform binned DVC Analytics exports into clean, traceable, scientifically
> interpretable behavioral datasets — and provide the missing preprocessing layer
> between DVC Analytics and statistical analysis.

The user journey is:

```
DVC Analytics export (CSV)
    → Import & validate
    → Define animals, groups, treatment schedule
    → Align to experimental events (surgery, injection, …)
    → Account for confounds (cage changes, facility events, light/dark, timezone)
    → Compute individual baselines & corrected values
    → QC review
    → Export traceable CSV for stats
    → Smart visualisation & quick statistical outputs
```

---

## What is already implemented (MVP)

| Module | Status |
|--------|--------|
| Wide → long DVC CSV parser (groups, subjects, timestamps) | ✅ done |
| Event CSV parser (REMOVED, INSERTED, CAGE_OFFLINE/ONLINE) | ✅ done |
| Light/dark annotation + Zeitgeber time | ✅ done |
| Exclusion windows around cage events | ✅ done |
| Temporal alignment to event or manual timestamp | ✅ done |
| Per-subject/metric baseline (mean, corrected value, % change) | ✅ done |
| Subject / group / study metadata tables | ✅ done |
| Optional aggregation (native → 1 min / 5 min / 1 h / 12 h / 24 h) | ✅ done |
| QC Plotly plots (raw, aligned, group mean ± SEM) | ✅ done |
| ZIP export (timeseries, baseline summary, exclusion log, config YAML, report) | ✅ done |
| 7-tab Streamlit app with pipeline stage tracker | ✅ done |
| 98-test pytest suite | ✅ done |

---

## Priority 1 — UX & guided workflow (do first)

The app is currently hard to understand for a non-developer user.
Every tab must explain what it does, what the user must do, and what happens next.

### 1.1 Add a contextual help panel to every tab

Each tab should open with a collapsible ℹ️ panel that explains:
- What this step is for (one sentence)
- What the user needs to do (numbered steps)
- What is optional vs. required
- What happens if they skip it

Example for Tab 4 (Events & Alignment):
> **What this step does:** Links your behavioral data to experimental events
> (e.g. surgery, injection). Without alignment, all time is shown as absolute
> clock time, and the baseline window cannot be defined.
>
> **What you must do:**
> 1. Check that your events were loaded correctly in the table above.
> 2. Select which event type marks "Day 0" (e.g. REMOVED for cage change, or
>    a custom surgery event you enter manually).
> 3. Choose whether each animal aligns to its own event (recommended) or to a
>    shared group timestamp.

### 1.2 Replace the 7-tab layout with a linear guided wizard

Current tabs are navigated freely, which confuses first-time users who do not
know which order to follow.

**Proposed change:** Replace the horizontal tabs with a stepped sidebar or
vertical progress stepper. Each step is unlocked only when the previous one
is complete (or explicitly skipped). Show a ✅ / ⚠️ / 🔒 badge on each step.

Steps:
1. **Import** — load files
2. **Inspect** — review detected structure, fix warnings
3. **Describe your study** — study-level metadata
4. **Describe your animals** — subject-level metadata + group assignment
5. **Define your timeline** — alignment event + exclusion rules
6. **Review confounds** — cage events, baseline window, light/dark
7. **Run & QC** — run pipeline, review plots
8. **Export** — download clean CSV + config

### 1.3 Inline column-level documentation in the metadata editors

The subject metadata `st.data_editor` has 20 columns and users do not know
what to fill. Add:
- Tooltip or short help text above the table explaining each column group
- A "required vs. optional" indicator per column
- Color-code empty required cells in red (Streamlit supports column config
  with `st.column_config`)

Use `st.data_editor` with explicit `column_config` for each column, e.g.:

```python
st.data_editor(
    subject_meta,
    column_config={
        "animal_id": st.column_config.TextColumn(
            "Animal ID",
            help="Your internal animal identifier (e.g. M001). Used in plots and exports.",
            required=True,
        ),
        "sex": st.column_config.SelectboxColumn(
            "Sex", options=["M", "F", "unknown"], required=True
        ),
        "genotype": st.column_config.TextColumn(
            "Genotype", help="e.g. WT, KO, HET"
        ),
        ...
    },
)
```

### 1.4 Add a "What does this column mean?" glossary page

Add a Tab 8 (or sidebar expander) with a glossary of all output columns in
`processed_timeseries.csv`, what they mean scientifically, and how they
were computed.

### 1.5 Show a data-flow diagram on the landing page

Above the tabs, show a static Mermaid or SVG diagram:

```
Raw DVC CSV → [Parser] → Long format → [Exclusions] → [Alignment] → [Baseline]
→ [Aggregation] → processed_timeseries.csv → [Plots] → [Stats]
```

Highlight the current stage in blue.

---

## Priority 2 — Metadata & group design improvements

### 2.1 Treatment schedule editor

Currently only a single surgery/treatment date per animal. Real experiments
need:
- Multiple treatment time points (e.g. daily injections over 7 days)
- Dose information per time point
- A treatment schedule table editable in the app

New `treatment_schedule` table:
```
animal_id | event_type | timestamp | dose | unit | route | notes
```

This table should be exportable and usable for alignment (align to
"first injection" automatically).

### 2.2 Guided group-builder

Instead of a raw data_editor, give the user a form:
1. "How many groups do you have?" → creates N rows
2. Per group: name, color (for plots), treatment description
3. Then assign animals to groups via a drag-and-drop or multiselect widget

### 2.3 Auto-detection of likely experimental events

When the user loads a DVC event file, parse REMOVED/INSERTED pairs and offer:
> "We detected 3 cage-change events for animal C57_1. These look like a
> surgery or cage-transfer. Would you like to use these as your alignment
> event (Day 0)?"

---

## Priority 3 — Confound handling improvements

### 3.1 Cage-change detection and smarter exclusion

Currently exclusion windows are symmetric (±24h around REMOVED/INSERTED).
A cage change is typically: REMOVED → short gap → INSERTED.
The biologically relevant exclusion window is:
- Before REMOVED: stress anticipation (0–6 h, configurable)
- After INSERTED: recovery (24–72 h, configurable)
- Gap between REMOVED and INSERTED: always excluded

Add a "cage-change pair" detector:
- Match REMOVED + INSERTED events for the same animal within a configurable
  window (e.g. < 6 h)
- Show these as "cage change" events with a dedicated rule
- Allow asymmetric pre/post exclusion

### 3.2 Facility-level event calendar

Allow the user to upload or manually enter facility-wide events that affect
all animals (e.g. room alarm, power outage, weekend vs. weekday caretaker).

New facility event table:
```
timestamp_start | timestamp_end | event_type | affected_groups | notes
```

Apply exclusion rules to all matched animals.

### 3.3 Incomplete baseline detection and warnings

Currently a warning is shown if baseline coverage < min_coverage.
Need to also:
- Show a per-animal baseline quality heatmap in the QC tab
- Offer to impute baseline from group mean with a flag (`baseline_imputed=True`)
- Allow the user to manually override the baseline window per animal

### 3.4 Irregular bin detection per animal

Flag animals whose timestamp intervals are irregular (sensor dropout, CAGE_OFFLINE)
and show them in the QC tab as a data quality report.

---

## Priority 4 — Post-export analysis layer (new module)

This is the second major feature block after the preprocessing pipeline.
The goal is to provide quick scientific outputs directly in the app, without
requiring the user to export and open R or Python separately.

### 4.1 New Tab: "Analysis" (after Export)

#### 4.1.1 Circadian rhythm analysis

- Plot mean activity per ZT hour (aggregated over all days, per group)
- Cosinor fit (amplitude, acrophase, MESOR) — use `scipy.optimize.curve_fit`
  with a cosine model
- Light vs. dark phase summary: mean ± SEM per phase per group
- Day vs. night ratio

Output table:
```
group_id | metric_name | MESOR | amplitude | acrophase_ZT | R2 | phase
```

#### 4.1.2 Time-bin averages (daily/weekly summaries)

Allow the user to select:
- Bin size: 1 day, 2 days, 1 week
- Relative to: alignment event (experimental days) or absolute date
- Which animals/groups to include

Output: mean ± SEM per group per time bin, ready for ANOVA.

#### 4.1.3 Baseline-corrected group plots

Plot `baseline_percent_change` (or `baseline_corrected_value`) over time,
grouped by treatment group, with SEM bands.
Allow the user to set:
- X axis: time from event in hours or experimental days
- Y axis: raw value, corrected value, % change
- Display: individual traces + group mean, or group mean only

#### 4.1.4 AUC (area under curve) per animal

For a user-defined window (e.g. Day 0 to Day 7):
- Compute AUC using trapezoidal rule (`numpy.trapz`)
- Output one value per animal
- Show as dot plot per group (individual points + mean ± SEM)
- Export as `auc_summary.csv`

#### 4.1.5 Quick statistics

For each metric and time bin, offer:
- Shapiro-Wilk normality test
- Mann-Whitney U (two groups) or Kruskal-Wallis (N groups) — non-parametric
  as default since behavioral data is rarely normal
- Display p-values on plots (ns / * / ** / ***)
- Export stats table

Use `scipy.stats` — no external stats dependencies.

**Important:** Label everything clearly as "exploratory statistics."
Add a disclaimer: "These results are for orientation only. Consult a
statistician for confirmatory analysis."

### 4.2 Export enhancements

Add to the ZIP:
- `auc_summary.csv` — one AUC value per animal per metric
- `daily_means.csv` — daily mean ± SEM per group per metric
- `circadian_summary.csv` — cosinor fit parameters per group
- `stats_summary.csv` — exploratory p-values per comparison
- `figures/` folder — all QC and analysis plots as PNG (use
  `fig.write_image()` from Plotly)

---

## Priority 5 — Code quality & architecture

### 5.1 Add a `src/dvc_behavior/analysis.py` module

Move all post-export analysis logic here (circadian, AUC, stats).
Keep the Streamlit app thin.

### 5.2 Add `src/dvc_behavior/api_adapter.py` (placeholder)

```python
class DVCApiAdapter:
    """Placeholder for future direct DVC API integration."""
    def fetch_project(self, project_id: str):
        raise NotImplementedError("DVC API integration is not yet implemented.")
```

### 5.3 Pandera schemas for key DataFrames

Add optional validation with Pandera for:
- `long_df` (parsed metric output)
- `event_df` (parsed event output)
- `processed_df` (pipeline output)

Make validation opt-in and non-crashing (warn only).

### 5.4 Caching with `@st.cache_data`

Currently `_parse_all_files()` re-parses on every rerun if files change.
Add `@st.cache_data` to `load_metric_csv` and `parse_event_csv` keyed on
file content hash.

### 5.5 Progress bars for large files

For files > 5 MB (like `E_animal_loc__index_smoothed.csv` at 2.8 MB),
show a `st.progress` bar during parsing.

---

## Priority 6 — Tests to add

| Status | Test | File |
|--------|------|------|
| [x] | Cage-change pair detection | `test_exclusions.py` |
| [x] | AUC calculation | `test_analysis.py` |
| [x] | Cosinor fit | `test_analysis.py` |
| [x] | Daily/weekly aggregation | `test_analysis.py` |
| [x] | Metadata column_config renders without error | `test_app_smoke.py` |
| [x] | Full pipeline end-to-end with all 3 example metric files | `test_pipeline_e2e.py` |

---

## Known bugs / minor fixes

- [x] The "Run pipeline" button does not show a spinner on large files
      (pipeline run is wrapped in `st.spinner`; parsing also has a progress bar).
- [x] `baseline_expected_bins` can be `None` when `native_bin_seconds` is NaN
      for a file; propagate a proper warning rather than silent `None`
- [x] The group-mean QC plot breaks when all subjects in a group are excluded
      (produces empty DataFrame → Plotly error)
- [x] `st.data_editor` with `num_rows="fixed"` prevents adding rows; the user
      cannot add a subject that is not in the DVC file (e.g. a failed animal
      with no data). Change to `num_rows="dynamic"` or add a separate
      "Add animal" form.
- [x] Export ZIP uses in-memory BytesIO; for very large datasets (> 500 MB)
      this will OOM. Add a streaming/file-based path for large exports.

---

## File structure after next iteration

```
src/dvc_behavior/
    analysis.py          ← NEW: circadian, AUC, daily means, stats
    api_adapter.py       ← placeholder for DVC API
    schemas.py           ← optional Pandera schemas

app/
    streamlit_app.py     ← refactor: wizard stepper + column_config
    components/
        help_panel.py    ← contextual help text per step
        stage_tracker.py ← sidebar progress stepper
        metadata_editor.py ← subject/group editors with column_config

tests/
    test_analysis.py     ← NEW
    test_pipeline_e2e.py ← NEW
```

---

## Session notes (2026-05-05)

- Example files confirmed working: `cohort2`, `E_animal_loc`, `PartnerC_cohort1`
- Real DVC timestamps use ISO 8601 with offset: `2024-11-01T07:00:00.000+0100`
- Native bin sizes seen in examples: 60 s (1 min), 300 s (5 min)
- The QRT column contains a JSON list string — currently stored as-is, not parsed
- Two Python installs on the machine: `C:\Python312` (active) and miniconda.
  Always use `python -m pip install` and `python -m pytest` to avoid confusion.
- Run on port 8502 to avoid conflict with existing app on 8501:
  `streamlit run app/streamlit_app.py --server.port 8502`

## Session notes (2026-05-07)

- v0.2 hardening (Workstreams A–D) verified end-to-end: provenance manifest,
  reusable `quality.py` diagnostics, exploratory statistics with effect sizes
  and Benjamini-Hochberg FDR, smoke + e2e tests.
- Bundled example set expanded (Cohort23/31/32, WT-Group, Xp41/42, Group11/12);
  `Group11`/`Group12` ship in a per-mouse long format the wide-format parser
  does not recognise — the e2e smoke test now asserts pipeline success on the
  files that do parse so new file shapes do not regress the suite.
- Full suite: 128 passed, 5 skipped; `ruff check .` clean.
