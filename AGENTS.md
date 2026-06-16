# AGENTS.md — guide for AI agents and contributors

Orientation for anyone (human or agent) making changes to the **DVC Behavioral
Preprocessing Workbench**. Read this before editing; it captures the conventions
the codebase already follows so changes stay consistent.

> **DVC = Digital Ventilated Cage** (Tecniplast/TSE home-cage activity sensing),
> **not** Data Version Control.

---

## What this project is

A Python + Streamlit workbench that turns binned DVC behavioral CSV exports into
clean, traceable, scientifically interpretable datasets, plus an exploratory
analysis and a grounded LLM-insights layer. It runs **locally** — the headline
promise to users is *"your data stays on your computer."* Keep that promise.

User journey: **Import → Validate → Metadata → Events/Alignment → Baseline/Aggregation
→ QC → Export → Analysis (+ AI insights)**.

---

## Layout

```
app/
  streamlit_app.py        # guided GUI (thin; logic lives in src/)
  components/             # workflow help, stepper, metadata editors
src/dvc_behavior/
  parsing.py metadata.py events.py light_dark.py exclusions.py
  alignment.py baseline.py aggregation.py qc.py export.py reporting.py
  quality.py provenance.py schemas.py config.py
  analysis.py             # exploratory analytics (table-in/table-out)
  insights.py             # grounded, offline-first LLM narrative + tool-calling Q&A
  literature.py           # optional Europe PMC literature grounding (opt-in)
tests/                    # pytest suite (mirrors module names)
data/examples/            # bundled example CSVs
docs/ANALYTICS_AND_LLM_REVIEW.md   # the analytics/LLM roadmap + status
```

---

## Conventions (match the surrounding code)

- **Library/app split.** Put logic in `src/dvc_behavior/`; keep `streamlit_app.py`
  thin. Library modules must **not** import `streamlit`.
- **Table-in / table-out.** Analysis functions take a tidy DataFrame and return
  `tuple[pd.DataFrame, list[str]]` — the list is human-readable warnings, not
  exceptions. Degrade gracefully (return an empty frame + a warning) instead of
  raising on bad/missing columns.
- **`from __future__ import annotations`** at the top of every module; full type hints.
- **Optional heavy deps are lazy + guarded.** scipy is a hard dep but still wrapped
  with fallbacks; `anthropic` / `requests` (insights) are imported *inside* functions
  and must never be required for the default path or the tests.
- **Additive changes.** New columns/params are added without breaking existing public
  signatures or output columns. Existing tests must keep passing.
- **Determinism.** Seed any RNG (`np.random.default_rng(0)`); exports are reproducible
  given the same inputs + config.
- Add public names to each module's `__all__`.

---

## Analytics (`analysis.py`)

Cosinor (with a zero-amplitude **rhythmicity test** + amplitude/acrophase CIs),
light/dark summaries, time-bin summaries, per-animal AUC, **estimation-first**
exploratory stats (median difference + bootstrap CI, effect-size CI, `min_possible_p`,
`small_n_warning`), non-parametric circadian metrics (**IS/IV/RA, M10/L5**), activity
**bout/fragmentation** metrics, **window-summary contrasts**, and Lomb–Scargle
**period estimation**. Statistics are **exploratory only** — always carry
`EXPLORATORY_STATS_DISCLAIMER`, surface sample sizes, lead with estimation over p-values.

## AI insights (`insights.py`) — design rules

The LLM is an **interpretation layer, not a calculator**, and the privacy model is
non-negotiable:

1. **The model never sees raw time series** — only the small aggregated summary
   payload (`build_insight_payload`) or the result tables returned by tool calls.
2. **Offline by default.** `NullProvider` renders a deterministic templated narrative
   with no network/key/optional deps. `OllamaProvider` (local) and
   `AnthropicProvider` / `AnthropicToolProvider` (BYO-key) are opt-in.
3. **Grounded Q&A.** `answer_question` runs a bounded tool-calling loop that executes
   the *real* `analysis.py` functions (`execute_analysis_tool`) — the model requests
   computations, it does not invent numbers.
4. **No hardcoded model ids.** Providers require the caller to pass the model id (the
   UI has a field; key comes from the field or `ANTHROPIC_API_KEY`). Do not bake a
   specific Claude model id into code, tests, comments, or committed docs.
5. **Traceability.** Every output records the provider, model id, and payload hash;
   the export bundle ships `insights/narrative.md` + `insights/payload.json`.
6. **Literature grounding (`literature.py`)** is opt-in and off by default. Only
   generic topic keywords (never data, group names, or file names) are sent to
   Europe PMC; results are framed as suggestions to verify. The default
   `NullLiteratureProvider` is offline, and tests monkeypatch the `_http_get_json`
   seam — never hit the network in tests.

---

## Workflow for changes

```bash
ruff check .          # lint (line-length 100; E,F,W,I; E501/I001 ignored)
ruff format .         # format
pytest -q             # full suite (~180 tests); keep it green
```

- Develop on the designated feature branch; **commit with clear messages** and
  **push** when done. Do **not** open a pull request unless explicitly asked.
- Add/extend tests alongside any behavior change. Favor property-style tests for
  statistics (recover known parameters from synthetic data) and fully **offline**
  tests for the insights layer (use `ScriptedToolProvider` for the Q&A loop).
- Update `README.md`, the output-column glossary (`app/components/workflow.py`),
  `docs/ANALYTICS_AND_LLM_REVIEW.md`, and `TODO.md` when you add user-facing outputs.

## Scientific guardrails

- Exploratory, never confirmatory — no "significant"/"proven" language in outputs.
- Respect the configured photoperiod (do not assume 12:12).
- Subject is the experimental unit: aggregate to per-subject summaries before group
  comparisons; prefer one contrast per window over many autocorrelated per-bin tests.
- Guard unstable math (e.g. near-zero baselines) and flag it rather than emitting
  misleading numbers.
