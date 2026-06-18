# Contributing to the DVC Behavioral Preprocessing Workbench

Thanks for your interest in improving the workbench! This project turns binned
**Digital Ventilated Cage** (DVC, Tecniplast/TSE home-cage activity sensing — *not*
Data Version Control) behavioral exports into clean, traceable, scientifically
interpretable datasets. Contributions of all kinds are welcome: bug reports,
documentation, tests, and code.

Please also read [`AGENTS.md`](AGENTS.md) — it is the authoritative guide to the
codebase conventions and applies to human contributors as much as to AI agents.

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating,
you are expected to uphold it. Please report unacceptable behavior as described in
that document.

## Getting started

```bash
git clone https://github.com/dhuzard/DVC_workbench.git
cd DVC_workbench
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"        # core + pytest + ruff
# Optional AI circadian-scoring extras:
# pip install -e ".[ai]"
```

Run the app locally:

```bash
streamlit run app/streamlit_app.py
# or use the helper: ./run.sh  (run.bat on Windows)
```

## Development workflow

Before opening a pull request, make sure all three of these pass — CI runs the
same checks:

```bash
ruff check .          # lint (line-length 100; E,F,W,I; E501/I001 ignored)
ruff format .         # format
pytest -q             # full suite (~240 tests); keep it green
```

## Conventions (match the surrounding code)

These are summarized from `AGENTS.md`; that file is the source of truth.

- **Library / app split.** Put logic in `src/dvc_behavior/`; keep `streamlit_app.py`
  thin. Library modules must **not** import `streamlit`.
- **Table-in / table-out.** Analysis functions take a tidy DataFrame and return
  `tuple[pd.DataFrame, list[str]]` (the list is human-readable warnings, not
  exceptions). Degrade gracefully instead of raising on bad/missing columns.
- **`from __future__ import annotations`** at the top of every module; full type hints.
- **Optional heavy deps are lazy + guarded.** `anthropic` / `requests` are imported
  *inside* functions and must never be required for the default path or the tests.
- **Additive changes.** Add new columns/params without breaking existing public
  signatures or output columns. Existing tests must keep passing.
- **Determinism.** Seed any RNG (`np.random.default_rng(0)`); exports are reproducible
  given the same inputs + config.
- Add public names to each module's `__all__`.

## Privacy & scientific guardrails

This tool runs **locally** — the headline promise to users is *"your data stays on
your computer."* Keep that promise:

- The LLM insights layer is **offline by default**; all network integrations
  (Anthropic BYO-key, Ollama, Europe PMC literature, Circadiem) are **opt-in and off
  by default**. The model never sees raw time series — only small aggregated summaries.
- Do **not** bake any specific model id into code, tests, comments, or committed docs.
- Statistics are **exploratory, never confirmatory** — no "significant"/"proven"
  language in outputs. Respect the configured photoperiod (do not assume 12:12).
  Aggregate to per-subject summaries before group comparisons.

## Tests

- Add or extend tests alongside any behavior change.
- Favor property-style tests for statistics (recover known parameters from synthetic
  data) and fully **offline** tests for the insights layer (use the scripted providers
  — never hit the network in tests).

## Submitting changes

1. Fork the repo and create a feature branch.
2. Make your change, with tests and docs updated as needed (`README.md`, the output
   glossary in `app/components/workflow.py`, `docs/ANALYTICS_AND_LLM_REVIEW.md`, and
   `CHANGELOG.md`).
3. Ensure `ruff check .`, `ruff format .`, and `pytest -q` all pass.
4. Open a pull request against `main` with a clear description of the change and its
   motivation.

## Reporting bugs & requesting features

Please use the [issue templates](https://github.com/dhuzard/DVC_workbench/issues/new/choose).
For anything security- or privacy-sensitive, follow [`SECURITY.md`](SECURITY.md)
instead of filing a public issue.
