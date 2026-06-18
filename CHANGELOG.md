# Changelog

All notable changes to the DVC Behavioral Preprocessing Workbench are documented
here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Open-source release scaffolding: `LICENSE` (GPL-3.0-or-later), `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `SECURITY.md`, `CITATION.cff`,
  issue/PR templates, and this changelog.
- Continuous-integration workflow (`.github/workflows/ci.yml`) running `ruff check`,
  `ruff format --check`, and the full `pytest` suite on Python 3.10–3.12.
- Public-API `__all__` declarations across all `src/dvc_behavior/` modules.
- Project metadata in `pyproject.toml` (license, authors, keywords, trove
  classifiers, project URLs).

### Changed

- Applied repository-wide `ruff format` so the codebase passes its own formatting gate.

## [0.1.0]

Initial workbench: guided import → validate → metadata → events/alignment →
baseline/aggregation → QC → export → analysis pipeline, with exploratory analytics
(cosinor rhythmicity, IS/IV/RA, M10/L5, activity bout/fragmentation, estimation-first
statistics, Lomb–Scargle period estimation) and a grounded, offline-first LLM
insights layer. Optional, opt-in integrations: Anthropic (BYO-key), Ollama (local),
Europe PMC literature grounding, and Circadiem AI circadian scoring.

[Unreleased]: https://github.com/dhuzard/DVC_workbench/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dhuzard/DVC_workbench/releases/tag/v0.1.0
