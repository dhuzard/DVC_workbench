# Summary

<!-- What does this PR change, and why? Link any related issue (e.g. "Closes #12"). -->

## Type of change

- [ ] Bug fix
- [ ] New feature / capability
- [ ] Documentation
- [ ] Refactor / internal cleanup
- [ ] Other (describe below)

## Checklist

- [ ] `ruff check .` passes
- [ ] `ruff format .` applied (no diff)
- [ ] `pytest -q` passes locally
- [ ] Tests added/updated for the change
- [ ] Docs updated where relevant (`README.md`, output glossary in
      `app/components/workflow.py`, `docs/ANALYTICS_AND_LLM_REVIEW.md`, `CHANGELOG.md`)
- [ ] Public names added to the module's `__all__` (if applicable)

## Guardrails

- [ ] Keeps the local-first privacy model intact — no raw data, file names, group
      names, or subject ids leave the machine without explicit opt-in.
- [ ] No specific LLM model id is hardcoded in code, tests, comments, or docs.
- [ ] Any statistics are exploratory, not confirmatory (no "significant"/"proven"
      language in outputs); the configured photoperiod is respected.

## Notes for reviewers

<!-- Anything reviewers should focus on, trade-offs, follow-ups, etc. -->
