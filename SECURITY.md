# Security Policy

## Supported versions

The DVC Behavioral Preprocessing Workbench is under active development. Security
fixes are applied to the latest release and the `main` branch.

| Version | Supported          |
| ------- | ------------------ |
| `main` / latest release | :white_check_mark: |
| older tags | :x:             |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security or privacy vulnerabilities.**

Instead, report privately using one of:

- GitHub's [private vulnerability reporting](https://github.com/dhuzard/DVC_workbench/security/advisories/new)
  (preferred), or
- email **damien@metadatapp.net** with the subject line `SECURITY: DVC Workbench`.

Please include:

- a description of the issue and its impact,
- steps to reproduce (a minimal example or affected file path),
- the version / commit you tested, and
- any suggested remediation if you have one.

We aim to acknowledge reports within **5 business days** and to provide a remediation
timeline after triage. Please give us a reasonable opportunity to address the issue
before any public disclosure.

## Scope & privacy model

This is a **local-first** tool — its headline promise is *"your data stays on your
computer."* The following are explicitly **in scope** for security/privacy reports:

- Any path on which behavioral data, raw time series, file names, group names, or
  subject ids could leave the machine **without explicit opt-in**.
- Any default code path that performs network egress (the default insights path is
  offline; all network integrations — Anthropic BYO-key, Ollama, Europe PMC,
  Circadiem — must remain opt-in and off by default).
- Unsafe handling of user-supplied CSV input (deserialization, path traversal,
  code execution).
- Leaked credentials or secrets in the repository or build artifacts.

## Handling of API keys

When optional AI integrations are enabled, API keys are supplied by the user
(BYO-key) via the UI or environment variables and are **never persisted** by the
application. Never commit real keys; use `.env` (gitignored) based on `.env.example`.
If you believe a key has been exposed in the repository history, report it privately
as above.
