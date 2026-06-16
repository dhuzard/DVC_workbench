# Critical Review — Analytics, Scientific Insights & an LLM Layer

**Scope:** A critical overview of the DVC Behavioral Preprocessing Workbench with
concrete, prioritized recommendations to (a) strengthen the analytics and
scientific-insight layer and (b) turn "LLM insights" into a real, trustworthy
asset that fits the product's local-first, traceable ethos.

**Reviewed at commit:** branch `claude/analytics-llm-insights-review-u6ulya`
**Date:** 2026-06-15

---

## 1. Executive summary

The preprocessing core is mature and well-engineered. The codebase has a clean
separation between the reusable library (`src/dvc_behavior/`) and a thin
Streamlit app, strong provenance/reproducibility (`manifest.yaml`,
`analysis_config.yaml`, SHA256 hashing), responsible QC diagnostics
(`quality.py`), and an exploratory-stats layer that already does the *right*
scientifically-cautious things (effect sizes, Benjamini–Hochberg FDR, explicit
"exploratory only" disclaimers, subject-level aggregation before testing).

The weakest link is not the *plumbing* — it is the **scientific depth and
trustworthiness of the outputs**. The analytics stop at means ± SEM, a single
24 h cosinor, AUC, and per-bin non-parametric tests. For a home-cage activity
tool this is a thin slice of what reviewers and PIs expect, and several outputs
are presented in ways that can mislead (R² on averaged bins, p-stars without
n-context or estimation, hard-coded 12:12 photoperiod, near-zero % change blow-up).

"LLM insights" do not exist yet (no LLM code anywhere). They are a real
opportunity — but only if built as a **grounded interpretation layer over the
already-computed statistics**, opt-in, privacy-preserving, and as traceable as
the rest of the pipeline. A naive "summarize my data with ChatGPT" feature would
break the product's core promise ("your data stays on your computer") and invite
hallucinated statistics.

This document is organized as: strengths (§2), analytics gaps with severity and
fixes (§3), smaller correctness/quality nits (§4), the LLM design (§5),
a phased roadmap (§6).

---

## 2. What is already good (keep and build on)

- **Library/app separation.** `analysis.py` is table-in/table-out and has no
  Streamlit dependency, so every analytic is unit-testable and reusable. This is
  exactly the surface an LLM tool-calling layer needs (see §5.4).
- **Defensive numerics.** scipy-optional with graceful fallbacks (cosinor falls
  back to linear least squares; stats degrade to a structured warning table when
  scipy is absent). `np.trapezoid`/`np.trapz` compatibility shim.
- **Scientific caution is already encoded.** `EXPLORATORY_STATS_DISCLAIMER`,
  effect sizes (rank-biserial, epsilon-squared), BH FDR q-values, and
  subject-level means before group tests (avoids the most common
  pseudoreplication trap). This is better than many published pipelines.
- **Provenance.** Deterministic outputs, hashed inputs, config captured. The
  whole repo treats *traceability as a feature* — the right culture to extend to
  any AI output.

---

## 3. Analytics & scientific-insight gaps

Severity: **[H]** high (can mislead conclusions), **[M]** medium (missing value),
**[L]** low (nice-to-have).

### 3.1 [H] Cosinor reports parameters but no test of rhythmicity, and R² is optimistic

`summarize_circadian_cosinor` fits `MESOR + amplitude·cos(...)` and returns
amplitude, acrophase, MESOR, R². Two problems:

1. **No statistical test that a rhythm exists.** The standard cosinor output is
   the **zero-amplitude (rhythm-detection) test** — an F-test that amplitude ≠ 0,
   yielding a p-value and confidence intervals on amplitude and acrophase.
   Without it, an "amplitude = 0.4" on a flat, noisy trace looks identical to a
   real rhythm. Amplitude and acrophase should ship with **confidence
   intervals**, not bare point estimates.
2. **The fit is on ZT-bin *means* (`fit_data`), not raw observations.** Averaging
   into 1 h bins before fitting collapses within-bin variance, so the reported R²
   is inflated and CIs are impossible to compute correctly from the fit. Fit on
   the (subject-level) observations, or at minimum compute the rhythmicity test
   and CIs from the residual variance of the underlying points.

**Fix:** Add the zero-amplitude F-test + parameter CIs (closed-form from the
linear cosinor design matrix — no new dependency). Report `p_rhythm`,
`amplitude_ci_low/high`, `acrophase_ci_low/high`. Fit at the observation or
subject level.

### 3.2 [H] Hard-coded 12:12 photoperiod in circadian phase labeling

`_phase_from_zt()` labels `0 ≤ ZT%24 < 12 → light`, else dark. But the light
cycle is **user-configurable** in `light_dark.py` (`light_on`/`light_off`), and
non-12:12 photoperiods (16:8, 8:16, short/long-day designs) are common. The
cosinor `phase` column and any downstream light/dark logic that relies on this
constant will mislabel under any non-12 h photoperiod.

**Fix:** Derive the light/dark boundary from the same configured cycle used in
`annotate_light_dark`; thread the photoperiod length through instead of the `12.0`
constant.

### 3.3 [H] Non-parametric circadian metrics (IS / IV / RA, M10 / L5) are missing

Cosinor assumes a sinusoid. Rodent home-cage activity is **not** sinusoidal
(sharp dark-onset bursts, fragmented rest). The field-standard, distribution-free
descriptors are:

- **IS** (interdaily stability) — day-to-day reproducibility of the 24 h pattern.
- **IV** (intradaily variability) — fragmentation of the rest/activity rhythm.
- **RA** (relative amplitude) from **M10** (most-active 10 h) and **L5**
  (least-active 5 h).

These are robust, easy to compute from the same tidy table, directly comparable
to actigraphy literature, and far more informative for genotype/treatment effects
than a cosinor R². This is the single highest-value scientific addition.

### 3.4 [H] Activity is summarized only as means — no bout/fragmentation structure

The DVC locomotion index supports behaviorally meaningful structure that the tool
currently discards:

- **Active vs. inactive bouts** (count, mean duration, longest bout) given an
  activity threshold.
- **Total daily activity**, **% time active**, **dark/light activity ratio**
  (the ratio exists, but not the rest).
- **Day-to-day variability** per subject (a stability/consistency readout).

Effects often live in *fragmentation* or *bout structure* even when 24 h means are
unchanged. Means ± SEM alone can hide the actual phenotype.

### 3.5 [M] P-values without estimation, n-context, or power awareness

`quick_exploratory_stats` leads with Mann-Whitney/Kruskal p-values and p-stars.
For a tool that (correctly) refuses confirmatory claims, the emphasis is
backwards:

- **Lead with estimation.** Report the **group difference (or median
  difference) with a bootstrap CI** as the primary number; keep p as secondary.
  Effect sizes are computed but have **no uncertainty interval** — add bootstrap
  CIs to rank-biserial/epsilon-squared.
- **Small-n footgun.** With n = 3 vs 3 the smallest achievable two-sided
  Mann-Whitney p is 0.10 — *significance is impossible* regardless of effect.
  The tool should **warn when n is too small for the test to discriminate** and
  surface n prominently next to every p. Right now `n_subjects`/`n_total` are in
  the table but not used to gate or annotate.

### 3.6 [M] Per-bin testing ignores temporal autocorrelation; FDR family is implicit

`summarize_time_bins` + `quick_exploratory_stats` run an independent test per
time bin and BH-correct across them. Adjacent bins are autocorrelated (and the
input index is *already smoothed*), so the tests are not independent and the
effective number of comparisons is overstated. Also, the **FDR family scope**
(which set of metrics × bins × groups is corrected together) is not surfaced to
the user, so q-values are hard to interpret.

**Fix:** Prefer one summary statistic per subject per window (e.g. AUC or window
mean) → a single test per contrast, which both removes the autocorrelation
problem and makes the multiplicity explicit. Document the FDR family in the
output and the UI.

### 3.7 [M] Confidence/coverage is not propagated into the summaries — ✅ resolved

A group-bin mean computed from 1 surviving subject is rendered identically to one
from 8 (SEM widens, but nothing flags it). `n_subjects` is reported but there is
no gating, no "low confidence" annotation, and SEM hides n by construction.
Recommend: minimum-n annotations, optional CI bands instead of SEM, and a
"cells below n_min" flag in the analysis tables.

**Done:** the circadian and group-mean plots now default to a Student's t **95% CI**
band computed over per-subject means (`qc.confidence_halfwidth`), selectable as
95% CI / SEM / none, with hover showing n and hollow/open markers on bins with
fewer than three subjects. The group-mean plot now aggregates to subject level
before summarizing (previously it took SEM over raw bins).

### 3.8 [M] Near-zero baselines make `baseline_percent_change` explode

`baseline.py` guards **exact** zero (`np.where(bv != 0, ...)`) but not *near*-zero.
Light-phase locomotion baselines are frequently ~0, so percent change can blow up
to thousands of percent and dominate any group mean. Add a small-denominator
guard (e.g. floor on |baseline|, or report absolute change when baseline is below
a configurable epsilon) and flag affected rows.

### 3.9 [L] Single fixed 24 h period; no periodogram

Period is fixed at 24 h. Free-running (DD) or tau-mutant designs need period
*estimation* (e.g. Lomb–Scargle, which handles the uneven sampling left by
exclusions/dropouts). Even keeping cosinor, exposing the period as a swept
parameter and reporting the best-fit period would add real scientific reach.

### 3.10 [L] No machine-readable analysis summary

All analysis outputs are CSV/Markdown. There is no single structured
`analysis_results.json` capturing the headline numbers + metadata. This is a
small gap today but a **prerequisite** for the LLM layer (§5) and for any
external dashboard — the narrative layer should consume a compact, typed payload,
not re-parse CSVs.

---

## 4. Correctness & code-quality nits

- **`reporting.py:20`** — dead/confusing artifact:
  `datetime.now(...).strftime("%Y-%m-%d %Human:%M:%S UTC").replace("Human","")`
  is immediately overwritten by line 21. Delete line 20.
- **`_phase_from_zt` constant** — see §3.2; replace the literal `12.0`/`24.0`
  split with the configured photoperiod.
- **Statistical correctness tests** — add tests that *recover known parameters*
  from synthetic data (inject a cosine with known amplitude/acrophase, assert
  recovery within tolerance; inject a known group shift, assert the effect size).
  The current cosinor test exists but property-style recovery tests would lock in
  the new CI/rhythmicity math.
- **Effect-size sign/direction** — document the reference direction of
  rank-biserial (which group is "positive") in the column glossary so narratives
  and readers don't invert it.

---

## 5. Making "LLM insights" a real asset

### 5.1 The core tension: local-first vs. cloud LLMs

The product's headline promise is *"Your data stays on your computer… nothing is
uploaded to a server."* A naive LLM feature that ships raw per-minute data to a
cloud API **violates that promise** and would be a trust regression. Every design
decision below follows from resolving this tension:

> **The LLM never sees raw data. It sees small, derived, aggregated summary
> tables and a typed results payload — and it interprets, it does not compute.**

This is both a privacy stance and a hallucination defense: if the model is handed
the numbers and constrained to cite them, it cannot invent p-values.

### 5.2 Design principles

1. **Grounded, not generative.** The LLM is a *translation/interpretation* layer
   over `analysis_results.json` (§3.10). It must reference the provided numbers and
   is never the source of a statistic. Low temperature, structured output.
2. **Opt-in and egress-explicit.** Off by default. A button labeled with exactly
   what leaves the machine and to whom ("Sends the 6 summary tables — not your raw
   data — to <provider>"). A fully **offline templated narrative** is the default
   so the feature degrades gracefully and is testable with no network/keys.
3. **Pluggable providers.** Local (Ollama) for users who cannot send anything out;
   BYO-key cloud (Anthropic Claude recommended for narrative quality) for users who
   accept summary-level egress; a `NullProvider` that emits deterministic templated
   text. Default the cloud option to the latest Claude models.
4. **As traceable as everything else.** Persist the exact payload the model saw,
   the prompt, the model id, and token usage into the export bundle
   (`insights/payload.json`, `insights/narrative.md`) and record the model id +
   payload hash in `manifest.yaml`. A narrative without its inputs is not
   reproducible and doesn't belong in this repo.
5. **Caveats are non-negotiable.** Inject `EXPLORATORY_STATS_DISCLAIMER` into
   every narrative and forbid the model from upgrading "exploratory" to
   "significant/confirmatory." Surface n alongside every claim.

### 5.3 Features, in value order

1. **Plain-language results narrative (highest value).** Convert the analysis
   CSVs — which non-coder beta testers *cannot read* — into a paragraph: *"KO
   showed ~32% lower dark-phase locomotion than WT (rank-biserial 0.6, exploratory
   p = 0.03, n = 6/6); circadian amplitude was reduced and acrophase delayed ~1.5 h.
   Exploratory only."* This directly serves the stated audience.
2. **QC / confound triage assistant.** Feed `quality_report.csv` +
   `exclusion_log` + manifest → *"3 cages show irregular intervals after Nov 5;
   M07 is 40% missing in its baseline window (baseline imputed); consider
   excluding."* Grounded entirely in tables you already compute; turns QC tables
   into actions.
3. **Methods-section drafter.** From `analysis_config.yaml` + `manifest.yaml` →
   a reproducible Methods paragraph (binning, alignment, baseline window, cosinor
   period, tests, FDR). Near-zero hallucination risk (it transcribes config) and a
   real time-saver toward publication.
4. **Conversational Q&A with tool-calling over `analysis.py` (the asset
   multiplier).** Because the analytics are clean table-in/table-out functions,
   they can be exposed as **callable tools** so the model *computes by calling real
   functions* ("which animals drove the dark-phase difference?" → it calls
   `compute_auc_per_animal`) instead of inventing numbers. This is where the
   existing architecture (and the `api_adapter.py` placeholder) pays off, and it is
   the safest possible LLM-numeric integration.
5. **Hypothesis/anomaly flagging (explicitly speculative).** "Group 3's acrophase
   is an outlier vs. the others — worth checking," always framed as a hypothesis
   to test, never a finding.
6. **Literature grounding (stretch).** A PubMed/articles source could attach
   relevant references to a finding. Adds external egress and scope — defer.

### 5.4 Proposed architecture (small, testable, offline-first)

New module `src/dvc_behavior/insights.py`:

```text
build_insight_payload(analysis_tables, config, manifest, quality_report) -> dict
    # PURE + deterministic: the typed, compact context. Fully testable, no LLM.

class LLMProvider(Protocol):
    def complete(self, system: str, payload: dict) -> InsightResult: ...

NullProvider      # deterministic templated narrative (default; CI-testable)
OllamaProvider    # fully local, no data leaves the machine
AnthropicProvider # BYO-key cloud, latest Claude; egress = summary tables only

generate_narrative(payload, provider, *, disclaimers) -> InsightResult
    # returns text + payload_hash + model_id + token_usage
```

- **App:** an "Insights" expander on the Analysis page, behind an explicit
  egress-disclosing button; default provider = offline templated.
- **Export:** `insights/narrative.md`, `insights/payload.json`, and model
  metadata folded into `manifest.yaml`.
- **Tooling layer (feature 4):** register the `analysis.py` functions as tools and
  let `AnthropicProvider` use function-calling; keep the same registry usable by a
  local model that supports tools.

The key property: `build_insight_payload` and `NullProvider` give a **complete,
deterministic, network-free feature** that is unit-tested like the rest of the
library. The cloud/local model is an *enhancement* layered on a working offline
baseline — never a hard dependency.

---

## 6. Suggested roadmap (incremental, low-dependency)

> **Implementation status (2026-06-15).** Phases 1–3 are implemented, plus the
> period-estimation item from Phase 4. The remaining open items are the
> agentic tool-calling loop and optional literature grounding. See §8.

**Phase 1 — Scientific trust (no new deps, highest ROI)** — ✅ done
- [x] Cosinor rhythmicity F-test + amplitude/acrophase CIs; subject-level fit (§3.1).
- [x] Configurable photoperiod in phase labeling (§3.2).
- [x] Estimation-first stats: median difference + bootstrap CI; effect-size CIs;
  small-n warnings with `min_possible_p` and n surfaced (§3.5).
- [x] Near-zero baseline guard + `baseline_percent_change_unstable` flag (§3.8).
- [x] Remove `reporting.py:20`; add parameter-recovery tests (§4).

**Phase 2 — Scientific depth** — ✅ mostly done
- [x] Non-parametric circadian metrics IS/IV/RA, M10/L5 (§3.3).
- [x] Bout / fragmentation / daily-total / %-time-active metrics (§3.4).
- [x] Window-summary-per-subject contrasts (`compare_window_summaries`) to
  replace per-bin multiplicity (§3.6).
- [x] `insights/payload.json` structured summary (subsumes the JSON in §3.10).
- [x] Coverage/confidence annotations and CI *bands* on the plots (§3.7) —
  subject-level Student's t 95% CI bands (selectable CI/SEM/none) with low-n
  markers on the circadian and group-mean plots.

**Phase 3 — Grounded LLM layer** — ✅ done
- [x] `insights.py` with `build_insight_payload` + `NullProvider` (offline narrative).
- [x] Results narrative + QC triage + Methods drafter (§5.3 #1–3).
- [x] Provider plug-ins (Ollama local, Anthropic BYO-key) with egress disclosure
  and full traceability into the export bundle (§5.2, §5.4).

**Phase 4 — Agentic / stretch**
- [x] Period estimation / periodogram (`estimate_period`, Lomb–Scargle) (§3.9).
- [x] Tool-calling Q&A over `analysis.py` (§5.3 #4) — `build_tool_specs` +
  `execute_analysis_tool` + `answer_question` run a bounded agent loop that calls the
  real analysis functions; an offline `ScriptedToolProvider` keeps it fully testable.
  Wired into the Analysis page (tool-capable provider required).
- [x] Optional literature grounding (§5.3 #6) — `literature.py` (Europe PMC,
  opt-in, generic-keyword queries only); offline-default + monkeypatchable HTTP seam.

All roadmap items from this review are now implemented.

---

## 8. What shipped on `claude/analytics-llm-insights-review-u6ulya`

- **`analysis.py`:** cosinor rhythmicity test (`p_rhythm`) + amplitude/acrophase
  CIs + subject-level fit; configurable photoperiod; estimation-first columns in
  `quick_exploratory_stats`; new `summarize_nonparametric_circadian`,
  `summarize_activity_bouts`, `compare_window_summaries`, `estimate_period`.
- **`baseline.py` / `reporting.py`:** near-zero percent-change guard + flag;
  removed the stale timestamp line.
- **`insights.py` (new):** `build_insight_payload`, `payload_hash`,
  `InsightResult`, `LLMProvider`/`NullProvider`/`OllamaProvider`/`AnthropicProvider`,
  `generate_narrative`, `draft_methods_section`, `triage_quality`, and an
  analysis-function tool registry — offline-first, never sees raw data. Plus the
  grounded Q&A loop: `build_tool_specs`, `execute_analysis_tool`,
  `AnthropicToolProvider`/`ScriptedToolProvider`, and `answer_question`.
- **`export.py`:** generic `text_artifacts` channel writes the `insights/` bundle.
- **App:** new analysis tables surfaced on the Analysis page; an insight-engine
  selector (offline / Ollama / Anthropic) with egress disclosure; a grounded
  tool-calling Q&A panel; insights bundle folded into the export ZIP.
- **`qc.py`:** `confidence_halfwidth` (t-based CI) and subject-level group-mean
  aggregation; the circadian and group-mean plots gained selectable CI/SEM/none
  bands with low-n markers.
- **`literature.py` (new):** opt-in Europe PMC literature grounding —
  `build_literature_queries` (generic keywords only), `EuropePMCProvider` /
  `NullLiteratureProvider`, `find_supporting_literature`, markdown/JSON export.
- **Docs:** `AGENTS.md` contributor/agent guide; README "AI insights" section
  (incl. a concrete recommended starting model and the literature feature).
- **Tests:** `test_analysis_advanced.py` (incl. property-recovery), `test_insights.py`,
  `test_insights_agent.py` (offline agent loop), expanded
  baseline/reporting/export/app-smoke coverage.

---

## 7. One-paragraph recommendation

Spend the next iteration making the **existing numbers trustworthy and deeper**
(rhythmicity tests with CIs, IS/IV/RA + bout structure, estimation-first stats
with n-awareness, configurable photoperiod) and emit a compact
`analysis_results.json`. *Then* add the LLM as a thin, grounded, opt-in
interpretation layer that reads that JSON, never the raw data — defaulting to an
offline templated narrative, with Ollama (local) and Anthropic Claude (BYO-key)
as enhancements, and with every AI output persisted and hashed like the rest of
the pipeline. Done in that order, the LLM becomes a genuine asset (it turns
unreadable CSVs into publishable, caveated prose and triages QC) without
compromising the local-first, traceable promise that makes this tool credible.
