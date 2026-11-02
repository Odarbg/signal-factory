# signal-factory: design for milestone 1

Date: 2026-11-02. Status: design, builder-reviewed, approved by owner.
Owner: Radoslav Ivanov (personal, public; GitHub `Odarbg`).
Background: `~/Projects/personal/ideas/ai-os/` (vision, protocol, milestone-1
decisions). This document is self-contained; the background explains why.

## 1. Intent

A supervised research loop that asks an LLM to propose *derived signals* for fraud
detection from a data dictionary and domain notes, computes those signals without
leaking the future, and measures, with a fair baseline, whether they help a fixed
learner on unseen later time periods.

Research question:

> Can an LLM use domain descriptions and relational data to construct derived signals
> that improve a fixed learner's fraud detection on unseen time periods, beyond the
> original features and routine automated aggregations?

A negative answer, reliably shown, is a valid result. The system must be able to
conclude "no supported improvement" and must not manufacture one.

This is milestone 1 of a longer programme (lifecycle, replay with delayed labels, code
generation, a second domain). None of that is in scope here.

## 2. Goals, non-goals, anti-goals

Goals
- One reproducible ablation on IEEE-CIS with a pre-registered acceptance criterion.
- A point-in-time feature engine that provably does not leak.
- A proposer whose every attempt is recorded and whose data access is enforced in code.
- A report that is useful even if every proposal is rejected.

Non-goals (deferred)
- Label-delay simulation, replay, monitoring, rule replacement, rollback.
- Generating executable code from signals.
- An MCP server or any agent-host integration.
- Market data or any second domain.
- A UI.

Anti-goals (would make this a failure even if it "works")
- Any result that used validation or final labels to choose a threshold, a signal or a
  hyperparameter.
- A lift reported without the routine-aggregation baseline.
- Any company (Osprey) data, numbers or identifiers anywhere in the repo or prompts.
- Dataset files committed to git.

## 3. Constraints

- Public data only. IEEE-CIS competition rules: non-commercial, academic use, **no
  redistribution**. Data lives in a gitignored `data/` directory; a script downloads it
  via the Kaggle CLI after the owner accepts the rules under their own account.
- Pure Python 3.12, managed with `uv`. `mypy --strict`, `ruff`, `pytest`, `hypothesis`.
- Runs on a laptop (train CSV is 1.35 GB; use float32 and Parquet).
- Pre-registration: the numbers in section 6 are fixed before any candidate is
  evaluated. Changing one afterwards is a new experiment and is reported as such.
- The proposer's process cannot read validation or final data. Enforced by file layout
  and by the module boundary, not by prompting.
- The owner practises TDD; every component ships with tests written first.

## 4. Dataset

IEEE-CIS Fraud Detection (Kaggle, Vesta). **Train split only** (`train_transaction.csv`,
`train_identity.csv`); test labels are private. 590,540 transactions, 3.5% fraud,
`TransactionDT` in seconds over 183 days, `isFraud` label.

Column families (meanings documented by the host; exact semantics of C/D/V families are
partly masked): `TransactionAmt`, `ProductCD`, `card1–6` (card type, category, issuer
bank, country), `addr1/addr2` (billing region, country), `dist1/dist2`,
`P_emaildomain`, `R_emaildomain`, `C1–14` (counts), `D1–15` (day deltas), `M1–9` (match
flags), `V1–339` (Vesta-engineered), identity table `id_01–38`, `DeviceType`,
`DeviceInfo`. The identity table joins on `TransactionID` and covers about 24% of rows.

Pseudo-entity identifier: `uid = card1 + "_" + addr1 + "_" + D1n` where
`D1n = floor(TransactionDT / 86400) - D1`. This is an **inferred grouping, not a verified
card identity**; two cards can share it. Always say "pseudo-identifier", never "card".
Its collision rate is unknown; measure and report it (section 10, open item).

Cross-row keys available for signals: `uid`, `card1`, `card1+addr1`, `P_emaildomain`,
`R_emaildomain`, `DeviceInfo`, `id_30`/`id_31` (OS, browser), `addr1`.

Known pitfalls (encode as tests or documented handling):
- `D1n` encodes the card's start day; `D9` is hour of day. Calendar leakage is possible
  if D-columns are used naively across the split.
- `addr2` nearly duplicates `V79`.
- Three-decimal `TransactionAmt` values indicate foreign transactions.
- Labels are effectively per uid: 97% of multi-transaction uids are all-0 or all-1, so
  entity aggregates partly memorise identity.

Label semantics (host): a reported chargeback marks the transaction and posterior
transactions linked by account/email/billing address as fraud; nothing reported within
120 days means legit. There is no per-row label-arrival time.

**The label is never an input.** `isFraud` is absent from the data dictionary, from
`history.parquet`, and is not a legal `column` or `filter` column in any signal, nor a
DFS target. `count(isFraud=1) per uid` would dominate everything and answer nothing.
(Builder-check decision.)

**Row A columns.** All original transaction and identity columns except
`TransactionID`, `TransactionDT`, `uid`, `day`. D-columns are used raw; their calendar
drift is documented in the report, not corrected. Every ablation row shares this base,
so the drift affects all rows equally. (Builder-check decision.)

## 5. Chronological split

Day `d = floor(TransactionDT / 86400)`, day 1 = first day present.

| Period | Days | Use |
|---|---|---|
| Dev | 1–105 | Train rows; proposer may see evaluation results computed here |
| Validation | 106–145 | Select signals and freeze the threshold; never shown to the proposer |
| Final | 146–183 | Touched exactly once, at the end |

A 7-day **gap** precedes each boundary. Concretely: dev training rows are days 1–98;
validation rows are days 106–138; final rows are days 146–183. Days 99–105 and
139–145 are **history only**: excluded from training and from evaluation, but
available as feature history. Feature history for any row includes all rows with
earlier `TransactionDT` regardless of period. The purpose of the gap is that no
training row's label is within 7 days of an evaluation row, so window features and
label propagation (the host marks posterior linked transactions as fraud) do not
straddle a boundary.

**This is a retrospective experiment with fully observed labels.** The report says so
in its first paragraph. Real-world label latency (30–120 days) is not simulated.

Seen/unseen: for validation and final, every metric is reported separately for rows
whose `uid` appears in dev training rows and rows whose `uid` does not. The headline
claim must hold on the unseen subset.

## 6. Pre-registered evaluation

| Item | Value |
|---|---|
| Learner | LightGBM binary classifier, one fixed parameter set for every row of the ablation, 5 seeds, predictions averaged. Parameters are set once on dev with a small time-based holdout *before* any derived signal exists, then frozen. |
| Features handling | Categorical columns via LightGBM categorical support; missing values left as NaN; no target encoding. |
| Headline metric | Recall at 1% false-positive rate. The score threshold is chosen **once, on all validation rows**, to give FPR = 1%, frozen, then applied to final; final reports recall **and the actual FPR** at that threshold, overall and per seen/unseen subset. One threshold, not one per subset: a deployed system has one. (Builder-check decision.) |
| Model used on final | The model trained on dev training rows only (days 1–98), the same one the threshold was chosen on. No retraining on dev + validation. (Builder-check decision.) |
| Secondary metrics | PR-AUC, ROC-AUC. |
| Acceptance | Row C or D beats row B by ≥ 2.0 points of recall@1%FPR on the final period, overall **and** on the unseen-uid subset, with a paired bootstrap 95% interval (1,000 resamples over final rows, same resample applied to both rows) that excludes zero. |
| Uncertainty | Paired bootstrap as above; also report seed-to-seed standard deviation. |
| Outcome if not met | "No supported improvement." Report which of poor proposals, limited data, or an already strong baseline best explains it, with evidence. |

## 7. Ablation rows

| Row | Inputs to the learner |
|---|---|
| A. Baseline | Original transaction + identity columns |
| B. Routine aggregations | A + routine aggregations (featuretools DFS or the signal engine, see below) over a **fixed grid**: keys {`uid`, `card1+addr1`, `P_emaildomain`, `DeviceInfo`} × windows {24h, 7d, all history} × { `count`; `time_since_last`; `sum`/`mean`/`max`/`min` of `TransactionAmt`, `dist1`, `C1–C14`, `D1–D15`; `nunique` of `ProductCD`, `card2–card6`, `addr1`, `P_emaildomain`, `R_emaildomain`, `DeviceInfo` (excluding the grouping key itself) }. About 1,600 features. Pre-registered; not tuned. |
| C. AI, dictionary only | B + LLM-proposed signals; proposer sees only the data dictionary |
| D. AI, dictionary + domain notes | B + LLM-proposed signals; proposer also sees generic card-fraud notes |
| E. Expert (optional) | B + owner-proposed signals, same contract, same budget. Only if time permits. |

Same learner, parameters, seeds and periods for every row. Lift is C−B and D−B. D−C
tells whether domain knowledge adds anything beyond reading the dictionary.

Row B must pass the same leakage test as the signal engine (section 8.2). featuretools'
cutoff-time mechanism is tried first. **Fallback is permitted on either of two
triggers**: it fails the leakage test, or it does not finish the full grid on the
owner's laptop within a time box of 8 hours of compute. In either case, implement the
same grid in the signal engine instead (the grid is depth-1 and fits the spec exactly)
and say so in the report. Plan the engine path from the start; the grid is the same
either way. (Builder-check decision.)

**Why the grid bounds C and D.** The signal spec is depth-1 with the same primitives,
so the proposer's room beyond B is: other keys (`card1`, `R_emaildomain`, `id_30`,
`id_31`, `addr1`), other windows, filters, and columns the grid does not cover. That is
deliberate: the claim is semantic selection, not a richer function class.

## 8. Components

Package layout (one `uv` project, `src/signal_factory/`):

```
data/        split.py, download.py, schema (column dictionary as data)
signals/     spec.py (pydantic model + JSON schema), compile.py (spec → DuckDB SQL), engine.py
baseline/    learner.py, metrics.py, bootstrap.py
routine/     dfs.py (featuretools row B)
proposer/    prompt.py, client.py, budget.py, log.py
ablation/    run.py, report.py
```

### 8.1 Data

- `download.py`: wraps `kaggle competitions download -c ieee-fraud-detection`; fails
  with a clear message if the rules have not been accepted. Writes raw CSVs to
  `data/raw/`.
- Conversion to Parquet in `data/parquet/` with float32, `uid`, `day`, and the identity
  join applied (left join).
- `split.py`: writes `data/split/dev.parquet`, `validation.parquet`, `final.parquet`,
  and `history.parquet` (all rows, for feature computation). Period membership is a
  column; the files exist so that access can be restricted by file.
- Tests: boundary days, no row in two period files, gap rows absent from dev training
  and from evaluation rows, row counts sum correctly, uid construction on fixtures.

### 8.2 Signal engine

Signal spec (pydantic, exported as JSON schema), depth 1 only:

| Field | Type / allowed values |
|---|---|
| `id`, `version` | string, int |
| `hypothesis` | string, one or two sentences |
| `entity_key` | one of the keys in section 4 |
| `column` | a column name from the dictionary, or `*` for count |
| `aggregation` | `count`, `count_distinct`, `sum`, `mean`, `max`, `min`, `time_since_last` |
| `window` | `all` or `{hours: n}` / `{days: n}` preceding decision time |
| `filter` | optional, a single `column op literal` predicate on history rows |
| `missing` | `null` (abstain) only, in milestone 1 |
| `inputs` | list of columns used, each with `available_at: decision_time` (the only value allowed in milestone 1; the field exists so later milestones can declare delays) |

Compilation: spec → one DuckDB SQL expression using a window over `history` ordered by
`TransactionDT`, partitioned by `entity_key`, range-bounded by the window, **strictly
before** the current row's `TransactionDT` (ties excluded: a row never sees itself or
same-timestamp rows).

Leakage property test (Hypothesis): for random small histories and a random signal
spec, computing the signal at every row, then appending random later rows and
recomputing, must leave every earlier value unchanged. This test is an acceptance gate
for the engine and for row B.

Other tests: each aggregation and window against hand-computed fixtures; filter
semantics; abstain on empty history; invalid specs rejected with reasons; compiled SQL
is deterministic.

### 8.3 Baseline, metrics, bootstrap

- `learner.py`: train/predict with fixed params and seeds; returns per-row scores.
- `metrics.py`: `recall_at_fpr(scores, labels, fpr=0.01)` returning threshold too;
  `apply_threshold`; PR-AUC, ROC-AUC via scikit-learn.
- `bootstrap.py`: paired bootstrap over rows for the difference of two score vectors.
- Tests: metrics against scikit-learn on toy data; threshold round-trip; bootstrap
  interval contains the true difference on synthetic data; seen/unseen split of rows.

### 8.4 Routine aggregations (row B)

featuretools EntitySet with cutoff times from `TransactionDT`; feature matrix saved to
Parquet; passes the leakage test from 8.2 (sampled rows, since the full matrix is
large). Document the exact primitive and window set.

### 8.5 Proposer

- `prompt.py`: assembles the system and user prompt from (a) the data dictionary (a
  YAML file in the repo), (b) the domain notes file (arm D only), (c) the JSON schema,
  (d) dev-period evaluation results of previous rounds (metric deltas per signal, dev
  only), (e) the list of already-proposed signal ids to avoid repeats, (f) a compact
  description of the row-B grid (its keys, windows, primitives and columns, as rules,
  not 1,600 names) with the instruction to propose signals the grid does not cover.
  **Owner decision 2026-11-02:** the proposer is told what B computes, so the budget
  goes on signals beyond routine aggregations, which is the research question. The
  report states this; the result reads "given routine aggregates exist, can an LLM add
  value?", not "does an unguided LLM rediscover them?"
- `client.py`: calls the model through the Anthropic SDK. Model id, temperature and
  max tokens are configuration; defaults chosen at implementation time and recorded in
  every log entry.
- `budget.py`: 3 rounds × 10 proposals per arm. Round loop: propose → validate against
  schema → compute on history → evaluate on dev (learner trained on dev with the
  signal added to row-B features, measured on a time-based dev holdout) → feed dev
  results back → next round. Validation numbers are never shown to the proposer.
- Selection (pre-registered, builder-check decision): after all three rounds, each
  valid signal is scored **individually** by its marginal lift, recall@1%FPR of
  (row B + that one signal) minus row B, trained on dev training rows with 1 seed and
  evaluated on validation. Signals with lift ≤ 0 are discarded. The remaining signals,
  ranked by lift, form the arm's set, capped at 10. Exact duplicates of a row-B grid
  feature (same key, window, primitive, column, no filter) are discarded before
  scoring. **A zero-signal arm is a legal outcome**; the report then states "no signal
  with positive validation lift" for that arm and the final evaluation of that arm
  equals row B. Individual scoring, not greedy forward selection: about 60 extra fits
  per arm, and it keeps the procedure independent of order.
- `log.py`: append-only JSONL of every proposal: arm, round, prompt hash, model id,
  raw response, parsed spec or rejection reason, dev metrics, validation rank, tokens
  used.
- Data boundary: the proposer package imports only `signals.spec` and reads only
  `data/split/dev.parquet`, `data/dictionary.yaml`, `data/domain_notes.md`. A test
  asserts the package has no import path to validation/final files and that the
  evaluation call it receives is a dev-only evaluator.
- Domain notes (arm D): written by the owner from public sources only; generic
  statements such as "one card used under many names", "a card on many accounts with
  little money collected", "bursts of attempts in minutes". No company data, numbers or
  identifiers.

### 8.6 Ablation runner and report

- `run.py`: one command runs rows A–D (E if present): builds features per row, trains
  with 5 seeds, freezes thresholds on validation, evaluates on final once, writes
  `results/<timestamp>/` with a JSON of every metric, interval, seed, proposal and
  rejection, plus the Markdown report.
- The runner refuses to run the final evaluation twice for the same configuration hash
  (a marker file), to make "touched once" mechanical.
- Report sections: retrospective-labels statement; dataset and split; ablation table
  (overall and unseen-uid); acceptance decision with intervals; proposal statistics
  (valid / invalid / selected per arm, token usage); the signals selected, with their
  hypotheses; failure analysis if not accepted.

## 9. Work plan, in order

1. Repo, tooling, download script, Parquet conversion, uid, split with gaps. Tests.
2. Signal spec, compiler, engine, leakage property test.
3. Learner, metrics, bootstrap, seen/unseen split; row A result on validation.
4. Row B via featuretools with cutoff times; leakage test on samples.
5. Proposer: prompt, client, budget loop, logging, data-boundary test; arms C and D.
6. Ablation runner, final evaluation (once), report.
7. Write-up.

Steps 1–4 contain no LLM and are roughly half the work.

## 10. Open items the builder measures, not decides

- The `uid` collision rate (how many distinct `card1–6` combinations share a uid). Report
  it; if it is high, say so prominently and keep the guard as defined.
- Whether featuretools' cutoff handling passes the leakage test (section 7 fallback).
- LightGBM parameter set: chosen once on dev before signals exist, then frozen; the
  values are recorded in the report.

## 11. Left to the builder

- Parquet partitioning, DuckDB vs pandas for the learner frame, exact LightGBM params,
  prompt wording, CLI shape, report formatting, CI configuration.
- Any decision that does not change what is measured or how the boundary is enforced.
