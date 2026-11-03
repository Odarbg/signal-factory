# signal-factory Plan 2: Proposer, selection, final evaluation, report

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The LLM half of milestone 1: a proposer whose data access is enforced in code, the pre-registered selection of its signals on validation, the single final-period evaluation of rows A–D with paired bootstrap, and the report that is useful even if every proposal is rejected.

**Architecture:** A `proposer` package that imports only the signal contract and the knowledge files, reads only `dev.parquet`, talks to Claude through an injected client, and receives an injected dev-only evaluator. The `ablation` package owns everything that sees validation or final: the dev evaluator's construction, selection on validation, the once-only final runner guarded by a marker file, and the report. Signals proposed by the LLM are computed by the plan 1 engine over `history.parquet` on the ablation side; the proposer never computes anything outside dev.

**Tech Stack:** Plan 1 stack plus the `anthropic` Python SDK (structured outputs via `client.messages.parse`, prompt caching on the system prompt). Tests never call the API; a fake client returns canned responses.

**Spec:** `docs/superpowers/specs/2026-11-02-signal-factory-design.md` (§5, §6, §7, §8.5, §8.6). Plan 1 and its results: `docs/superpowers/plans/2026-11-02-signal-factory-plan-1-foundations.md`, `docs/results/plan-1-rows-a-b.md`.

## Global Constraints

- Everything in plan 1's Global Constraints still holds: Python 3.12 with `uv`; `make check` (pytest, ruff, `mypy --strict` over `src tests scripts`) green before every commit; `data/` and `results/` gitignored; `isFraud` never an input; strict-past signals; periods dev 1–98, gap 99–105, validation 106–138, gap 139–145, final 146–183.
- Pre-registered and not tuned (spec §6, §8.5): seeds `(0, 1, 2, 3, 4)`; FPR = 0.01; 1,000 bootstrap resamples; acceptance ≥ 2.0 points recall@1%FPR on final, overall **and** unseen, with both 95% intervals excluding zero; budget 3 rounds × 10 proposals per arm; selection by individual marginal lift on validation with 1 seed (seed 0), lift ≤ 0 discarded, ranked, capped at 10; frozen params from `src/signal_factory/baseline/params.json` (commit 5287438) for every fit in this plan.
- Dev-only feedback fits (spec §8.5) train on dev days 1–84 and measure on dev days 85–98, the same holdout used to freeze the parameters, 1 seed (seed 0).
- The proposer package (`src/signal_factory/proposer/`) imports, from this project, only `signal_factory.signals.spec`, `signal_factory.signals.columns`, `signal_factory.knowledge` and itself. It opens exactly one Parquet file, `dev.parquet`. It receives the grid description, the grid cell set, the dev evaluator and the model client as arguments. Enforced by tests in Task 10.
- Spec §8.5 names `data/dictionary.yaml` and `data/domain_notes.md`; `/data/` is gitignored, so these two files live in `src/signal_factory/knowledge/` as package data. Recorded here as a plan decision, not a spec change.
- `final.parquet` is opened by exactly one function, `run_final` (Task 12), once per configuration hash. Nothing in Tasks 1–11 or 13–14 opens it.
- Validation numbers never reach the proposer: no validation metric is written to the proposal log before the rounds end, and the prompt assembly has no validation input.
- Model access: `ANTHROPIC_API_KEY` (or an `ant auth login` profile) from the environment only. No key, prompt log or response is ever committed; proposal logs under `results/` are copied into `docs/results/` only after the owner checks they contain no row-level data (they contain specs, hypotheses and aggregate metrics).
- Default model configuration, recorded in every log entry: model `claude-opus-5-5`, `output_config.effort = "high"`, `max_tokens = 16000`, adaptive thinking (the API default; no `thinking` parameter sent). Current models reject `temperature`, so "temperature" in spec §8.5 is recorded as `effort` instead. No server-side fallback model: a `refusal` stop reason is logged and that round yields zero proposals.
- No company data, numbers, identifiers or internal hypotheses anywhere: not in the dictionary, the domain notes, the prompts, the tests or the report. Public-source card-fraud knowledge only.
- Collaboration: the owner chose "Claude writes tests and implements, commit on main" for plan 1 and asked to be asked once per plan. Ask once at the start of this plan, with an "apply to all tasks" option.

## Review Focus

Inputs the spec implies but no task's tests would otherwise exercise, most likely to bite first. Each has its test pinned in the owning task.

1. A model response that is truncated at `max_tokens`, is not JSON, or ends with `stop_reason == "refusal"` must be logged with the stop reason, consume that round's budget position, and let the next round proceed. → Task 9.
2. A proposal with a legal shape but a column the dictionary does not list, a `filter.value` string against a numeric column, or an id starting with the reserved `rb__` prefix must be rejected with a recorded reason, never computed. → Task 5.
3. A candidate feature Parquet that lacks some `TransactionID` of the split (stale file from an earlier proposer run) must raise before training, as plan 1's `_load_features` does; the selection step must write a file covering every history row. → Task 11.
4. A crash after `final.parquet` is read but before results are written must not leave a marker file; a second complete run with the same configuration hash is refused with a message naming the marker. → Task 12.
5. Two arms may propose the same signal id; the candidate columns are namespaced `c__<id>` and `d__<id>` so rows C and D never collide, and a zero-signal arm's final row equals row B exactly, including its threshold. → Tasks 11 and 12.

---

### Task 1: Knowledge files: data dictionary and domain notes

**Files:**
- Create: `src/signal_factory/knowledge/__init__.py`, `src/signal_factory/knowledge/dictionary.yaml`, `src/signal_factory/knowledge/domain_notes.md`, `src/signal_factory/knowledge/loader.py`, `tests/knowledge/__init__.py`, `tests/knowledge/test_loader.py`, `tests/knowledge/test_notes.py`

**Interfaces:**
- Consumes: `ENTITY_KEYS`, `FORBIDDEN_COLUMNS` from `signal_factory.signals.columns`.
- Produces:
  - `ColumnInfo` (frozen dataclass): `name: str, family: str, kind: Literal["numeric", "categorical"], description: str`.
  - `Dictionary` (frozen dataclass): `columns: dict[str, ColumnInfo]`, `keys: dict[str, str]` (entity key → one-line description), `families: dict[str, str]` (family → description).
  - `load_dictionary() -> Dictionary` reading the packaged YAML through `importlib.resources.files("signal_factory.knowledge")`.
  - `load_domain_notes() -> str`.
  - `RAW_FEATURE_COLUMNS: frozenset[str]`: the 431 IEEE-CIS columns a signal may use (434 raw minus `TransactionID`, `TransactionDT`, `isFraud`), generated from name patterns in `loader.py`.

YAML format (compact, ranges expanded by the loader):

```yaml
families:
  V: {kind: numeric, description: "Vesta-engineered features; semantics masked by the host"}
  ...
columns:
  - {name: TransactionAmt, family: amount, kind: numeric, description: "Transaction amount in USD; three decimals suggest a foreign-currency conversion"}
  - {range: {prefix: V, start: 1, end: 339}, family: V}
  - {range: {prefix: id_, start: 1, end: 38, pad: 2}, family: identity}
  ...
keys:
  uid: "pseudo-identifier card1 + addr1 + card start day; an inferred grouping, two cards can share it"
  ...
```

Descriptions come from the public Kaggle data description and the host's public forum notes only (spec §4 column families and pitfalls: `D9` is hour of day, `addr2` near-duplicates `V79`, identity covers about 24% of rows).

- [ ] **Step 1: Write the tests**

```python
def test_dictionary_covers_exactly_the_raw_feature_columns() -> None:
    d = load_dictionary()
    assert set(d.columns) == RAW_FEATURE_COLUMNS and len(d.columns) == 431
def test_dictionary_never_lists_forbidden_columns() -> None:
    assert not (set(load_dictionary().columns) & FORBIDDEN_COLUMNS)
def test_dictionary_keys_match_entity_keys() -> None:
    assert set(load_dictionary().keys) == ENTITY_KEYS
def test_every_column_has_family_kind_and_description() -> None: ...
def test_range_expansion_pads_identity_columns() -> None:
    assert "id_01" in load_dictionary().columns and "id_1" not in load_dictionary().columns
def test_domain_notes_are_generic() -> None:
    notes = load_domain_notes()
    assert len(notes) > 500
    assert "osprey" not in notes.lower()
    assert not re.search(r"\b\d{2,}\b", notes)  # no magnitudes: generic statements only
```

- [ ] **Step 2: Run to verify they fail** (`uv run pytest tests/knowledge -q`; expected: ImportError)

- [ ] **Step 3: Write `dictionary.yaml`, `domain_notes.md`, `loader.py`**

Domain notes: eight to twelve short generic patterns of card-not-present fraud from public sources (card testing bursts, one card under many names or emails, new device or browser for an established card, mismatch between billing region and purchase region, amount just under round thresholds, many distinct cards from one device or email domain, first transaction of a pseudo-identity, recurring-product fraud on free-trial style products). No numbers, no company names, no dataset-specific findings from validation.

- [ ] **Step 4: Run tests; `make check`**

- [ ] **Step 5: Commit** `git commit -m "Add data dictionary and generic domain notes as package data"`

- [ ] **Step 6 (owner gate): read `domain_notes.md` and confirm or edit it before arm D is ever run. Record the confirmation date at the top of the file in the Task 14 commit.**

---

### Task 2: Proposer contracts and dev-only resources

**Files:**
- Create: `src/signal_factory/proposer/__init__.py`, `src/signal_factory/proposer/types.py`, `src/signal_factory/proposer/resources.py`, `tests/proposer/__init__.py`, `tests/proposer/conftest.py`, `tests/proposer/test_resources.py`

**Interfaces:**
- Consumes: `SignalSpec`, `Filter`, `SignalInput`, `WindowHours`, `WindowDays` (plan 1 Task 5); `Dictionary` (Task 1).
- Produces (`types.py`):
  - `Arm = Literal["C", "D"]`; `ROUNDS = 3`; `PROPOSALS_PER_ROUND = 10`; `SELECTION_CAP = 10`.
  - `RawProposal` (pydantic, `extra="forbid"`): the SignalSpec fields with loose types: `id: str, version: int, hypothesis: str, entity_key: str, column: str, aggregation: str, window: Literal["all"] | WindowHours | WindowDays, filter: Filter | None = None, missing: Literal["null"] = "null", inputs: list[SignalInput]`. `ProposalBatch` (pydantic): `proposals: list[RawProposal]`.
  - `Rejection` (frozen dataclass): `proposal_id: str | None, reason: str, raw: dict[str, Any] | None`.
  - `DevMetrics` (frozen dataclass): `recall_delta_points: float, pr_auc_delta: float, holdout_rows: int, holdout_positives: int`.
  - `DevEvaluator = Callable[[SignalSpec], DevMetrics]`.
  - `ProposerResponse` (frozen dataclass): `text: str, model: str, effort: str, max_tokens: int, stop_reason: str, input_tokens: int, output_tokens: int, cache_read_tokens: int`.
  - `ProposerClient` (Protocol): `def propose(self, system: str, user: str) -> ProposerResponse`.
  - `FeedbackItem` (frozen dataclass): `proposal_id: str, status: Literal["evaluated", "rejected"], metrics: DevMetrics | None, reason: str | None`; `RoundFeedback`: `round: int, items: tuple[FeedbackItem, ...]`.
- Produces (`resources.py`):
  - `ColumnStats` (frozen dataclass): `null_rate: float, distinct: int, minimum: float | None, maximum: float | None`.
  - `Resources` (frozen dataclass): `dictionary: Dictionary, domain_notes: str | None, dev_stats: dict[str, ColumnStats], grid_description: str, grid_cells: frozenset[tuple[str, str, str, str]]` (key, window name, aggregation, column).
  - `dev_column_stats(dev: pd.DataFrame, columns: Iterable[str]) -> dict[str, ColumnStats]`.
  - `load_resources(dev_parquet: Path, arm: Arm, grid_description: str, grid_cells: frozenset[...]) -> Resources`: reads `dev_parquet` (the only file read anywhere in the package), the dictionary and, for arm D only, the domain notes.
- `tests/proposer/conftest.py`: reuses `make_synthetic_all` and `synthetic_split` from `tests/ablation/conftest.py` (import them); adds `FakeProposerClient(responses: list[str])` returning each canned text in turn with `stop_reason="end_turn"`, model `"fake"`, and a `fake_resources(arm)` helper with a two-family dictionary subset. Because the synthetic data has only a few columns, tests build `Dictionary` objects directly rather than loading the packaged one.

- [ ] **Step 1: Write the tests**

```python
def test_load_resources_opens_only_dev_parquet(synthetic_split, monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(pd, "read_parquet", record_then_call(opened))
    load_resources(synthetic_split / "dev.parquet", "C", "grid", frozenset())
    assert [Path(p).name for p in opened] == ["dev.parquet"]
def test_arm_c_has_no_domain_notes_and_arm_d_has_them(synthetic_split) -> None: ...
def test_dev_column_stats_null_rate_and_distinct() -> None:
    df = pd.DataFrame({"a": [1.0, None, 3.0, 3.0], "b": pd.Series(["x", "y", None, "x"], dtype=pd.StringDtype("pyarrow"))})
    s = dev_column_stats(df, ["a", "b"])
    assert s["a"] == ColumnStats(null_rate=0.25, distinct=2, minimum=1.0, maximum=3.0)
    assert s["b"].distinct == 2 and s["b"].minimum is None
def test_dev_column_stats_skips_columns_absent_from_frame() -> None: ...
def test_raw_proposal_forbids_extra_fields() -> None: ...
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement `types.py`, `resources.py`**

- [ ] **Step 4: Run tests; `make check`**

- [ ] **Step 5: Commit** `git commit -m "Add proposer contracts and dev-only resource loading"`

---

### Task 3: Grid description and cell set for the proposer

**Files:**
- Modify: `src/signal_factory/routine/grid.py`
- Test: `tests/routine/test_grid.py`

**Interfaces:**
- Consumes: `routine_grid()`, grid constants (plan 1 Task 12).
- Produces:
  - `grid_cells() -> frozenset[tuple[str, str, str, str]]`: `(entity_key, window_name, aggregation, column)` for every grid spec, window names `"all" | "24h" | "7d"`.
  - `window_name(window: Window) -> str`: `"all"`, `"<n>h"`, `"<n>d"`.
  - `describe_grid() -> str`: the grid as rules (keys, windows, primitives, column families), under 60 lines, ending with the instruction that proposals must lie outside it: other keys (`card1`, `R_emaildomain`, `id_30`, `id_31`, `addr1`), other windows, a filter, or columns the grid does not aggregate. States that six cells are constant by construction.

- [ ] **Step 1: Write the tests**

```python
def test_grid_cells_match_specs() -> None:
    assert len(grid_cells()) == len(routine_grid()) and ("uid", "24h", "count", "*") in grid_cells()
def test_window_name_round_trip() -> None:
    assert window_name(WindowHours(hours=24)) == "24h" and window_name("all") == "all"
def test_describe_grid_names_every_key_and_window_and_no_feature_ids() -> None:
    text = describe_grid()
    assert all(k in text for k in GRID_KEYS) and "rb__" not in text and len(text.splitlines()) < 60
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run tests; `make check`**

- [ ] **Step 5: Commit** `git commit -m "Describe the row B grid as rules for the proposer"`

---

### Task 4: Prompt assembly

**Files:**
- Create: `src/signal_factory/proposer/prompt.py`, `tests/proposer/test_prompt.py`

**Interfaces:**
- Consumes: `Resources`, `RoundFeedback`, `Arm` (Task 2); `signal_json_schema()` (plan 1).
- Produces:
  - `PromptBundle` (frozen dataclass): `system: str, user: str, prompt_hash: str` (sha256 hex of `system + "\n\n" + user`).
  - `build_prompt(arm: Arm, resources: Resources, round_index: int, feedback: Sequence[RoundFeedback], proposed_ids: Sequence[str], n: int = PROPOSALS_PER_ROUND) -> PromptBundle`.
  - System prompt sections, in this order (stable across rounds so the cache prefix holds): role and research framing (retrospective, generic, propose signals a fixed learner can use); the JSON schema of `ProposalBatch`; the signal semantics (strict past, ties excluded, NULL key abstains, count 0 on empty history, `!=` filter excludes NULL, `*` only for count/time_since_last, `inputs` must list every used column, ids `^[a-z][A-Za-z0-9_]{2,63}$` and never starting with `rb__`); the dictionary rendered by family with per-column dev-period stats; the grid description; domain notes (arm D only).
  - User prompt: round number, exactly `n` proposals requested, the previous rounds' feedback (per id: recall delta in points and PR-AUC delta on the dev holdout, or the rejection reason), the list of ids already proposed, and the sentence that the dev holdout is the only evaluation the proposer will see.

- [ ] **Step 1: Write the tests**

```python
def test_system_prompt_is_identical_across_rounds(fake_resources) -> None:
    a = build_prompt("C", fake_resources("C"), 1, [], [])
    b = build_prompt("C", fake_resources("C"), 2, [some_feedback], ["x_sig"])
    assert a.system == b.system and a.user != b.user and a.prompt_hash != b.prompt_hash
def test_arm_d_system_prompt_contains_notes_and_arm_c_does_not(fake_resources) -> None: ...
def test_user_prompt_lists_feedback_and_proposed_ids(fake_resources) -> None:
    fb = RoundFeedback(1, (FeedbackItem("amt_mean_card1", "evaluated", DevMetrics(0.4, 0.002, 1000, 40), None),
                           FeedbackItem("bad", "rejected", None, "unknown column Z")))
    p = build_prompt("C", fake_resources("C"), 2, [fb], ["amt_mean_card1", "bad"])
    assert "amt_mean_card1" in p.user and "+0.4" in p.user and "unknown column Z" in p.user
def test_prompt_never_mentions_validation_or_final(fake_resources) -> None:
    p = build_prompt("D", fake_resources("D"), 1, [], [])
    assert "validation" not in (p.system + p.user).lower() and "final" not in (p.system + p.user).lower()
def test_prompt_contains_schema_grid_and_every_dictionary_column(fake_resources) -> None: ...
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement `prompt.py`**

- [ ] **Step 4: Run tests; `make check`**

- [ ] **Step 5: Commit** `git commit -m "Assemble proposer prompts with a cache-stable system section"`

---

### Task 5: Parsing and validating proposals

**Files:**
- Create: `src/signal_factory/proposer/validate.py`, `tests/proposer/test_validate.py`

**Interfaces:**
- Consumes: `RawProposal`, `ProposalBatch`, `Rejection` (Task 2); `SignalSpec` (plan 1); `grid_cells` shape (Task 3, passed in via `Resources`).
- Produces:
  - `RESERVED_ID_PREFIX = "rb__"`.
  - `parse_batch(text: str) -> ProposalBatch | Rejection`: `Rejection(None, "response is not a ProposalBatch: <pydantic message>", None)` on failure.
  - `validate_proposal(raw: RawProposal, resources: Resources, already: Mapping[str, SignalSpec]) -> SignalSpec | Rejection`, applying in order: `SignalSpec.model_validate(raw.model_dump())` (pydantic reasons verbatim); `spec.validate_columns(resources.dictionary.columns)`; filter value type matches the column kind (string value on a numeric column and vice versa rejected); id prefix `rb__` rejected; same `(entity_key, window name, aggregation, column)` with no filter present in `resources.grid_cells` rejected as "duplicate of the row B grid"; identical spec (all fields except id, version, hypothesis) to one in `already` rejected as "duplicate of <id>"; duplicate id rejected.
  - `validate_batch(batch: ProposalBatch, resources, already) -> tuple[list[SignalSpec], list[Rejection]]`, order preserved, `already` extended as it goes.

- [ ] **Step 1: Write the tests**

```python
def test_parse_batch_rejects_non_json_and_wrong_shape() -> None: ...
def test_valid_proposal_becomes_signal_spec(fake_resources) -> None: ...
def test_forbidden_and_unknown_columns_are_rejected_with_reason(fake_resources) -> None:
    r = validate_proposal(raw(column="isFraud"), fake_resources("C"), {})
    assert isinstance(r, Rejection) and "forbidden" in r.reason
def test_string_filter_value_on_numeric_column_is_rejected(fake_resources) -> None: ...
def test_reserved_prefix_is_rejected(fake_resources) -> None: ...
def test_grid_duplicate_is_rejected_but_filtered_variant_is_not(fake_resources) -> None: ...
def test_identical_spec_under_new_id_is_rejected_as_duplicate(fake_resources) -> None: ...
def test_validate_batch_keeps_order_and_counts(fake_resources) -> None: ...
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement `validate.py`**

- [ ] **Step 4: Run tests; `make check`**

- [ ] **Step 5: Commit** `git commit -m "Validate proposals against the contract, the dictionary and the grid"`

---

### Task 6: Append-only proposal log

**Files:**
- Create: `src/signal_factory/proposer/log.py`, `tests/proposer/test_log.py`

**Interfaces:**
- Produces:
  - `ProposalRecord` (pydantic, `extra="forbid"`): `kind: Literal["proposal", "round", "selection"]`, `arm: Arm`, `round: int`, `position: int | None`, `timestamp: str` (ISO-8601 UTC), `prompt_hash: str | None`, `model: str | None`, `effort: str | None`, `max_tokens: int | None`, `stop_reason: str | None`, `input_tokens: int | None`, `output_tokens: int | None`, `cache_read_tokens: int | None`, `raw_response: str | None`, `spec: dict[str, Any] | None`, `rejection: str | None`, `dev_metrics: dict[str, float | int] | None`, `validation_lift_points: float | None`, `validation_rank: int | None`, `selected: bool | None`.
  - `ProposalLog(path: Path)`: `append(record: ProposalRecord) -> None` (opens in `"a"`, one JSON line, flush); `read() -> list[ProposalRecord]`; `proposals(arm: Arm) -> list[ProposalRecord]` (kind `"proposal"` only).
  - `utc_now() -> str`.

- [ ] **Step 1: Write the tests**

```python
def test_append_is_append_only_across_instances(tmp_path) -> None:
    p = tmp_path / "log.jsonl"
    ProposalLog(p).append(rec(position=0)); ProposalLog(p).append(rec(position=1))
    assert [r.position for r in ProposalLog(p).read()] == [0, 1]
def test_record_round_trips_spec_and_metrics(tmp_path) -> None: ...
def test_read_rejects_unknown_fields(tmp_path) -> None: ...
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement `log.py`**

- [ ] **Step 4: Run tests; `make check`**

- [ ] **Step 5: Commit** `git commit -m "Add append-only JSONL proposal log"`

---

### Task 7: Anthropic client behind the `ProposerClient` protocol

**Files:**
- Create: `src/signal_factory/proposer/client.py`, `tests/proposer/test_client.py`
- Modify: `pyproject.toml` (add `anthropic` to `dependencies`; add `anthropic.*` to the mypy override list only if its stubs prove incomplete)

**Interfaces:**
- Consumes: `ProposerClient`, `ProposerResponse`, `ProposalBatch` (Task 2).
- Produces:
  - `DEFAULT_MODEL = "claude-opus-5-5"`, `DEFAULT_EFFORT = "high"`, `DEFAULT_MAX_TOKENS = 16000`.
  - `AnthropicProposerClient(client: anthropic.Anthropic | None = None, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT, max_tokens: int = DEFAULT_MAX_TOKENS)` implementing `propose(system, user) -> ProposerResponse`. Calls `client.messages.parse(model=..., max_tokens=..., system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}], messages=[{"role": "user", "content": user}], output_format=ProposalBatch, output_config={"effort": effort})`. Returns the first text block's text (or `""` when none, e.g. on refusal), `stop_reason`, and `usage.input_tokens / output_tokens / cache_read_input_tokens` (0 when `None`). Never raises on `refusal` or `max_tokens`; lets `anthropic.APIError` propagate.
  - The constructor builds `anthropic.Anthropic()` lazily when `client is None`, so importing the module never needs a key.

- [ ] **Step 1: Write the tests** (a stub object with a `messages.parse(**kwargs)` method that records kwargs and returns a stand-in with `content`, `stop_reason`, `usage`)

```python
def test_propose_sends_cached_system_prompt_and_schema(stub) -> None:
    AnthropicProposerClient(client=stub).propose("SYS", "USER")
    kw = stub.calls[0]
    assert kw["model"] == DEFAULT_MODEL and kw["output_config"] == {"effort": "high"}
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"} and kw["output_format"] is ProposalBatch
def test_propose_returns_text_stop_reason_and_usage(stub) -> None: ...
def test_refusal_returns_empty_text_without_raising(stub_refusal) -> None: ...
def test_importing_client_module_needs_no_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import signal_factory.proposer.client  # noqa: F401
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: `uv add anthropic`; implement `client.py`**

- [ ] **Step 4: Run tests; `make check`**

- [ ] **Step 5: Commit** `git commit -m "Add Anthropic proposer client with structured output and cached system prompt"`

---

### Task 8: Dev-only evaluator (ablation side)

**Files:**
- Create: `src/signal_factory/ablation/dev_eval.py`, `tests/ablation/test_dev_eval.py`

**Interfaces:**
- Consumes: `compute_signals` (plan 1 Task 7, with a `pd.DataFrame` as history); `train_predict_single`, `load_params` (Task 8 of plan 1); `recall_at_fpr`, `recall_at_threshold`, `pr_auc`, `FPR`; `base_features`, `labels`; `_load_features` pattern from `row_a.py`; `DevMetrics`, `DevEvaluator` (Task 2).
- Produces:
  - `DEV_TRAIN_END_DAY = 84` (holdout = dev rows with `day > 84`, i.e. 85–98; identical to `scripts/freeze_params.py`).
  - `make_dev_evaluator(dev_parquet: Path, row_b_parquet: Path, params: dict[str, Any] | None = None, seed: int = 0) -> DevEvaluator`: reads `dev.parquet` and the dev rows of `row_b.parquet` once; fits the row B baseline once (train 1–84, score 85–98, threshold at FPR 0.01 on the holdout); each call computes the one signal with `compute_signals(dev_df, [spec])` (dev rows are their own history: every earlier row of a dev row is a dev row), appends the column, refits with the same seed, and returns `DevMetrics(recall_delta_points=100 * (recall_signal - recall_baseline), pr_auc_delta=..., holdout_rows, holdout_positives)` where each recall is at its own FPR-0.01 threshold on the holdout.
  - The returned callable holds no reference to any path other than the two it was given (tests check `__closure__` contents by name).

- [ ] **Step 1: Write the tests**

```python
def test_evaluator_reads_only_dev_and_row_b_files(synthetic_split, tmp_path, monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(pd, "read_parquet", record_then_call(opened))
    make_dev_evaluator(synthetic_split / "dev.parquet", row_b_small(tmp_path, synthetic_split), params=FAST)
    assert {Path(p).name for p in opened} == {"dev.parquet", "row_b_small.parquet"}
def test_baseline_fit_once_and_one_fit_per_signal(synthetic_split, tmp_path, monkeypatch) -> None:
    calls = count_calls(monkeypatch, dev_eval, "train_predict_single")
    ev = make_dev_evaluator(...); ev(spec1); ev(spec2)
    assert calls() == 3
def test_metrics_are_finite_and_holdout_is_days_85_to_98(synthetic_split, tmp_path) -> None: ...
def test_signal_values_use_only_dev_rows(synthetic_split, tmp_path, monkeypatch) -> None:
    # compute_signals must be called with a frame whose max day is <= 98
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement `dev_eval.py`**

- [ ] **Step 4: Run tests; `make check`**

- [ ] **Step 5: Commit** `git commit -m "Add dev-only evaluator for proposer feedback"`

---

### Task 9: Budget loop

**Files:**
- Create: `src/signal_factory/proposer/budget.py`, `tests/proposer/test_budget.py`

**Interfaces:**
- Consumes: Tasks 2, 4, 5, 6, 7 (through the protocol), 8 (through `DevEvaluator`).
- Produces:
  - `ArmResult` (frozen dataclass): `arm: Arm, valid: tuple[SignalSpec, ...], rejected: tuple[Rejection, ...], dev_metrics: dict[str, DevMetrics], rounds_completed: int, input_tokens: int, output_tokens: int`.
  - `run_arm(arm: Arm, resources: Resources, client: ProposerClient, evaluate: DevEvaluator, log: ProposalLog, rounds: int = ROUNDS, per_round: int = PROPOSALS_PER_ROUND) -> ArmResult`. Per round: build prompt → `client.propose` → one `kind="round"` record (prompt hash, model, effort, max_tokens, stop reason, tokens, raw response) → if `stop_reason` is `"refusal"` or the batch does not parse, the round contributes zero proposals and `per_round` positions are logged as rejections with that reason → else `validate_batch`; each valid spec is evaluated and logged with `dev_metrics`; each rejection logged with `rejection`; more than `per_round` proposals: the surplus is logged as rejected "over budget"; fewer: the missing positions are logged as rejected "not proposed". Feedback for the next round is built from this round's records. Validation is never touched here.

- [ ] **Step 1: Write the tests** (FakeProposerClient; a `fake_evaluator` returning deterministic `DevMetrics` keyed by id)

```python
def test_three_rounds_log_thirty_positions_and_feedback_flows(synthetic_resources, tmp_path) -> None:
    client = FakeProposerClient([batch_json(10, prefix="r1"), batch_json(10, prefix="r2"), batch_json(10, prefix="r3")])
    res = run_arm("C", synthetic_resources, client, fake_evaluator, ProposalLog(tmp_path / "c.jsonl"))
    assert res.rounds_completed == 3 and len(res.valid) + len(res.rejected) == 30
    assert "r1_sig0" in client.calls[1].user and "r1_sig0" not in client.calls[0].user
def test_refusal_round_consumes_budget_and_next_round_runs(...) -> None:
    client = FakeProposerClient([refusal(), batch_json(10), batch_json(10)])
    res = run_arm(...); assert res.rounds_completed == 3 and sum(r.reason == "refusal" for r in res.rejected) == 10
def test_non_json_response_is_logged_not_raised(...) -> None: ...
def test_surplus_and_missing_positions_are_logged(...) -> None: ...
def test_no_validation_record_is_written(...) -> None:
    assert all(r.validation_lift_points is None and r.kind != "selection" for r in log.read())
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement `budget.py`**

- [ ] **Step 4: Run tests; `make check`**

- [ ] **Step 5: Commit** `git commit -m "Add the three-round proposer budget loop with dev-only feedback"`

---

### Task 10: Data-boundary tests for the proposer package

**Files:**
- Create: `tests/proposer/test_boundary.py`
- Modify (only if a test fails): files under `src/signal_factory/proposer/`

**Interfaces:**
- Consumes: the whole `signal_factory.proposer` package; `run_arm` (Task 9); `make_dev_evaluator` (Task 8).

- [ ] **Step 1: Write the tests**

```python
ALLOWED_PROJECT_IMPORTS = {"signal_factory.signals.spec", "signal_factory.signals.columns", "signal_factory.knowledge", "signal_factory.knowledge.loader"}
def test_proposer_imports_only_the_contract_and_knowledge() -> None:
    for path in Path(proposer.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = imported_module_names(node)  # helper over Import / ImportFrom
            for n in names:
                if n.startswith("signal_factory") and not n.startswith("signal_factory.proposer"):
                    assert n in ALLOWED_PROJECT_IMPORTS, f"{path.name} imports {n}"
def test_proposer_source_never_names_other_period_files() -> None:
    src = "".join(p.read_text() for p in Path(proposer.__file__).parent.glob("*.py"))
    assert "validation.parquet" not in src and "final.parquet" not in src and "history.parquet" not in src
def test_full_arm_run_opens_only_dev_parquet(synthetic_split, tmp_path, monkeypatch) -> None:
    opened: list[str] = []
    evaluate = make_dev_evaluator(synthetic_split / "dev.parquet", row_b_small(...), params=FAST)
    monkeypatch.setattr(pd, "read_parquet", record_then_call(opened))   # after the evaluator is built
    resources = load_resources(synthetic_split / "dev.parquet", "C", describe_grid(), grid_cells())
    run_arm("C", resources, FakeProposerClient([...]), evaluate, ProposalLog(tmp_path / "c.jsonl"), rounds=1, per_round=2)
    assert {Path(p).name for p in opened} == {"dev.parquet"}
def test_dev_evaluator_closure_holds_no_other_paths(synthetic_split, tmp_path) -> None: ...
```

- [ ] **Step 2: Run them.** Expected: PASS if Tasks 2–9 kept the boundary; any failure is a boundary breach to fix in the offending module, with the test left as the gate.

- [ ] **Step 3: `make check`**

- [ ] **Step 4: Commit** `git commit -m "Pin the proposer data boundary with import and file-access tests"`

---

### Task 11: Selection on validation

**Files:**
- Create: `src/signal_factory/ablation/select.py`, `tests/ablation/test_select.py`

**Interfaces:**
- Consumes: `compute_signals` (Path history), `run_row`-style feature loading (`_load_features`, `_features` from `row_a.py`, made importable without underscore: rename to `load_feature_files`, `assemble_features` and keep the old names as aliases), `train_predict_single`, `recall_at_fpr`, `seen_mask`; `ArmResult` (Task 9); `ProposalLog` (Task 6); `SELECTION_CAP`.
- Produces:
  - `candidate_column(arm: Arm, signal_id: str) -> str` = `f"{arm.lower()}__{signal_id}"`.
  - `compute_candidates(arm: Arm, specs: Sequence[SignalSpec], history: Path, out_path: Path) -> int`: `compute_signals` over the full history (float32), columns renamed with `candidate_column`, written with `TransactionID`; returns the column count. Raises `ValueError` if `out_path` exists with a different column set (stale file; Review Focus 3).
  - `SelectionResult` (frozen dataclass): `arm: Arm, baseline_recall: float, ranked: tuple[tuple[str, float], ...]` (id, lift in points, descending), `selected: tuple[SignalSpec, ...]`, `selected_parquet: Path | None` (None when zero signals), `fits: int`.
  - `select_signals(arm: Arm, specs: Sequence[SignalSpec], split_dir: Path, row_b_parquet: Path, candidates_parquet: Path, out_dir: Path, log: ProposalLog | None = None, params=None, seed: int = 0, cap: int = SELECTION_CAP) -> SelectionResult`: trains row B once (dev → validation, seed 0, threshold at FPR 0.01 on all validation rows), then row B + one candidate column per spec, lift = 100 × (recall_signal − recall_B) each at its own FPR-0.01 threshold on all validation rows; drops lift ≤ 0; ranks; caps; writes `out_dir/selected_<arm>.parquet` (TransactionID + selected columns, every history row) and `out_dir/selection_<arm>.json`; appends one `kind="selection"` record per evaluated spec to `log` with `validation_lift_points`, `validation_rank`, `selected`. Never opens `final.parquet`.

- [ ] **Step 1: Write the tests** (synthetic split, FAST params, two or three hand-made specs over `card1` and `addr1` keys; a `fake_lifts` monkeypatch of the inner fit to force known lifts for the ranking tests)

```python
def test_compute_candidates_namespaces_columns_and_covers_every_history_row(...) -> None: ...
def test_compute_candidates_refuses_stale_file_with_other_columns(...) -> None: ...
def test_selection_drops_non_positive_ranks_and_caps(...) -> None:
    # forced lifts {a: 3.0, b: -1.0, c: 0.0, d: 5.0, ...12 ids} -> selected ids ranked d, a, ..., len <= cap, no b or c
def test_zero_signal_arm_is_legal(...) -> None:
    r = select_signals("C", [], ...); assert r.selected == () and r.selected_parquet is None and r.fits == 1
def test_selection_never_opens_final(...) -> None: ...
def test_selection_records_are_appended_after_rounds(...) -> None: ...
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement `select.py`; rename helpers in `row_a.py` (aliases kept)**

- [ ] **Step 4: Run tests; `make check`**

- [ ] **Step 5: Commit** `git commit -m "Select proposed signals by individual validation lift, capped at ten"`

---

### Task 12: Final runner, touched once

**Files:**
- Create: `src/signal_factory/ablation/run.py`, `tests/ablation/test_run.py`

**Interfaces:**
- Consumes: `train_predict_single`, `SEEDS`, `load_params`, `FIXED_PARAMS`; `recall_at_fpr`, `recall_at_threshold`, `pr_auc`, `roc_auc`, `seen_mask`; `paired_recall_diff`, `N_RESAMPLES`; feature helpers from `row_a.py`; `SelectionResult` JSON (Task 11); split constants.
- Produces:
  - `RowSpec` (frozen dataclass): `name: Literal["A", "B", "C", "D"], feature_parquets: tuple[Path, ...]`.
  - `FinalRowResult` (frozen dataclass): `name, threshold, validation: Metrics, final: Metrics, final_seen: Metrics, final_unseen: Metrics, recall_seed_std: float, feature_count: int, alias_of: str | None` (`"B"` for a zero-signal arm).
  - `Lift` (frozen dataclass): `arm, points_overall: float, points_unseen: float, bootstrap_overall: BootstrapResult, bootstrap_unseen: BootstrapResult, accepted: bool`.
  - `accepted(points_overall, points_unseen, bootstrap_overall, bootstrap_unseen, minimum_points: float = 2.0) -> bool` = both point lifts ≥ 2.0 and both `lo > 0`.
  - `config_hash(params, seeds, rows: Sequence[RowSpec], selections: Mapping[str, Sequence[str]]) -> str` (sha256 over a canonical JSON; the selections map arm → selected signal ids, so a different proposer run is a different configuration).
  - `AblationResult` (frozen dataclass): `config_hash, params, fixed_params, seeds, rows: dict[str, FinalRowResult], lifts: dict[str, Lift], selections: dict[str, list[dict[str, Any]]]` (the selected specs as dicts), `n_resamples, timestamp`.
  - `run_final(split_dir: Path, rows: Sequence[RowSpec], selections: Mapping[str, SelectionResult], out_dir: Path, params=None, seeds=SEEDS, n_resamples=N_RESAMPLES) -> AblationResult`. Order: compute the hash; refuse if `out_dir / f"{hash}.done"` exists (`RuntimeError` naming the marker); read dev and validation; for each row train one model per seed on dev (storing per-seed validation scores), average, freeze the threshold on all validation rows; **then** read `final.parquet` once; for each row score final per seed, average, report recall and actual FPR at the frozen threshold overall / seen (uid in dev) / unseen, PR-AUC, ROC-AUC, seed std of final recall; a zero-signal arm reuses row B's scores and threshold with `alias_of="B"`; paired bootstrap of C−B and D−B on final overall and on the unseen subset; write `scores_<row>.parquet` (TransactionID, period, per-seed and mean scores), `ablation.json`, and only then the marker. Marker written last so a crash leaves none (Review Focus 4).

- [ ] **Step 1: Write the tests** (synthetic split, FAST params, seeds `(0, 1)`, 50 resamples; a tiny `selected_C.parquet` with one informative column and arm D empty)

```python
def test_final_parquet_is_read_exactly_once_and_after_thresholds(...) -> None:
    opened: list[str] = []  # record_then_call; assert opened.count(final) == 1 and final appears after validation
def test_second_run_with_same_config_is_refused_and_different_selection_is_not(...) -> None: ...
def test_crash_before_results_leaves_no_marker(..., monkeypatch) -> None:
    # monkeypatch paired_recall_diff to raise; assert no *.done file
def test_zero_signal_arm_aliases_row_b(...) -> None:
    assert res.rows["D"].alias_of == "B" and res.rows["D"].threshold == res.rows["B"].threshold
def test_accepted_requires_both_subsets_and_both_intervals() -> None:
    br = BootstrapResult(0.03, 0.01, 0.05, 1000); bad = BootstrapResult(0.03, -0.01, 0.05, 1000)
    assert accepted(3.0, 2.5, br, br) and not accepted(3.0, 1.9, br, br) and not accepted(3.0, 2.5, br, bad)
def test_seed_std_and_metrics_are_finite(...) -> None: ...
def test_ablation_json_round_trips(...) -> None: ...
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement `run.py`**

- [ ] **Step 4: Run tests; `make check`**

- [ ] **Step 5: Commit** `git commit -m "Add the once-only final evaluation with paired bootstrap and acceptance"`

---

### Task 13: Report

**Files:**
- Create: `src/signal_factory/ablation/report.py`, `tests/ablation/test_report.py`

**Interfaces:**
- Consumes: `AblationResult` (Task 12); `ProposalLog` (Task 6); `docs/decisions/row-b-backend.md` facts as constants passed in (`row_b_backend: str`).
- Produces:
  - `ProposalStats` (frozen dataclass): `arm, proposed: int, valid: int, rejected: int, rejection_reasons: dict[str, int]` (reason prefix → count), `selected: int, input_tokens: int, output_tokens: int, cache_read_tokens: int, model: str`.
  - `proposal_stats(log: ProposalLog, arm: Arm) -> ProposalStats`.
  - `render_report(result: AblationResult, stats: Mapping[str, ProposalStats], row_b_backend: str = "signal engine") -> str`, Markdown with these sections in order (spec §8.6): first paragraph states that this is a retrospective experiment with fully observed labels and no label latency; dataset and split (counts from `result.rows[...].final.n` etc.); ablation table overall and unseen (recall at the frozen threshold, actual FPR, PR-AUC, ROC-AUC, seed std) for A–D; acceptance decision per arm with point lift and 95% interval, overall and unseen, and the sentence "No supported improvement" when not accepted; proposal statistics per arm; the selected signals with their hypotheses (or "no signal with positive validation lift" for a zero-signal arm); failure analysis section listing which of poor proposals, limited data, or an already strong baseline the evidence supports (rule: poor proposals if fewer than half the valid proposals had positive dev lift; strong baseline if row B's unseen recall is within 1 point of row A's and the selected signals' validation lifts were all under 1 point; limited data if the unseen interval width exceeds 4 points); a note that row B was produced by `row_b_backend` and that the proposer was told what row B computes (spec §8.5 owner decision); the frozen parameters.

- [ ] **Step 1: Write the tests**

```python
def test_first_paragraph_states_retrospective_labels(result_fixture, stats_fixture) -> None:
    text = render_report(result_fixture, stats_fixture)
    assert "retrospective" in text.split("\n\n")[1].lower()
def test_sections_appear_in_order(...) -> None: ...
def test_not_accepted_says_no_supported_improvement(...) -> None: ...
def test_zero_signal_arm_wording(...) -> None: ...
def test_proposal_stats_count_rejections_by_reason(tmp_path) -> None: ...
def test_report_contains_no_row_level_values(...) -> None:
    assert "TransactionID" not in text
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement `report.py`**

- [ ] **Step 4: Run tests; `make check`**

- [ ] **Step 5: Commit** `git commit -m "Render the milestone-1 report from the ablation result and proposal logs"`

---

### Task 14: CLI, real runs, close-out

**Files:**
- Modify: `src/signal_factory/cli.py`, `tests/test_cli.py`, `README.md`, `docs/superpowers/specs/2026-11-02-signal-factory-design.md` (§10: nothing new to measure; §9 steps 5–7 marked done with dates), `src/signal_factory/knowledge/domain_notes.md` (owner confirmation date)
- Create: `docs/results/plan-2-proposals.md`, `docs/results/plan-2-final.md`, `docs/results/plan-2-proposals-C.jsonl`, `docs/results/plan-2-proposals-D.jsonl`, `docs/results/milestone-1-report.md`

**Interfaces:**
- Consumes: Tasks 1–13.
- Produces CLI subcommands (argparse, same style as plan 1):
  - `signal-factory propose ARM --split data/split --row-b data/features/row_b.parquet --log results/proposals/ARM.jsonl [--model ... --effort ... --max-tokens ...]`: `load_resources` → `AnthropicProposerClient` → `make_dev_evaluator` → `run_arm`; prints valid / rejected counts and token totals.
  - `signal-factory select ARM --split data/split --history data/split/history.parquet --row-b ... --log ... --out data/features`: rebuilds the arm's valid specs from the log, `compute_candidates`, `select_signals`; prints the ranking.
  - `signal-factory run-final --split data/split --row-b ... --features data/features --out results/final`: rows A, B, C (`selected_C.parquet` if present), D; prints the ablation table.
  - `signal-factory report --final results/final/ablation.json --logs results/proposals --out docs/results/milestone-1-report.md`.

- [ ] **Step 1: Write the CLI tests** (monkeypatched runners; each subcommand parses its arguments and calls the right function with the right paths; `run-final` refuses to accept a `--final` path argument, i.e. there is no way to point it at another final file)

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement the subcommands**

- [ ] **Step 4: Run tests; `make check`**

- [ ] **Step 5: Commit** `git commit -m "Add propose, select, run-final and report commands"`

- [ ] **Step 6 (owner gate): confirm `domain_notes.md` (Task 1 Step 6) and that `ANTHROPIC_API_KEY` or an `ant auth login` profile is available. Note the expected cost: 6 model calls with a cached system prompt (cents to a few dollars), and about 120 single-seed LightGBM fits for dev feedback and selection at roughly 8 minutes each on the plan 1 laptop, plus about 2 hours for the final run. Run overnight in two sessions.**

- [ ] **Step 7 (owner, real data): `signal-factory propose C`, then `signal-factory propose D`. Commit nothing yet.**

- [ ] **Step 8 (owner, real data): `signal-factory select C`, `signal-factory select D`. Check the two JSONL logs contain only specs, hypotheses, aggregate metrics and token counts; copy them to `docs/results/plan-2-proposals-C.jsonl` and `-D.jsonl`. Write `docs/results/plan-2-proposals.md`: counts per arm, rejection reasons, dev feedback summary, the ranked validation lifts, the selected signals. Commit** `git commit -m "Record proposer rounds and validation selection for arms C and D"`

- [ ] **Step 9 (owner, real data, once): `signal-factory run-final`, then `signal-factory report`. Write `docs/results/plan-2-final.md` (the ablation table, intervals, acceptance decision) and commit the generated `docs/results/milestone-1-report.md`. Update README "Status" with the headline numbers and the acceptance decision; mark spec §9 steps 5–7 done. Commit** `git commit -m "Close plan 2: final evaluation and milestone-1 report"`

---

## Self-review notes

- Spec coverage: §8.5 prompt (Task 4), client (Task 7), budget (Task 9), selection (Task 11), log (Task 6), boundary (Tasks 2, 8, 10), domain notes (Task 1); §8.6 runner, marker, report (Tasks 12, 13); §6 acceptance and uncertainty (Task 12); §5 seen/unseen on final (Task 12); §7 rows C and D, row E omitted as optional; §9 step 7 write-up (Task 14). Row E is not planned; if the owner wants it, it is one more `RowSpec` with an owner-written spec list through `compute_candidates`.
- Known cost: the dev feedback fits are spec-literal (frozen params, full row B features). Reducing their cost would mean a cheaper learner for feedback only, which is a spec change; not made here.
- Type consistency checked: `DevMetrics`, `DevEvaluator`, `ProposerResponse`, `Rejection`, `ArmResult`, `SelectionResult`, `RowSpec`, `AblationResult` are defined once each (Tasks 2, 9, 11, 12) and consumed by name.
