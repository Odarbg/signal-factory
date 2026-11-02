# signal-factory Plan 1: Foundations (data, signal engine, baseline, row B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Everything in milestone 1 that involves no LLM: a leakage-tested point-in-time signal engine, the chronological split, the fixed LightGBM baseline with pre-registered metrics, and the routine-aggregation row B.

**Architecture:** One `uv` project, `src/signal_factory/`. Data flows CSV → Parquet (with `uid`, `day`, `card1_addr1`) → period files. Signals are pydantic specs compiled to DuckDB window-function SQL and evaluated against `history.parquet`. The learner, metrics and bootstrap are plain functions over NumPy arrays. Row B is the fixed grid from the design, expressed as signal specs and computed by the same engine (featuretools is a time-boxed attempt whose outcome is recorded).

**Tech Stack:** Python 3.12, uv, pandas, pyarrow, DuckDB, LightGBM, scikit-learn, pydantic v2, pytest, Hypothesis, mypy (strict), ruff. featuretools only in Task 12.

**Spec:** `docs/superpowers/specs/2026-11-02-signal-factory-design.md`

## Global Constraints

- Python 3.12; dependencies managed with `uv`; `uv run` for every command.
- `mypy --strict` and `ruff check` clean on `src/` and `tests/` before every commit.
- `data/` and `results/` are gitignored; no dataset file or derived row-level file is ever committed.
- `isFraud` is never an input: absent from `history.parquet`, rejected as a signal `column` or `filter` column (spec §4).
- A signal value at a row uses only history rows with `TransactionDT` strictly less than the row's own (ties excluded) (spec §8.2).
- Periods, in days: dev training 1–98; gap 99–105; validation 106–138; gap 139–145; final 146–183 (spec §5).
- Pre-registered values: 5 seeds `(0, 1, 2, 3, 4)`; FPR = 0.01; 1,000 bootstrap resamples; acceptance +2.0 points recall (spec §6). This plan never touches `final.parquet` beyond writing it.
- No company data, numbers or identifiers anywhere.
- Collaboration: the owner chooses per task whether they write the tests and Claude implements, the reverse, or review-only. Ask before each task.

## Review Focus

Inputs the spec implies but no task's tests would otherwise exercise, most likely to bite first. Each has its test pinned in the owning task.

1. A row whose entity key is NULL (e.g. missing `addr1` → NULL `uid`) must yield NULL for every signal, never join with other NULL-key rows as one giant group. → Task 6.
2. Two rows with identical `TransactionDT` and the same key must not see each other (ties excluded), or the leakage test passes on strict-time data but production data leaks. → Task 6.
3. The 7-day gap rows must be present in `history.parquet` (feature history) but absent from `dev.parquet` training rows and from `validation.parquet`/`final.parquet` evaluation rows. → Task 4.
4. A validation row whose `uid` is NULL must be classified "unseen", not crash the seen/unseen split and not be dropped from overall metrics. → Task 9.
5. `recall_at_fpr` on a score vector with many ties at the threshold must return a threshold whose actual FPR is ≤ 0.01 (choose the conservative side), and `recall_at_threshold` must report the realised FPR, not assume 0.01. → Task 9.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/signal_factory/__init__.py`, `src/signal_factory/py.typed`, `tests/__init__.py`, `tests/test_smoke.py`, `Makefile`
- Modify: `README.md` (add "Development" section)

**Interfaces:**
- Produces: the package `signal_factory`; commands `make test`, `make lint`, `make typecheck`.

- [ ] **Step 1: Write `pyproject.toml`**

`[project]` name `signal-factory`, `requires-python = ">=3.12"`, dependencies: `pandas`, `pyarrow`, `duckdb`, `lightgbm`, `scikit-learn`, `pydantic>=2`, `numpy`, `pyyaml`. Dev group: `pytest`, `hypothesis`, `mypy`, `ruff`, `pandas-stubs`, `types-PyYAML`. Build backend `hatchling`, `[tool.hatch.build.targets.wheel] packages = ["src/signal_factory"]`. `[tool.mypy] strict = true`, `[[tool.mypy.overrides]] module = ["lightgbm.*", "duckdb.*", "sklearn.*", "featuretools.*"] ignore_missing_imports = true`. `[tool.ruff] line-length = 100`, `target-version = "py312"`. `[tool.pytest.ini_options] testpaths = ["tests"]`.

- [ ] **Step 2: Write the smoke test**

```python
# tests/test_smoke.py
import signal_factory

def test_package_imports() -> None:
    assert signal_factory.__version__ == "0.1.0"
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv sync --all-groups && uv run pytest tests/test_smoke.py -v`
Expected: FAIL, `AttributeError: __version__`

- [ ] **Step 4: Add `__version__ = "0.1.0"` to `src/signal_factory/__init__.py`; create empty `py.typed`**

- [ ] **Step 5: Write `Makefile`** with targets `test` (`uv run pytest`), `lint` (`uv run ruff check src tests`), `typecheck` (`uv run mypy src tests`), `check` (all three).

- [ ] **Step 6: Run `make check`**

Expected: 1 passed; ruff and mypy report no issues.

- [ ] **Step 7: Add a "Development" section to `README.md`** listing `uv sync --all-groups`, `make check`.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src tests Makefile README.md
git commit -m "Add Python project scaffold with uv, pytest, mypy strict and ruff"
```

---

### Task 2: Dataset download script

**Files:**
- Create: `src/signal_factory/data/__init__.py`, `src/signal_factory/data/download.py`, `tests/data/__init__.py`, `tests/data/test_download.py`

**Interfaces:**
- Produces: `download(dest: Path, runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = subprocess.run) -> None`. Writes `train_transaction.csv` and `train_identity.csv` into `dest`. Raises `RulesNotAcceptedError(RuntimeError)` with a message containing the URL `https://www.kaggle.com/competitions/ieee-fraud-detection/rules` when the Kaggle CLI returns 403.

- [ ] **Step 1: Write the failing tests**

```python
def test_download_invokes_kaggle_cli_and_unzips(tmp_path) -> None:
    calls: list[list[str]] = []
    def fake_run(cmd): calls.append(cmd); (tmp_path / "ieee-fraud-detection.zip").write_bytes(make_zip({"train_transaction.csv": b"a", "train_identity.csv": b"b"})); return CompletedProcess(cmd, 0, "", "")
    download(tmp_path, runner=fake_run)
    assert calls[0][:3] == ["kaggle", "competitions", "download"]
    assert (tmp_path / "train_transaction.csv").read_bytes() == b"a"

def test_download_explains_rules_on_403(tmp_path) -> None:
    def fake_run(cmd): return CompletedProcess(cmd, 1, "", "403 - Forbidden")
    with pytest.raises(RulesNotAcceptedError, match="competitions/ieee-fraud-detection/rules"):
        download(tmp_path, runner=fake_run)
```

`make_zip` is a 5-line helper in the test module using `zipfile`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_download.py -v`
Expected: FAIL, `ImportError`

- [ ] **Step 3: Implement `download` in `download.py`**

Command: `["kaggle", "competitions", "download", "-c", "ieee-fraud-detection", "-p", str(dest)]`. Unzip with `zipfile`, keep only the two train files, delete the zip and any test files.

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit** `git commit -m "Add Kaggle download script with rules-not-accepted guidance"`

---

### Task 3: CSV to Parquet with `uid`, `day`, `card1_addr1`

**Files:**
- Create: `src/signal_factory/data/convert.py`, `tests/data/test_convert.py`, `tests/data/fixtures/mini_transaction.csv`, `tests/data/fixtures/mini_identity.csv`

**Interfaces:**
- Consumes: raw CSVs from Task 2.
- Produces: `convert(raw_dir: Path, out_path: Path) -> None` writing one Parquet file with all original columns (identity left-joined on `TransactionID`), floats as float32, object columns as string, plus `day: int32`, `uid: string | null`, `card1_addr1: string | null`. Constants `SECONDS_PER_DAY = 86400`. Helper `add_derived_columns(df: pd.DataFrame) -> pd.DataFrame` (pure, tested directly).

Definitions (spec §4, §5): `day = TransactionDT // 86400 - (min(TransactionDT) // 86400) + 1`; `D1n = day - D1`; `uid = f"{card1}_{addr1}_{D1n}"` and `card1_addr1 = f"{card1}_{addr1}"`, each **NULL if any component is NULL** (decision: a NULL key abstains; it must not form a shared group).

- [ ] **Step 1: Create the fixtures**

`mini_transaction.csv`: 8 rows, columns `TransactionID,isFraud,TransactionDT,TransactionAmt,ProductCD,card1,card2,addr1,D1,P_emaildomain`; include one row with empty `addr1`, one with empty `D1`, two rows sharing `TransactionDT`, and `TransactionDT` values 86400, 86460, 172800, …. `mini_identity.csv`: 3 rows, `TransactionID,DeviceInfo,id_30`, matching 3 of the 8 ids.

- [ ] **Step 2: Write the failing tests**

```python
def test_day_starts_at_one_and_uid_components() -> None:
    df = pd.read_csv(FIXTURES / "mini_transaction.csv")
    out = add_derived_columns(df)
    assert out["day"].min() == 1
    row = out.loc[out.TransactionID == 1].iloc[0]
    assert row["uid"] == f"{int(row.card1)}_{int(row.addr1)}_{int(row.day - row.D1)}"

def test_null_component_gives_null_uid() -> None:
    out = add_derived_columns(pd.read_csv(FIXTURES / "mini_transaction.csv"))
    assert out.loc[out.addr1.isna(), "uid"].isna().all()
    assert out.loc[out.D1.isna(), "uid"].isna().all()
    assert out.loc[out.addr1.isna(), "card1_addr1"].isna().all()

def test_convert_joins_identity_and_downcasts(tmp_path) -> None:
    convert(FIXTURES, tmp_path / "all.parquet")
    df = pd.read_parquet(tmp_path / "all.parquet")
    assert len(df) == 8
    assert df["DeviceInfo"].notna().sum() == 3
    assert df["TransactionAmt"].dtype == "float32"
    assert {"day", "uid", "card1_addr1"} <= set(df.columns)
```

- [ ] **Step 3: Run tests to verify they fail**

- [ ] **Step 4: Implement `add_derived_columns` and `convert`**

Read with `pd.read_csv`; left-merge identity on `TransactionID`; downcast float64 → float32; cast object → `string[pyarrow]`; write with `df.to_parquet(out_path, index=False)`.

- [ ] **Step 5: Run tests to verify they pass; run `make check`**

- [ ] **Step 6: Commit** `git commit -m "Add CSV to Parquet conversion with uid, day and card1_addr1"`

---

### Task 4: Chronological split with gaps

**Files:**
- Create: `src/signal_factory/data/split.py`, `tests/data/test_split.py`

**Interfaces:**
- Consumes: the Parquet from Task 3.
- Produces:
  - `Period = Literal["dev", "gap", "validation", "final"]`
  - `assign_period(day: int) -> Period` with boundaries `DEV_END = 98`, `VALIDATION_START = 106`, `VALIDATION_END = 138`, `FINAL_START = 146`, `FINAL_END = 183`; days 99–105 and 139–145 → `"gap"`; day > 183 → `ValueError`.
  - `write_split(all_parquet: Path, out_dir: Path) -> SplitCounts` writing `dev.parquet`, `validation.parquet`, `final.parquet` (period rows only, all columns including `isFraud`) and `history.parquet` (**all** rows, `isFraud` column dropped). `SplitCounts` is a dataclass with `dev`, `gap`, `validation`, `final`, `history: int`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("day,period", [(1,"dev"),(98,"dev"),(99,"gap"),(105,"gap"),(106,"validation"),(138,"validation"),(139,"gap"),(145,"gap"),(146,"final"),(183,"final")])
def test_assign_period_boundaries(day, period) -> None: ...

def test_assign_period_rejects_day_after_183() -> None: ...

def test_write_split_files_are_disjoint_and_history_has_no_label(tmp_path) -> None:
    # synthetic parquet: one row per day 1..183 plus duplicates
    counts = write_split(src, tmp_path)
    dev, val, fin, hist = (pd.read_parquet(tmp_path / f) for f in ("dev.parquet","validation.parquet","final.parquet","history.parquet"))
    assert set(dev.day) <= set(range(1, 99)) and set(val.day) <= set(range(106, 139)) and set(fin.day) <= set(range(146, 184))
    assert not (set(dev.TransactionID) & set(val.TransactionID)) and not (set(val.TransactionID) & set(fin.TransactionID))
    assert "isFraud" not in hist.columns and len(hist) == counts.history == counts.dev + counts.gap + counts.validation + counts.final
    assert set(hist.day) >= {99, 105, 139, 145}   # Review Focus 3: gap rows are history
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement `split.py`** (pandas filter per period; `history` = all rows with `isFraud` dropped).

- [ ] **Step 4: Run tests; `make check`**

- [ ] **Step 5: Commit** `git commit -m "Add chronological split with 7-day gaps and label-free history"`

---

### Task 5: Signal spec and JSON schema

**Files:**
- Create: `src/signal_factory/signals/__init__.py`, `src/signal_factory/signals/columns.py`, `src/signal_factory/signals/spec.py`, `tests/signals/__init__.py`, `tests/signals/test_spec.py`

**Interfaces:**
- Produces, in `columns.py`: `ENTITY_KEYS: frozenset[str] = {"uid", "card1", "card1_addr1", "P_emaildomain", "R_emaildomain", "DeviceInfo", "id_30", "id_31", "addr1"}`; `FORBIDDEN_COLUMNS: frozenset[str] = {"isFraud", "TransactionID", "TransactionDT", "uid", "day", "card1_addr1"}`.
- Produces, in `spec.py` (pydantic v2, `model_config = ConfigDict(extra="forbid")`):
  - `Aggregation = Literal["count", "count_distinct", "sum", "mean", "max", "min", "time_since_last"]`
  - `Window`: `Literal["all"] | WindowHours(hours: int > 0) | WindowDays(days: int > 0)` as a discriminated union; method `seconds() -> int | None` (`None` for all).
  - `Filter(column: str, op: Literal["==", "!=", "<", "<=", ">", ">="], value: float | str)`
  - `SignalInput(column: str, available_at: Literal["decision_time"])`
  - `SignalSpec(id: str (pattern `^[a-z][a-z0-9_]{2,63}$`), version: int ≥ 1, hypothesis: str (10–400 chars), entity_key: str, column: str, aggregation: Aggregation, window: Window, filter: Filter | None = None, missing: Literal["null"] = "null", inputs: list[SignalInput])`
  - Validators: `entity_key in ENTITY_KEYS`; `column == "*"` iff `aggregation == "count"`; `column` and `filter.column` not in `FORBIDDEN_COLUMNS`; `aggregation == "time_since_last"` requires `column == "*"`; `inputs` must list `column` (if not `*`) and `filter.column` (if any) and `entity_key`.
  - `SignalSpec.validate_columns(known: Iterable[str]) -> None` raising `ValueError` naming any unknown column.
  - `signal_json_schema() -> dict[str, Any]` = `SignalSpec.model_json_schema()`.

- [ ] **Step 1: Write the failing tests**

```python
def test_valid_count_spec_parses() -> None: SignalSpec.model_validate(VALID_COUNT)
def test_label_is_never_a_column() -> None:
    with pytest.raises(ValidationError, match="isFraud"): SignalSpec.model_validate({**VALID_SUM, "column": "isFraud"})
def test_label_is_never_a_filter_column() -> None: ...
def test_unknown_entity_key_rejected() -> None: ...
def test_count_requires_star_and_star_requires_count() -> None: ...
def test_inputs_must_cover_used_columns() -> None: ...
def test_validate_columns_names_unknown() -> None:
    with pytest.raises(ValueError, match="nope"): SignalSpec.model_validate({**VALID_SUM, "column": "nope", ...}).validate_columns(["TransactionAmt"])
def test_json_schema_has_required_fields() -> None:
    assert set(signal_json_schema()["required"]) >= {"id","version","hypothesis","entity_key","column","aggregation","window","inputs"}
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement `columns.py` and `spec.py`**

- [ ] **Step 4: Run tests; `make check`**

- [ ] **Step 5: Commit** `git commit -m "Add depth-1 signal spec with validation and JSON schema"`

---

### Task 6: Compile a signal spec to DuckDB SQL

**Files:**
- Create: `src/signal_factory/signals/compile.py`, `tests/signals/test_compile.py`, `tests/signals/conftest.py` (DuckDB connection with a small `history` table built from a pandas fixture)

**Interfaces:**
- Consumes: `SignalSpec` (Task 5).
- Produces: `compile_signal(spec: SignalSpec) -> str` returning a single SQL **select-list expression** (no `SELECT`/`FROM`) that evaluates to the signal's value for the current row of a table aliased `h`, using a window `PARTITION BY h.<entity_key> ORDER BY h.TransactionDT RANGE BETWEEN <n> PRECEDING AND 1 PRECEDING` (`UNBOUNDED PRECEDING` for `all`). Also `build_query(specs: Sequence[SignalSpec], table: str = "history") -> str` producing `SELECT h.TransactionID, <expr_1> AS <id_1>, … FROM <table> h`.

Semantics to encode (spec §8.2 and decisions):
- Ties excluded: `1 PRECEDING` on an integer `TransactionDT` excludes same-timestamp rows.
- Window `[t − n, t)`: `RANGE BETWEEN n PRECEDING AND 1 PRECEDING`.
- Filter: aggregate over `CASE WHEN <filter> THEN <column> END` (for `count`: `CASE WHEN <filter> THEN 1 END`).
- `count` → `count(...)`, which yields `0` on empty history (decision: a count of zero prior rows is information; the spec's "abstain on empty history" applies to the other aggregations, which yield NULL naturally). Record this in the spec's §8.2 in Step 6.
- `time_since_last` → `h.TransactionDT - max(h.TransactionDT) OVER (...)`.
- NULL key → `CASE WHEN h.<entity_key> IS NULL THEN NULL ELSE <window expr> END` (Review Focus 1).
- String literals in filters are quoted with single quotes, doubled inside.

- [ ] **Step 1: Write the fixture in `conftest.py`**

A `history` table of ~12 rows over 2 keys (`uid` values `"a"`, `"b"`, and one NULL), integer `TransactionDT` including a tie, `TransactionAmt`, `ProductCD` in {"W","C"}, `P_emaildomain`. Hand-compute expected values in the test file comments.

- [ ] **Step 2: Write the failing tests**

```python
def test_count_all_history_excludes_self_and_ties(con) -> None: ...   # Review Focus 2
def test_sum_window_24h_is_half_open(con) -> None: ...                # row exactly 86400s earlier is included; row at 86401s is not
def test_count_distinct_with_filter(con) -> None: ...
def test_time_since_last_is_null_when_no_history(con) -> None: ...
def test_count_is_zero_when_no_history(con) -> None: ...
def test_null_key_yields_null_not_group(con) -> None: ...            # Review Focus 1
def test_build_query_names_columns_by_spec_id(con) -> None: ...
def test_compile_is_deterministic() -> None: assert compile_signal(s) == compile_signal(s)
```

Each test runs `con.execute(build_query([spec])).df()` and asserts the exact per-`TransactionID` values.

- [ ] **Step 3: Run to verify they fail**

- [ ] **Step 4: Implement `compile.py`**

- [ ] **Step 5: Run tests; `make check`**

- [ ] **Step 6: Amend spec §8.2** "abstain on empty history" to "NULL on empty history for every aggregation except `count`, which is 0" with one line of reason. Commit the spec change with the code.

- [ ] **Step 7: Commit** `git commit -m "Compile signal specs to DuckDB window SQL with strict-past semantics"`

---

### Task 7: Signal engine and the leakage property test

**Files:**
- Create: `src/signal_factory/signals/engine.py`, `tests/signals/test_engine.py`, `tests/signals/test_leakage.py`

**Interfaces:**
- Consumes: `build_query` (Task 6), `history.parquet` layout (Task 4).
- Produces: `compute_signals(history: Path | pd.DataFrame, specs: Sequence[SignalSpec], batch_size: int = 200) -> pd.DataFrame` with columns `TransactionID` plus one float column per spec id, in spec order; batches specs into several queries of at most `batch_size` expressions and joins on `TransactionID`; `connect(history: Path | pd.DataFrame) -> duckdb.DuckDBPyConnection` registering the table as `history`.

- [ ] **Step 1: Write the engine tests**

```python
def test_compute_signals_from_dataframe_matches_single_query(tiny_history) -> None: ...
def test_compute_signals_batches_and_preserves_order(tiny_history) -> None:
    out = compute_signals(tiny_history, specs_x5, batch_size=2)
    assert list(out.columns) == ["TransactionID", *[s.id for s in specs_x5]]
def test_compute_signals_from_parquet_path(tmp_path, tiny_history) -> None: ...
```

- [ ] **Step 2: Write the leakage property test (the acceptance gate)**

```python
@settings(max_examples=200, deadline=None)
@given(history=histories(), later=later_rows(), spec=signal_specs())
def test_appending_later_rows_never_changes_earlier_values(history, later, spec) -> None:
    before = compute_signals(history, [spec]).set_index("TransactionID")[spec.id]
    after = compute_signals(pd.concat([history, later]), [spec]).set_index("TransactionID")[spec.id]
    pd.testing.assert_series_equal(after.loc[before.index], before, check_names=False)
```

Strategies in the same file: `histories()` builds 1–30 rows with `uid` from `{"a","b","c",None}`, `TransactionDT` integers 0–200_000 with duplicates allowed, `TransactionAmt` floats, `ProductCD` from `{"W","C"}`; `later_rows()` builds rows with `TransactionDT` strictly greater than the history's max; `signal_specs()` samples aggregation, window and optional filter over those columns with `entity_key="uid"`.

- [ ] **Step 3: Run to verify they fail**

- [ ] **Step 4: Implement `engine.py`**

- [ ] **Step 5: Run all signal tests; `make check`**

Run: `uv run pytest tests/signals -v`
Expected: all pass including 200 Hypothesis examples.

- [ ] **Step 6: Commit** `git commit -m "Add signal engine with Hypothesis leakage property test"`

---

### Task 8: LightGBM learner with one-off parameter freeze

**Files:**
- Create: `src/signal_factory/baseline/__init__.py`, `src/signal_factory/baseline/learner.py`, `src/signal_factory/baseline/params.json`, `scripts/freeze_params.py`, `tests/baseline/__init__.py`, `tests/baseline/test_learner.py`
- Create: `src/signal_factory/ablation/__init__.py`, `src/signal_factory/ablation/features.py`, `tests/ablation/__init__.py`, `tests/ablation/test_features.py`

**Interfaces:**
- Produces, in `features.py`: `base_features(df: pd.DataFrame) -> pd.DataFrame` dropping `TransactionID, TransactionDT, uid, day, card1_addr1, isFraud` (if present) and casting string columns to `category`; `DROP_COLUMNS` constant. `labels(df) -> np.ndarray` (int8).
- Produces, in `learner.py`: `SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)`; `load_params() -> dict[str, Any]` reading `params.json`; `train_predict(x_train: pd.DataFrame, y_train: np.ndarray, x_eval: pd.DataFrame, params: dict[str, Any] | None = None, seeds: Sequence[int] = SEEDS) -> np.ndarray` returning seed-averaged probabilities; `train_predict_single(..., seed: int)`.
- Produces: `scripts/freeze_params.py`, run once by the owner on dev (train days 1–84, holdout 85–98) over a fixed grid of 6 candidates (`num_leaves ∈ {31, 127}` × `learning_rate ∈ {0.05, 0.02}` with `n_estimators` 500/1000 respectively, plus `feature_fraction 0.5`, `bagging_fraction 0.8`, `min_child_samples 100`); writes the winner by holdout PR-AUC to `params.json` and prints the table. The committed `params.json` is the frozen value (spec §10).

- [ ] **Step 1: Write the tests**

```python
def test_base_features_drops_ids_and_label_and_categorises() -> None: ...
def test_train_predict_returns_probabilities_in_unit_interval(synthetic_xy) -> None: ...
def test_train_predict_is_seed_deterministic(synthetic_xy) -> None:
    assert np.allclose(train_predict_single(..., seed=0), train_predict_single(..., seed=0))
def test_train_predict_averages_seeds(synthetic_xy) -> None: ...   # mean of singles == averaged
def test_load_params_has_required_keys() -> None: assert {"num_leaves","learning_rate","n_estimators"} <= load_params().keys()
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement `features.py`, `learner.py`, a placeholder `params.json` (`num_leaves 31, learning_rate 0.05, n_estimators 500`, plus the fixed fractions) and `scripts/freeze_params.py`**

- [ ] **Step 4: Run tests; `make check`**

- [ ] **Step 5: Commit** `git commit -m "Add fixed-parameter LightGBM learner and parameter freeze script"`

- [ ] **Step 6 (owner, once data is downloaded): run `uv run python scripts/freeze_params.py data/split/dev.parquet`, commit the resulting `params.json` with the printed table in the commit message**

---

### Task 9: Metrics and seen/unseen split

**Files:**
- Create: `src/signal_factory/baseline/metrics.py`, `tests/baseline/test_metrics.py`

**Interfaces:**
- Produces: `recall_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr: float = 0.01) -> ThresholdResult` (dataclass: `threshold: float, recall: float, actual_fpr: float`) choosing the **smallest threshold whose actual FPR ≤ fpr** (conservative under ties); `recall_at_threshold(scores, labels, threshold) -> ThresholdResult` (flag = `score >= threshold`); `pr_auc(scores, labels) -> float`; `roc_auc(scores, labels) -> float` (both via scikit-learn); `seen_mask(eval_uids: pd.Series, train_uids: Iterable[str]) -> np.ndarray` (bool; NULL uid → `False`).

- [ ] **Step 1: Write the tests**

```python
def test_recall_at_fpr_matches_sklearn_roc_curve() -> None: ...
def test_recall_at_fpr_with_ties_never_exceeds_budget() -> None:      # Review Focus 5
    scores = np.array([0.9]*50 + [0.1]*950); labels = ...
    r = recall_at_fpr(scores, labels, 0.01); assert r.actual_fpr <= 0.01
def test_recall_at_threshold_reports_realised_fpr() -> None: ...
def test_seen_mask_null_uid_is_unseen() -> None:                       # Review Focus 4
    m = seen_mask(pd.Series(["a", None, "b"]), ["a"]); assert m.tolist() == [True, False, False]
def test_pr_auc_and_roc_auc_match_sklearn() -> None: ...
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement `metrics.py`**

- [ ] **Step 4: Run tests; `make check`**

- [ ] **Step 5: Commit** `git commit -m "Add recall-at-FPR with frozen thresholds and seen/unseen masks"`

---

### Task 10: Paired bootstrap

**Files:**
- Create: `src/signal_factory/baseline/bootstrap.py`, `tests/baseline/test_bootstrap.py`

**Interfaces:**
- Consumes: `recall_at_threshold` (Task 9).
- Produces: `paired_recall_diff(scores_a, scores_b, labels, threshold_a: float, threshold_b: float, n_resamples: int = 1000, seed: int = 0) -> BootstrapResult` (dataclass: `mean_diff: float, lo: float, hi: float, n_resamples: int`), percentile 2.5/97.5 interval of recall(b) − recall(a) over row resamples with replacement, the **same** resample applied to both; `N_RESAMPLES = 1000`.

- [ ] **Step 1: Write the tests**

```python
def test_identical_scores_give_zero_interval() -> None: r = paired_recall_diff(s, s, y, t, t); assert r.lo <= 0 <= r.hi and abs(r.mean_diff) < 1e-12
def test_interval_contains_known_difference_on_synthetic() -> None: ...
def test_is_seed_deterministic() -> None: ...
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement `bootstrap.py`** (NumPy `default_rng(seed).integers`, vectorised over resamples).

- [ ] **Step 4: Run tests; `make check`**

- [ ] **Step 5: Commit** `git commit -m "Add paired bootstrap for recall differences"`

---

### Task 11: Row A end-to-end on validation, plus uid collision statistic

**Files:**
- Create: `src/signal_factory/data/stats.py`, `src/signal_factory/ablation/row_a.py`, `src/signal_factory/cli.py`, `tests/data/test_stats.py`, `tests/ablation/test_row_a.py`
- Modify: `pyproject.toml` (add `[project.scripts] signal-factory = "signal_factory.cli:main"`)

**Interfaces:**
- Consumes: Tasks 3, 4, 8, 9.
- Produces:
  - `uid_collision_rate(df: pd.DataFrame) -> CollisionStats` (dataclass: `uids: int, colliding: int, rate: float`) = share of non-NULL uids with more than one distinct `(card2, card3, card4, card5, card6)` tuple.
  - `run_row_a(split_dir: Path, out_dir: Path, seeds=SEEDS) -> RowResult` (dataclass: `name: str, validation: Metrics, validation_seen: Metrics, validation_unseen: Metrics, threshold: float, params: dict`) where `Metrics` = `recall, actual_fpr, pr_auc, roc_auc, n`. Trains on `dev.parquet`, freezes the threshold on all validation rows, writes `out_dir/row_a.json`. **Never reads `final.parquet`.**
  - `cli.py`: `signal-factory convert RAW OUT`, `signal-factory split ALL OUT_DIR`, `signal-factory stats ALL`, `signal-factory row-a SPLIT_DIR OUT_DIR` (argparse).

- [ ] **Step 1: Write the tests**

```python
def test_uid_collision_rate_counts_multi_card_uids() -> None: ...
def test_run_row_a_writes_json_and_never_opens_final(tmp_path, synthetic_split, monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(pd, "read_parquet", record_then_call(opened))
    res = run_row_a(synthetic_split, tmp_path)
    assert not any("final" in p for p in opened)
    assert (tmp_path / "row_a.json").exists() and 0 <= res.validation.recall <= 1
def test_cli_row_a_invokes_runner(monkeypatch) -> None: ...
```

`synthetic_split` is a fixture writing a 400-row synthetic dataset through `write_split` (Task 4) with a few informative columns.

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement `stats.py`, `row_a.py`, `cli.py`**

- [ ] **Step 4: Run tests; `make check`**

- [ ] **Step 5: Commit** `git commit -m "Add row A validation run, uid collision statistic and CLI"`

- [ ] **Step 6 (owner, with real data): `signal-factory stats data/parquet/all.parquet` and `signal-factory row-a data/split results/row_a`; paste the collision rate and row-A validation metrics into `docs/results/plan-1-row-a.md` and commit that file (aggregate numbers only).**

---

### Task 12: Row B grid and backend decision

**Files:**
- Create: `src/signal_factory/routine/__init__.py`, `src/signal_factory/routine/grid.py`, `src/signal_factory/routine/compute.py`, `tests/routine/__init__.py`, `tests/routine/test_grid.py`, `tests/routine/test_compute.py`, `scripts/try_featuretools.py`, `docs/decisions/row-b-backend.md`

**Interfaces:**
- Consumes: `SignalSpec` (Task 5), `compute_signals` (Task 7).
- Produces:
  - `routine_grid() -> list[SignalSpec]`: the fixed grid from spec §7: keys `("uid", "card1_addr1", "P_emaildomain", "DeviceInfo")` × windows `("all", WindowHours(24), WindowDays(7))` × { `count` of `*`; `time_since_last`; `sum/mean/max/min` of `TransactionAmt`, `dist1`, `C1..C14`, `D1..D15`; `count_distinct` of `ProductCD`, `card2..card6`, `addr1`, `P_emaildomain`, `R_emaildomain`, `DeviceInfo` excluding the grouping key }. Ids follow `rb__<key>__<window>__<agg>__<column>`; hypothesis text is the fixed string "routine aggregation (row B grid)".
  - `compute_row_b(history: Path, out_path: Path, batch_size: int = 150) -> int` writing a Parquet of `TransactionID` + all grid columns, computed by `compute_signals` in batches, returning the column count.
  - `scripts/try_featuretools.py`: time-boxed featuretools DFS attempt on the same grid with cutoff times; prints wall time and whether a 2,000-row sample matches `compute_row_b` values; exits non-zero on mismatch or timeout (8 h).

- [ ] **Step 1: Write the grid tests**

```python
def test_grid_size_and_shape() -> None:
    g = routine_grid(); assert 1500 <= len(g) <= 1700 and len({s.id for s in g}) == len(g)
def test_grid_never_uses_label_or_grouping_key_as_column() -> None: ...
def test_grid_contains_expected_members() -> None:
    ids = {s.id for s in routine_grid()}
    assert "rb__uid__24h__count__star" in ids and "rb__P_emaildomain__all__count_distinct__DeviceInfo" in ids
    assert "rb__DeviceInfo__all__count_distinct__DeviceInfo" not in ids
def test_grid_specs_all_validate() -> None: ...
```

- [ ] **Step 2: Write the compute test** (`compute_row_b` on `tiny_history` writes the expected number of columns; values for two chosen cells equal `compute_signals` on the same specs).

- [ ] **Step 3: Run to verify they fail**

- [ ] **Step 4: Implement `grid.py`, `compute.py`, `scripts/try_featuretools.py`** (featuretools added as an optional dependency group `routine-ft`).

- [ ] **Step 5: Run tests; `make check`**

- [ ] **Step 6: Commit** `git commit -m "Add row B routine-aggregation grid computed by the signal engine"`

- [ ] **Step 7 (owner, with real data): run `scripts/try_featuretools.py` under the 8-hour time box; record outcome (finished / timed out / leakage mismatch) and the engine's own wall time in `docs/decisions/row-b-backend.md`; commit. Per spec §7 either outcome is acceptable and the report will say which backend produced row B.**

- [ ] **Step 8 (owner, with real data): `compute_row_b` on `data/split/history.parquet` → `data/features/row_b.parquet`; run row B on validation by reusing `run_row_a` logic with the extra features (small addition: `run_row(name, feature_parquets: list[Path], …)` generalising `run_row_a`; add a test that `run_row("A", [])` equals `run_row_a`). Record validation metrics for A and B in `docs/results/plan-1-rows-a-b.md`.**

---

### Task 13: Plan-1 close-out

**Files:**
- Modify: `README.md` (status, how to reproduce rows A and B on validation), `docs/superpowers/specs/2026-11-02-signal-factory-design.md` (§10 open items: fill in the uid collision rate and the row-B backend decision)

- [ ] **Step 1: Update README and spec §10 with measured values (aggregate numbers only)**
- [ ] **Step 2: `make check`; confirm `git status` shows nothing under `data/` or `results/`**
- [ ] **Step 3: Commit** `git commit -m "Close plan 1: rows A and B reproducible on validation"`

Plan 2 (proposer, selection, ablation runner, final evaluation lock, report) starts from here.

---

## Self-review notes

- Spec coverage: §4 (dataset, uid, label-never-input, row A columns) → Tasks 3, 5, 8; §5 (split, gaps, seen/unseen) → Tasks 4, 9; §6 (learner, metrics, threshold, bootstrap, model-on-final) → Tasks 8–10 (final evaluation itself is plan 2); §7 rows A, B and the fallback → Tasks 11, 12; §8.1–8.4 → Tasks 2–12; §8.5–8.6 → plan 2; §10 measurements → Tasks 11, 12, 13.
- Deviation recorded: `count` on empty history is 0 (Task 6, Step 6 amends the spec).
- Types: `SignalSpec`, `Window`, `compute_signals`, `ThresholdResult`, `BootstrapResult`, `RowResult`, `SEEDS` are defined once and referenced by name afterwards.
