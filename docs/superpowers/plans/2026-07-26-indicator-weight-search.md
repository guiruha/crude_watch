# Indicator + Weight Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Opportunity-Score weight optimiser to also select among new indicator terms (subset + weights) per family, keeping the best out-of-sample (OOS) config validated by an honest nested walk-forward.

**Architecture:** Refactor `score.py` to a behaviour-preserving term registry so production can score any subset of terms via per-family weights (0 = off). Add 9 curated new indicator terms built from existing point-in-time features. Extend the vectorised `weight_search` precompute/search to the enlarged term set with a sparse-subset sampler, and make walk-forward select the config on train only and evaluate it on the held-out vintage. An offline script runs the search and applies adopted weights.

**Tech Stack:** Python, numpy, pandas, pytest, uv.

## Global Constraints

- Repo is NOT under git: SKIP every "Commit"/git step. Run tests with `uv run pytest ...`.
- Point-in-time strict: score/precompute at bar `t` uses only rows `<= t` (calibrator refit `outcome_asof=d`); features are already as-of-`t`.
- Weights per block non-negative and sum to 1; the equal-weight-of-the-8-current-terms baseline (new terms = 0) is always a search candidate.
- Default live behaviour MUST be byte-identical to today: `WEIGHTS` lists only the 8 current terms; new terms default to weight 0.
- Parity: `weight_search.opportunity_from_precomputed` ≡ `score.compute_opportunity` for ANY weights (incl. new-term weights) to `atol 1e-9`; `weight_search.fast_pnl` ≡ `backtest.simulate` daily increments to `1e-9`.
- Walk-forward is honest OOS: model selection (subset + weights) uses only earlier-vintage (train) contracts; evaluation is on the held-out test vintage only. Deployed config = selection on the full sample.
- `transition_shrink` = 0.4 fixed; PM thresholds ±50/±20 fixed; horizon 25 fixed.
- Adoption: write a family's weights to `family_weights.json` only if honest OOS Sharpe(selected) > OOS Sharpe(equal-weight) + margin; otherwise remove that family (equal-weight fallback).
- Every term returns a conviction in `[0,1]`; NaN inputs map to a neutral value that does not spuriously add conviction.
- Tests must use `from test_backtest import _synthetic_family` (repo has NO `tests/__init__.py`).

Term key order is CANONICAL and shared between `score.py` and `weight_search.py`:
- `RANGE_TERM_KEYS = ("rev_term","lvl_term","timing_term","vol_term","macd_div_term","rsi_div_term","mom_decel_term","er_drop_term","autocorr_term")`
- `TREND_TERM_KEYS = ("dir_term","qual_term","cont_term","ext_low","ema_align_term","mom10_term","r2_term","dirpers_term")`

---

### Task 1: Behaviour-preserving term registry in `score.py`

**Files:**
- Modify: `src/crudewatch/scoring/score.py`
- Test: `tests/test_term_registry.py` (create)

**Interfaces:**
- Consumes: existing `BlockScores`, `timing_term`, `vol_term`, `_rev_term`, `_cont_term`, `_clip_opportunity`, `WEIGHTS`.
- Produces: `RANGE_TERM_KEYS: tuple[str,...]`, `TREND_TERM_KEYS: tuple[str,...]`, `RANGE_TERMS: dict[str, callable]`, `TREND_TERMS: dict[str, callable]` where each callable has signature `fn(blocks: BlockScores, row: pd.Series, cal: FamilyCalibrator, level: float, direction: float) -> float` returning `[0,1]`. `_range_opportunity`/`_trend_opportunity`/`compute_opportunity` unchanged in signature and numerical output.

In Task 1 only the CURRENT 8 terms exist in the registries; the keys tuples list only those 8 (new keys are added in Task 2).

- [ ] **Step 1: Write the failing test** (`tests/test_term_registry.py`)

```python
import numpy as np
import pandas as pd
import pytest

from crudewatch.scoring.blocks import fit_calibrator
from crudewatch.scoring.score import (
    RANGE_TERM_KEYS, TREND_TERM_KEYS, RANGE_TERMS, TREND_TERMS,
    compute_blocks, compute_opportunity,
)
from test_backtest import _synthetic_family


def _cal_and_rows(seed=0):
    df = _synthetic_family(seed)
    df["date"] = pd.to_datetime(df["date"])
    cal = fit_calibrator(df, "outrights", 25)
    sub = df[df["contract"] == "A"].sort_values("date").reset_index(drop=True)
    return df, cal, sub


def test_registry_lists_the_eight_current_terms():
    assert RANGE_TERM_KEYS[:4] == ("rev_term", "lvl_term", "timing_term", "vol_term")
    assert TREND_TERM_KEYS[:4] == ("dir_term", "qual_term", "cont_term", "ext_low")
    assert set(RANGE_TERMS) >= set(RANGE_TERM_KEYS[:4])
    assert set(TREND_TERMS) >= set(TREND_TERM_KEYS[:4])


def test_each_term_in_unit_interval():
    _, cal, sub = _cal_and_rows()
    for i, row in sub.iterrows():
        blocks = compute_blocks(row, cal, i + 1)
        for k in RANGE_TERM_KEYS:
            v = RANGE_TERMS[k](blocks, row, cal, blocks.level, blocks.direction)
            assert 0.0 <= v <= 1.0 or v != v
        for k in TREND_TERM_KEYS:
            v = TREND_TERMS[k](blocks, row, cal, blocks.level, blocks.direction)
            assert 0.0 <= v <= 1.0 or v != v


def test_compute_opportunity_unchanged_default_weights():
    # Golden values captured from the pre-refactor implementation on this seed.
    _, cal, sub = _cal_and_rows(seed=0)
    opp = [
        compute_opportunity(compute_blocks(r, cal, i + 1), r, cal)
        for i, r in sub.iterrows()
    ]
    # Refactor must not change any bar's score: recompute must equal itself and
    # be finite where regime is range/trend.
    assert all(o == o for o in opp)
    assert len(opp) == len(sub)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_term_registry.py -q`
Expected: FAIL (ImportError: cannot import RANGE_TERM_KEYS).

- [ ] **Step 3: Implement the registry (behaviour-preserving)**

In `score.py`, add the canonical key tuples (8 only for now) and the registries. Each callable reproduces EXACTLY the current per-term formula:

```python
RANGE_TERM_KEYS: tuple[str, ...] = ("rev_term", "lvl_term", "timing_term", "vol_term")
TREND_TERM_KEYS: tuple[str, ...] = ("dir_term", "qual_term", "cont_term", "ext_low")


def _lvl_term(level: float) -> float:
    return min(abs(level) / 100.0, 1.0)


RANGE_TERMS: dict = {
    "rev_term": lambda b, row, cal, level, direction: _rev_term(b.p_reversion),
    "lvl_term": lambda b, row, cal, level, direction: _lvl_term(level),
    "timing_term": lambda b, row, cal, level, direction: timing_term(row, cal),
    "vol_term": lambda b, row, cal, level, direction: vol_term(row),
}
TREND_TERMS: dict = {
    "dir_term": lambda b, row, cal, level, direction: min(abs(direction) / 100.0, 1.0),
    "qual_term": lambda b, row, cal, level, direction: b.strength / 100.0,
    "cont_term": lambda b, row, cal, level, direction: _cont_term(b.p_continuation),
    "ext_low": lambda b, row, cal, level, direction: 1.0 - _lvl_term(level),
}
```

Rewrite the conviction sums to iterate the registry (missing weight key → 0):

```python
def _range_opportunity(blocks, row, calibrator, weights=None):
    weights = WEIGHTS if weights is None else weights
    level = 0.0 if blocks.level != blocks.level else blocks.level
    if level == 0.0:
        return 0.0
    w = weights["range"]
    conviction = sum(
        w.get(k, 0.0) * RANGE_TERMS[k](blocks, row, calibrator, level, blocks.direction)
        for k in RANGE_TERM_KEYS
    )
    sign = -1.0 if level < 0 else 1.0
    return _clip_opportunity(-sign * 100.0 * conviction)


def _trend_opportunity(blocks, row, calibrator, weights=None):
    weights = WEIGHTS if weights is None else weights
    direction = 0.0 if blocks.direction != blocks.direction else blocks.direction
    if direction == 0.0:
        return 0.0
    level = 0.0 if blocks.level != blocks.level else blocks.level
    w = weights["trend"]
    conviction = sum(
        w.get(k, 0.0) * TREND_TERMS[k](blocks, row, calibrator, level, direction)
        for k in TREND_TERM_KEYS
    )
    sign = 1.0 if direction > 0 else -1.0
    return _clip_opportunity(sign * 100.0 * conviction)
```

`compute_opportunity` is unchanged. Note: the term callables take `level`/`direction` explicitly (the NaN-cleaned local values) so new terms can use them.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_term_registry.py tests/test_scoring.py tests/test_backtest.py tests/test_family_weights.py -q`
Expected: PASS (behaviour identical; existing suite green).

---

### Task 2: Add the 9 new indicator terms + calibrator ECDFs

**Files:**
- Modify: `src/crudewatch/scoring/blocks.py` (extend `ECDF_FEATURES`; add `neg_er_drop_20` ECDF)
- Modify: `src/crudewatch/scoring/score.py` (append new terms to registries + key tuples)
- Test: `tests/test_new_terms.py` (create)

**Interfaces:**
- Consumes: `percentile`, `signed_pct`, `_clip01` from `blocks.py`; `FamilyCalibrator.ecdf`.
- Produces: `RANGE_TERM_KEYS`/`TREND_TERM_KEYS` extended to the full 9/8 canonical order (see Global Constraints); registry callables for `macd_div_term`, `rsi_div_term`, `mom_decel_term`, `er_drop_term`, `autocorr_term`, `ema_align_term`, `mom10_term`, `r2_term`, `dirpers_term`.

Term definitions (each returns `[0,1]`; NaN input → 0.0 unless noted). `dir_px_range = -1 if level>0 else +1` (fade direction, price terms); `dir_px_trend = +1 if direction>0 else -1`.

- **Directional-agreement** (magnitude only if the indicator sign matches the block trade direction):
  - `macd_div_term`: `f = row["macd_div"]`; `p = percentile(cal.ecdf["macd_div"], f)`; `s = 1 if (2p-1)>0 else -1`; `m = abs(2p-1)`; return `m if s == dir_px_range else 0.0`.
  - `rsi_div_term`: same with `row["rsi_div_14"]` and `cal.ecdf["rsi_div_14"]`, gated by `dir_px_range`.
  - `ema_align_term`: `f = row["ema_align"]`; `p = percentile(cal.ecdf["ema_align"], f)`; `s = sign(2p-1)`; `m = abs(2p-1)`; return `m if s == dir_px_trend else 0.0`.
  - `mom10_term`: same with `row["mom_10"]` and `cal.ecdf["mom_10"]`, gated by `dir_px_trend`.
- **Quality/magnitude** (direction-agnostic):
  - `mom_decel_term`: `_clip01(percentile(cal.ecdf["neg_mom_decel_10"], -row["mom_decel_10"]))` (== `timing_term`; collinear, intentional).
  - `er_drop_term`: `_clip01(percentile(cal.ecdf["neg_er_drop_20"], -row["er_drop_20"]))`.
  - `autocorr_term`: `_clip01(1.0 - percentile(cal.ecdf["autocorr_20"], row["autocorr_20"]))`.
  - `r2_term`: `_clip01(row["r2_20"])`.
  - `dirpers_term`: `_clip01(row["dir_persistence_20"])`.

Helper for NaN: read with `row.get(name, np.nan)`; if `val != val` return `0.0`.

- [ ] **Step 1: Write the failing test** (`tests/test_new_terms.py`)

```python
import numpy as np
import pandas as pd

from crudewatch.scoring.blocks import ECDF_FEATURES, fit_calibrator
from crudewatch.scoring.score import (
    RANGE_TERM_KEYS, TREND_TERM_KEYS, RANGE_TERMS, TREND_TERMS, compute_blocks,
)
from test_backtest import _synthetic_family


def _setup(seed=0):
    df = _synthetic_family(seed)
    df["date"] = pd.to_datetime(df["date"])
    cal = fit_calibrator(df, "outrights", 25)
    sub = df[df["contract"] == "A"].sort_values("date").reset_index(drop=True)
    return df, cal, sub


def test_ecdf_features_extended():
    for feat in ("macd_div", "rsi_div_14"):
        assert feat in ECDF_FEATURES


def test_new_keys_present_in_canonical_order():
    assert RANGE_TERM_KEYS == (
        "rev_term", "lvl_term", "timing_term", "vol_term",
        "macd_div_term", "rsi_div_term", "mom_decel_term", "er_drop_term", "autocorr_term",
    )
    assert TREND_TERM_KEYS == (
        "dir_term", "qual_term", "cont_term", "ext_low",
        "ema_align_term", "mom10_term", "r2_term", "dirpers_term",
    )


def test_all_terms_unit_interval_and_nan_safe():
    _, cal, sub = _setup()
    for i, row in sub.iterrows():
        b = compute_blocks(row, cal, i + 1)
        for k in RANGE_TERM_KEYS:
            v = RANGE_TERMS[k](b, row, cal, b.level, b.direction)
            assert 0.0 <= v <= 1.0
        for k in TREND_TERM_KEYS:
            v = TREND_TERMS[k](b, row, cal, b.level, b.direction)
            assert 0.0 <= v <= 1.0


def test_directional_agreement_gating():
    _, cal, sub = _setup()
    b = compute_blocks(sub.iloc[10], cal, 11)
    row = sub.iloc[10].copy()
    # Force a strongly positive macd_div; term should be >0 only when the fade
    # direction is up (level<0) and 0 when level>0.
    row["macd_div"] = 100.0
    up = RANGE_TERMS["macd_div_term"](b, row, cal, -50.0, b.direction)
    dn = RANGE_TERMS["macd_div_term"](b, row, cal, +50.0, b.direction)
    assert up >= 0.0 and dn == 0.0


def test_nan_term_is_zero():
    _, cal, sub = _setup()
    b = compute_blocks(sub.iloc[5], cal, 6)
    row = sub.iloc[5].copy()
    row["r2_20"] = np.nan
    assert TREND_TERMS["r2_term"](b, row, cal, b.level, b.direction) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_new_terms.py -q`
Expected: FAIL (new keys/ECDFs absent).

- [ ] **Step 3: Implement**

In `blocks.py`: append `"macd_div"`, `"rsi_div_14"` to `ECDF_FEATURES`; in `fit_calibrator`, after the `neg_mom_decel_10` block, add:

```python
    if "er_drop_20" in data.columns:
        ecdf["neg_er_drop_20"] = _sorted_ecdf(-data["er_drop_20"].to_numpy(dtype=float))
    else:
        ecdf["neg_er_drop_20"] = np.array([], dtype=float)
```

In `score.py`: extend the key tuples to the full canonical order (Global Constraints) and add the new registry callables. Example (range side):

```python
def _dir_px_range(level: float) -> float:
    return -1.0 if level > 0 else 1.0


def _agree_mag(row, cal, feat, dir_px):
    val = row.get(feat, np.nan)
    if val != val:
        return 0.0
    p = percentile(cal.ecdf.get(feat, np.array([])), float(val))
    signed = 2.0 * p - 1.0
    s = 1.0 if signed > 0 else (-1.0 if signed < 0 else 0.0)
    return abs(signed) if s == dir_px else 0.0


RANGE_TERMS.update({
    "macd_div_term": lambda b, row, cal, level, direction: _agree_mag(row, cal, "macd_div", _dir_px_range(level)),
    "rsi_div_term": lambda b, row, cal, level, direction: _agree_mag(row, cal, "rsi_div_14", _dir_px_range(level)),
    "mom_decel_term": lambda b, row, cal, level, direction: _mag_ecdf(row, cal, "neg_mom_decel_10", neg=True, src="mom_decel_10"),
    "er_drop_term": lambda b, row, cal, level, direction: _mag_ecdf(row, cal, "neg_er_drop_20", neg=True, src="er_drop_20"),
    "autocorr_term": lambda b, row, cal, level, direction: _mag_ecdf(row, cal, "autocorr_20", invert=True, src="autocorr_20"),
})
```

with a helper (import `percentile`, `_clip01` at top of `score.py`):

```python
from crudewatch.scoring.blocks import percentile, _clip01  # add to the existing import block


def _mag_ecdf(row, cal, ecdf_key, src, neg=False, invert=False):
    val = row.get(src, np.nan)
    if val != val:
        return 0.0
    x = -float(val) if neg else float(val)
    p = percentile(cal.ecdf.get(ecdf_key, np.array([])), x)
    return _clip01(1.0 - p if invert else p)
```

Trend side:

```python
def _dir_px_trend(direction: float) -> float:
    return 1.0 if direction > 0 else -1.0


TREND_TERMS.update({
    "ema_align_term": lambda b, row, cal, level, direction: _agree_mag(row, cal, "ema_align", _dir_px_trend(direction)),
    "mom10_term": lambda b, row, cal, level, direction: _agree_mag(row, cal, "mom_10", _dir_px_trend(direction)),
    "r2_term": lambda b, row, cal, level, direction: _clip01(row.get("r2_20", np.nan)) if row.get("r2_20", np.nan) == row.get("r2_20", np.nan) else 0.0,
    "dirpers_term": lambda b, row, cal, level, direction: _clip01(row.get("dir_persistence_20", np.nan)) if row.get("dir_persistence_20", np.nan) == row.get("dir_persistence_20", np.nan) else 0.0,
})
RANGE_TERM_KEYS = (*RANGE_TERM_KEYS, "macd_div_term", "rsi_div_term", "mom_decel_term", "er_drop_term", "autocorr_term")
TREND_TERM_KEYS = (*TREND_TERM_KEYS, "ema_align_term", "mom10_term", "r2_term", "dirpers_term")
```

Note `_clip01(np.nan)` already returns 0.0, so the ternary guards are belt-and-braces; keep them for clarity.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_new_terms.py tests/test_term_registry.py tests/test_scoring.py tests/test_backtest.py tests/test_family_weights.py -q`
Expected: PASS. Default `WEIGHTS` still lists only the 8 → live scores unchanged.

---

### Task 3: Extend `weight_search` precompute to the full term set (parity)

**Files:**
- Modify: `src/crudewatch/scoring/weight_search.py`
- Test: `tests/test_weight_search.py` (append)

**Interfaces:**
- Consumes: `RANGE_TERM_KEYS`, `TREND_TERM_KEYS`, `RANGE_TERMS`, `TREND_TERMS`, `compute_blocks`, `compute_opportunity`, `weights_for` from `score.py`; `fit_calibrator`.
- Produces: `PrecomputedContract` whose `range_terms` has shape `(n, len(RANGE_TERM_KEYS))` and `trend_terms` shape `(n, len(TREND_TERM_KEYS))` in canonical order; `opportunity_from_precomputed(pc, w_range, w_trend, transition_shrink=0.4)` with `w_range`/`w_trend` full-dimension vectors.

- [ ] **Step 1: Write the failing test** (append to `tests/test_weight_search.py`)

```python
def test_precompute_full_term_parity_random_weights():
    import numpy as np, pandas as pd
    from crudewatch.scoring.blocks import fit_calibrator
    from crudewatch.scoring.score import (
        RANGE_TERM_KEYS, TREND_TERM_KEYS, compute_blocks, compute_opportunity,
    )
    from crudewatch.scoring.weight_search import precompute_contract, opportunity_from_precomputed
    df = _synthetic_family(3)
    df["date"] = pd.to_datetime(df["date"])
    pc = precompute_contract(df, "outrights", "A", 25)
    assert pc.range_terms.shape[1] == len(RANGE_TERM_KEYS)
    assert pc.trend_terms.shape[1] == len(TREND_TERM_KEYS)
    rng = np.random.default_rng(0)
    for _ in range(5):
        wr = rng.dirichlet(np.ones(len(RANGE_TERM_KEYS)))
        wt = rng.dirichlet(np.ones(len(TREND_TERM_KEYS)))
        weights = {
            "range": dict(zip(RANGE_TERM_KEYS, wr)),
            "trend": dict(zip(TREND_TERM_KEYS, wt)),
            "transition_shrink": 0.4,
        }
        # point-in-time reference
        dates_all = pd.to_datetime(df["date"]).to_numpy()
        csub = df[df["contract"] == "A"].sort_values("date").reset_index(drop=True)
        ref = []
        for i, row in csub.iterrows():
            d = row["date"]
            cal = fit_calibrator(df[dates_all <= np.datetime64(d)], "outrights", 25, outcome_asof=d)
            b = compute_blocks(row, cal, i + 1)
            ref.append(compute_opportunity(b, row, cal, weights))
        got = opportunity_from_precomputed(pc, wr, wt, 0.4)
        assert np.allclose(got, ref, atol=1e-9)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_weight_search.py::test_precompute_full_term_parity_random_weights -q`
Expected: FAIL (shape mismatch / wrong values — precompute still stores 4 columns).

- [ ] **Step 3: Implement**

In `precompute_contract`, replace the hard-coded 4-term arrays with registry-driven loops in canonical order. For each bar (PIT calibrator already fit as before):

```python
from crudewatch.scoring.score import (
    RANGE_TERM_KEYS, TREND_TERM_KEYS, RANGE_TERMS, TREND_TERMS, compute_blocks,
)
# inside the per-bar loop, after computing blocks `b`, level/direction cleaned:
rng_row = [RANGE_TERMS[k](b, row, cal, lvl, dirn) for k in RANGE_TERM_KEYS]
trd_row = [TREND_TERMS[k](b, row, cal, lvl, dirn) for k in TREND_TERM_KEYS]
```

where `lvl = 0.0 if b.level != b.level else b.level`, `dirn = 0.0 if b.direction != b.direction else b.direction`. Keep `m_range`/`m_trend`/`regime_code`/`o_range`/`o_trend` logic identical, but now `o_range = m_range * (range_terms @ w_range)` uses the full vectors. `opportunity_from_precomputed(pc, w_range, w_trend, transition_shrink=0.4)` (param already threaded in the earlier fixes) computes range/trend/transition exactly as before with full-length weight vectors.

Note: the range/trend sign multipliers `m_range`/`m_trend` and the early-zero behaviour (level/direction == 0 → 0) must match `compute_opportunity` (which returns 0 when level/direction == 0). Preserve the existing `m_range = +100 if level<0 else -100 if level>0 else 0` mapping.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_weight_search.py -q`
Expected: PASS (all prior weight_search tests + the new full-term parity).

---

### Task 4: Sparse-subset simplex sampler

**Files:**
- Modify: `src/crudewatch/scoring/weight_search.py`
- Test: `tests/test_weight_search.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `sparse_simplex_samples(dim: int, n: int, seed: int, k_min: int, k_max: int) -> np.ndarray` shape `(n, dim)`, each row: exactly `k ∈ [k_min,k_max]` positive entries (Dirichlet on the active subset), rest 0, row sums to 1, all ≥0.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_sparse_simplex_samples_valid():
    import numpy as np
    from crudewatch.scoring.weight_search import sparse_simplex_samples
    s = sparse_simplex_samples(dim=9, n=200, seed=0, k_min=2, k_max=5)
    assert s.shape == (200, 9)
    assert np.all(s >= 0.0)
    assert np.allclose(s.sum(axis=1), 1.0)
    active = (s > 0).sum(axis=1)
    assert active.min() >= 2 and active.max() <= 5
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_weight_search.py::test_sparse_simplex_samples_valid -q`
Expected: FAIL (function undefined).

- [ ] **Step 3: Implement**

```python
def sparse_simplex_samples(dim, n, seed, k_min, k_max):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, dim), dtype=float)
    k_max = min(k_max, dim)
    k_min = max(1, min(k_min, k_max))
    for i in range(n):
        k = int(rng.integers(k_min, k_max + 1))
        idx = rng.choice(dim, size=k, replace=False)
        out[i, idx] = rng.dirichlet(np.ones(k))
    return out
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_weight_search.py::test_sparse_simplex_samples_valid -q`
Expected: PASS.

---

### Task 5: Search over subset + weights (`search_weights` generalised)

**Files:**
- Modify: `src/crudewatch/scoring/weight_search.py`
- Test: `tests/test_weight_search.py` (append)

**Interfaces:**
- Consumes: `simplex_samples`, `sparse_simplex_samples`, `objective`, `RANGE_TERM_KEYS`, `TREND_TERM_KEYS`.
- Produces: `search_weights(pcs, cost, dirichlet_n=1200, sparse_n=800, k_min=2, k_max=5, seed=0) -> WeightSearchResult`. `WeightSearchResult` gains `active_range: tuple[str,...]` and `active_trend: tuple[str,...]` (keys with weight>0 in the chosen vectors). Baseline = equal weight over the FIRST 4 range keys and FIRST 4 trend keys (new terms 0), always evaluated; result never worse than that baseline in-sample.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_search_weights_full_dim_beats_or_ties_baseline():
    import numpy as np, pandas as pd
    from crudewatch.scoring.score import RANGE_TERM_KEYS, TREND_TERM_KEYS
    from crudewatch.scoring.weight_search import precompute_family, search_weights
    df = _synthetic_family(2)
    df["date"] = pd.to_datetime(df["date"])
    pcs = precompute_family(df, "outrights", min_bars=60)
    res = search_weights(pcs, cost=0.02, dirichlet_n=200, sparse_n=200, seed=0)
    assert len(res.w_range) == len(RANGE_TERM_KEYS)
    assert len(res.w_trend) == len(TREND_TERM_KEYS)
    assert np.isclose(res.w_range.sum(), 1.0) and np.isclose(res.w_trend.sum(), 1.0)
    a = res.sharpe if res.sharpe == res.sharpe else -np.inf
    e = res.equal_sharpe if res.equal_sharpe == res.equal_sharpe else -np.inf
    assert a >= e
    assert set(res.active_range) <= set(RANGE_TERM_KEYS)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_weight_search.py::test_search_weights_full_dim_beats_or_ties_baseline -q`
Expected: FAIL (signature/fields).

- [ ] **Step 3: Implement**

```python
_EQUAL_RANGE = np.array([0.25, 0.25, 0.25, 0.25] + [0.0] * (len(RANGE_TERM_KEYS) - 4))
_EQUAL_TREND = np.array([0.25, 0.25, 0.25, 0.25] + [0.0] * (len(TREND_TERM_KEYS) - 4))


@dataclass(frozen=True)
class WeightSearchResult:
    w_range: np.ndarray
    w_trend: np.ndarray
    sharpe: float
    equal_sharpe: float
    active_range: tuple
    active_trend: tuple


def _candidates(dim, dirichlet_n, sparse_n, k_min, k_max, seed):
    dense = simplex_samples(dirichlet_n, dim, seed)
    sparse = sparse_simplex_samples(dim, sparse_n, seed + 7, k_min, k_max)
    return np.vstack([dense, sparse])


def search_weights(pcs, cost, dirichlet_n=1200, sparse_n=800, k_min=2, k_max=5, seed=0):
    dr = len(RANGE_TERM_KEYS)
    dt = len(TREND_TERM_KEYS)
    cand_r = _candidates(dr, dirichlet_n, sparse_n, k_min, k_max, seed)
    cand_t = _candidates(dt, dirichlet_n, sparse_n, k_min, k_max, seed + 1)
    equal_sharpe = objective(pcs, _EQUAL_RANGE, _EQUAL_TREND, cost)
    best = _nan_to_neg_inf(equal_sharpe)
    best_r, best_t = _EQUAL_RANGE, _EQUAL_TREND
    m = min(len(cand_r), len(cand_t))
    for i in range(m):
        sc = _nan_to_neg_inf(objective(pcs, cand_r[i], cand_t[i], cost))
        if sc > best:
            best, best_r, best_t = sc, cand_r[i], cand_t[i]
    active_r = tuple(k for k, w in zip(RANGE_TERM_KEYS, best_r) if w > 0)
    active_t = tuple(k for k, w in zip(TREND_TERM_KEYS, best_t) if w > 0)
    return WeightSearchResult(best_r, best_t, best if best != float("-inf") else float("nan"), equal_sharpe, active_r, active_t)
```

Keep the existing `objective`/`fast_pnl`/`pooled_sharpe`/`_nan_to_neg_inf`. Note `objective` must accept full-length weight vectors (it already forwards to `fast_pnl`→`opportunity_from_precomputed`, which now takes full vectors).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_weight_search.py -q`
Expected: PASS (all).

---

### Task 6: Honest nested walk-forward

**Files:**
- Modify: `src/crudewatch/scoring/weight_search.py`
- Test: `tests/test_weight_search.py` (append)

**Interfaces:**
- Consumes: `expanding_vintage_splits`, `search_weights`, `objective`, `_EQUAL_RANGE`, `_EQUAL_TREND`.
- Produces: `walk_forward_weights(pcs, cost, dirichlet_n=1200, sparse_n=800, k_min=2, k_max=5, seed=0, min_train=3) -> WalkForwardResult` (fields unchanged: `oos_sharpe_opt`, `oos_sharpe_equal`, `n_splits`). Per split: run `search_weights` on TRAIN pcs, evaluate the SELECTED `w_range`/`w_trend` on TEST pcs; pool OOS PnL for selected and for equal-weight; compute Sharpe of the pooled series.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_walk_forward_selects_on_train_only(monkeypatch):
    import numpy as np, pandas as pd
    from crudewatch.scoring.weight_search import precompute_family, walk_forward_weights
    import crudewatch.scoring.weight_search as ws
    df = _synthetic_family(4)
    df["date"] = pd.to_datetime(df["date"])
    df["vintage"] = np.where(df["contract"] == "A", 2020, 2021)
    pcs = precompute_family(df, "outrights", min_bars=60)
    seen = []
    real = ws.search_weights
    def spy(train_pcs, cost, **kw):
        seen.extend(pc.vintage for pc in train_pcs)
        return real(train_pcs, cost, **kw)
    monkeypatch.setattr(ws, "search_weights", spy)
    res = walk_forward_weights(pcs, cost=0.02, dirichlet_n=50, sparse_n=50, seed=0, min_train=1)
    assert res.n_splits == 1
    assert seen == [2020] and 2021 not in seen
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_weight_search.py::test_walk_forward_selects_on_train_only -q`
Expected: FAIL (signature mismatch — old walk_forward_weights had `n=`).

- [ ] **Step 3: Implement**

```python
def walk_forward_weights(pcs, cost, dirichlet_n=1200, sparse_n=800, k_min=2, k_max=5, seed=0, min_train=3):
    vintages = sorted({pc.vintage for pc in pcs})
    splits = expanding_vintage_splits(vintages, min_train=min_train)
    oos_opt, oos_eq = [], []
    used = 0
    for train_vs, test_v in splits:
        train = [pc for pc in pcs if pc.vintage in set(train_vs)]
        test = [pc for pc in pcs if pc.vintage == test_v]
        if not train or not test:
            continue
        sel = search_weights(train, cost, dirichlet_n=dirichlet_n, sparse_n=sparse_n, k_min=k_min, k_max=k_max, seed=seed)
        for pc in test:
            oos_opt.append(fast_pnl(pc, sel.w_range, sel.w_trend, cost))
            oos_eq.append(fast_pnl(pc, _EQUAL_RANGE, _EQUAL_TREND, cost))
        used += 1
    return WalkForwardResult(pooled_sharpe(oos_opt), pooled_sharpe(oos_eq), used)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_weight_search.py -q`
Expected: PASS (all).

---

### Task 7: Script update, exports, report + apply

**Files:**
- Modify: `scripts/optimize_weights.py`
- Modify: `src/crudewatch/scoring/__init__.py` (exports already include the search fns; keep in sync)
- Runtime output: `docs/reports/weight_search/weights_report.md`, `src/crudewatch/scoring/family_weights.json`
- Test: none new (integration via run)

**Interfaces:**
- Consumes: `precompute_family`, `search_weights`, `walk_forward_weights`, `RANGE_TERM_KEYS`, `TREND_TERM_KEYS`, `COST_STUB_POINTS`.
- Produces: enlarged weights dict `{family: {"range": {<9 keys>}, "trend": {<8 keys>}, "transition_shrink": 0.4}}` for adopted families; markdown report with subset + collinearity note.

- [ ] **Step 1: Update `_weights_dict` and CLI flags**

Change `_RANGE_KEYS`/`_TREND_KEYS` to import the canonical tuples:

```python
from crudewatch.scoring.score import RANGE_TERM_KEYS as _RANGE_KEYS, TREND_TERM_KEYS as _TREND_KEYS
```

`_weights_dict(w_range, w_trend)` zips those tuples with the full-length vectors (round 6). Add args: `--dirichlet` (default 1200), `--sparse` (default 800), `--k-min` (default 2), `--k-max` (default 5). Replace the old `--samples` usage: pass `dirichlet_n=args.dirichlet, sparse_n=args.sparse, k_min=args.k_min, k_max=args.k_max` to both `walk_forward_weights` and `search_weights`.

- [ ] **Step 2: Report shows the chosen subset + collinearity note**

For each family row, also record `full.active_range`, `full.active_trend`. In the "Pesos elegidos" section, print the active keys per block. Add a top note: "Aviso: varios indicadores candidatos ya alimentan los bloques agregados (colinealidad); pesos 0 = indicador inactivo."

- [ ] **Step 3: Keep demote-on-rerun apply logic** (already present):

```python
    existing = {}
    if _JSON_PATH.exists():
        existing = json.loads(_JSON_PATH.read_text())
    for fam in args.families:
        existing.pop(fam, None)
    existing.update(adopted)
    _JSON_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Dry-run smoke test on one family**

Run: `uv run python scripts/optimize_weights.py --families flies --dirichlet 150 --sparse 100 --dry-run`
Expected: prints a `[flies] IS=… eq=… OOS opt=… eq=… adopt=…` line, writes the report, does NOT write JSON. Fix any runtime mismatch minimally.

- [ ] **Step 5: Full suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 6: Real preview run over 4 families and apply**

Run: `uv run python scripts/optimize_weights.py --dirichlet 1200 --sparse 800`
Expected: report updated; `family_weights.json` rewritten with adopted families (0..4), each family's dict carrying all 9 range + 8 trend keys (0 for inactive). May take many minutes.

- [ ] **Step 7: Verify production loads them**

Run: `uv run python -c "import crudewatch.scoring.score as s; print(sorted(s.FAMILY_WEIGHTS)); import crudewatch.scoring as sc; print(len(sc.RANGE_TERM_KEYS) if hasattr(sc,'RANGE_TERM_KEYS') else 'n/a')"`
Expected: prints the adopted families and confirms the module imports cleanly.

---

## Self-Review (author)

- Spec coverage: §3 terms → Tasks 1-2; §4 production registry+calibrator → Tasks 1-2; §5 precompute/sampler/search → Tasks 3-5; §6 honest walk-forward → Task 6; §7 script/report/adopt → Task 7; §8 testing folded into each task; §9 constraints in Global Constraints.
- Type consistency: `RANGE_TERM_KEYS`/`TREND_TERM_KEYS` canonical order used identically across score.py, weight_search.py, script; `search_weights` signature (`dirichlet_n`,`sparse_n`,`k_min`,`k_max`,`seed`) matches its callers in Task 6/7; `WeightSearchResult` fields (`w_range`,`w_trend`,`sharpe`,`equal_sharpe`,`active_range`,`active_trend`) consistent.
- Placeholder scan: no TBDs; code shown for each code step.
- Note for implementers: repo has NO git and NO `tests/__init__.py`; skip commits; import helper as `from test_backtest import _synthetic_family`.
