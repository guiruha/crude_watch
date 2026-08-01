# Indicator Bucket Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline sweep that, for every contract family, buckets all 24 indicators into terciles using **only prior-date information**, forms every combination of up to four indicators, and reports forward-price-difference statistics per joint bucket cell at horizons D+1/2/3/5/10/15/20.

**Architecture:** A pure library module `backtesting/research/bucket_sweep.py` (point-in-time cutoffs, bucketing, combination sweep, per-cell statistics — no file IO) driven by a thin CLI `scripts/run_bucket_sweep.py` that owns argument parsing, the family loop, parallelism and writing parquet/CSV. Tests live in `backtesting/tests/test_bucket_sweep.py` and exercise the library on synthetic frames, so they need no data workbook.

**Tech Stack:** Python ≥3.10, pandas ≥2.0, numpy ≥1.24, pyarrow ≥12.0 (parquet), pytest ≥8.0. All already in `pyproject.toml` — **no new dependencies**.

**Spec:** `docs/superpowers/specs/2026-08-01-indicator-bucket-sweep-design.md`

## Global Constraints

- **NO LOOK-AHEAD.** Every quantity determining a row's bucket must derive from dates strictly before that row's date. Bucket cutoffs are **expanding quantiles over strictly prior dates** — never `panel[col].quantile(q)` over the whole panel. Task 2 contains the binding test; it must never be weakened or skipped.
- The `backtesting` package is **offline only** and runs in place from the repo root. It must never be imported by `src/crudewatch/` or `app/`.
- The library module performs **no file IO** and **no printing**. All IO lives in `scripts/run_bucket_sweep.py`.
- Forward outcome definition is fixed: `fwd_h = close[t+h] − open[t+1]`, produced by `crudewatch.research.targets.add_forward_returns` with `horizons=(1, 2, 3, 5, 10, 15, 20)`.
- Bucket labels are exactly `("low", "mid", "high")` for the default 3 buckets. `MISSING_CODE = -1` marks a row that cannot be bucketed (inside warmup, or degenerate edges).
- Default combination depth is `max_k = 4`; values above 4 are refused unless `--force`.
- Defaults: `min_samples = 30`, `min_history = 2000`, `n_buckets = 3`.
- Indicator pool is `crudewatch.research.features.FEATURE_NAMES` (all 24), never a hand-picked subset.
- **Theme exclusivity.** Every indicator belongs to exactly one of 7 themes, and a combination contains **at most one indicator per theme**. All cross-theme combinations are swept; no within-theme pair ever is. This yields 5,015 combinations and 321,975 cells per family at `max_k=4`.
- Combinations are unordered, deterministic in order, and each appears exactly once.
- Output directory default: `docs/reports/bucket_sweep/`.
- Scripts insert both `ROOT / "src"` and `ROOT` onto `sys.path` before importing, matching `scripts/run_research.py:22-24`.

---

### Task 1: Point-in-time cutoffs from strictly prior dates

**Files:**
- Create: `backtesting/research/bucket_sweep.py`
- Test: `backtesting/tests/test_bucket_sweep.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `BUCKET_LABELS: tuple[str, ...] = ("low", "mid", "high")`
  - `MISSING_CODE: int = -1`
  - `class InsufficientData(RuntimeError)`
  - `expanding_cutoffs(panel: pd.DataFrame, indicators: Sequence[str], n_buckets: int = 3, min_history: int = 2000, date_col: str = "date") -> pd.DataFrame` — tidy frame with columns `date, indicator, edge_index, value`, one row per (unique date × indicator × interior edge). `value` is NaN where prior history is shorter than `min_history`.

- [ ] **Step 1: Write the failing test**

Create `backtesting/tests/test_bucket_sweep.py`:

```python
"""Tests for the offline indicator bucket sweep.

The load-bearing property under test is that nothing which decides a row's
bucket may depend on that row's date or any later date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.research.bucket_sweep import BUCKET_LABELS, MISSING_CODE, expanding_cutoffs


def _dated(values, contract="A", start="2015-01-01"):
    """One row per business day, one contract, with the given indicator values."""
    return pd.DataFrame(
        {
            "date": pd.bdate_range(start, periods=len(values)),
            "contract": contract,
            "ind": np.asarray(values, dtype=float),
        }
    )


def test_cutoffs_use_only_strictly_prior_dates():
    """Edges applied on date d are the quantiles of the rows before d, nothing else."""
    values = [5, 1, 9, 3, 7, 2, 8, 4, 6, 0]
    panel = _dated(values)

    cuts = expanding_cutoffs(panel, ["ind"], n_buckets=3, min_history=3)
    on_date_3 = cuts[cuts["date"] == panel["date"].iloc[3]].sort_values("edge_index")

    # Prior rows are [5, 1, 9]; linear-interpolated 1/3 and 2/3 quantiles.
    assert on_date_3["value"].tolist() == pytest.approx([3.666667, 6.333333], abs=1e-5)


def test_cutoffs_are_nan_inside_the_warmup():
    panel = _dated([5, 1, 9, 3, 7, 2, 8, 4, 6, 0])

    cuts = expanding_cutoffs(panel, ["ind"], n_buckets=3, min_history=4)

    early = cuts[cuts["date"].isin(panel["date"].iloc[:4])]
    assert early["value"].isna().all()
    late = cuts[cuts["date"] == panel["date"].iloc[4]]
    assert late["value"].notna().all()


def test_same_date_rows_do_not_influence_each_other():
    """Two contracts share every date; changing one must not move the other's edges."""
    a = _dated([5, 1, 9, 3, 7, 2, 8, 4, 6, 0], contract="A")
    b = _dated([4, 2, 8, 1, 6, 3, 9, 5, 7, 1], contract="B")
    panel = pd.concat([a, b], ignore_index=True)

    base = expanding_cutoffs(panel, ["ind"], n_buckets=3, min_history=4)

    bumped = panel.copy()
    last_date = panel["date"].max()
    bumped.loc[(bumped["contract"] == "B") & (bumped["date"] == last_date), "ind"] = 999.0
    after = expanding_cutoffs(bumped, ["ind"], n_buckets=3, min_history=4)

    pd.testing.assert_frame_equal(base, after)


def test_module_constants():
    assert BUCKET_LABELS == ("low", "mid", "high")
    assert MISSING_CODE == -1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backtesting/tests/test_bucket_sweep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtesting.research.bucket_sweep'`

- [ ] **Step 3: Write minimal implementation**

Create `backtesting/research/bucket_sweep.py`:

```python
"""Indicator bucket sweep (offline): forward outcomes per joint indicator state.

Buckets each continuous indicator into quantile terciles, forms every
combination of up to ``max_k`` indicators, and reports the distribution of
forward price differences inside each joint bucket cell.

**No look-ahead.** Bucket cutoffs are expanding quantiles over *strictly prior
dates*: the edges applied to a row dated ``d`` come from rows dated ``< d``
only. A plain ``panel[col].quantile(q)`` would silently calibrate on the whole
history, including the future of every early row — that is the mistake this
module exists to avoid. Features are as-of ``t`` by construction and forward
outcomes never enter bucketing, so the appended-future-bars test in the suite
covers the whole chain.

This module performs no file IO and no printing — ``scripts/run_bucket_sweep.py``
owns all of that.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

BUCKET_LABELS: tuple[str, ...] = ("low", "mid", "high")
MISSING_CODE: int = -1


class InsufficientData(RuntimeError):
    """Raised when a family has too few usable rows to sweep meaningfully."""


def _interior_quantiles(n_buckets: int) -> list[float]:
    if n_buckets < 2:
        raise ValueError(f"n_buckets must be at least 2, got {n_buckets}")
    return [i / n_buckets for i in range(1, n_buckets)]


def expanding_cutoffs(
    panel: pd.DataFrame,
    indicators: Sequence[str],
    n_buckets: int = 3,
    min_history: int = 2000,
    date_col: str = "date",
) -> pd.DataFrame:
    """Per-date interior quantile edges computed from strictly prior dates.

    The panel is sorted by date, so for the first row of any date every earlier
    row is strictly earlier in time. Reading an expanding quantile at the index
    just *before* that first row therefore yields a quantile over ``date < d``
    only — which also means sibling contracts trading on the same date cannot
    influence one another's edges.

    Returns a tidy ``date, indicator, edge_index, value`` frame. ``value`` is
    NaN while fewer than ``min_history`` prior rows exist.
    """
    quantiles = _interior_quantiles(n_buckets)
    ordered = panel.sort_values(date_col, kind="mergesort")
    dates = ordered[date_col].to_numpy()

    # Index of the first row of each date, and the dates themselves.
    is_first = np.r_[True, dates[1:] != dates[:-1]]
    first_idx = np.flatnonzero(is_first)
    unique_dates = dates[first_idx]

    # Rows strictly before date d end at first_idx - 1. The very first date has
    # no prior history at all.
    has_prior = first_idx > 0
    prior_end = first_idx - 1

    frames: list[pd.DataFrame] = []
    for name in indicators:
        series = ordered[name].reset_index(drop=True)
        for edge_index, q in enumerate(quantiles):
            expanding = series.expanding(min_periods=min_history).quantile(q).to_numpy()
            values = np.full(len(first_idx), np.nan)
            values[has_prior] = expanding[prior_end[has_prior]]
            frames.append(
                pd.DataFrame(
                    {
                        "date": unique_dates,
                        "indicator": name,
                        "edge_index": edge_index,
                        "value": values,
                    }
                )
            )

    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backtesting/tests/test_bucket_sweep.py -v`
Expected: PASS — 4 passed

- [ ] **Step 5: Commit**

```bash
git add backtesting/research/bucket_sweep.py backtesting/tests/test_bucket_sweep.py
git commit -m "feat(bucket-sweep): point-in-time quantile cutoffs from prior dates only"
```

---

### Task 2: Bucketing, and the look-ahead guard

**Files:**
- Modify: `backtesting/research/bucket_sweep.py`
- Test: `backtesting/tests/test_bucket_sweep.py`

**Interfaces:**
- Consumes: `expanding_cutoffs`, `MISSING_CODE` from Task 1.
- Produces:
  - `bucketize(panel: pd.DataFrame, indicators: Sequence[str], n_buckets: int = 3, min_history: int = 2000, date_col: str = "date") -> tuple[pd.DataFrame, pd.DataFrame]` — returns `(codes, cutoffs)`. `codes` is an `int8` frame indexed like the date-sorted `panel`, one column per indicator, values in `0..n_buckets-1` or `MISSING_CODE`. `cutoffs` is Task 1's tidy frame.

- [ ] **Step 1: Write the failing test**

Append to `backtesting/tests/test_bucket_sweep.py`:

```python
from backtesting.research.bucket_sweep import bucketize


def test_bucket_codes_ignore_future_bars():
    """THE binding test: appending future history must not re-bucket earlier rows.

    This is what separates point-in-time cutoffs from a full-sample
    `.quantile()`. If it ever fails, look-ahead has been reintroduced.
    """
    rng = np.random.default_rng(3)
    past = _dated(rng.normal(size=200))
    codes_past, _ = bucketize(past, ["ind"], n_buckets=3, min_history=20)

    future = _dated(rng.normal(size=80) + 25.0, start="2016-06-01")  # a regime shift
    extended = pd.concat([past, future], ignore_index=True)
    codes_extended, _ = bucketize(extended, ["ind"], n_buckets=3, min_history=20)

    original = codes_past["ind"].to_numpy()
    recomputed = codes_extended["ind"].to_numpy()[: len(past)]
    np.testing.assert_array_equal(original, recomputed)


def test_bucketize_marks_the_warmup_as_missing():
    panel = _dated([5, 1, 9, 3, 7, 2, 8, 4, 6, 0])

    codes, _ = bucketize(panel, ["ind"], n_buckets=3, min_history=4)

    assert (codes["ind"].to_numpy()[:4] == MISSING_CODE).all()
    assert (codes["ind"].to_numpy()[4:] != MISSING_CODE).all()


def test_bucketize_assigns_codes_against_prior_date_edges():
    panel = _dated([5, 1, 9, 3, 7, 2, 8, 4, 6, 0])

    codes, _ = bucketize(panel, ["ind"], n_buckets=3, min_history=3)

    # Row 3 has value 3.0 against edges 3.666667 / 6.333333 -> below both -> low.
    assert codes["ind"].iloc[3] == 0
    # Row 6 has value 8.0; prior [5,1,9,3,7,2] -> edges 2.666667 / 6.333333 -> high.
    assert codes["ind"].iloc[6] == 2


def test_bucketize_ignores_forward_outcome_columns():
    """Perturbing outcomes never changes bucket codes — outcomes cannot leak in."""
    rng = np.random.default_rng(1)
    panel = _dated(rng.normal(size=200))
    codes_a, _ = bucketize(panel, ["ind"], n_buckets=3, min_history=20)

    panel["fwd_1"] = rng.normal(size=200) * 1000.0
    codes_b, _ = bucketize(panel, ["ind"], n_buckets=3, min_history=20)

    pd.testing.assert_frame_equal(codes_a, codes_b)


def test_bucketize_marks_degenerate_edges_as_missing():
    """A constant prior window gives equal edges, which is not a real split."""
    panel = _dated([2.0] * 6 + [1.0, 5.0, 3.0, 4.0])

    codes, _ = bucketize(panel, ["ind"], n_buckets=3, min_history=3)

    # Rows 3-5 look back on a constant window -> both edges are 2.0 -> missing.
    assert (codes["ind"].iloc[3:6] == MISSING_CODE).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backtesting/tests/test_bucket_sweep.py -v`
Expected: FAIL — `ImportError: cannot import name 'bucketize' from 'backtesting.research.bucket_sweep'`

- [ ] **Step 3: Write minimal implementation**

Append to `backtesting/research/bucket_sweep.py`:

```python
def bucketize(
    panel: pd.DataFrame,
    indicators: Sequence[str],
    n_buckets: int = 3,
    min_history: int = 2000,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign each row a bucket code per indicator using that date's prior edges.

    A row's code is the number of its date's edges the value meets or exceeds
    (0 = low, 1 = mid, 2 = high for terciles). Rows whose date has no usable
    edges — still inside the ``min_history`` warmup, or a prior window so
    degenerate the edges are not strictly increasing — get ``MISSING_CODE``
    rather than a fabricated bucket.

    Returns ``(codes, cutoffs)``; ``codes`` is indexed like the date-sorted
    ``panel`` so it can be aligned straight back onto it.
    """
    cutoffs = expanding_cutoffs(panel, indicators, n_buckets, min_history, date_col)
    ordered = panel.sort_values(date_col, kind="mergesort")
    row_dates = ordered[date_col]

    wide = cutoffs.pivot(index="date", columns=["indicator", "edge_index"], values="value")

    code_cols: dict[str, np.ndarray] = {}
    for name in indicators:
        values = ordered[name].to_numpy(dtype=float)
        edges = [
            wide[(name, j)].reindex(row_dates).to_numpy(dtype=float)
            for j in range(n_buckets - 1)
        ]

        usable = ~np.isnan(edges[0])
        for j in range(1, n_buckets - 1):
            usable &= ~np.isnan(edges[j])
            usable &= edges[j] > edges[j - 1]  # strictly increasing, else no real split

        code = np.zeros(len(values), dtype=np.int16)
        for edge in edges:
            code += (values >= edge).astype(np.int16)

        code_cols[name] = np.where(usable, code, MISSING_CODE).astype(np.int8)

    codes = pd.DataFrame(code_cols, index=ordered.index)
    return codes, cutoffs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backtesting/tests/test_bucket_sweep.py -v`
Expected: PASS — 9 passed

- [ ] **Step 5: Commit**

```bash
git add backtesting/research/bucket_sweep.py backtesting/tests/test_bucket_sweep.py
git commit -m "feat(bucket-sweep): bucket assignment with look-ahead guard"
```

---

### Task 3: Per-cell statistics for a single combination

**Files:**
- Modify: `backtesting/research/bucket_sweep.py`
- Test: `backtesting/tests/test_bucket_sweep.py`

**Interfaces:**
- Consumes: `BUCKET_LABELS` from Task 1.
- Produces:
  - `RESULT_COLUMNS: tuple[str, ...] = ("k", "themes", "indicators", "buckets", "horizon", "n", "mean", "median", "std", "hit_rate", "t_stat")`
  - `decode_cell(cell: int, k: int, n_buckets: int) -> str`
  - `cell_stats(data: pd.DataFrame, code: pd.Series, horizons: Sequence[int], min_samples: int) -> pd.DataFrame` — columns `cell, horizon, n, mean, median, std, hit_rate, t_stat`. `data` must contain `fwd_{h}` and `hit_{h}` columns for every horizon.

- [ ] **Step 1: Write the failing test**

Append to `backtesting/tests/test_bucket_sweep.py`:

```python
from backtesting.research.bucket_sweep import cell_stats, decode_cell


def _tiny_data():
    """Six rows, one horizon. Cell 0 = [1, 2, 3]; cell 1 = [-1, -2, 0.5]."""
    fwd = pd.Series([1.0, 2.0, 3.0, -1.0, -2.0, 0.5])
    return pd.DataFrame({"fwd_1": fwd, "hit_1": (fwd > 0).astype(float)})


def test_cell_stats_match_hand_computed_values():
    out = cell_stats(
        _tiny_data(), pd.Series([0, 0, 0, 1, 1, 1]), horizons=[1], min_samples=3
    ).set_index("cell")

    low = out.loc[0]
    assert low["n"] == 3
    assert low["mean"] == pytest.approx(2.0)
    assert low["median"] == pytest.approx(2.0)
    assert low["std"] == pytest.approx(1.0)          # sample std of 1, 2, 3
    assert low["hit_rate"] == pytest.approx(1.0)
    assert low["t_stat"] == pytest.approx(2.0 / (1.0 / np.sqrt(3)))

    mid = out.loc[1]
    assert mid["n"] == 3
    assert mid["mean"] == pytest.approx(-0.8333333, abs=1e-6)
    assert mid["median"] == pytest.approx(-1.0)
    assert mid["std"] == pytest.approx(1.2583057, abs=1e-6)
    assert mid["hit_rate"] == pytest.approx(1 / 3)


def test_cell_stats_drops_cells_below_min_samples():
    """A thin cell is absent from the output entirely, not present with NaN."""
    out = cell_stats(
        _tiny_data(), pd.Series([0, 0, 0, 0, 0, 1]), horizons=[1], min_samples=3
    )

    assert out["cell"].tolist() == [0]


def test_decode_cell_maps_joint_code_to_labels():
    # cell = b0 + 3*b1 + 9*b2 with b0=2 (high), b1=0 (low), b2=1 (mid)
    assert decode_cell(2 + 3 * 0 + 9 * 1, k=3, n_buckets=3) == "high|low|mid"
    assert decode_cell(0, k=1, n_buckets=3) == "low"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backtesting/tests/test_bucket_sweep.py -v`
Expected: FAIL — `ImportError: cannot import name 'cell_stats' from 'backtesting.research.bucket_sweep'`

- [ ] **Step 3: Write minimal implementation**

Append to `backtesting/research/bucket_sweep.py`:

```python
RESULT_COLUMNS: tuple[str, ...] = (
    "k", "themes", "indicators", "buckets", "horizon",
    "n", "mean", "median", "std", "hit_rate", "t_stat",
)


def decode_cell(cell: int, k: int, n_buckets: int) -> str:
    """Turn a joint integer cell code back into ``|``-joined bucket labels.

    Inverse of the mixed-radix encoding ``b0 + n*b1 + n^2*b2 + ...`` used by the
    sweep, so position ``i`` of the result is the bucket of the ``i``-th
    indicator of the combination.
    """
    labels = [BUCKET_LABELS[(cell // n_buckets**i) % n_buckets] for i in range(k)]
    return "|".join(labels)


def cell_stats(
    data: pd.DataFrame,
    code: pd.Series,
    horizons: Sequence[int],
    min_samples: int,
) -> pd.DataFrame:
    """Per-cell forward-outcome statistics for one combination, all horizons.

    ``data`` holds ``fwd_{h}`` (forward price difference) and ``hit_{h}``
    (1.0 where ``fwd_{h} > 0``) columns; ``code`` is the joint integer cell per
    row. Cells with fewer than ``min_samples`` rows are omitted entirely rather
    than emitted with NaN statistics.
    """
    agg_map: dict[str, list[str]] = {}
    for h in horizons:
        agg_map[f"fwd_{h}"] = ["count", "mean", "median", "std"]
        agg_map[f"hit_{h}"] = ["mean"]

    desc = data.groupby(code, observed=True, sort=False).agg(agg_map)

    frames: list[pd.DataFrame] = []
    for h in horizons:
        keep = desc[(f"fwd_{h}", "count")] >= min_samples
        if not keep.any():
            continue
        kept = desc.loc[keep]
        n = kept[(f"fwd_{h}", "count")].to_numpy(dtype=float)
        mean = kept[(f"fwd_{h}", "mean")].to_numpy(dtype=float)
        std = kept[(f"fwd_{h}", "std")].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = mean / (std / np.sqrt(n))
        frames.append(
            pd.DataFrame(
                {
                    "cell": kept.index.to_numpy(),
                    "horizon": h,
                    "n": n.astype(np.int64),
                    "mean": mean,
                    "median": kept[(f"fwd_{h}", "median")].to_numpy(dtype=float),
                    "std": std,
                    "hit_rate": kept[(f"hit_{h}", "mean")].to_numpy(dtype=float),
                    "t_stat": t_stat,
                }
            )
        )

    if not frames:
        return pd.DataFrame(
            {c: pd.Series(dtype="float64") for c in
             ("cell", "horizon", "n", "mean", "median", "std", "hit_rate", "t_stat")}
        )
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backtesting/tests/test_bucket_sweep.py -v`
Expected: PASS — 12 passed

- [ ] **Step 5: Commit**

```bash
git add backtesting/research/bucket_sweep.py backtesting/tests/test_bucket_sweep.py
git commit -m "feat(bucket-sweep): per-cell forward outcome statistics"
```

---

### Task 4: Themes and cross-theme combinations

**Files:**
- Modify: `backtesting/research/bucket_sweep.py`
- Test: `backtesting/tests/test_bucket_sweep.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `THEMES: dict[str, tuple[str, ...]]` — theme name → its indicator names. Exactly 7 entries partitioning all 24 of `FEATURE_NAMES`.
  - `INDICATOR_THEME: dict[str, str]` — the inverted map, indicator → theme.
  - `theme_combinations(indicators: Sequence[str], max_k: int, theme_of: Mapping[str, str] = INDICATOR_THEME) -> Iterator[tuple[str, ...]]` — every cross-theme combination of size 1..`max_k`, at most one indicator per theme, each unordered combination yielded once in deterministic order.
  - `count_theme_combinations(indicators: Sequence[str], max_k: int, theme_of: Mapping[str, str] = INDICATOR_THEME) -> int` — the count, without materialising the combinations (used for progress totals and CLI validation).

- [ ] **Step 1: Write the failing test**

Append to `backtesting/tests/test_bucket_sweep.py`:

```python
from crudewatch.research.features import FEATURE_NAMES

from backtesting.research.bucket_sweep import (
    INDICATOR_THEME,
    THEMES,
    count_theme_combinations,
    theme_combinations,
)


def test_themes_partition_every_indicator_exactly_once():
    """Adding an indicator to FEATURES without a theme must fail here, loudly."""
    assigned = [name for names in THEMES.values() for name in names]

    assert len(assigned) == len(set(assigned)), "an indicator appears in two themes"
    assert set(assigned) == set(FEATURE_NAMES)
    assert len(THEMES) == 7
    assert INDICATOR_THEME["z_20"] == "level"
    assert INDICATOR_THEME["mom_10"] == "direction"
    assert INDICATOR_THEME["vol_ratio"] == "volatility"


def test_no_combination_mixes_two_indicators_of_one_theme():
    for combo in theme_combinations(FEATURE_NAMES, max_k=4):
        themes = [INDICATOR_THEME[name] for name in combo]
        assert len(themes) == len(set(themes)), f"{combo} repeats a theme"


def test_combination_count_matches_symmetric_polynomial():
    """Theme sizes [6, 6, 4, 3, 2, 2, 1] -> e1..e4 = 24, 235, 1212, 3544."""
    per_k = {k: 0 for k in range(1, 5)}
    for combo in theme_combinations(FEATURE_NAMES, max_k=4):
        per_k[len(combo)] += 1

    assert per_k == {1: 24, 2: 235, 3: 1212, 4: 3544}
    assert sum(per_k.values()) == 5015
    assert count_theme_combinations(FEATURE_NAMES, max_k=4) == 5015


def test_combinations_are_unordered_and_unique():
    small = ["a1", "a2", "b1", "c1"]
    theme_of = {"a1": "A", "a2": "A", "b1": "B", "c1": "C"}

    combos = list(theme_combinations(small, max_k=3, theme_of=theme_of))

    # (1 + 2x)(1 + x)(1 + x) = 1 + 4x + 5x^2 + 2x^3 -> e1=4, e2=5, e3=2
    assert len(combos) == 11
    assert len(set(combos)) == len(combos)
    assert ("a1", "a2") not in set(combos)      # same theme, never emitted
    assert ("a1", "b1") in set(combos)
    assert ("b1", "a1") not in set(combos)      # unordered: only one orientation
    assert count_theme_combinations(small, max_k=3, theme_of=theme_of) == 11
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backtesting/tests/test_bucket_sweep.py -v`
Expected: FAIL — `ImportError: cannot import name 'INDICATOR_THEME' from 'backtesting.research.bucket_sweep'`

- [ ] **Step 3: Write minimal implementation**

Add to the imports at the top of `backtesting/research/bucket_sweep.py`:

```python
from collections.abc import Callable, Iterator, Mapping, Sequence
from itertools import combinations, product
```

(Replace whatever `collections.abc` import line is currently there.)

Then append:

```python
# Indicator -> theme. Mirrors the scoring engine's blocks (block_trendiness,
# block_direction, block_strength, block_level in crudewatch.scoring.blocks) and
# the app's component screens, so the sweep speaks the project's vocabulary.
#
# A combination takes at most one indicator per theme. Two indicators from the
# same theme mostly restate each other — z_20 and z_50 are one measurement at
# two windows — so pairing them buys a narrower cell, not a new question.
THEMES: dict[str, tuple[str, ...]] = {
    "level": ("z_10", "z_20", "z_50", "pctb_20_2", "pctb_10_1_5", "keltner_dist_20"),
    "direction": ("slope_20", "macd_hist", "ema_align", "mom_5", "mom_10", "mom_20"),
    "exhaustion": ("rsi_div_14", "macd_div", "mom_decel_10", "er_drop_20"),
    "regime": ("er_20", "variance_ratio_5", "autocorr_20"),
    "quality": ("r2_20", "dir_persistence_20"),
    "oscillator": ("rsi_2", "rsi_14"),
    "volatility": ("vol_ratio",),
}

INDICATOR_THEME: dict[str, str] = {
    name: theme for theme, names in THEMES.items() for name in names
}


def _by_theme(
    indicators: Sequence[str], theme_of: Mapping[str, str]
) -> list[tuple[str, list[str]]]:
    """Group ``indicators`` by theme, preserving first-seen order for determinism."""
    grouped: dict[str, list[str]] = {}
    for name in indicators:
        grouped.setdefault(theme_of[name], []).append(name)
    return list(grouped.items())


def theme_combinations(
    indicators: Sequence[str],
    max_k: int,
    theme_of: Mapping[str, str] = INDICATOR_THEME,
) -> Iterator[tuple[str, ...]]:
    """Yield every cross-theme combination of size 1..``max_k``.

    A combination picks k distinct themes and one indicator from each, so no two
    of its indicators share a theme. Themes are chosen with ``combinations`` (so
    unordered, each set once) and the per-theme picks are enumerated in order,
    making the whole sequence deterministic.
    """
    grouped = _by_theme(indicators, theme_of)

    for k in range(1, max_k + 1):
        for theme_group in combinations(grouped, k):
            yield from product(*(names for _, names in theme_group))


def count_theme_combinations(
    indicators: Sequence[str],
    max_k: int,
    theme_of: Mapping[str, str] = INDICATOR_THEME,
) -> int:
    """How many combinations ``theme_combinations`` will yield, without building them.

    This is the sum of the elementary symmetric polynomials ``e_1..e_max_k`` of
    the theme sizes, accumulated by expanding ``prod(1 + size_i * x)``.
    """
    sizes = [len(names) for _, names in _by_theme(indicators, theme_of)]
    poly = [1]  # coefficients of x^0, x^1, ...
    for size in sizes:
        poly = poly + [0]
        for i in range(len(poly) - 1, 0, -1):
            poly[i] += size * poly[i - 1]
    return sum(poly[1 : max_k + 1])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backtesting/tests/test_bucket_sweep.py -v`
Expected: PASS — 16 passed

- [ ] **Step 5: Commit**

```bash
git add backtesting/research/bucket_sweep.py backtesting/tests/test_bucket_sweep.py
git commit -m "feat(bucket-sweep): theme partition and cross-theme combinations"
```

---

### Task 5: The combination sweep

**Files:**
- Modify: `backtesting/research/bucket_sweep.py`
- Test: `backtesting/tests/test_bucket_sweep.py`

**Interfaces:**
- Consumes: `decode_cell`, `cell_stats`, `RESULT_COLUMNS` from Task 3; `INDICATOR_THEME`, `theme_combinations`, `count_theme_combinations` from Task 4.
- Produces:
  - `sweep(codes: pd.DataFrame, forwards: pd.DataFrame, max_k: int = 4, min_samples: int = 30, n_buckets: int = 3, theme_of: Mapping[str, str] = INDICATOR_THEME, progress: Callable[[int, int], None] | None = None) -> pd.DataFrame` — one row per surviving cell × horizon, columns exactly `RESULT_COLUMNS`. `forwards` columns must be named `fwd_{h}`. `codes` must already be free of `MISSING_CODE` (Task 6 filters them out). `progress`, if given, is called as `progress(done, total)` after each combination.

- [ ] **Step 1: Write the failing test**

Append to `backtesting/tests/test_bucket_sweep.py`:

```python
from backtesting.research.bucket_sweep import RESULT_COLUMNS, sweep

_SMALL_THEMES = {"a1": "A", "a2": "A", "b1": "B", "c1": "C"}


def test_sweep_covers_every_cross_theme_combination_exactly_once():
    """Theme sizes [2, 1, 1] at max_k=3 -> 4 + 5 + 2 = 11 combinations."""
    rng = np.random.default_rng(0)
    names = list(_SMALL_THEMES)
    codes = pd.DataFrame({n: np.zeros(50, dtype=np.int8) for n in names})
    forwards = pd.DataFrame({"fwd_1": rng.normal(size=50)})

    out = sweep(codes, forwards, max_k=3, min_samples=1, n_buckets=3,
                theme_of=_SMALL_THEMES)

    assert len(out) == 11        # all codes are 0 -> one cell per combination
    assert out["indicators"].nunique() == 11
    assert not out["indicators"].duplicated().any()
    emitted = set(out["indicators"])
    assert "a1|a2" not in emitted   # same theme, never paired
    assert "a1|b1" in emitted
    assert "b1|a1" not in emitted   # unordered: one orientation only


def test_sweep_labels_themes_alongside_indicators():
    codes = pd.DataFrame({"a1": np.zeros(6, dtype=np.int8),
                          "b1": np.zeros(6, dtype=np.int8)})
    forwards = pd.DataFrame({"fwd_1": [1.0, 2.0, 3.0, -1.0, -2.0, 0.5]})

    out = sweep(codes, forwards, max_k=2, min_samples=3, n_buckets=3,
                theme_of=_SMALL_THEMES)

    pair = out[out["indicators"] == "a1|b1"]
    assert pair["themes"].iloc[0] == "A|B"


def test_sweep_returns_expected_columns_and_k():
    codes = pd.DataFrame({"a1": np.array([0, 0, 0, 1, 1, 1], dtype=np.int8),
                          "b1": np.zeros(6, dtype=np.int8)})
    forwards = pd.DataFrame({"fwd_1": [1.0, 2.0, 3.0, -1.0, -2.0, 0.5]})

    out = sweep(codes, forwards, max_k=2, min_samples=3, n_buckets=3,
                theme_of=_SMALL_THEMES)

    assert list(out.columns) == list(RESULT_COLUMNS)
    assert set(out["k"]) == {1, 2}
    single = out[(out["indicators"] == "a1") & (out["buckets"] == "low")]
    assert single["mean"].iloc[0] == pytest.approx(2.0)


def test_sweep_reports_progress():
    codes = pd.DataFrame({"a1": np.zeros(10, dtype=np.int8),
                          "b1": np.zeros(10, dtype=np.int8)})
    forwards = pd.DataFrame({"fwd_1": np.arange(10, dtype=float)})
    seen: list[tuple[int, int]] = []

    sweep(codes, forwards, max_k=2, min_samples=1, n_buckets=3,
          theme_of=_SMALL_THEMES,
          progress=lambda done, total: seen.append((done, total)))

    assert seen == [(1, 3), (2, 3), (3, 3)]  # a1, b1, then (a1, b1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backtesting/tests/test_bucket_sweep.py -v`
Expected: FAIL — `ImportError: cannot import name 'sweep' from 'backtesting.research.bucket_sweep'`

- [ ] **Step 3: Write minimal implementation**

Append to `backtesting/research/bucket_sweep.py`:

```python
def sweep(
    codes: pd.DataFrame,
    forwards: pd.DataFrame,
    max_k: int = 4,
    min_samples: int = 30,
    n_buckets: int = 3,
    theme_of: Mapping[str, str] = INDICATOR_THEME,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Statistics for every joint bucket cell of every cross-theme combination.

    Combinations come from :func:`theme_combinations`, so each is unordered,
    appears once, and never contains two indicators of the same theme. The joint
    cell of a combination is the mixed-radix code ``b0 + n*b1 + n^2*b2 + ...``,
    which collapses a k-way grouping into a single integer key — grouping on one
    integer column is markedly faster in pandas than grouping on k separate keys.
    """
    names = list(codes.columns)
    horizons = [int(c.split("_")[1]) for c in forwards.columns]

    hits = (forwards > 0).astype(float)
    hits.columns = [f"hit_{h}" for h in horizons]
    data = pd.concat([forwards, hits], axis=1)

    total = count_theme_combinations(names, max_k, theme_of)
    done = 0
    parts: list[pd.DataFrame] = []

    for combo in theme_combinations(names, max_k, theme_of):
        k = len(combo)
        code = codes[combo[0]].astype(np.int32)
        for i, col in enumerate(combo[1:], start=1):
            code = code + codes[col].astype(np.int32) * (n_buckets**i)

        stats = cell_stats(data, code, horizons, min_samples)
        done += 1
        if progress is not None:
            progress(done, total)
        if stats.empty:
            continue

        stats["k"] = k
        stats["themes"] = "|".join(theme_of[name] for name in combo)
        stats["indicators"] = "|".join(combo)
        stats["buckets"] = [decode_cell(int(c), k, n_buckets) for c in stats["cell"]]
        parts.append(stats.drop(columns="cell"))

    if not parts:
        return pd.DataFrame(
            {c: pd.Series(dtype="object"
                          if c in ("themes", "indicators", "buckets") else "float64")
             for c in RESULT_COLUMNS}
        )
    return pd.concat(parts, ignore_index=True)[list(RESULT_COLUMNS)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backtesting/tests/test_bucket_sweep.py -v`
Expected: PASS — 20 passed

- [ ] **Step 5: Commit**

```bash
git add backtesting/research/bucket_sweep.py backtesting/tests/test_bucket_sweep.py
git commit -m "feat(bucket-sweep): combination sweep over indicator buckets"
```

---

### Task 6: Family orchestration

**Files:**
- Modify: `backtesting/research/bucket_sweep.py`
- Modify: `backtesting/research/__init__.py`
- Test: `backtesting/tests/test_bucket_sweep.py`

**Interfaces:**
- Consumes: `bucketize`, `sweep`, `InsufficientData`, `MISSING_CODE` from Tasks 1–5.
- Produces:
  - `HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 10, 15, 20)`
  - `run_family(family: str, frame: pd.DataFrame, horizons: Sequence[int] = HORIZONS, n_buckets: int = 3, max_k: int = 4, min_samples: int = 30, min_history: int = 2000, progress: Callable[[int, int], None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]` — returns `(results, cutoffs)`, both with a leading `family` column. Raises `InsufficientData` when too few rows survive.

- [ ] **Step 1: Write the failing test**

Append to `backtesting/tests/test_bucket_sweep.py`:

```python
from backtesting.research.bucket_sweep import HORIZONS, InsufficientData, run_family


def _synthetic_frame(n_contracts: int = 6, n_rows: int = 400) -> pd.DataFrame:
    """A price panel shaped like a family frame: date, contract, open, close."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2015-01-01", periods=n_rows)
    parts = []
    for c in range(n_contracts):
        close = 50.0 + np.cumsum(rng.normal(0, 0.5, size=n_rows))
        parts.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "contract": f"C{c}",
                    "open": close + rng.normal(0, 0.05, size=n_rows),
                    "close": close,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_horizons_match_the_spec():
    assert HORIZONS == (1, 2, 3, 5, 10, 15, 20)


def test_run_family_produces_labelled_results():
    results, cutoffs = run_family(
        "synthetic", _synthetic_frame(),
        max_k=2, min_samples=5, n_buckets=3, min_history=200,
    )

    assert not results.empty
    assert results.columns[0] == "family"
    assert set(results["family"]) == {"synthetic"}
    assert set(results["horizon"]) <= set(HORIZONS)
    assert set(results["k"]) == {1, 2}
    assert (results["n"] >= 5).all()
    assert cutoffs.columns[0] == "family"
    # Every k=1 combination names exactly one indicator.
    singles = results[results["k"] == 1]
    assert not singles["indicators"].str.contains(r"\|").any()
    # No MISSING_CODE ever reaches a bucket label.
    assert set(results["buckets"].str.split("|").explode()) <= set(BUCKET_LABELS)
    # Themes are real and never repeat inside a combination.
    assert set(results["themes"].str.split("|").explode()) <= set(THEMES)
    repeated = results["themes"].str.split("|").apply(lambda t: len(t) != len(set(t)))
    assert not repeated.any()


def test_run_family_raises_on_a_thin_panel():
    with pytest.raises(InsufficientData):
        run_family("thin", _synthetic_frame(n_contracts=1, n_rows=130),
                   max_k=4, min_samples=30, n_buckets=3, min_history=2000)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backtesting/tests/test_bucket_sweep.py -v`
Expected: FAIL — `ImportError: cannot import name 'HORIZONS' from 'backtesting.research.bucket_sweep'`

- [ ] **Step 3: Write minimal implementation**

Add to the imports at the top of `backtesting/research/bucket_sweep.py`, after the numpy/pandas imports:

```python
from crudewatch.research.features import FEATURE_NAMES, add_features
from crudewatch.research.targets import add_forward_returns
```

Then append:

```python
HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 10, 15, 20)


def run_family(
    family: str,
    frame: pd.DataFrame,
    horizons: Sequence[int] = HORIZONS,
    n_buckets: int = 3,
    max_k: int = 4,
    min_samples: int = 30,
    min_history: int = 2000,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full sweep for one contract family: features -> outcomes -> buckets -> cells.

    Rows with a NaN in any indicator or any forward column are dropped up front,
    then rows the point-in-time bucketing could not code (quantile warmup or
    degenerate edges) are dropped too. What survives is a single complete-case
    panel, so every combination is measured on an identical sample and cells are
    comparable across combinations.
    """
    featured = add_features(frame)
    full = add_forward_returns(featured, horizons=tuple(horizons))

    fwd_cols = [f"fwd_{h}" for h in horizons]
    panel = full.dropna(subset=list(FEATURE_NAMES) + fwd_cols)

    if len(panel) < min_history:
        raise InsufficientData(
            f"{family}: {len(panel)} complete-case rows, fewer than the "
            f"min_history={min_history} needed before any date can be bucketed"
        )

    codes, cutoffs = bucketize(panel, FEATURE_NAMES, n_buckets, min_history)
    codeable = (codes != MISSING_CODE).all(axis=1)
    codes = codes.loc[codeable]
    panel = panel.loc[codes.index]

    required = min_samples * n_buckets**max_k
    if len(codes) < required:
        raise InsufficientData(
            f"{family}: {len(codes)} codeable rows after warmup, need at least "
            f"{required} for max_k={max_k}, n_buckets={n_buckets}, "
            f"min_samples={min_samples}"
        )

    results = sweep(
        codes, panel[fwd_cols], max_k=max_k, min_samples=min_samples,
        n_buckets=n_buckets, progress=progress,
    )

    results.insert(0, "family", family)
    cutoffs.insert(0, "family", family)
    return results, cutoffs
```

- [ ] **Step 4: Document the module in the package docstring**

Replace the contents of `backtesting/research/__init__.py` with:

```python
"""Walk-forward research, strategy simulation and their HTML reports (offline).

Consumers import the concrete submodules directly (``evaluate``, ``regime``,
``strategy``, ``diagnostics``, ``composite``, ``costs``, ``quality``, ``report``,
``strategy_report``, ``bucket_sweep``). The shared, app-facing feature/dataset
pipeline lives in :mod:`crudewatch.research` instead.
"""
```

- [ ] **Step 5: Run the full test file to verify it passes**

Run: `uv run pytest backtesting/tests/test_bucket_sweep.py -v`
Expected: PASS — 23 passed

- [ ] **Step 6: Commit**

```bash
git add backtesting/research/bucket_sweep.py backtesting/research/__init__.py backtesting/tests/test_bucket_sweep.py
git commit -m "feat(bucket-sweep): per-family orchestration over features and targets"
```

---

### Task 7: CLI runner

**Files:**
- Create: `scripts/run_bucket_sweep.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `HORIZONS`, `InsufficientData`, `run_family` from Task 6.
- Produces: an executable script. No importable API — nothing depends on this task.

- [ ] **Step 1: Write the script**

Create `scripts/run_bucket_sweep.py`:

```python
"""Sweep forward outcomes across every combination of indicator bucket states.

    uv run python scripts/run_bucket_sweep.py
    uv run python scripts/run_bucket_sweep.py --families quarterly flies --max-k 2
    uv run python scripts/run_bucket_sweep.py --jobs 4 --resume

For each contract family: bucket all 24 indicators into terciles whose cutoffs
are expanding quantiles over *strictly prior dates* (no look-ahead), form every
cross-theme combination of up to --max-k indicators (at most one indicator per
theme, so near-duplicates like z_20 with z_50 are never paired), and report the
distribution of forward price differences (close[t+h] - open[t+1]) inside each
joint bucket cell, for h in 1, 2, 3, 5, 10, 15, 20.

Writes, per family, to docs/reports/bucket_sweep/:

    <family>.parquet              one row per surviving cell x horizon
    <family>_cutoffs.parquet      the per-date edge series behind every bucket
    <family>_cutoffs_latest.csv   the edges in force on the final date

plus a pooled top_cells.csv of the strongest cells by |t_stat|.

Descriptive only. Every input is point-in-time, but cells are chosen by
inspection: with ~322k cells per family thousands clear |t| > 3 on noise alone,
and overlapping forward windows inflate it further. Rank with it, then confirm
on held-out dates before believing it.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # the offline `backtesting` package lives at repo root

from crudewatch.data_preparation import build_all  # noqa: E402
from crudewatch.infra import load_raw  # noqa: E402
from crudewatch.research.features import FEATURE_NAMES  # noqa: E402
from backtesting.research.bucket_sweep import (  # noqa: E402
    HORIZONS,
    THEMES,
    InsufficientData,
    count_theme_combinations,
    run_family,
)

FRAME_NAMES = [
    "outrights", "calendars", "cracks", "brent_wti",
    "quarterly", "semestral", "yearly", "flies",
]
RAW = ROOT / "data" / "raw_files.xlsx"
OUT_DIR = ROOT / "docs" / "reports" / "bucket_sweep"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--families", nargs="+", default=FRAME_NAMES, choices=FRAME_NAMES)
    p.add_argument("--max-k", type=int, default=4, help="combination depth (default 4)")
    p.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    p.add_argument("--n-buckets", type=int, default=3, help="buckets per indicator")
    p.add_argument("--min-samples", type=int, default=30, help="minimum rows per cell")
    p.add_argument("--min-history", type=int, default=2000,
                   help="prior rows required before a date can be bucketed")
    p.add_argument("--top-n", type=int, default=500, help="rows in top_cells.csv")
    p.add_argument("--jobs", type=int, default=1, help="worker processes")
    p.add_argument("--resume", action="store_true", help="skip families already written")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--force", action="store_true", help="allow --max-k above 4")
    return p.parse_args(argv)


def validate(args: argparse.Namespace) -> None:
    """Fail fast on inputs that would waste hours or blow up the output."""
    if args.max_k < 1:
        raise SystemExit("--max-k must be at least 1")
    if args.max_k > len(THEMES):
        raise SystemExit(
            f"--max-k {args.max_k} exceeds the {len(THEMES)} themes. A combination "
            f"holds at most one indicator per theme, so no combination can be "
            f"larger than {len(THEMES)}."
        )
    if args.max_k > 4 and not args.force:
        cells = 0
        for k in range(1, args.max_k + 1):
            n_k = (count_theme_combinations(FEATURE_NAMES, k)
                   - count_theme_combinations(FEATURE_NAMES, k - 1))
            cells += n_k * args.n_buckets**k
        raise SystemExit(
            f"--max-k {args.max_k} would produce {cells:,} cells per family. "
            f"Re-run with --force if that is genuinely what you want."
        )
    if not RAW.exists():
        raise SystemExit(f"Missing raw workbook: {RAW}")


def sweep_one(family: str, frame: pd.DataFrame, args: argparse.Namespace):
    """Run one family, returning (family, results, cutoffs, error_message)."""
    try:
        results, cutoffs = run_family(
            family,
            frame,
            horizons=args.horizons,
            n_buckets=args.n_buckets,
            max_k=args.max_k,
            min_samples=args.min_samples,
            min_history=args.min_history,
        )
    except InsufficientData as exc:
        return family, None, None, str(exc)
    return family, results, cutoffs, None


def write_family(out_dir: Path, family: str, results: pd.DataFrame,
                 cutoffs: pd.DataFrame) -> Path:
    """Write one family's outputs as soon as it finishes, so --resume can skip it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{family}.parquet"
    results.to_parquet(path, index=False)
    cutoffs.to_parquet(out_dir / f"{family}_cutoffs.parquet", index=False)
    latest = cutoffs[cutoffs["date"] == cutoffs["date"].max()]
    latest.to_csv(out_dir / f"{family}_cutoffs_latest.csv", index=False)
    return path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate(args)

    families = list(args.families)
    if args.resume:
        skipped = [f for f in families if (args.out_dir / f"{f}.parquet").exists()]
        for f in skipped:
            print(f"[resume] {f}: already written, skipping")
        families = [f for f in families if f not in skipped]
    if not families:
        print("Nothing to do.")
        return

    print(f"Loading raw workbook: {RAW}")
    frames = build_all(load_raw(RAW))

    written: list[Path] = []
    started = time.time()

    def report(family: str, results: pd.DataFrame, path: Path) -> None:
        print(f"[done] {family}: {len(results):,} cells -> {path}  "
              f"({time.time() - started:.0f}s elapsed)")

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(sweep_one, f, frames[f], args): f for f in families}
            for fut in as_completed(futures):
                family, results, cutoffs, err = fut.result()
                if err:
                    print(f"[skip] {err}")
                    continue
                path = write_family(args.out_dir, family, results, cutoffs)
                written.append(path)
                report(family, results, path)
    else:
        for family in families:
            print(f"[run ] {family} ...")
            _, results, cutoffs, err = sweep_one(family, frames[family], args)
            if err:
                print(f"[skip] {err}")
                continue
            path = write_family(args.out_dir, family, results, cutoffs)
            written.append(path)
            report(family, results, path)

    if not written:
        print("No results produced.")
        return

    pooled = pd.concat([pd.read_parquet(p) for p in written], ignore_index=True)
    top = pooled.reindex(
        pooled["t_stat"].abs().sort_values(ascending=False).index
    ).head(args.top_n)
    top_path = args.out_dir / "top_cells.csv"
    top.to_csv(top_path, index=False)
    print(f"Wrote {top_path}  ({len(top)} rows)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the CLI refuses an unsafe depth**

Run: `uv run python scripts/run_bucket_sweep.py --max-k 5`
Expected: exits non-zero printing `--max-k 5 would produce ... cells per family. Re-run with --force if that is genuinely what you want.`

- [ ] **Step 3: Verify a real end-to-end run on one family at shallow depth**

Run: `uv run python scripts/run_bucket_sweep.py --families flies --max-k 2 --min-samples 50`
Expected: prints `[run ] flies ...` then `[done] flies: <N> cells -> .../flies.parquet`, and creates `docs/reports/bucket_sweep/flies.parquet`, `flies_cutoffs.parquet`, `flies_cutoffs_latest.csv` and `top_cells.csv`.

- [ ] **Step 4: Inspect the output to confirm the schema**

Run:
```bash
uv run python -c "
import pandas as pd
d = pd.read_parquet('docs/reports/bucket_sweep/flies.parquet')
print(d.columns.tolist()); print(d.head()); print(d['n'].min(), sorted(d['horizon'].unique()))
c = pd.read_parquet('docs/reports/bucket_sweep/flies_cutoffs.parquet')
print(c.columns.tolist()); print(c['date'].nunique(), 'dates')
"
```
Expected: results columns `['family','k','themes','indicators','buckets','horizon','n','mean','median','std','hit_rate','t_stat']`, `n.min() >= 50`, horizons `[1, 2, 3, 5, 10, 15, 20]`; cutoff columns `['family','date','indicator','edge_index','value']` across many dates.

- [ ] **Step 5: Confirm the cutoffs actually move over time**

Run:
```bash
uv run python -c "
import pandas as pd
c = pd.read_parquet('docs/reports/bucket_sweep/flies_cutoffs.parquet')
s = c[(c['indicator']=='rsi_14') & (c['edge_index']==0)].dropna(subset=['value'])
print(s['value'].describe())
assert s['value'].nunique() > 1, 'cutoffs are constant - bucketing may not be expanding'
print('OK: cutoffs vary over time')
"
```
Expected: prints a describe with `std > 0` and `OK: cutoffs vary over time`. A constant series here would mean the point-in-time calibration silently degenerated into a fixed split.

- [ ] **Step 6: Document the script in the README**

In `README.md`, in the `## Layout` code block, change the `scripts/` line to:

```
scripts/              # run_backtests.py / run_research.py / run_strategy.py / run_bucket_sweep.py (offline report generators)
```

Then add this section immediately before `## Setup`:

````markdown
## Indicator bucket sweep (offline)

Descriptive study of what followed each *joint indicator state*. Buckets all 24
indicators into terciles, forms every combination of up to four of them, and
reports forward price differences (`close[t+h] − open[t+1]`, for
h = 1, 2, 3, 5, 10, 15, 20) per bucket cell.

Indicators are grouped into 7 themes (level, direction, exhaustion, regime,
quality, oscillator, volatility) and a combination takes **at most one indicator
per theme** — so themes are crossed against each other, but `z_20` is never
paired with `z_50`. That is 5,015 combinations and 321,975 cells per family.

Cutoffs are **expanding quantiles over strictly prior dates**, so a row's bucket
never depends on its own date or any later one — "low" means low relative to
what was knowable at the time.

```bash
uv run python scripts/run_bucket_sweep.py --families flies --max-k 2   # quick look
uv run python scripts/run_bucket_sweep.py --jobs 4 --resume            # full run
```

Output lands in `docs/reports/bucket_sweep/`. A full `--max-k 4` run is ~322k
cells per family and takes hours — use `--jobs` and `--resume`.

> Every input is point-in-time, but cells are picked by inspection, so `t_stat`
> is still selection-biased and thousands of cells clear |t| > 3 by chance.
> Rank candidates with it; confirm them on held-out dates.
````

- [ ] **Step 7: Run the whole suite to confirm nothing regressed**

Run: `uv run pytest`
Expected: PASS — all pre-existing tests plus the 23 new ones.

- [ ] **Step 8: Commit**

```bash
git add scripts/run_bucket_sweep.py README.md
git commit -m "feat(bucket-sweep): CLI runner and README documentation"
```

---

## Verification

After Task 7, all of the following must hold:

- [ ] `uv run pytest` passes with 23 new tests in `backtesting/tests/test_bucket_sweep.py`.
- [ ] `test_bucket_codes_ignore_future_bars` passes — appending future bars does not re-bucket earlier rows.
- [ ] `docs/reports/bucket_sweep/flies.parquet` exists with the exact `RESULT_COLUMNS` schema prefixed by `family`.
- [ ] `flies_cutoffs.parquet` shows edges that **vary across dates** (Task 7 Step 5).
- [ ] `--max-k 5` is refused without `--force`.
- [ ] `--resume` skips a family whose parquet already exists.
- [ ] No file under `src/crudewatch/` or `app/` imports `backtesting.research.bucket_sweep`.
