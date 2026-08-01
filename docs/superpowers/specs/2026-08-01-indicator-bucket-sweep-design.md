# Indicator Bucket Sweep — Design

**Date:** 2026-08-01
**Status:** Approved, pending implementation plan

## Purpose

Measure what happens to price after every observable *combination of indicator
states*. For each contract family, bucket all 24 continuous indicators into
terciles, form every combination of up to four indicators, and for each joint
bucket cell report the distribution of forward price differences at horizons
D+1, +2, +3, +5, +10, +15 and +20.

This is a descriptive study, not a trading system. It answers "when these
indicators are simultaneously in these states, what has historically followed?"
and surfaces candidate conditions for later, honest evaluation.

## Scope

- **Indicators:** all 24 in `FEATURES` (`crudewatch.research.features`).
- **Combination depth:** k = 1, 2, 3, 4. No 5-way or deeper.
- **Families:** all eight — `outrights`, `calendars`, `cracks`, `brent_wti`,
  `quarterly`, `semestral`, `yearly`, `flies` — each swept and reported
  independently. No pooled cross-family table.
- **Horizons:** 1, 2, 3, 5, 10, 15, 20 trading bars.

Explicitly out of scope: position sizing, cost modelling, weight fitting, and
any claim of out-of-sample validity. Cells are selected by inspection, so the
reported statistics are in-sample by construction.

## Combinatorics

24 indicators × 3 buckets, k ≤ 4:

| k | combinations | cells per combination | cells |
|---|---|---|---|
| 1 | 24 | 3 | 72 |
| 2 | 276 | 9 | 2,484 |
| 3 | 2,024 | 27 | 54,648 |
| 4 | 10,626 | 81 | 860,706 |
| **total** | **12,950** | | **917,910** |

×7 horizons ×8 families ≈ 51M output rows before filtering. This is why output
is parquet and why the `--min-samples` floor exists.

**Runtime.** The sweep uses one pandas `groupby` per combination (approach
chosen deliberately over a vectorised `np.bincount` sweep). Expect roughly 7–19
hours for all eight families single-threaded; `--jobs` parallelises across
families and `--resume` makes an interrupted run cheap to restart.

## Architecture

Offline research code lives outside `src/` and runs in place from the repo
root, matching the existing `backtesting/` + `scripts/` split.

```
backtesting/research/bucket_sweep.py    # library: bucketing, sweep, statistics
scripts/run_bucket_sweep.py             # CLI: args, family loop, IO, progress
backtesting/tests/test_bucket_sweep.py  # tests
docs/reports/bucket_sweep/              # output
```

### Units

**`bucket_sweep.bucketize(panel, indicators, n_buckets)`**
Cuts each indicator at its own quantiles over the supplied rows, returning an
`int8` code frame (0 = low, 1 = mid, 2 = high) plus a cutoff table. Depends only
on pandas/numpy. Indicators that are constant over the family are dropped and
reported, never collapsed into a single degenerate bucket.

**`bucket_sweep.sweep(codes, forwards, max_k, min_samples)`**
Iterates combinations of size 1..`max_k`, groups by the joint code, and returns
one row per surviving cell × horizon. Knows nothing about families, files or
indicators beyond their column names.

**`bucket_sweep.run_family(family, ...)`**
Orchestrates load → features → targets → complete-case → bucketize → sweep for
one family, returning the result frame and the cutoff table.

The CLI owns argument parsing, the family loop, parallelism and all file IO; the
library performs no IO. This keeps `sweep` testable on a synthetic frame with no
data workbook present.

## Data flow (per family)

1. `load_raw` → `build_all` → select the family frame.
2. `add_features` → the 24 continuous indicator columns (all as-of `t`).
3. `add_forward_returns(horizons=(1, 2, 3, 5, 10, 15, 20))` → `fwd_1 … fwd_20`.
4. **Complete-case filter** — drop any row with a NaN in the 24 indicator
   columns or the 7 forward columns.
5. **Bucketing** — per family, per indicator, at the 33.3 / 66.7 percentiles of
   the surviving rows.
6. **Sweep** — per combination, one `groupby(code, observed=True, sort=False)`
   aggregating all 7 forward columns at once. Keep cells with `n ≥ min_samples`.
7. **Write** — `<family>.parquet` and `<family>_cutoffs.csv`, written as the
   family completes.

### Forward outcome definition

`fwd_h = close[t + h] − open[t + 1]`.

The indicator state is known at `close[t]`; the outcome is anchored at the next
bar's open, matching the executable basis established in
`crudewatch.research.targets`. Results are therefore directly comparable to the
existing backtest and strategy reports, and carry no same-bar look-ahead.

### Why complete-case

Every combination is measured on an identical row set. Without this, a cell
conditioning on `ema_align` (100-bar warmup) would be evaluated on a different
sample than one conditioning on `rsi_2` (2-bar warmup), and their means would
not be comparable. The cost is the leading warmup (~100 bars) and trailing 20
bars of each contract.

### Why per-family cutoffs

The 24 indicators live on wildly different scales (RSI 0–100, z-scores, ATR
units, ratios), and the eight families differ just as much in volatility and
behaviour. Family-relative quantiles adapt to both without a hand-tuned
threshold table, and guarantee balanced cells at k = 1.

## Output

`docs/reports/bucket_sweep/<family>.parquet`, one row per cell × horizon:

| column | meaning |
|---|---|
| `family` | contract family |
| `k` | number of indicators in the combination |
| `indicators` | `\|`-joined indicator names, lexicographic |
| `buckets` | `\|`-joined bucket labels (`low`/`mid`/`high`), aligned to `indicators` |
| `horizon` | 1, 2, 3, 5, 10, 15 or 20 |
| `n` | rows in the cell |
| `mean` | mean forward difference, price points |
| `median` | median forward difference |
| `std` | standard deviation |
| `hit_rate` | fraction with `fwd_h > 0` |
| `t_stat` | `mean / (std / √n)` |

`<family>_cutoffs.csv` records each indicator's actual threshold values, so any
cell reads back as a concrete rule. `top_cells.csv` collects the strongest cells
by `|t_stat|` across families for inspection.

**Interpreting `t_stat`.** With ~918k cells per family, thousands will show
|t| > 3 by chance alone. It is a ranking aid, not evidence. Overlapping forward
windows also autocorrelate the outcomes, inflating it further. Anything
promising needs separate out-of-sample evaluation.

## Error handling

- Missing `data/raw_files.xlsx` → fail fast before any sweep begins.
- Family with fewer than `min_samples × 81` complete-case rows → warn, skip the
  family, continue. One thin family must not abort a multi-hour run.
- Constant indicator within a family → drop from that family's sweep, record in
  the cutoff table, warn.
- `--max-k > 4` → refuse unless `--force`, reporting the resulting cell count.
- Parquet written per family on completion; `--resume` skips families already on
  disk.

## Testing

`backtesting/tests/test_bucket_sweep.py`, under the configured `testpaths`.

- **Bucketing** — terciles on a known series give the expected codes and roughly
  equal counts; a constant column is dropped rather than collapsed.
- **Statistics** — on a small hand-built panel, `n`, `mean`, `std` and
  `hit_rate` for a named cell match hand-computed values. The core assertion.
- **Combination coverage** — 5 indicators at `max_k=4` emit exactly
  C(5,1)+C(5,2)+C(5,3)+C(5,4) = 30 combinations, no duplicates, no ordering
  variants.
- **Look-ahead guard** — appending future bars leaves already-computed bucket
  codes unchanged, mirroring the existing feature-stability test.
- **min-samples** — cells below the floor are absent from the output, not
  present with NaN.

## Decisions taken

| Decision | Choice | Rationale |
|---|---|---|
| Depth | k ≤ 4 | User's call; `--min-samples` guards thin cells |
| Cutoffs | Per-family quantile terciles | Scale-free, balanced, no hand tuning |
| Anchor | `close[t+h] − open[t+1]` | Matches repo's executable basis |
| Families | All 8, reported separately | Preserves per-family scale and behaviour |
| Output | Full stats, parquet, `n ≥ 30` | Keeps size sane, drops noise-only cells |
| Compute | pandas `groupby` per combination | User's call over vectorised bincount |
