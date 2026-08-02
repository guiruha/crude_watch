# Indicator Bucket Sweep — Design

**Date:** 2026-08-01
**Status:** Implemented on branch `indicator-bucket-sweep` (160 tests passing)

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
- **Theme exclusivity:** each indicator belongs to exactly one theme, and a
  combination may take **at most one indicator per theme**. Every cross-theme
  combination is swept; no within-theme pairing ever is.
- **Families:** all eight — `outrights`, `calendars`, `cracks`, `brent_wti`,
  `quarterly`, `semestral`, `yearly`, `flies` — each swept and reported
  independently. No pooled cross-family table.
- **Horizons:** 1, 2, 3, 5, 10, 15, 20 trading bars.

Explicitly out of scope: position sizing, cost modelling, weight fitting, and
any claim of out-of-sample validity. Cells are selected by inspection, so the
reported statistics remain a *selection*-biased in-sample read even though every
input is point-in-time.

## No look-ahead — the governing constraint

Every quantity that determines a row's bucket must be computable from
information available strictly before that row's date. Four separate places
could leak, and each is closed explicitly:

1. **Features** — as-of `t` by construction (`crudewatch.research.features`
   uses only rolling/EWM windows, never `shift(-k)`), and already covered by an
   existing stability test.
2. **Outcomes** — `fwd_h = close[t+h] − open[t+1]` is strictly future, and never
   participates in bucketing. Enforced by a test that perturbs the forward
   columns and asserts bucket codes are unchanged.
3. **Bucket cutoffs** — **expanding-window quantiles over strictly prior dates**
   (see below). This is the one that a naive implementation gets wrong, because
   the obvious `panel[col].quantile(1/3)` silently uses the whole history.
4. **The complete-case filter itself.** Dropping a row because its *forward*
   columns are NaN removes it from the pooled prior-history quantile pool — and
   whether a row has a valid `fwd_20` depends on data 20 bars later. A contract
   expiring near date `d` would therefore change other contracts' cutoffs at
   `d`. Bucketing runs on the **feature**-complete panel; only the sweep uses
   the forward-complete subset. Guarded by a staggered-contract-tail test.

The binding test for the first three: appending future bars to a panel must
leave every already-computed bucket code byte-identical. The fourth needs its
own test, because a panel whose contracts all share a date range cannot expose
it — the guard extends one contract's tail and asserts earlier buckets hold.

### Point-in-time cutoffs

Sort the family panel by date. For a row on date `d`, its cutoffs are the
quantiles of every row with date **strictly less than `d`** — pooled across the
family's contracts, matching how the buckets are interpreted.

Implementation: take `expanding().quantile(q)` over the date-sorted rows and
read it at the row index immediately *before* each date's first row, then
broadcast that value to all rows sharing that date. Because the panel is sorted
by date, every row at or before that index is strictly earlier, so same-day rows
of sibling contracts cannot leak into one another either.

**Warmup.** A date needs enough prior history for its quantiles to mean
anything. Rows whose prior history is shorter than `--min-history` pooled rows
(default 2000) are dropped, on top of the existing per-contract feature warmup.

**Consequence:** cutoffs drift over time, so the same raw RSI value may be
`mid` in 2014 and `low` in 2023. That is correct — the bucket means "low
relative to what was knowable then," which is the only version of the statement
a trader could have acted on.

## Themes

Each indicator belongs to exactly one theme, and a combination may contain at
most one indicator from each. The taxonomy follows the scoring engine's existing
blocks (`block_trendiness`, `block_direction`, `block_strength`, `block_level`
in `crudewatch.scoring.blocks`) and the app's six component screens, so the
vocabulary matches what the rest of the project already speaks.

| Theme | Indicators | n |
|---|---|---|
| `level` | `z_10`, `z_20`, `z_50`, `pctb_20_2`, `pctb_10_1_5`, `keltner_dist_20` | 6 |
| `direction` | `slope_20`, `macd_hist`, `ema_align`, `mom_5`, `mom_10`, `mom_20` | 6 |
| `exhaustion` | `rsi_div_14`, `macd_div`, `mom_decel_10`, `er_drop_20` | 4 |
| `regime` | `er_20`, `variance_ratio_5`, `autocorr_20` | 3 |
| `quality` | `r2_20`, `dir_persistence_20` | 2 |
| `oscillator` | `rsi_2`, `rsi_14` | 2 |
| `volatility` | `vol_ratio` | 1 |

The partition must be exhaustive and disjoint — every one of the 24 indicators
appears exactly once — and this is asserted in the tests, so adding an indicator
to `FEATURES` without assigning it a theme fails loudly rather than silently
dropping it from the sweep.

**Why.** Two indicators from the same theme largely restate each other:
`z_20` and `z_50` are the same measurement at different windows, `mom_5` and
`mom_10` likewise. Conditioning on both spends a combination slot to learn
almost nothing, and produces cells that look impressively selective while merely
being narrow. Barring within-theme pairs removes those, and what remains is the
genuinely informative question — how themes interact.

## Combinatorics

Choosing k distinct themes and one indicator from each, the number of
combinations of size k is the elementary symmetric polynomial `e_k` of the theme
sizes `[6, 6, 4, 3, 2, 2, 1]`:

| k | combinations | cells per combination | cells |
|---|---|---|---|
| 1 | 24 | 3 | 72 |
| 2 | 235 | 9 | 2,115 |
| 3 | 1,212 | 27 | 32,724 |
| 4 | 3,544 | 81 | 287,064 |
| **total** | **5,015** | | **321,975** |

(Unrestricted, these would be 12,950 combinations and 917,910 cells — theme
exclusivity removes roughly two thirds, almost all of it near-duplicate.)

×7 horizons ×8 families ≈ 18M output rows before filtering. This is why output
is parquet and why the `--min-samples` floor exists.

**Runtime.** The sweep uses one pandas `groupby` per combination (approach
chosen deliberately over a vectorised `np.bincount` sweep). Expect roughly 3–7
hours for all eight families single-threaded; `--jobs` parallelises across
families and `--resume` makes an interrupted run cheap to restart. The expanding
quantile pass adds 48 passes per family (24 indicators × 2 edges) — a couple of
minutes, negligible against the sweep.

## Architecture

Offline research code lives outside `src/` and runs in place from the repo
root, matching the existing `backtesting/` + `scripts/` split.

```
backtesting/research/bucket_sweep.py    # library: cutoffs, bucketing, sweep, statistics
scripts/run_bucket_sweep.py             # CLI: args, family loop, IO, progress
backtesting/tests/test_bucket_sweep.py  # tests
docs/reports/bucket_sweep/              # output
```

### Units

**`expanding_cutoffs(panel, indicators, n_buckets, min_history)`**
Returns, per date, the interior quantile edges computed from strictly prior
dates. The single place where look-ahead could enter, so it is small, isolated
and directly tested.

**`bucketize(panel, indicators, n_buckets, min_history)`**
Applies those per-date cutoffs to produce an `int8` code frame (0 = low,
1 = mid, 2 = high) plus the cutoff time series. Rows inside the warmup, and
indicators whose edges are not strictly increasing on a given date, yield a
sentinel code that excludes the row from that indicator's combinations.

**`theme_combinations(indicators, max_k)`**
Yields every cross-theme combination of size 1..`max_k`, at most one indicator
per theme, each unordered combination once, in a deterministic order. Pure and
tiny, so the exclusivity rule is testable in isolation from the sweep.

**`sweep(codes, forwards, max_k, min_samples)`**
Iterates the combinations from `theme_combinations`, groups by the joint code,
returns one row per surviving cell × horizon. Knows nothing about families,
files or dates.

**`run_family(family, frame, ...)`**
Orchestrates load → features → targets → complete-case → bucketize → sweep for
one family, returning the result frame and the cutoff series.

The CLI owns argument parsing, the family loop, parallelism and all file IO; the
library performs no IO. This keeps `sweep` testable on a synthetic frame with no
data workbook present.

## Data flow (per family)

1. `load_raw` → `build_all` → select the family frame.
2. `add_features` → the 24 continuous indicator columns (all as-of `t`).
3. `add_forward_returns(horizons=(1, 2, 3, 5, 10, 15, 20))` → `fwd_1 … fwd_20`.
4. **Feature-complete filter** — drop rows with a NaN in any of the 24 indicator
   columns. Forward columns are deliberately *not* filtered yet: doing so here
   would let a contract's expiry date influence other contracts' cutoffs.
5. **Point-in-time bucketing** — sort by date, compute expanding quantile edges
   from strictly prior dates, drop the `min_history` warmup, assign codes.
   Rows that cannot be coded (`MISSING_CODE`) are dropped.
6. **Forward-complete intersection** — now drop rows with a NaN in any of the 7
   forward columns, keeping `codes` and the panel aligned on identical labels.
7. **Sweep** — per combination, one `groupby(code, observed=True, sort=False)`
   aggregating all 7 forward columns at once. Keep cells with `n ≥ min_samples`.
8. **Write** — `<family>.parquet`, `<family>_cutoffs.parquet` and
   `<family>_cutoffs_latest.csv`, written atomically as the family completes.

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
bars of each contract, plus the `min_history` quantile warmup.

### Why per-family cutoffs

The 24 indicators live on wildly different scales (RSI 0–100, z-scores, ATR
units, ratios), and the eight families differ just as much in volatility and
behaviour. Family-relative quantiles adapt to both without a hand-tuned
threshold table.

Note that with expanding cutoffs the buckets are no longer balanced by
construction — early dates calibrate on less history, and a drifting
distribution makes cell counts uneven. That is the honest cost of point-in-time
calibration, and `n` is reported per cell precisely so it is visible.

## Output

`docs/reports/bucket_sweep/<family>.parquet`, one row per cell × horizon:

| column | meaning |
|---|---|
| `family` | contract family |
| `k` | number of indicators in the combination |
| `themes` | `\|`-joined theme names, aligned to `indicators` |
| `indicators` | `\|`-joined indicator names, in theme-group order (aligned to `themes`/`buckets`, not lexicographic) |
| `buckets` | `\|`-joined bucket labels (`low`/`mid`/`high`), aligned to `indicators` |
| `horizon` | 1, 2, 3, 5, 10, 15 or 20 |
| `n` | rows in the cell |
| `mean` | mean forward difference, price points |
| `median` | median forward difference |
| `std` | standard deviation |
| `hit_rate` | fraction with `fwd_h > 0` |
| `t_stat` | `mean / (std / √n)` |

`<family>_cutoffs.parquet` holds the full per-date edge series
(`date, indicator, edge_index, value`), so any historical cell reads back as the
concrete threshold that applied on that date. `<family>_cutoffs_latest.csv` is
the final date's edges — the thresholds a rule would use today.
`top_cells.csv` collects the strongest cells by `|t_stat|` across families.

**Interpreting `t_stat`.** With ~322k cells per family, thousands will show
|t| > 3 by chance alone. It is a ranking aid, not evidence. Overlapping forward
windows also autocorrelate the outcomes, inflating it further. Every input is
now point-in-time, which removes calibration look-ahead — but it does not remove
**selection** bias: choosing a cell because it looks good is itself fitting.
Anything promising still needs out-of-sample confirmation on held-out dates.

## Error handling

- Missing `data/raw_files.xlsx` → fail fast before any sweep begins.
- Family with fewer than `min_samples × 81` post-warmup rows → warn, skip the
  family, continue. One thin family must not abort a multi-hour run.
- Family with fewer than `min_history` rows total → warn and skip; no date has
  enough prior history to bucket.
- Indicator with non-increasing edges on a given date (constant over the prior
  window), or a NaN reading → the row gets `MISSING_CODE` rather than a
  fabricated bucket. `run_family` then drops such rows from **all** combinations
  (complete-case, per "Why complete-case" above); `sweep` independently excludes
  sentinel rows from any combination naming that indicator.
- `--max-k > 4` → refuse unless `--force`, reporting the resulting cell count.
- Parquet written per family on completion; `--resume` skips families already on
  disk.

## Testing

`backtesting/tests/test_bucket_sweep.py`, under the configured `testpaths`.

- **Look-ahead guard (the binding test)** — compute codes on a panel, append
  future bars, recompute: every originally-computed bucket code is unchanged.
  This is what distinguishes the design from the naive full-sample version and
  must fail if `expanding_cutoffs` is ever replaced by a plain `.quantile()`.
- **Prior-dates-only cutoffs** — on a hand-built panel with known values, the
  edges applied on date `d` equal the quantiles of the rows before `d`, and same
  date rows of different contracts do not influence each other.
- **Outcome isolation** — perturbing the forward columns leaves every bucket
  code unchanged.
- **Warmup** — rows with less than `min_history` prior history are absent from
  the swept panel.
- **Statistics** — on a small hand-built panel, `n`, `mean`, `std` and
  `hit_rate` for a named cell match hand-computed values.
- **Theme partition** — the theme map covers every name in `FEATURE_NAMES`
  exactly once, with no extras. Adding an indicator without a theme fails here.
- **Theme exclusivity** — no emitted combination contains two indicators of the
  same theme, and on a small hand-built theme map the combination count equals
  the elementary symmetric polynomial of the theme sizes.
- **Combination coverage** — combinations are unordered and appear exactly once;
  with the real taxonomy at `max_k=4` the total is 5,015.
- **min-samples** — cells below the floor are absent from the output, not
  present with NaN.

## Decisions taken

| Decision | Choice | Rationale |
|---|---|---|
| Depth | k ≤ 4 | User's call; `--min-samples` guards thin cells |
| Themes | 7 themes, at most one indicator each per combination | Kills near-duplicate pairings; cuts cells by ~65% |
| Cutoffs | Per-family **expanding quantiles over strictly prior dates** | No look-ahead; a bucket means what was knowable at the time |
| Warmup | `--min-history`, default 2000 pooled rows | Early dates lack the history for meaningful quantiles |
| Buckets | Terciles only; `--n-buckets` other than 3 is refused | `BUCKET_LABELS` is a fixed ("low","mid","high") tuple |
| Anchor | `close[t+h] − open[t+1]` | Matches repo's executable basis |
| Families | All 8, reported separately | Preserves per-family scale and behaviour |
| Output | Full stats, parquet, `n ≥ 30` | Keeps size sane, drops noise-only cells |
| Compute | pandas `groupby` per combination | User's call over vectorised bincount |
