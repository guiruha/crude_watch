# Per-contract Opportunity-Score backtest — design

**Date:** 2026-07-26
**Status:** design (awaiting user review)
**Author:** pairing session

## Problem

We can score a contract's Opportunity Score point-in-time, but we cannot yet
answer: *"if a PM had followed the score over this contract's life, what would
the results have been?"* This spec designs a **per-contract backtest** that
walks a single contract bar-by-bar, computes the Opportunity Score strictly
point-in-time, simulates a stateful trading rule that follows the score, and
reports equity vs buy-and-hold, the trade ledger, summary stats, and a price
chart with entry/exit markers.

Out of scope (v1): family-wide aggregation, learned/optimised thresholds,
position sizing by score magnitude, real (non-stub) costs.

## Decisions (agreed)

| Topic | Decision |
|-------|----------|
| Unit | The **currently selected contract** only |
| Trading rule | **Stateful hysteresis**: enter at `±50`, exit at `±20` |
| Outputs | Equity vs buy&hold · trades table · summary stats · price chart with entry/exit markers |
| Costs | **Always net** with the per-family `COST_STUB_POINTS` |
| Calibration | **Strict per-bar point-in-time** refit |
| Placement | New **"Backtest"** screen, **last** entry in the nav |

## Architecture (Approach A: library engine + thin screen)

```
enriched_frame(family)                      [app/core/scoring.py, cached]
  → backtest_contract(data, family, contract, horizon, params)
        ├── score_series(...)   # per-bar PIT Opportunity Score
        └── simulate(...)       # stateful hysteresis → trades + equity
  → BacktestResult (score_df, trades, equity, benchmark, stats)
  → backtest_contract_cached(...)           [app/core/scoring.py, @st.cache_data]
  → BacktestScreen.display(selection)       [app/screens/backtest.py]
```

Pure, testable logic lives in `src/crudewatch/scoring/backtest.py`. The Streamlit
screen is a thin renderer over a cached wrapper. This mirrors the existing
lib/app split (`scoring/` is pure; `app/core/scoring.py` caches; `app/screens/*`
renders).

### New module: `src/crudewatch/scoring/backtest.py`

**`score_series(data, family, contract, horizon) -> pd.DataFrame`**

For each bar `d` of `contract` (ordered by date):
1. `sub = data[date <= d]`
2. `cal = fit_calibrator(sub, family, horizon, outcome_asof=d)`  *(strict PIT)*
3. `row = contract's last row <= d`; `n_bars = count of contract bars <= d`
4. `blocks = compute_blocks(row, cal, n_bars)`; `opp = compute_opportunity(blocks, row, cal)`

Returns a frame indexed by date with columns: `close`, `open`, `opportunity`,
`regime`. Bars where the score is undefined (insufficient history / NaN
features) are dropped from the front; the backtest starts at the first defined
score. **No look-ahead:** the score at `d` depends only on data `<= d`.

*Performance:* strict per-bar refit is ~250–300 calibrator fits for a typical
outrights contract. The whole `BacktestResult` is cached in the app layer and
computed behind a spinner + progress bar. A future `calibrate_every: int`
(monthly cadence) can be added without changing the interface.

**`simulate(score_df, cost, enter=50.0, exit=20.0) -> (trades, equity, benchmark)`**

State machine over bars, state ∈ {flat, long, short}:

- `flat → long` when `opp >= +enter`; `flat → short` when `opp <= -enter`
- `long → flat` when `opp < +exit`; `short → flat` when `opp > -exit`
- **Flip** allowed: `long → short` when `opp <= -enter` (and symmetric) — modelled
  as a close + immediate open (two cost legs)

Execution basis (anti-lookahead, matches `research/targets.py`): a state change
decided at `close[t]` executes at **`open[t+1]`** (fallback `close[t]` if no
`open`). Position is **unit** `+1 / -1 / 0` (no magnitude scaling in v1).

- **Equity (net, price points):** marked at each bar's close;
  `Δequity_t = position_t · (close_t − close_{t-1})`, minus `cost` charged on each
  executed entry/exit leg (a completed round trip costs one `COST_STUB_POINTS`;
  a flip costs two legs). `cost = COST_STUB_POINTS[family]`.
- **Benchmark (buy&hold):** `close_t − close_first` over the simulated span.
- **Trades ledger** (one row per closed position): entry date/price, exit
  date/price, side (`long`/`short`), bars held, gross pnl, cost, net pnl, and
  realised MFE/MAE over the holding window (from bar highs/lows). An open
  position at the last bar is closed at the last close and flagged `open=True`.

**`backtest_contract(data, family, contract, horizon, enter=50, exit=20) -> BacktestResult`**

Orchestrates `score_series` → `simulate`, computes `stats`, returns a dataclass:

```python
@dataclass
class BacktestResult:
    contract: str
    horizon: int
    score_df: pd.DataFrame      # date, close, open, opportunity, regime
    trades: pd.DataFrame        # ledger (see above)
    equity: pd.Series           # net strategy equity, indexed by date
    benchmark: pd.Series        # buy&hold equity, indexed by date
    stats: dict                 # summary metrics
```

**`stats`:** `n_trades`, `win_rate`, `avg_pnl`, `total_pnl` (net), `sharpe`
(annualised daily equity-return Sharpe, √252 — implemented locally, NOT imported
from `backtesting/`), `max_drawdown`, `avg_mfe`, `avg_mae`, `time_in_market`
(fraction of bars in a position), `bench_total` (buy&hold total), and
`excess_total = total_pnl − bench_total`.

### App layer

**`app/core/scoring.py`** — add:
```python
@st.cache_data(show_spinner="Simulando backtest…")
def backtest_contract_cached(family, contract, horizon) -> BacktestResult: ...
```

**`app/screens/backtest.py`** — `BacktestScreen(frames)` with
`display(self, selection)`:
1. Guard: needs a selected contract; otherwise `st.info(...)`.
2. **Summary stat cards** (reuse existing card styling).
3. **Equity vs buy&hold** line chart (Plotly, theme; strategy emerald, benchmark
   muted).
4. **Price chart with markers**: close line + entry markers (long = up triangle
   emerald, short = down triangle red) and exit markers (hollow).
5. **Trades table**: net-pnl-coloured rows (reuse the custom HTML table style
   from the metric table, or `st.dataframe`).

**`app/main.py`** — register as the **last** `SCREENS` entry: `"Backtest"`.

## Correctness / point-in-time

- Score at `d` uses only data `<= d` (calibrator `outcome_asof=d`).
- Execution at `open[t+1]`; equity marks at close.
- Costs charged per executed leg; flips = two legs.

## Edge cases

- Front bars with undefined score → dropped; backtest starts when score exists.
- Contract with `< 2` scored bars → empty-state message, no simulation.
- Position open at contract end → closed at last close, `open=True` flag.
- No trades triggered → show score/price context + "sin operaciones" note.
- Missing `open` column → execute at `close`.

## Testing (`tests/test_backtest.py`)

- `simulate` on a synthetic score series → asserts exact trades, entry/exit
  bars, flip handling, cost accounting, and equity path.
- Benchmark equals cumulative close change over the span.
- Cost accounting: round trip = one stub; flip = two.
- Point-in-time no-leak: `score_series(..).opportunity.loc[d]` is unchanged when
  rows `> d` are appended to `data`.
- Stats sanity: `win_rate ∈ [0,1]`, `time_in_market ∈ [0,1]`, drawdown `<= 0`.
- References: `tests/test_scoring.py`, `tests/test_targets.py`.

## Reused building blocks

- `compute_blocks`, `compute_opportunity`, `fit_calibrator` (`scoring/`)
- `enriched_frame` targets `fwd/mfe/mae`, bar `high/low` for MFE/MAE
- `COST_STUB_POINTS` (`crudewatch.research`)
- Theme/plot helpers and card/table CSS in `app/theme/palette.py`
- Screen registration pattern in `app/main.py`

## Defaults fixed for v1 (YAGNI)

Enter `±50`, exit `±20`, unit position, per-family stub cost, per-bar PIT.
`horizon` comes from the selection (feeds the calibrator's forward-probability
window). Configurable thresholds / calibration cadence / magnitude sizing are
explicit future extensions, not built now.
