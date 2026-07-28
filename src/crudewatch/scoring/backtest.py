"""Per-contract backtest: follow the Opportunity Score over a contract's life.

Walks a single contract bar-by-bar, computes the Opportunity Score strictly
point-in-time (calibrator refit on the family history ``<=`` that date), then
simulates a **stateful hysteresis** rule that follows the score's sign, and
reports equity vs buy-and-hold, a trade ledger and summary statistics.

Conventions
-----------
* Signal decided at ``close[t]`` (uses only information ``<= t``); the resulting
  position change fills at ``open[t+1]`` (falls back to that bar's close when no
  ``open`` column exists), matching the executable basis of
  :mod:`crudewatch.research.targets`.
* Position is **unit** ``+1 / -1 / 0`` — no magnitude sizing.
* PnL is in **price points**. Costs use the per-family ``COST_STUB_POINTS``
  round-trip stub, charged as ``cost / 2`` per traded leg (a normal round trip =
  one stub; a direct flip = two legs = one stub).
* Point-in-time: the score at ``t`` never depends on rows after ``t``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crudewatch.research.dataset import COST_STUB_POINTS
from crudewatch.scoring.blocks import fit_calibrator
from crudewatch.scoring.score import compute_blocks, compute_opportunity, weights_for

DEFAULT_ENTER: float = 50.0
DEFAULT_EXIT: float = 20.0

TRADE_COLUMNS: tuple[str, ...] = (
    "entry_date", "exit_date", "side", "bars",
    "entry_price", "exit_price", "gross_pnl", "cost", "net_pnl",
    "mfe", "mae", "open",
)


@dataclass(frozen=True)
class BacktestResult:
    """Everything the UI needs to present a single-contract score backtest."""

    contract: str
    horizon: int
    enter_at: float
    exit_at: float
    cost: float
    score_df: pd.DataFrame          # date, close, open, high, low, opportunity, regime
    trades: pd.DataFrame            # ledger (TRADE_COLUMNS)
    equity: pd.Series               # net strategy equity in points, indexed by date
    benchmark: pd.Series            # buy&hold equity in points, indexed by date
    stats: dict = field(default_factory=dict)


def score_series(
    data: pd.DataFrame,
    family: str,
    contract: str,
    horizon: int = 25,
) -> pd.DataFrame:
    """Strictly point-in-time Opportunity Score for every bar of ``contract``.

    For each bar dated ``d`` the family calibrator is refit on rows ``<= d``
    (``outcome_asof=d``), so the score never peeks past ``d``. Returns a frame
    with ``date``, ``close``, ``open``, ``high``, ``low``, ``opportunity`` and
    ``regime`` (one row per contract bar, ordered by date).
    """
    df = data.reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])
    dates_all = df["date"].to_numpy()
    csub = df[df["contract"] == contract].sort_values("date").reset_index(drop=True)
    if csub.empty:
        raise KeyError(f"contract {contract!r} not found in data")

    fam_weights = weights_for(family)
    has = {c: (c in csub.columns) for c in ("open", "high", "low", "close")}
    records: list[dict] = []
    for i, row in csub.iterrows():
        d = row["date"]
        window = df[dates_all <= np.datetime64(d)]
        cal = fit_calibrator(window, family, horizon, outcome_asof=d)
        blocks = compute_blocks(row, cal, i + 1)
        opp = compute_opportunity(blocks, row, cal, fam_weights)
        records.append(
            {
                "date": d,
                "close": float(row["close"]) if has["close"] else np.nan,
                "open": float(row["open"]) if has["open"] and row["open"] == row["open"] else np.nan,
                "high": float(row["high"]) if has["high"] and row["high"] == row["high"] else np.nan,
                "low": float(row["low"]) if has["low"] and row["low"] == row["low"] else np.nan,
                "opportunity": float(opp),
                "regime": blocks.regime,
            }
        )
    return pd.DataFrame.from_records(records)


def _hysteresis(opp: np.ndarray, enter_at: float, exit_at: float) -> np.ndarray:
    """Stateful position target in ``{-1, 0, +1}`` from the score series.

    Enter long at ``>= +enter_at``; short at ``<= -enter_at``; leave a long when
    the score drops below ``+exit_at``; leave a short when it rises above
    ``-exit_at``; a direct flip is allowed when the opposite entry is crossed.
    """
    state = 0
    out = np.zeros(len(opp), dtype=int)
    for t, raw in enumerate(opp):
        x = 0.0 if raw != raw else float(raw)
        if state == 0:
            if x >= enter_at:
                state = 1
            elif x <= -enter_at:
                state = -1
        elif state == 1:
            if x <= -enter_at:
                state = -1
            elif x < exit_at:
                state = 0
        else:  # state == -1
            if x >= enter_at:
                state = 1
            elif x > -exit_at:
                state = 0
        out[t] = state
    return out


def _mfe_mae(side: int, hi: np.ndarray, lo: np.ndarray, entry: float) -> tuple[float, float]:
    """Max favourable / adverse excursion over the holding window, side-oriented."""
    if len(hi) == 0 or np.all(np.isnan(hi)) or np.all(np.isnan(lo)):
        return np.nan, np.nan
    if side > 0:
        return float(np.nanmax(hi) - entry), float(np.nanmin(lo) - entry)
    return float(entry - np.nanmin(lo)), float(entry - np.nanmax(hi))


def _sharpe(daily: np.ndarray) -> float:
    """Annualised Sharpe of a daily PnL-point series (idle days included)."""
    if daily is None or len(daily) < 2:
        return float("nan")
    sd = float(np.std(daily, ddof=1))
    if not sd or sd <= 0:
        return float("nan")
    return float(np.mean(daily) / sd * np.sqrt(252.0))


def simulate(
    score_df: pd.DataFrame,
    cost: float,
    enter_at: float = DEFAULT_ENTER,
    exit_at: float = DEFAULT_EXIT,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """Run the hysteresis rule over ``score_df`` and return trades, equity, benchmark, stats.

    See the module docstring for execution / cost / PnL conventions. The equity
    curve and the trade ledger share the same open-fill + close-mark basis, so
    the ledger's net PnL sums to the final equity.
    """
    df = score_df.reset_index(drop=True)
    n = len(df)
    dates = pd.to_datetime(df["date"])
    empty_trades = pd.DataFrame(columns=list(TRADE_COLUMNS))
    if n < 2:
        eq = pd.Series(np.zeros(n), index=dates, dtype=float)
        return empty_trades, eq, eq.copy(), _empty_stats()

    close = df["close"].to_numpy(dtype=float)
    openp = df["open"].to_numpy(dtype=float) if "open" in df.columns else np.full(n, np.nan)
    fill = np.where(np.isnan(openp), close, openp)
    high = df["high"].to_numpy(dtype=float) if "high" in df.columns else np.full(n, np.nan)
    low = df["low"].to_numpy(dtype=float) if "low" in df.columns else np.full(n, np.nan)
    opp = df["opportunity"].to_numpy(dtype=float)
    date_arr = dates.to_numpy()

    desired = _hysteresis(opp, enter_at, exit_at)
    # Execution lag: a change decided at close[t] takes effect on the next bar's
    # open, so the position effective during bar t is the target set at t-1.
    effective = np.empty(n, dtype=int)
    effective[0] = 0
    effective[1:] = desired[:-1]

    leg_cost = cost / 2.0
    state = 0
    entry_price = np.nan
    entry_idx = -1
    entry_cost = 0.0
    realized = 0.0
    cost_accum = 0.0
    equity = np.zeros(n)
    trades: list[dict] = []
    open_trade: dict | None = None

    for t in range(n):
        new_state = int(effective[t])
        if new_state != state:
            price = fill[t]
            if state != 0 and open_trade is not None:
                gross = state * (price - entry_price)
                exit_cost = leg_cost * abs(state)
                realized += gross
                cost_accum += exit_cost
                mfe, mae = _mfe_mae(state, high[entry_idx : t + 1], low[entry_idx : t + 1], entry_price)
                open_trade.update(
                    exit_date=date_arr[t], exit_price=float(price), bars=int(t - entry_idx),
                    gross_pnl=float(gross), cost=float(entry_cost + exit_cost),
                    net_pnl=float(gross - entry_cost - exit_cost), mfe=mfe, mae=mae, open=False,
                )
                trades.append(open_trade)
                open_trade = None
            if new_state != 0:
                entry_price = price
                entry_idx = t
                entry_cost = leg_cost * abs(new_state)
                cost_accum += entry_cost
                open_trade = {
                    "entry_date": date_arr[t], "entry_price": float(price),
                    "side": "long" if new_state > 0 else "short",
                }
            state = new_state
        unreal = state * (close[t] - entry_price) if state != 0 else 0.0
        equity[t] = realized + unreal - cost_accum

    # Close any position still open at the last bar (mark to last close, flagged).
    if state != 0 and open_trade is not None:
        last = n - 1
        gross = state * (close[last] - entry_price)
        mfe, mae = _mfe_mae(state, high[entry_idx : last + 1], low[entry_idx : last + 1], entry_price)
        open_trade.update(
            exit_date=date_arr[last], exit_price=float(close[last]), bars=int(last - entry_idx),
            gross_pnl=float(gross), cost=float(entry_cost),
            net_pnl=float(gross - entry_cost), mfe=mfe, mae=mae, open=True,
        )
        trades.append(open_trade)

    trades_df = pd.DataFrame(trades, columns=list(TRADE_COLUMNS)) if trades else empty_trades
    equity_s = pd.Series(equity, index=dates, dtype=float)
    benchmark_s = pd.Series(close - close[0], index=dates, dtype=float)

    daily = np.diff(equity, prepend=0.0)
    run_max = np.maximum.accumulate(equity)
    net = trades_df["net_pnl"].to_numpy(dtype=float) if len(trades_df) else np.array([])
    stats = {
        "n_trades": int(len(trades_df)),
        "win_rate": float(np.mean(net > 0)) if len(net) else float("nan"),
        "avg_pnl": float(np.mean(net)) if len(net) else float("nan"),
        "total_pnl": float(equity[-1]),
        "sharpe": _sharpe(daily),
        "max_drawdown": float(np.min(equity - run_max)),
        "avg_mfe": float(np.nanmean(trades_df["mfe"])) if len(trades_df) else float("nan"),
        "avg_mae": float(np.nanmean(trades_df["mae"])) if len(trades_df) else float("nan"),
        "time_in_market": float(np.mean(effective != 0)),
        "bench_total": float(benchmark_s.iloc[-1]),
        "excess_total": float(equity[-1] - benchmark_s.iloc[-1]),
    }
    return trades_df, equity_s, benchmark_s, stats


def _empty_stats() -> dict:
    keys = (
        "n_trades", "win_rate", "avg_pnl", "total_pnl", "sharpe", "max_drawdown",
        "avg_mfe", "avg_mae", "time_in_market", "bench_total", "excess_total",
    )
    return {k: (0 if k == "n_trades" else float("nan")) for k in keys}


def backtest_contract(
    data: pd.DataFrame,
    family: str,
    contract: str,
    horizon: int = 25,
    enter_at: float = DEFAULT_ENTER,
    exit_at: float = DEFAULT_EXIT,
    cost: float | None = None,
) -> BacktestResult:
    """Backtest following the Opportunity Score on a single ``contract``.

    ``cost`` defaults to the family's ``COST_STUB_POINTS`` round-trip stub.
    """
    if cost is None:
        cost = COST_STUB_POINTS.get(family, 0.0)
    sdf = score_series(data, family, contract, horizon)
    trades, equity, benchmark, stats = simulate(sdf, cost, enter_at, exit_at)
    return BacktestResult(
        contract=str(contract),
        horizon=int(horizon),
        enter_at=float(enter_at),
        exit_at=float(exit_at),
        cost=float(cost),
        score_df=sdf,
        trades=trades,
        equity=equity,
        benchmark=benchmark,
        stats=stats,
    )
