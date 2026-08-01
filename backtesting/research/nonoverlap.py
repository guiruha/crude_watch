"""Non-overlapping Opportunity Score backtest (offline).

The score backtest that ships with the app (:mod:`crudewatch.scoring.backtest`)
runs a stateful hysteresis rule that is effectively always in the market, per
contract. That is the right thing for the UI, but it makes every reported
statistic dependent: consecutive bars share a position, and sibling contracts
hold correlated positions on the same days. It is the same defect that inflated
the bucket sweep's t-statistics roughly 59x over chance.

This module answers the narrower, harder question: **if the score is traded so
that no two trades ever share a day, does it still make money?**

Rules
-----
* At most **one open position per family** at any time.
* Enter when the point-in-time score clears ``enter_at``; among contracts
  eligible on that date, take the highest conviction.
* Signal is decided at ``close[t]``; the fill is ``open[t+1]`` — the executable
  basis used throughout the project.
* Hold a **fixed** number of bars, then exit at that bar's open. No stop, no
  target, no re-entry until the position is closed.
* The next signal must fall strictly after the exit date, so holding periods are
  pairwise disjoint.

Because trades are independent, the Sharpe may be annualised by the realised
number of trades per year rather than by an assumed 252 — no overlap correction
is needed, which is the whole point.

Offline only; performs no file IO and no printing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADE_COLUMNS: tuple[str, ...] = (
    "contract", "entry_date", "exit_date", "side", "bars",
    "entry_price", "exit_price", "opp", "gross_pnl", "cost", "net_pnl",
)


def _price(row: pd.Series) -> float:
    """Fill price: the bar's open, falling back to its close when open is absent."""
    o = row.get("open", np.nan)
    if o == o:
        return float(o)
    c = row.get("close", np.nan)
    return float(c) if c == c else np.nan


def nonoverlapping_trades(
    panel: pd.DataFrame,
    hold_bars: int = 20,
    enter_at: float = 50.0,
    cost: float = 0.0,
) -> pd.DataFrame:
    """Walk ``panel`` and emit trades whose holding periods never overlap.

    ``panel`` needs ``date``, ``contract``, ``opp`` (the point-in-time score,
    positive = long) and ``open`` / ``close``. Returns one row per trade with
    :data:`TRADE_COLUMNS`; an empty but correctly-typed frame if nothing fires.
    """
    if hold_bars < 1:
        raise ValueError(f"hold_bars must be at least 1, got {hold_bars}")

    panel = panel.sort_values(["contract", "date"], kind="mergesort")
    by_contract = {c: g.reset_index(drop=True) for c, g in panel.groupby("contract", sort=False)}
    # date -> row position, per contract, so a signal date maps to its own bar
    pos = {c: {d: i for i, d in enumerate(g["date"])} for c, g in by_contract.items()}

    trades: list[dict] = []
    busy_until = None

    for date, day in panel.sort_values("date", kind="mergesort").groupby("date", sort=True):
        if busy_until is not None and date <= busy_until:
            continue
        eligible = day[day["opp"].abs() >= enter_at]
        if eligible.empty:
            continue

        signal = eligible.loc[eligible["opp"].abs().idxmax()]
        contract = signal["contract"]
        g = by_contract[contract]
        i = pos[contract][date]

        entry_i = i + 1                      # fill at the next bar's open
        exit_i = entry_i + hold_bars
        if exit_i >= len(g):                 # contract ends before the hold completes
            continue

        entry_price = _price(g.iloc[entry_i])
        exit_price = _price(g.iloc[exit_i])
        if entry_price != entry_price or exit_price != exit_price:
            continue

        side = 1 if signal["opp"] > 0 else -1
        gross = side * (exit_price - entry_price)
        trades.append({
            "contract": contract,
            "entry_date": g.iloc[entry_i]["date"],
            "exit_date": g.iloc[exit_i]["date"],
            "side": side,
            "bars": exit_i - entry_i,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "opp": float(signal["opp"]),
            "gross_pnl": gross,
            "cost": cost,
            "net_pnl": gross - cost,
        })
        busy_until = g.iloc[exit_i]["date"]

    if not trades:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in TRADE_COLUMNS})
    return pd.DataFrame(trades)[list(TRADE_COLUMNS)]


def summarize(trades: pd.DataFrame) -> dict:
    """Headline statistics for a set of non-overlapping trades.

    ``sharpe`` is annualised by the **realised trades per year**, which is valid
    precisely because the trades do not overlap. Annualising by ``sqrt(252)``
    here would reintroduce the inflation this backtest exists to avoid.
    """
    n = len(trades)
    if n == 0:
        return {"trades": 0, "total_pnl": 0.0, "mean_pnl": np.nan, "hit_rate": np.nan,
                "sharpe": np.nan, "trades_per_year": np.nan, "years": np.nan,
                "pnl_per_year": np.nan, "worst": np.nan, "best": np.nan}

    pnl = trades["net_pnl"].to_numpy(dtype=float)
    span_days = (trades["exit_date"].max() - trades["entry_date"].min()).days
    years = max(span_days / 365.25, 1e-9)
    per_year = n / years
    sd = pnl.std(ddof=1) if n > 1 else np.nan

    return {
        "trades": n,
        "total_pnl": float(pnl.sum()),
        "mean_pnl": float(pnl.mean()),
        "hit_rate": float((pnl > 0).mean()),
        "sharpe": float((pnl.mean() / sd) * np.sqrt(per_year)) if sd and sd == sd else np.nan,
        "trades_per_year": float(per_year),
        "years": float(years),
        "pnl_per_year": float(pnl.sum() / years),
        "worst": float(pnl.min()),
        "best": float(pnl.max()),
    }


def panel_from_precomputed(pcs, w_range: np.ndarray, w_trend: np.ndarray,
                           transition_shrink: float = 0.4) -> pd.DataFrame:
    """Build the walker's input from ``weight_search.precompute_family`` output.

    Reuses the precomputed point-in-time term matrices, so the score is exactly
    the one the app would show on that date, without refitting a calibrator per
    bar.
    """
    from crudewatch.scoring.weight_search import _opportunity_matrix

    frames = []
    for pc in pcs:
        opp = _opportunity_matrix(pc, w_range[:, None], w_trend[:, None], transition_shrink)
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(pc.date),
            "contract": pc.contract,
            "opp": opp[:, 0],
            "open": pc.open,
            "close": pc.close,
        }))
    if not frames:
        return pd.DataFrame(columns=["date", "contract", "opp", "open", "close"])
    return pd.concat(frames, ignore_index=True)
