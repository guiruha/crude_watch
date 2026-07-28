"""Tests for the strategy simulator + ledger (range-regime confirmed reversion)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.research.evaluate import _accumulate_trades, _bucket_edges
from backtesting.research.strategy import _ledger_trades, simulate_strategy, strategy_stats

RNG = np.random.default_rng(7)


def test_ledger_matches_accumulate_trades():
    """The ledger must reproduce EXACTLY the trades of evaluate._accumulate_trades
    (same side rule, lock-out and NaN gate), only adding dates."""
    n = 30
    feat = np.linspace(-3.0, 3.0, n)
    fwd = -feat + RNG.normal(0, 0.1, n)          # reversion: cheap -> up
    mfe = np.abs(fwd)
    mae = -np.abs(fwd)
    mfe[5] = np.nan                               # a NaN excursion must drop that trade in BOTH
    df = pd.DataFrame({
        "contract": ["C1"] * n,
        "date": pd.date_range("2020-01-01", periods=n, freq="B"),
        "f": feat, "fwd_5": fwd, "mfe_5": mfe, "mae_5": mae,
    })
    edges = _bucket_edges(feat, 5)
    sense = -0.5

    pnls, favs, advs, sides = [], [], [], []
    _accumulate_trades(df, "f", edges, sense, 5, 5, 0.01, "fwd_5", "mfe_5", "mae_5",
                       pnls, favs, advs, sides)
    led = _ledger_trades(df, "f", edges, sense, 5, 5, 0.01, "fwd_5")

    assert len(led) == len(pnls) and len(pnls) > 0
    assert [round(t["pnl"], 9) for t in led] == [round(p, 9) for p in pnls]
    assert [t["side"] for t in led] == sides


def _reversion_frame() -> pd.DataFrame:
    """Synthetic family: in the range regime, cheap z_20 reverts up; level agrees."""
    rows = []
    for v in range(2010, 2019):
        per = 120
        z = RNG.normal(0, 1, per)
        rows.append(pd.DataFrame({
            "vintage": [v] * per,
            "contract": [f"C{v}"] * per,
            "date": pd.date_range(f"{v}-01-01", periods=per, freq="B"),
            "er_20": RNG.uniform(0.0, 0.2, per),      # spread so a bottom tercile exists
            "z_20": z,
            "level_z": z + RNG.normal(0, 0.2, per),   # level agrees with the oscillator
            "fwd_10": -1.6 * z + RNG.normal(0, 0.3, per),
            "mfe_10": np.abs(-1.6 * z) + 0.1,
            "mae_10": -(np.abs(-1.6 * z) + 0.1),
        }))
    return pd.concat(rows, ignore_index=True)


def test_simulate_strategy_produces_profitable_ledger():
    data = _reversion_frame()
    led = simulate_strategy(data, "quarterly", 10, cost=0.01, confirm=True, min_fold_rows=15)
    assert not led.empty
    # Ledger is time-ordered and the running total is a genuine cumsum.
    assert led["date"].is_monotonic_increasing
    assert np.isclose(led["cum_pnl"].iloc[-1], led["pnl"].sum())
    assert set(led["side"].unique()).issubset({-1.0, 1.0})

    st = strategy_stats(led, 10)
    assert st["n_trades"] == len(led)
    assert st["n_long"] + st["n_short"] == st["n_trades"]
    assert st["sharpe"] > 0            # the reversion edge is real -> positive OOS Sharpe
    assert st["win_rate"] > 0.5


def test_confirmation_filters_trades():
    data = _reversion_frame()
    confirmed = simulate_strategy(data, "quarterly", 10, cost=0.01, confirm=True, min_fold_rows=15)
    unconfirmed = simulate_strategy(data, "quarterly", 10, cost=0.01, confirm=False, min_fold_rows=15)
    # The level confirmation can only remove trades, never add them.
    assert len(confirmed) <= len(unconfirmed)


def test_strategy_stats_empty_ledger():
    st = strategy_stats(pd.DataFrame(columns=["family", "date", "pnl", "side", "cum_pnl"]), 10)
    assert st["n_trades"] == 0
    assert np.isnan(st["sharpe"])
