"""Tests for the per-contract follow-the-score backtest engine."""
from __future__ import annotations

import numpy as np
import pandas as pd

from crudewatch.scoring.backtest import (
    _hysteresis,
    backtest_contract,
    score_series,
    simulate,
)


def _score_df(opp, close, open_=None, high=None, low=None) -> pd.DataFrame:
    n = len(opp)
    open_ = close if open_ is None else open_
    high = [max(c, o) for c, o in zip(close, open_)] if high is None else high
    low = [min(c, o) for c, o in zip(close, open_)] if low is None else low
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "close": list(map(float, close)),
            "open": list(map(float, open_)),
            "high": list(map(float, high)),
            "low": list(map(float, low)),
            "opportunity": list(map(float, opp)),
            "regime": ["range"] * n,
        }
    )


# -- hysteresis state machine ------------------------------------------------

def test_hysteresis_enter_and_exit():
    opp = np.array([0, 60, 30, 10, 0], dtype=float)
    assert list(_hysteresis(opp, 50.0, 20.0)) == [0, 1, 1, 0, 0]


def test_hysteresis_short_side():
    opp = np.array([0, -60, -30, -10, 0], dtype=float)
    assert list(_hysteresis(opp, 50.0, 20.0)) == [0, -1, -1, 0, 0]


def test_hysteresis_direct_flip():
    opp = np.array([60, -60, 60], dtype=float)
    assert list(_hysteresis(opp, 50.0, 20.0)) == [1, -1, 1]


def test_hysteresis_nan_is_flat():
    opp = np.array([np.nan, 60, np.nan], dtype=float)
    # NaN treated as 0: stays flat, then enters long, then holds (NaN=0 < exit? 0<20 -> exit)
    assert list(_hysteresis(opp, 50.0, 20.0)) == [0, 1, 0]


# -- simulate: a single clean long trade -------------------------------------

def test_simulate_single_long_trade():
    # Enter decided at t1 (opp 60), fills at open[2]; exit decided at t3 (opp 10),
    # fills at open[4]. Execution lag = one bar.
    sdf = _score_df(
        opp=[0, 60, 30, 10, 0, 0],
        close=[100, 101, 103, 104, 102, 101],
        open_=[100, 101, 102, 103, 104, 105],
        high=[100, 101, 103, 105, 104, 105],
        low=[100, 100, 101, 102, 101, 100],
    )
    trades, equity, benchmark, stats = simulate(sdf, cost=0.10, enter_at=50.0, exit_at=20.0)

    assert len(trades) == 1
    tr = trades.iloc[0]
    assert tr["side"] == "long"
    assert tr["entry_price"] == 102.0          # open[2]
    assert tr["exit_price"] == 104.0           # open[4]
    assert tr["bars"] == 2
    assert tr["gross_pnl"] == 2.0              # 104 - 102
    assert abs(tr["cost"] - 0.10) < 1e-9       # round trip = one stub
    assert abs(tr["net_pnl"] - 1.90) < 1e-9
    assert tr["mfe"] == 3.0                    # max high 105 - 102
    assert tr["mae"] == -1.0                   # min low 101 - 102
    assert bool(tr["open"]) is False

    assert abs(equity.iloc[-1] - 1.90) < 1e-9
    assert list(benchmark.to_numpy()) == [0, 1, 3, 4, 2, 1]
    assert abs(stats["max_drawdown"] - (-0.05)) < 1e-9
    assert stats["n_trades"] == 1
    assert stats["win_rate"] == 1.0
    assert abs(stats["time_in_market"] - (2 / 6)) < 1e-9
    assert abs(stats["excess_total"] - (1.90 - 1.0)) < 1e-9


def test_ledger_net_sums_to_final_equity():
    sdf = _score_df(
        opp=[0, 60, 30, 10, 0, 0],
        close=[100, 101, 103, 104, 102, 101],
        open_=[100, 101, 102, 103, 104, 105],
    )
    trades, equity, _, _ = simulate(sdf, cost=0.10)
    assert abs(trades["net_pnl"].sum() - equity.iloc[-1]) < 1e-9


# -- simulate: flip charges two legs; trailing position left open ------------

def test_simulate_flip_and_open_position_costs():
    sdf = _score_df(
        opp=[0, 60, 60, -60, -60, 0],
        close=[100, 101, 102, 103, 104, 105],
        open_=[100, 101, 102, 103, 104, 105],
    )
    trades, equity, _, _ = simulate(sdf, cost=0.10, enter_at=50.0, exit_at=20.0)

    # Long entered @ open[2], flipped to short @ open[4]; short still open at end.
    assert len(trades) == 2
    long_tr, short_tr = trades.iloc[0], trades.iloc[1]
    assert long_tr["side"] == "long" and bool(long_tr["open"]) is False
    assert abs(long_tr["cost"] - 0.10) < 1e-9          # entry + exit legs
    assert short_tr["side"] == "short" and bool(short_tr["open"]) is True
    assert abs(short_tr["cost"] - 0.05) < 1e-9          # entry leg only (never exited)
    # Ledger still reconciles with the equity path.
    assert abs(trades["net_pnl"].sum() - equity.iloc[-1]) < 1e-9


# -- degenerate inputs -------------------------------------------------------

def test_simulate_too_short_is_empty():
    sdf = _score_df(opp=[0], close=[100])
    trades, equity, benchmark, stats = simulate(sdf, cost=0.10)
    assert trades.empty
    assert list(equity.to_numpy()) == [0.0]
    assert stats["n_trades"] == 0


def test_simulate_no_signal_no_trades():
    sdf = _score_df(opp=[0, 10, -10, 5, 0], close=[100, 101, 100, 99, 100])
    trades, equity, _, stats = simulate(sdf, cost=0.10)
    assert trades.empty
    assert stats["n_trades"] == 0
    assert (equity == 0.0).all()


# -- score_series point-in-time (no look-ahead) ------------------------------

def _synthetic_family(seed: int = 0) -> pd.DataFrame:
    """Minimal enriched-like family frame with the numeric columns the scorer reads."""
    rng = np.random.default_rng(seed)
    feats = [
        "er_20", "slope_20", "macd_hist", "level_pct", "mom_decel_10", "ema_align",
        "mom_5", "mom_10", "mom_20", "variance_ratio_5", "autocorr_20", "r2_20",
        "dir_persistence_20", "level_z", "z_10", "z_20", "z_50", "keltner_dist_20",
        "rsi_2", "rsi_14", "pctb_20_2", "rsi_div_14", "vol_ratio",
    ]
    parts = []
    for contract in ("A", "B"):
        n = 60
        dates = pd.date_range("2023-01-02", periods=n, freq="B")
        df = pd.DataFrame({"date": dates, "contract": contract})
        df["close"] = 100.0 + np.cumsum(rng.normal(0, 1, n))
        df["open"] = df["close"].shift(1).fillna(df["close"])
        df["high"] = df[["open", "close"]].max(axis=1) + 0.5
        df["low"] = df[["open", "close"]].min(axis=1) - 0.5
        for f in feats:
            if f in ("er_20", "level_pct", "r2_20", "dir_persistence_20"):
                df[f] = rng.uniform(0.0, 1.0, n)
            elif f in ("rsi_2", "rsi_14"):
                df[f] = rng.uniform(0.0, 100.0, n)
            else:
                df[f] = rng.normal(0.0, 1.0, n)
        df["fwd_25"] = rng.normal(0.0, 2.0, n)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def test_score_series_is_point_in_time():
    base = _synthetic_family()
    s_base = score_series(base, "outrights", "A", horizon=25)

    # Append purely future rows (a new contract that only trades later). They must
    # not change any of A's already-computed, as-of scores.
    future = base[base["contract"] == "B"].copy()
    future["contract"] = "C"
    future["date"] = future["date"] + pd.Timedelta(days=400)
    extended = pd.concat([base, future], ignore_index=True)
    s_ext = score_series(extended, "outrights", "A", horizon=25)

    merged = s_base.merge(s_ext, on="date", suffixes=("_base", "_ext"))
    assert len(merged) == len(s_base)
    assert np.allclose(
        merged["opportunity_base"].to_numpy(),
        merged["opportunity_ext"].to_numpy(),
        equal_nan=True,
    )


def test_backtest_contract_smoke():
    base = _synthetic_family(seed=3)
    res = backtest_contract(base, "outrights", "A", horizon=25, cost=0.02)
    assert res.contract == "A"
    assert len(res.equity) == len(res.score_df)
    assert set(["n_trades", "total_pnl", "sharpe", "max_drawdown"]).issubset(res.stats)
    # Equity always reconciles with the ledger when there are trades.
    if not res.trades.empty:
        assert abs(res.trades["net_pnl"].sum() - res.equity.iloc[-1]) < 1e-6

