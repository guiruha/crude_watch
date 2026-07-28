"""Tests for WS4 (regime anatomy), WS5 (direction/quality), WS8 (costs)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.research.costs import cost_operability
from backtesting.research.quality import direction_breakdown, trend_quality_gradient
from backtesting.research.regime import regime_profile

RNG = np.random.default_rng(7)


# --------------------------------------------------------------------------- #
# WS4 - regime anatomy
# --------------------------------------------------------------------------- #
def _regime_frame(n: int = 900, contracts: int = 3) -> pd.DataFrame:
    rows = []
    per = n // contracts
    for c in range(contracts):
        er = RNG.uniform(0, 1, per)
        rows.append(pd.DataFrame({
            "contract": [f"C{c}"] * per,
            "date": pd.date_range("2018-01-01", periods=per, freq="B"),
            "er_20": er,
            "fwd_10": RNG.normal(0, 1, per),
        }))
    return pd.concat(rows, ignore_index=True)


def test_regime_profile_occupancy_and_transitions_are_probabilities():
    data = _regime_frame()
    profile, transitions = regime_profile(data, "fam", 10)

    # Occupancy of the three regimes sums to 1.
    assert np.isclose(profile["occupancy"].sum(), 1.0)
    assert set(profile["regime"]) == {"range", "dead", "trend"}

    # Every transition row (from-regime with data) sums to 1 across destinations.
    for frm, block in transitions.groupby("from"):
        if block["n"].iloc[0] > 0:
            assert np.isclose(block["prob"].sum(), 1.0)


def test_regime_transitions_do_not_bleed_across_contracts():
    # Contract A always low ER (range), contract B always high ER (trend). With no
    # within-contract regime changes, off-diagonal transitions must stay ~0.
    a = pd.DataFrame({
        "contract": ["A"] * 100, "date": pd.date_range("2018-01-01", periods=100, freq="B"),
        "er_20": np.full(100, 0.05), "fwd_10": RNG.normal(0, 1, 100),
    })
    b = pd.DataFrame({
        "contract": ["B"] * 100, "date": pd.date_range("2018-01-01", periods=100, freq="B"),
        "er_20": np.full(100, 0.95), "fwd_10": RNG.normal(0, 1, 100),
    })
    _, transitions = regime_profile(pd.concat([a, b], ignore_index=True), "fam", 10)
    piv = transitions.pivot_table(index="from", columns="to", values="prob")
    # range -> trend should never happen (contracts never switch).
    assert piv.loc["range", "trend"] == 0.0
    assert piv.loc["trend", "range"] == 0.0


# --------------------------------------------------------------------------- #
# WS5 - direction + trend quality
# --------------------------------------------------------------------------- #
def _trade_frame() -> pd.DataFrame:
    rows = []
    for v in range(2010, 2017):  # 7 vintages -> several OOS folds
        for c in range(2):
            per = 40
            feat = RNG.normal(0, 1, per)
            fwd = -feat + RNG.normal(0, 0.2, per)  # reversion: low feat -> +fwd
            rows.append(pd.DataFrame({
                "vintage": [v] * per,
                "contract": [f"{v}-{c}"] * per,
                "date": pd.date_range("2010-01-01", periods=per, freq="B"),
                "z_20": feat,
                "fwd_10": fwd,
                "mfe_10": np.abs(fwd),
                "mae_10": -np.abs(fwd),
            }))
    return pd.concat(rows, ignore_index=True)


def test_direction_breakdown_legs_sum_to_total():
    data = _trade_frame()
    res = direction_breakdown(data, "z_20", 10, cost=0.0, min_fold_rows=20)
    assert res is not None
    assert res["n_long"] + res["n_short"] == res["n_trades"]
    # A clean reversion edge should win on both legs.
    assert res["n_long"] > 0 and res["n_short"] > 0
    assert 0.0 <= res["win_long"] <= 1.0
    assert 0.0 <= res["win_short"] <= 1.0


def test_trend_quality_gradient_deters_low_er_reversion():
    n = 1500
    er = RNG.uniform(0, 1, n)
    z = RNG.normal(0, 1, n)
    # Reversion signal only in the low-ER (choppy) rows.
    fwd = np.where(er < 0.4, -2.0 * z, 0.0) + RNG.normal(0, 0.3, n)
    data = pd.DataFrame({"er_20": er, "z_20": z, "slope_20": RNG.normal(0, 1, n), "fwd_10": fwd})

    grad = trend_quality_gradient(data, "fam", 10, n_quantiles=5)
    rev = grad[grad["kind"] == "reversion"].set_index("er_bin")["ic"]
    assert rev.loc[1] < -0.3        # strong reversion in Q1 (low ER)
    assert abs(rev.loc[5]) < 0.2    # ~flat in Q5 (high ER)


# --------------------------------------------------------------------------- #
# WS8 - costs & operability
# --------------------------------------------------------------------------- #
def test_cost_operability_breakeven_and_sensitivity():
    # avg_pnl at 1x stub = 0.08, stub = 0.02 -> gross = 0.10.
    out = cost_operability(0.08, 1.5, n_trades=200, stub_cost=0.02, years=4.0, horizon=10)
    assert np.isclose(out["gross_pnl"], 0.10)
    assert np.isclose(out["breakeven_cost"], 0.10)
    assert np.isclose(out["safety_margin"], 5.0)          # 0.10 / 0.02
    assert np.isclose(out["trades_per_year"], 50.0)       # 200 / 4
    assert out["holding_days"] == 10

    # Sensitivity: 0x keeps full gross, 1x matches input, higher cost shrinks P&L.
    assert np.isclose(out["pnl_0x"], 0.10)
    assert np.isclose(out["pnl_1x"], 0.08)
    assert np.isclose(out["pnl_2x"], 0.06)
    # Sharpe scales with the mean (dispersion is cost-invariant).
    assert np.isclose(out["sharpe_1x"], 1.5)
    assert np.isclose(out["sharpe_2x"], 1.5 * 0.06 / 0.08)


def test_cost_operability_safety_margin_below_one_flags_weak_edge():
    # Gross barely above cost: safety margin < 1 means the stub already kills it.
    out = cost_operability(-0.005, 0.2, n_trades=50, stub_cost=0.02, years=3.0, horizon=5)
    assert out["gross_pnl"] < out["stub_cost"]
    assert out["safety_margin"] < 1.0
