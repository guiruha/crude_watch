"""Tests for the walk-forward evaluation harness."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.research.evaluate import (
    _confidence,
    calendar_daily_sharpe,
    evaluate_feature,
    walk_forward_splits,
)


def test_confidence_is_zero_without_trades():
    # A strong, stable IC that never actually trades must score zero confidence
    # (regression guard for the trade_factor bug that returned 1.0 at n_trades=0).
    ics = [-0.3, -0.32, -0.29, -0.31]
    conf = _confidence(ics, ic_t=-6.0, n_folds=len(ics), n_trades=0)
    assert conf["confidence"] == 0.0


def test_calendar_daily_sharpe_sums_concurrent_trades():
    dates = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03"])
    # Two winning trades on day 1, then steady positive days -> positive Sharpe.
    assert calendar_daily_sharpe([1.0, 1.0, 1.0, 1.0], dates) > 0
    # Degenerate inputs are NaN, not an exception.
    assert np.isnan(calendar_daily_sharpe([1.0], dates[:1]))


def test_walk_forward_splits_expanding():
    splits = walk_forward_splits([2015, 2011, 2011, 2013, 2014, 2012], min_train=3)
    # Distinct sorted vintages: 2011..2015. min_train=3 -> test 2014, then 2015.
    assert splits == [([2011, 2012, 2013], 2014), ([2011, 2012, 2013, 2014], 2015)]


def _synthetic_reversion(beta: float = -0.6, seed: int = 0) -> pd.DataFrame:
    """Multi-vintage panel where a high feature predicts a negative forward move."""
    rng = np.random.default_rng(seed)
    parts = []
    for v in range(2008, 2020):  # 12 vintages
        n = 250
        feat = rng.normal(0, 1, n)
        fwd = beta * feat + rng.normal(0, 1, n)
        parts.append(pd.DataFrame({"vintage": v, "z_20": feat, "fwd_5": fwd}))
    return pd.concat(parts, ignore_index=True)


def test_reversion_feature_has_negative_ic_and_monotonicity():
    df = _synthetic_reversion(beta=-0.6)
    stats = evaluate_feature(df, "z_20", horizon=5, n_buckets=5, min_train=3, cost=0.0)

    assert stats is not None
    assert stats["ic_mean"] < -0.2          # strong negative IC (mean reversion)
    assert stats["ic_t"] < -3               # stable across walk-forward folds
    assert stats["monotonicity"] < -0.8     # monotone decreasing bucket profile
    assert stats["gross_spread"] > 0        # cheap bucket outperforms expensive bucket
    assert stats["n_vintages"] == 12
    assert stats["n_folds"] == 9            # 12 vintages, min_train=3 -> 9 OOS folds


def test_noise_feature_has_near_zero_ic():
    df = _synthetic_reversion(beta=0.0)  # feature carries no information
    stats = evaluate_feature(df, "z_20", horizon=5, n_buckets=5, min_train=3, cost=0.0)
    assert stats is not None
    assert abs(stats["ic_mean"]) < 0.1
    assert abs(stats["ic_t"]) < 3           # not a stable edge


def test_reversion_probability_and_confidence():
    # Confidence now requires the edge to actually TRADE (trade_factor=0 when
    # there are no trades), so use the contract-aware panel.
    df = _synthetic_reversion_contracts(beta=-0.6)
    stats = evaluate_feature(df, "z_20", 5, n_buckets=5, min_train=3)
    # Extreme buckets revert more often than not, and the edge is confident.
    assert stats["p_reversion"] > 0.5
    assert np.isclose(stats["p_reversion"] + stats["p_continuation"], 1.0)
    assert stats["n_trades"] > 0
    assert stats["confidence"] > 50
    assert stats["sign_consistency"] > 0.8
    assert stats["ic_ci_high"] < 0          # whole IC CI is negative (reversion)


def test_noise_has_low_confidence_and_even_probability():
    df = _synthetic_reversion(beta=0.0)
    stats = evaluate_feature(df, "z_20", 5, n_buckets=5, min_train=3)
    assert abs(stats["p_reversion"] - 0.5) < 0.1
    assert stats["confidence"] < 40


def test_cost_reduces_net_spread():
    df = _synthetic_reversion(beta=-0.6)
    gross = evaluate_feature(df, "z_20", 5, cost=0.0)
    netted = evaluate_feature(df, "z_20", 5, cost=0.3)
    assert np.isclose(netted["net_spread"], gross["net_spread"] - 0.3)


def test_trade_stats_absent_without_contract_columns():
    # A bare (feature, target) panel has no contract/date/MFE/MAE -> no trades.
    df = _synthetic_reversion(beta=-0.6)
    stats = evaluate_feature(df, "z_20", 5)
    assert stats["n_trades"] == 0
    assert np.isnan(stats["sharpe"])
    assert np.isnan(stats["mfe_mean"])
    assert np.isnan(stats["mae_mean"])


def _synthetic_reversion_contracts(beta: float = -0.6, seed: int = 0) -> pd.DataFrame:
    """Reversion panel with real contract/date and MFE/MAE columns for trading."""
    rng = np.random.default_rng(seed)
    parts = []
    for v in range(2008, 2020):
        n = 250
        feat = rng.normal(0, 1, n)
        fwd = beta * feat + rng.normal(0, 1, n)
        noise = np.abs(rng.normal(0, 1, n))
        parts.append(pd.DataFrame({
            "vintage": v,
            "contract": f"CL{v}",
            "date": pd.date_range("2007-01-01", periods=n, freq="B"),
            "z_20": feat,
            "fwd_5": fwd,
            "mfe_5": np.maximum(fwd, 0.0) + noise,   # favourable-for-long excursion
            "mae_5": np.minimum(fwd, 0.0) - noise,   # adverse-for-long excursion
        }))
    return pd.concat(parts, ignore_index=True)


def test_trade_stats_present_and_oriented_with_contracts():
    df = _synthetic_reversion_contracts(beta=-0.6)
    stats = evaluate_feature(df, "z_20", 5, n_buckets=5, min_train=3)

    assert stats["n_trades"] > 0
    assert 0.0 <= stats["win_rate"] <= 1.0
    # Reversion edge should be profitable and give a positive Sharpe.
    assert stats["avg_pnl"] > 0
    assert stats["sharpe"] > 0
    # Excursions are oriented by trade side: favourable >= 0 >= adverse.
    assert stats["mfe_mean"] >= 0
    assert stats["mae_mean"] <= 0


def test_non_overlapping_trades_are_capped_per_contract():
    # 250 business days, horizon 5, one contract per vintage: at most ceil(250/5)
    # extreme-bucket trades per contract even if every row were a signal.
    df = _synthetic_reversion_contracts(beta=-0.6)
    stats = evaluate_feature(df, "z_20", 5, min_train=3)
    n_test_contracts = 12 - 3  # 12 vintages, min_train=3 -> 9 OOS folds/contracts
    assert stats["n_trades"] <= n_test_contracts * (250 // 5 + 1)
