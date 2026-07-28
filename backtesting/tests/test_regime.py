"""Tests for the regime-gated backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.research.regime import (
    _regime_mask,
    _regime_thresholds,
    evaluate_feature_regime,
    evaluate_gated_strategy,
)


def test_regime_thresholds_and_mask_terciles():
    er = np.arange(1, 10, dtype=float)  # 1..9 -> terciles at 3.67 and 6.33
    lo, hi = _regime_thresholds(er)
    assert lo < hi
    rng = _regime_mask(er, lo, hi, "range")
    trd = _regime_mask(er, lo, hi, "trend")
    dead = ~(rng | trd)
    # Bottom tercile is range, top is trend, middle is the untraded dead zone.
    assert rng.sum() > 0 and trd.sum() > 0 and dead.sum() > 0
    assert er[rng].max() <= lo and er[trd].min() >= hi


def _regime_panel(seed: int = 0) -> pd.DataFrame:
    """Panel where reversion pays ONLY in the low-ER (range) regime and
    continuation pays ONLY in the high-ER (trend) regime."""
    rng = np.random.default_rng(seed)
    parts = []
    for v in range(2005, 2020):  # 15 vintages
        n = 300
        er = rng.uniform(0, 1, n)
        is_trend = er >= 0.66
        is_range = er <= 0.33
        z = rng.normal(0, 1, n)          # reversion feature
        slope = rng.normal(0, 1, n)      # continuation feature
        noise = rng.normal(0, 1, n)
        fwd = noise.copy()
        # Reversion (negative loading on z) only where the regime is 'range'.
        fwd[is_range] += -1.2 * z[is_range]
        # Continuation (positive loading on slope) only where regime is 'trend'.
        fwd[is_trend] += 1.2 * slope[is_trend]
        mfe = np.maximum(fwd, 0.0) + np.abs(rng.normal(0, 0.5, n))
        mae = np.minimum(fwd, 0.0) - np.abs(rng.normal(0, 0.5, n))
        parts.append(pd.DataFrame({
            "vintage": v,
            "contract": f"CL{v}",
            "date": pd.date_range("2004-01-01", periods=n, freq="B"),
            "er_20": er,
            "z_20": z,
            "slope_20": slope,
            "fwd_5": fwd,
            "mfe_5": mfe,
            "mae_5": mae,
        }))
    return pd.concat(parts, ignore_index=True)


def test_reversion_edge_is_stronger_inside_range_regime():
    df = _regime_panel()
    in_range = evaluate_feature_regime(df, "z_20", 5, "range", min_train=3)
    in_trend = evaluate_feature_regime(df, "z_20", 5, "trend", min_train=3)
    assert in_range is not None and in_trend is not None
    assert in_range["regime"] == "range"
    # z_20 reverts strongly in range, but carries no edge in trend.
    assert in_range["ic_mean"] < -0.2
    assert in_range["ic_mean"] < in_trend["ic_mean"]
    assert abs(in_trend["ic_mean"]) < 0.15


def test_continuation_edge_is_stronger_inside_trend_regime():
    df = _regime_panel()
    in_range = evaluate_feature_regime(df, "slope_20", 5, "range", min_train=3)
    in_trend = evaluate_feature_regime(df, "slope_20", 5, "trend", min_train=3)
    assert in_range is not None and in_trend is not None
    assert in_trend["ic_mean"] > 0.2
    assert in_trend["ic_mean"] > in_range["ic_mean"]


def test_gated_strategy_picks_the_right_feature_per_regime_and_profits():
    df = _regime_panel()
    res = evaluate_gated_strategy(
        df, 5, reversion=["z_20"], continuation=["slope_20"], min_train=3,
    )
    assert res is not None
    assert res["range_feature"] == "z_20"
    assert res["trend_feature"] == "slope_20"
    assert res["n_trades"] > 0
    assert res["avg_pnl"] > 0        # both legs trade with their true edge
    assert res["sharpe"] > 0
    assert res["mfe_mean"] >= 0 >= res["mae_mean"]


def _regime_panel_with_level(seed: int = 0) -> pd.DataFrame:
    """Regime panel plus a ``level_z`` that only sometimes agrees with the signal."""
    df = _regime_panel(seed)
    rng = np.random.default_rng(seed + 7)
    df["level_z"] = rng.normal(0, 1, len(df))
    return df


def test_confirmation_reduces_trade_count():
    df = _regime_panel_with_level()
    base = evaluate_gated_strategy(
        df, 5, reversion=["z_20"], continuation=["slope_20"], min_train=3, confirm=False,
    )
    conf = evaluate_gated_strategy(
        df, 5, reversion=["z_20"], continuation=["slope_20"], min_train=3,
        confirm=True, confirm_feature="level_z",
    )
    assert base is not None and conf is not None
    assert conf["confirmed"] is True and base["confirmed"] is False
    # A random confirmator only lets a subset of signals through.
    assert conf["n_trades"] < base["n_trades"]


def test_confirmation_noop_when_feature_absent():
    # No level_z column -> confirm silently disabled, identical to unconfirmed.
    df = _regime_panel()
    base = evaluate_gated_strategy(df, 5, reversion=["z_20"], continuation=["slope_20"], min_train=3)
    asked = evaluate_gated_strategy(
        df, 5, reversion=["z_20"], continuation=["slope_20"], min_train=3, confirm=True,
    )
    assert asked is not None and asked["confirmed"] is False
    assert asked["n_trades"] == base["n_trades"]


def test_gated_strategy_none_when_no_edge():
    # Pure noise target -> no feature has a usable signed IC on train -> no trades.
    rng = np.random.default_rng(1)
    parts = []
    for v in range(2005, 2018):
        n = 300
        parts.append(pd.DataFrame({
            "vintage": v,
            "contract": f"CL{v}",
            "date": pd.date_range("2004-01-01", periods=n, freq="B"),
            "er_20": rng.uniform(0, 1, n),
            "z_20": rng.normal(0, 1, n),
            "slope_20": rng.normal(0, 1, n),
            "fwd_5": rng.normal(0, 1, n),
            "mfe_5": np.abs(rng.normal(0, 1, n)),
            "mae_5": -np.abs(rng.normal(0, 1, n)),
        }))
    df = pd.concat(parts, ignore_index=True)
    res = evaluate_gated_strategy(df, 5, reversion=["z_20"], continuation=["slope_20"], min_train=3)
    # Either no trades at all, or a Sharpe indistinguishable from zero.
    assert res is None or abs(res["avg_pnl"]) < 0.1
