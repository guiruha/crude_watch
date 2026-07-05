"""Unit tests for the Phase 4 regime / Bollinger / dynamic TP-SL analytics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crudewatch.analytics.regime import (
    Regime,
    StrategyMode,
    bollinger_layers,
    detect_regime,
    risk_metrics,
    strategy_mode,
    trade_levels,
)
from crudewatch.analytics.structures import Level


def test_detect_regime_by_sign():
    assert detect_regime(pd.Series([0.4, 0.5, 0.6] * 10)) == Regime.BACKWARDATION
    assert detect_regime(pd.Series([-0.4, -0.5, -0.6] * 10)) == Regime.CONTANGO


def test_strategy_mode_mapping():
    assert strategy_mode(Regime.CONTANGO, Level.MONTHLY) == StrategyMode.MEAN_REVERSION
    assert strategy_mode(Regime.BACKWARDATION, Level.MONTHLY) == StrategyMode.BREAKOUT
    # butterflies always mean-revert, regardless of curve regime
    assert strategy_mode(Regime.BACKWARDATION, Level.BUTTERFLY) == StrategyMode.MEAN_REVERSION


def test_bollinger_layers_ordering():
    s = pd.Series(np.random.default_rng(0).normal(size=100))
    row = bollinger_layers(s, 20).iloc[-1]
    assert row["u3"] > row["u2"] > row["u1"] > row["mid"] > row["l1"] > row["l2"] > row["l3"]


def _touch(upper: bool, n: int = 60) -> pd.Series:
    """Near-flat series whose final point sits clearly beyond the ±2σ band."""
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(0, 0.05, n))
    s.iloc[-1] = 1.0 if upper else -1.0
    return s


def test_trade_levels_mean_reversion_short_above_upper():
    s = _touch(upper=True)
    bb = bollinger_layers(s, 20).iloc[-1]
    tl = trade_levels(s, StrategyMode.MEAN_REVERSION)
    assert tl.side == "Short"
    assert tl.take_profit == pytest.approx(bb["mid"])   # TP = the mean
    assert tl.stop_loss == pytest.approx(bb["u3"])      # SL = far band
    assert tl.risk_reward is not None and tl.risk_reward > 0


def test_trade_levels_mean_reversion_long_below_lower():
    s = _touch(upper=False)
    bb = bollinger_layers(s, 20).iloc[-1]
    tl = trade_levels(s, StrategyMode.MEAN_REVERSION)
    assert tl.side == "Long"
    assert tl.take_profit == pytest.approx(bb["mid"])
    assert tl.stop_loss == pytest.approx(bb["l3"])


def test_trade_levels_breakout_long_above_upper():
    s = _touch(upper=True)
    bb = bollinger_layers(s, 20).iloc[-1]
    tl = trade_levels(s, StrategyMode.BREAKOUT)
    assert tl.side == "Long"
    assert tl.take_profit == pytest.approx(bb["u3"])    # target ±3σ
    assert tl.stop_loss == pytest.approx(bb["u2"])      # stop back at ±2σ


def test_trade_levels_no_setup_inside_bands():
    flat = pd.Series(np.random.default_rng(2).normal(0, 0.2, 60))
    tl = trade_levels(flat, StrategyMode.MEAN_REVERSION)
    assert tl.side == "None"
    assert tl.risk_reward is None


def test_trade_levels_insufficient_history():
    assert trade_levels(pd.Series([1.0, 2, 3]), StrategyMode.BREAKOUT).side == "None"


def test_risk_metrics_bounds():
    s = pd.Series(np.random.default_rng(3).normal(size=120).cumsum())
    rm = risk_metrics(s)
    assert rm.atr is not None and rm.atr >= 0
    assert 0.0 <= rm.atr_percentile <= 100.0


def test_risk_metrics_insufficient_history_is_none():
    rm = risk_metrics(pd.Series([1.0, 2, 3]))
    assert rm.atr is None and rm.pctb is None
