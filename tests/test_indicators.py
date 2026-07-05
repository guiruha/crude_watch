"""Unit tests for the Phase 2 CORE indicator library."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crudewatch.analytics.indicators import (
    Bias,
    IndicatorSignal,
    atr,
    bollinger,
    ema,
    ma_signal,
    macd,
    rsi,
    signal_panel,
    sma,
    supertrend,
    supertrend_signal,
    zscore,
    zscore_signal,
)


@pytest.fixture
def uptrend() -> pd.Series:
    return pd.Series(np.linspace(0.0, 25.0, 260))


@pytest.fixture
def downtrend() -> pd.Series:
    return pd.Series(np.linspace(25.0, 0.0, 260))


def test_sma_matches_manual():
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert sma(s, 3).iloc[-1] == pytest.approx(4.0)


def test_ema_first_value_is_seed():
    s = pd.Series([1.0, 2, 3, 4])
    assert ema(s, 3).iloc[0] == pytest.approx(1.0)


def test_rsi_bounds_and_all_gains():
    up = pd.Series(np.arange(1, 50, dtype=float))
    r = rsi(up)
    assert r.dropna().between(0, 100).all()
    assert r.iloc[-1] == pytest.approx(100.0)


def test_rsi_all_losses_is_zero():
    down = pd.Series(np.arange(50, 1, -1, dtype=float))
    assert rsi(down).iloc[-1] == pytest.approx(0.0)


def test_zscore_matches_definition():
    s = pd.Series(np.arange(100, dtype=float))
    z = zscore(s, 20)
    window = s.iloc[-20:]
    expected = (s.iloc[-1] - window.mean()) / window.std(ddof=0)
    assert z.iloc[-1] == pytest.approx(expected)


def test_bollinger_pctb_at_bands():
    s = pd.Series(np.random.default_rng(0).normal(size=200))
    bb = bollinger(s, 20)
    pctb = bb["pctb"].dropna()
    assert (pctb.between(-0.5, 1.5)).mean() > 0.95  # mostly within a sane range


def test_atr_positive():
    s = pd.Series(np.random.default_rng(1).normal(size=100).cumsum())
    assert (atr(s).dropna() >= 0).all()


def test_macd_positive_hist_on_uptrend(uptrend):
    assert macd(uptrend)["hist"].iloc[-1] > 0


def test_ma_signal_bullish_on_uptrend(uptrend):
    sig = ma_signal(uptrend)
    assert sig.bias == Bias.BULLISH


def test_ma_signal_bearish_on_downtrend(downtrend):
    assert ma_signal(downtrend).bias == Bias.BEARISH


def test_supertrend_direction_domain(uptrend):
    d = supertrend(uptrend)["direction"]
    assert set(d.unique()) <= {-1, 1}
    assert supertrend_signal(uptrend).bias == Bias.BULLISH


def test_zscore_signal_mean_reversion_direction():
    base = pd.Series(np.arange(61, dtype=float) * 0.001)
    rich = base.copy()
    rich.iloc[-1] += 5.0
    cheap = base.copy()
    cheap.iloc[-1] -= 5.0
    assert zscore_signal(rich).bias == Bias.BEARISH   # rich -> expect fall
    assert zscore_signal(cheap).bias == Bias.BULLISH  # cheap -> expect rise


def test_zscore_signal_insufficient_history_is_neutral():
    assert zscore_signal(pd.Series([1.0, 2, 3])).bias == Bias.NEUTRAL


def test_signals_are_indicator_signals(uptrend):
    panel = signal_panel(uptrend)
    families = {v.family for v in panel}
    assert families == {"Trend", "Momentum", "Volatility", "Statistical"}
    for verdict in panel:
        assert all(isinstance(s, IndicatorSignal) for s in verdict.signals)
        assert 0.0 <= verdict.conviction <= 1.0
        assert verdict.bias in (Bias.BULLISH, Bias.BEARISH, Bias.NEUTRAL)


def test_signal_panel_trend_bullish_on_uptrend(uptrend):
    trend = next(v for v in signal_panel(uptrend) if v.family == "Trend")
    assert trend.bias == Bias.BULLISH
    assert trend.conviction > 0
