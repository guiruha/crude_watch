"""Unit tests for the Phase 5 QUANT statistics and composite scoring engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crudewatch.analytics.indicators import Bias, IndicatorSignal
from crudewatch.analytics.indicators.statistical import (
    half_life,
    hurst_exponent,
    regime_diagnostics,
    variance_ratio,
)
from crudewatch.analytics.scoring import Regime, composite_score, weight_for


def _mean_reverting(n: int = 400, seed: int = 0) -> pd.Series:
    """AR(1) with strong mean reversion (phi ~ 0.3)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.3 * x[t - 1] + rng.normal(0, 1)
    return pd.Series(x)


def _random_walk(n: int = 400, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 1, n).cumsum())


# -- statistics ----------------------------------------------------------------

def test_hurst_lower_for_mean_reverting_than_random_walk():
    h_mr = hurst_exponent(_mean_reverting())
    h_rw = hurst_exponent(_random_walk())
    assert h_mr is not None and h_rw is not None
    assert h_mr < h_rw
    assert h_mr < 0.5           # anti-persistent


def test_half_life_positive_for_mean_reverting():
    hl = half_life(_mean_reverting())
    assert hl is not None and hl > 0


def test_variance_ratio_below_one_for_mean_reverting():
    vr = variance_ratio(_mean_reverting())
    assert vr is not None and vr < 1.0


def test_diagnostics_flags_mean_reversion():
    diag = regime_diagnostics(_mean_reverting())
    assert diag.is_mean_reverting is True


def test_statistics_none_on_short_series():
    short = pd.Series([1.0, 2, 3, 4])
    assert hurst_exponent(short) is None
    assert half_life(short) is None
    assert variance_ratio(short) is None


# -- composite scoring ---------------------------------------------------------

def _sig(name: str, bias: Bias) -> IndicatorSignal:
    return IndicatorSignal(name, bias)


def test_weight_regime_dependence():
    # statistical weighted up in contango, down in backwardation; trend the reverse
    assert weight_for("Z-Score", Regime.CONTANGO) > weight_for("Z-Score", Regime.BACKWARDATION)
    assert weight_for("MA ribbon", Regime.BACKWARDATION) > weight_for("MA ribbon", Regime.CONTANGO)


def test_composite_all_bullish_is_max():
    sigs = [_sig(n, Bias.BULLISH) for n in ("MA ribbon", "MACD", "Bollinger %B", "Z-Score")]
    cs = composite_score(sigs, Regime.CONTANGO)
    assert cs.value == pytest.approx(100.0)
    assert cs.color_token == "strong_up"


def test_composite_all_bearish_is_min():
    sigs = [_sig(n, Bias.BEARISH) for n in ("MA ribbon", "MACD", "Bollinger %B", "Z-Score")]
    cs = composite_score(sigs, Regime.CONTANGO)
    assert cs.value == pytest.approx(-100.0)


def test_composite_bounded_and_neutral_center():
    mixed = [_sig("MA ribbon", Bias.BULLISH), _sig("Z-Score", Bias.BEARISH),
             _sig("MACD", Bias.NEUTRAL), _sig("Bollinger %B", Bias.NEUTRAL)]
    cs = composite_score(mixed, Regime.CONTANGO)
    assert -100.0 <= cs.value <= 100.0


def test_composite_regime_tilts_score():
    # One bullish trend vote vs one bearish statistical vote: contango favours the
    # statistical (bearish) read; backwardation favours the trend (bullish) read.
    sigs = [_sig("MA ribbon", Bias.BULLISH), _sig("Z-Score", Bias.BEARISH)]
    contango = composite_score(sigs, Regime.CONTANGO).value
    backwardation = composite_score(sigs, Regime.BACKWARDATION).value
    assert contango < 0 < backwardation


def test_composite_empty_is_zero():
    cs = composite_score([], Regime.CONTANGO)
    assert cs.value == 0.0
    assert cs.color_token == "neutral"
