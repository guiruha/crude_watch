"""Tests for the continuous feature matrix, including a look-ahead guard."""
from __future__ import annotations

import numpy as np
import pandas as pd

from crudewatch.research.features import (
    FEATURE_NAMES,
    add_features,
    bollinger_pctb,
    divergence,
    momentum_deceleration,
    vol_ratio,
    zscore,
)


def _series(closes: list[float], contract: str = "A") -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="B"),
            "contract": [contract] * n,
            "close": closes,
        }
    )


def test_zscore_values():
    c = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = zscore(c, 3)
    # At index 2, window [1,2,3]: mean 2, std (ddof=1) = 1 -> (3-2)/1 = 1.
    assert z.iloc[2] == 1.0
    assert np.isnan(z.iloc[0])  # warmup


def test_bollinger_pctb_bounds():
    rng = np.random.default_rng(0)
    c = pd.Series(rng.normal(50, 5, 300))
    pctb = bollinger_pctb(c, 20, 2.0)
    mid = pctb.dropna()
    # Most values fall inside the bands (0..1); it is a bounded-ish oscillator.
    assert (mid.between(-0.5, 1.5)).mean() > 0.95


def test_add_features_columns_present():
    rng = np.random.default_rng(1)
    df = _series(list(50 + np.cumsum(rng.normal(0, 1, 120))))
    out = add_features(df)
    for name in FEATURE_NAMES:
        assert name in out.columns
    # Late rows should have all features defined (past the longest warmup).
    assert out[FEATURE_NAMES].iloc[-1].notna().all()


def test_no_lookahead_appending_future_bars():
    """Appending future bars must not change any past feature value."""
    rng = np.random.default_rng(2)
    closes = list(50 + np.cumsum(rng.normal(0, 1, 200)))

    short = add_features(_series(closes[:150]))
    long = add_features(_series(closes))  # 50 extra future bars

    a = short[FEATURE_NAMES].to_numpy()
    b = long[FEATURE_NAMES].iloc[:150].to_numpy()
    # Identical where both are defined (NaNs line up in the warmup).
    assert np.allclose(a, b, equal_nan=True)


def test_bloque_e_features_registered_and_defined():
    """The new exhaustion/reversion features exist and resolve past their warmup."""
    for name in ("rsi_div_14", "macd_div", "mom_decel_10", "er_drop_20", "vol_ratio"):
        assert name in FEATURE_NAMES
    rng = np.random.default_rng(3)
    df = _series(list(50 + np.cumsum(rng.normal(0, 1, 200))))
    out = add_features(df)
    for name in ("rsi_div_14", "macd_div", "mom_decel_10", "er_drop_20", "vol_ratio"):
        assert out[name].iloc[-1] == out[name].iloc[-1]  # not NaN past warmup


def test_divergence_self_agreement_is_zero():
    # A feature diverging from itself (or an affine copy) shows no divergence.
    rng = np.random.default_rng(4)
    close = pd.Series(50 + np.cumsum(rng.normal(0, 1, 200)))
    same = divergence(close, close, 14, 60).dropna()
    affine = divergence(close, 2.0 * close + 3.0, 14, 60).dropna()
    assert np.allclose(same, 0.0, atol=1e-9)
    assert np.allclose(affine, 0.0, atol=1e-9)  # z-score is scale/shift invariant


def test_vol_ratio_detects_expansion():
    rng = np.random.default_rng(5)
    low = rng.normal(0, 0.2, 150)   # long calm regime
    high = rng.normal(0, 6.0, 20)   # short volatile burst (< the 50-bar long window)
    close = pd.Series(50 + np.cumsum(np.r_[low, high]))
    vr = vol_ratio(close, 10, 50).dropna()
    assert vr.iloc[-1] > 1.3  # recent burst more volatile than the long run


def test_momentum_deceleration_negative_when_slowing():
    # sqrt path: ever-shrinking increments -> momentum always decelerating.
    close = pd.Series(50 + np.sqrt(np.arange(1, 201)) * 5.0)
    md = momentum_deceleration(close, 10, 5, 10).dropna()
    assert md.iloc[-1] < 0


def test_direction_regime_features_registered_and_defined():
    """Direction / regime enrichment features exist and resolve past warmup."""
    for name in (
        "ema_align",
        "mom_5",
        "mom_10",
        "mom_20",
        "variance_ratio_5",
        "autocorr_20",
        "r2_20",
    ):
        assert name in FEATURE_NAMES
    rng = np.random.default_rng(6)
    df = _series(list(50 + np.cumsum(rng.normal(0, 1, 200))))
    out = add_features(df)
    for name in (
        "ema_align",
        "mom_5",
        "mom_10",
        "mom_20",
        "variance_ratio_5",
        "autocorr_20",
        "r2_20",
    ):
        assert out[name].iloc[-1] == out[name].iloc[-1]  # not NaN past warmup


def test_r2_and_momentum_on_increasing_path():
    """Strictly increasing prices -> high r2_20 and positive mom_10."""
    closes = list(50.0 + np.arange(1, 101))
    out = add_features(_series(closes))
    r2 = out["r2_20"].dropna()
    assert r2.iloc[-1] == r2.iloc[-1]
    assert r2.iloc[-1] > 0.95
    mom = out["mom_10"].dropna()
    assert mom.iloc[-1] > 0


def test_no_leak_across_contracts():
    a = _series(list(np.arange(1.0, 61.0)), contract="A")
    b = _series(list(np.arange(100.0, 40.0, -1.0)), contract="B")
    out = add_features(pd.concat([a, b], ignore_index=True))
    # Each contract's slope sign reflects only its own path.
    a_slope = out.loc[out["contract"] == "A", "slope_20"].dropna()
    b_slope = out.loc[out["contract"] == "B", "slope_20"].dropna()
    assert (a_slope > 0).all()
    assert (b_slope < 0).all()
