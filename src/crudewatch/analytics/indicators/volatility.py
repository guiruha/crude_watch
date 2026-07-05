"""Volatility & statistical indicators for a close-only spread series.

A fixed-date spread is a linear combination of outright *closes*, so it has no
meaningful intraday high/low. True range therefore falls back to the
close-to-close move ``|Δclose|`` — the standard proxy for close-only series.

The star of this module is the Z-Score: for mean-reverting spreads a reading of
``+2σ`` flags "rich" (bias to fall, i.e. bearish on the spread) and ``-2σ``
flags "cheap" (bias to rise, bullish).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crudewatch.analytics.indicators import Bias, IndicatorSignal


def true_range(close: pd.Series) -> pd.Series:
    """Close-to-close true range, ``|Δclose|`` (no high/low available)."""
    return close.diff().abs()


def atr(close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range via Wilder smoothing of the close-based true range."""
    return true_range(close).ewm(alpha=1 / period, adjust=False).mean()


def bollinger(close: pd.Series, period: int = 20, mult: float = 2.0) -> pd.DataFrame:
    """Bollinger bands plus %B and bandwidth.

    ``pctb`` is 0 at the lower band and 1 at the upper; ``bandwidth`` is the
    band width normalised by the mid (its contraction marks a squeeze).
    """
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + mult * std
    lower = mid - mult * std
    width = (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        "mid": mid,
        "upper": upper,
        "lower": lower,
        "pctb": (close - lower) / width,
        "bandwidth": (upper - lower) / mid.replace(0, np.nan),
    })


def zscore(close: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: standard deviations from the trailing mean."""
    mean = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0).replace(0, np.nan)
    return (close - mean) / std


def zscore_dual(close: pd.Series, short: int = 20, long: int = 60) -> pd.DataFrame:
    """Short- and long-window z-scores side by side."""
    return pd.DataFrame({f"z{short}": zscore(close, short), f"z{long}": zscore(close, long)})


def zscore_signal(close: pd.Series, window: int = 60, threshold: float = 2.0) -> IndicatorSignal:
    """Mean-reversion vote: rich (+σ) -> bearish, cheap (-σ) -> bullish."""
    z = zscore(close, window).iloc[-1] if len(close) >= window else np.nan
    if pd.isna(z):
        return IndicatorSignal("Z-Score", Bias.NEUTRAL, None, "insufficient history")
    if z >= threshold:
        return IndicatorSignal("Z-Score", Bias.BEARISH, float(z), f"+{z:.1f}σ rich")
    if z <= -threshold:
        return IndicatorSignal("Z-Score", Bias.BULLISH, float(z), f"{z:.1f}σ cheap")
    return IndicatorSignal("Z-Score", Bias.NEUTRAL, float(z), f"{z:+.1f}σ")


def bollinger_signal(close: pd.Series, period: int = 20, mult: float = 2.0) -> IndicatorSignal:
    """Mean-reversion vote from %B: above the upper band -> bearish, below lower -> bullish."""
    if len(close) < period:
        return IndicatorSignal("Bollinger %B", Bias.NEUTRAL, None, "insufficient history")
    pctb = bollinger(close, period, mult)["pctb"].iloc[-1]
    if pd.isna(pctb):
        return IndicatorSignal("Bollinger %B", Bias.NEUTRAL, None, "insufficient history")
    if pctb >= 1:
        return IndicatorSignal("Bollinger %B", Bias.BEARISH, float(pctb), "above upper band")
    if pctb <= 0:
        return IndicatorSignal("Bollinger %B", Bias.BULLISH, float(pctb), "below lower band")
    return IndicatorSignal("Bollinger %B", Bias.NEUTRAL, float(pctb), f"%B={pctb:.2f}")


__all__ = [
    "true_range", "atr", "bollinger", "zscore", "zscore_dual",
    "zscore_signal", "bollinger_signal",
]
