"""Momentum indicators for a close-only spread series.

RSI uses Wilder smoothing and is made *adaptive* by wrapping Bollinger bands
around the RSI itself, so overbought/oversold thresholds calibrate to the
spread's own history instead of fixed 70/30. MACD carries a lightweight
pivot-based divergence detector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crudewatch.analytics.indicators import Bias, IndicatorSignal


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI in [0, 100]."""
    change = close.diff()
    gain = change.clip(lower=0)
    loss = (-change).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(avg_loss != 0, 100.0)


def adaptive_rsi_bands(close: pd.Series, period: int = 14, bb_period: int = 20, mult: float = 2.0) -> pd.DataFrame:
    """RSI with Bollinger bands on the RSI (adaptive overbought/oversold)."""
    r = rsi(close, period)
    mid = r.rolling(bb_period, min_periods=bb_period).mean()
    std = r.rolling(bb_period, min_periods=bb_period).std(ddof=0)
    return pd.DataFrame({"rsi": r, "upper": mid + mult * std, "lower": mid - mult * std})


def rsi_signal(close: pd.Series, period: int = 14, bb_period: int = 20, mult: float = 2.0) -> IndicatorSignal:
    """Mean-reversion vote using adaptive bands, with a 70/30 fallback for short series."""
    if len(close) < period + 1:
        return IndicatorSignal("RSI (adaptive)", Bias.NEUTRAL, None, "insufficient history")
    bands = adaptive_rsi_bands(close, period, bb_period, mult).iloc[-1]
    r = bands["rsi"]
    if pd.isna(r):
        return IndicatorSignal("RSI (adaptive)", Bias.NEUTRAL, None, "insufficient history")
    upper, lower = bands["upper"], bands["lower"]
    if pd.isna(upper) or pd.isna(lower):
        upper, lower = 70.0, 30.0
    if r >= upper:
        return IndicatorSignal("RSI (adaptive)", Bias.BEARISH, float(r), f"RSI {r:.0f} overbought")
    if r <= lower:
        return IndicatorSignal("RSI (adaptive)", Bias.BULLISH, float(r), f"RSI {r:.0f} oversold")
    return IndicatorSignal("RSI (adaptive)", Bias.NEUTRAL, float(r), f"RSI {r:.0f}")


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    macd_line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": macd_line - signal_line})


def _pivots(series: pd.Series, left: int = 5, right: int = 5, kind: str = "low") -> pd.Series:
    """Boolean series marking local pivot lows (or highs) with ``left/right`` bars each side."""
    values = series.to_numpy()
    flags = np.zeros(len(values), dtype=bool)
    for i in range(left, len(values) - right):
        window = values[i - left:i + right + 1]
        centre = values[i]
        if np.isnan(centre) or np.isnan(window).any():
            continue
        if kind == "low" and centre == window.min():
            flags[i] = True
        elif kind == "high" and centre == window.max():
            flags[i] = True
    return pd.Series(flags, index=series.index)


def macd_divergence(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9,
                    left: int = 5, right: int = 5) -> int:
    """Latest regular divergence between price pivots and the MACD line.

    Returns +1 (bullish: lower price low, higher MACD low), -1 (bearish: higher
    price high, lower MACD high) or 0. Compares the two most recent pivots.
    """
    line = macd(close, fast, slow, signal)["macd"]
    for kind, sign in (("low", 1), ("high", -1)):
        piv = _pivots(close, left, right, kind)
        idx = list(np.flatnonzero(piv.to_numpy()))
        if len(idx) < 2:
            continue
        a, b = idx[-2], idx[-1]
        pa, pb = close.iloc[a], close.iloc[b]
        ma, mb = line.iloc[a], line.iloc[b]
        if pd.isna(ma) or pd.isna(mb):
            continue
        if kind == "low" and pb < pa and mb > ma:
            return 1
        if kind == "high" and pb > pa and mb < ma:
            return -1
    return 0


def macd_signal(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> IndicatorSignal:
    """Directional vote from the histogram, reinforced by divergence."""
    if len(close) < slow + signal:
        return IndicatorSignal("MACD", Bias.NEUTRAL, None, "insufficient history")
    hist = macd(close, fast, slow, signal)["hist"].iloc[-1]
    if pd.isna(hist):
        return IndicatorSignal("MACD", Bias.NEUTRAL, None, "insufficient history")
    div = macd_divergence(close, fast, slow, signal)
    if div == 1:
        return IndicatorSignal("MACD", Bias.BULLISH, float(hist), "bullish divergence")
    if div == -1:
        return IndicatorSignal("MACD", Bias.BEARISH, float(hist), "bearish divergence")
    bias = Bias.BULLISH if hist > 0 else Bias.BEARISH if hist < 0 else Bias.NEUTRAL
    return IndicatorSignal("MACD", bias, float(hist), f"hist {hist:+.3f}")


__all__ = [
    "rsi", "adaptive_rsi_bands", "rsi_signal",
    "macd", "macd_divergence", "macd_signal",
]
