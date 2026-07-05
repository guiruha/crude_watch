"""Trend & structure indicators for a close-only spread series.

The triple moving-average ribbon is the CORE trend read; ADX/DMI is the regime
filter (trend vs. range) that conditions how much to trust directional signals;
Supertrend gives a visual trailing stop. ADX and Supertrend normally use
high/low — here they use the close-based true range (see ``volatility``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crudewatch.analytics.indicators import Bias, IndicatorSignal
from crudewatch.analytics.indicators.volatility import atr, true_range


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def ma_ribbon(close: pd.Series, fast: int = 20, mid: int = 50, slow: int = 200) -> pd.DataFrame:
    """The triple MA ribbon: EMA-fast, EMA-mid, SMA-slow."""
    return pd.DataFrame({
        f"ema{fast}": ema(close, fast),
        f"ema{mid}": ema(close, mid),
        f"sma{slow}": sma(close, slow),
    })


def ma_signal(close: pd.Series, fast: int = 20, mid: int = 50, slow: int = 200) -> IndicatorSignal:
    """Stacked-ribbon vote: fully bullish/bearish alignment, else neutral.

    Falls back to the fast-vs-mid cross when there is not enough history for the
    slow SMA, so short vintages still get a read.
    """
    f, m = ema(close, fast).iloc[-1], ema(close, mid).iloc[-1]
    s = sma(close, slow).iloc[-1] if len(close) >= slow else np.nan
    if pd.isna(f) or pd.isna(m):
        return IndicatorSignal("MA ribbon", Bias.NEUTRAL, None, "insufficient history")
    if pd.isna(s):
        bias = Bias.BULLISH if f > m else Bias.BEARISH if f < m else Bias.NEUTRAL
        return IndicatorSignal("MA ribbon", bias, None, "fast/mid only")
    if f > m > s:
        return IndicatorSignal("MA ribbon", Bias.BULLISH, None, "stacked up")
    if f < m < s:
        return IndicatorSignal("MA ribbon", Bias.BEARISH, None, "stacked down")
    return IndicatorSignal("MA ribbon", Bias.NEUTRAL, None, "mixed")


def dmi_adx(close: pd.Series, period: int = 14) -> pd.DataFrame:
    """+DI / -DI / ADX from close moves (directional movement approximated by Δclose)."""
    change = close.diff()
    up = change.clip(lower=0)
    down = (-change).clip(lower=0)
    tr = true_range(close)
    alpha = 1 / period
    atr_ = tr.ewm(alpha=alpha, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * up.ewm(alpha=alpha, adjust=False).mean() / atr_
    minus_di = 100 * down.ewm(alpha=alpha, adjust=False).mean() / atr_
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx})


def adx_signal(close: pd.Series, period: int = 14, trend: float = 25.0, range_: float = 20.0) -> IndicatorSignal:
    """Regime filter: strong ADX -> directional (by DI dominance); weak ADX -> range/neutral."""
    if len(close) < period + 1:
        return IndicatorSignal("ADX/DMI", Bias.NEUTRAL, None, "insufficient history")
    row = dmi_adx(close, period).iloc[-1]
    adx = row["adx"]
    if pd.isna(adx) or adx < range_:
        return IndicatorSignal("ADX/DMI", Bias.NEUTRAL, float(adx) if pd.notna(adx) else None, "range")
    if adx >= trend:
        bias = Bias.BULLISH if row["plus_di"] > row["minus_di"] else Bias.BEARISH
        return IndicatorSignal("ADX/DMI", bias, float(adx), f"ADX {adx:.0f} trending")
    return IndicatorSignal("ADX/DMI", Bias.NEUTRAL, float(adx), f"ADX {adx:.0f}")


def supertrend(close: pd.Series, period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Close-based Supertrend line and direction (+1 up-trend, -1 down-trend)."""
    atr_ = atr(close, period)
    upper = close + mult * atr_
    lower = close - mult * atr_

    final_upper = upper.copy()
    final_lower = lower.copy()
    idx = close.index
    for i in range(1, len(close)):
        pu, pl = final_upper.iloc[i - 1], final_lower.iloc[i - 1]
        cu, cl = upper.iloc[i], lower.iloc[i]
        pc = close.iloc[i - 1]
        final_upper.iloc[i] = cu if (cu < pu or pc > pu) else pu
        final_lower.iloc[i] = cl if (cl > pl or pc < pl) else pl

    direction = pd.Series(1, index=idx, dtype=int)
    line = pd.Series(np.nan, index=idx)
    for i in range(1, len(close)):
        prev_dir = direction.iloc[i - 1]
        c = close.iloc[i]
        if prev_dir == 1:
            direction.iloc[i] = -1 if c < final_lower.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if c > final_upper.iloc[i] else -1
        line.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]
    return pd.DataFrame({"supertrend": line, "direction": direction})


def supertrend_signal(close: pd.Series, period: int = 10, mult: float = 3.0) -> IndicatorSignal:
    if len(close) < period + 1:
        return IndicatorSignal("Supertrend", Bias.NEUTRAL, None, "insufficient history")
    direction = supertrend(close, period, mult)["direction"].iloc[-1]
    bias = Bias.BULLISH if direction > 0 else Bias.BEARISH
    return IndicatorSignal("Supertrend", bias, float(direction), "up-trend" if direction > 0 else "down-trend")


__all__ = [
    "ema", "sma", "ma_ribbon", "ma_signal",
    "dmi_adx", "adx_signal", "supertrend", "supertrend_signal",
]
