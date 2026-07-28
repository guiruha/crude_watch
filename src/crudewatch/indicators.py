"""Shared indicator math (continuous values), used by both sides of the codebase.

These are the raw, continuous indicator computations — *not* the long/flat
trading signals. They are imported by the live feature matrix
(:mod:`crudewatch.research.features`, which powers the Opportunity Score) and by
the offline backtesting engine, so they live in a neutral module that neither
side owns. Everything here uses only information up to and including ``t`` (no
``shift(-k)``), so it is look-ahead free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def moving_average(series: pd.Series, window: int, kind: str = "ema") -> pd.Series:
    """Simple (``sma``) or exponential (``ema``) moving average of ``series``.

    Values before ``window`` observations are ``NaN`` so the warmup period is
    explicit and never traded on.
    """
    kind = kind.lower()
    if kind == "sma":
        return series.rolling(window, min_periods=window).mean()
    if kind == "ema":
        return series.ewm(span=window, adjust=False, min_periods=window).mean()
    raise ValueError(f"kind must be 'sma' or 'ema', got {kind!r}")


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index (0–100)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return the MACD line, its signal line and the histogram."""
    fast_ema = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def linreg_slope(close: pd.Series, window: int = 20) -> pd.Series:
    """Slope of an ordinary least-squares line fitted to the last ``window`` closes."""
    n = window
    x = np.arange(n)
    sx = float(x.sum())
    sxx = float((x * x).sum())
    denom = n * sxx - sx * sx
    sy = close.rolling(n).sum()
    sxy = sum(j * close.shift(n - 1 - j) for j in range(n))
    return (n * sxy - sx * sy) / denom


def efficiency_ratio(close: pd.Series, window: int = 20) -> pd.Series:
    """Kaufman Efficiency Ratio in [0, 1]: net move over total path length.

    Near 1 means a clean directional move (trend); near 0 means choppy, back-and
    forth price action (a ranging / mean-reverting regime).
    """
    change = (close - close.shift(window)).abs()
    volatility = close.diff().abs().rolling(window).sum()
    return change / volatility.replace(0.0, np.nan)
