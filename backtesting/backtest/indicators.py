"""Technical indicators and the long/flat positions they produce.

Every ``*_signal`` function returns a position series (``1`` = long, ``0`` =
flat) already shifted one bar, so a signal seen at yesterday's close is only
acted on at today's price — no look-ahead. The warmup period (before the
indicator is defined) always stays flat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Continuous indicator math shared with the live feature matrix lives in the
# app-facing package, so both sides compute identical values from one source.
from crudewatch.indicators import (
    efficiency_ratio,
    linreg_slope,
    macd,
    moving_average,
    rsi,
)


def _hold(signal: pd.Series, valid: pd.Series) -> pd.Series:
    """Turn discrete entry(1)/exit(0) events into a held long/flat position.

    ``signal`` carries ``1`` on entries, ``0`` on exits and ``NaN`` elsewhere;
    the position is forward-filled between events. Bars where ``valid`` is False
    (indicator warmup) are forced flat. The result is shifted one bar.
    """
    pos = signal.ffill().fillna(0.0)
    pos[~valid] = 0.0
    return pos.shift(1).fillna(0.0)


def crossover_signal(close: pd.Series, fast: int, slow: int, kind: str = "ema") -> pd.Series:
    """Long/flat position (``1`` = long, ``0`` = flat) from a fast/slow crossover.

    Long while the fast average sits above the slow one. The position is shifted
    one bar so a signal seen at yesterday's close is only acted on at today's
    price — this removes look-ahead bias. The warmup period (before the slow
    average exists) stays flat.
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be smaller than slow ({slow})")

    fast_ma = moving_average(close, fast, kind)
    slow_ma = moving_average(close, slow, kind)

    long = (fast_ma > slow_ma).astype("float")
    long[fast_ma.isna() | slow_ma.isna()] = float("nan")  # no position during warmup
    return long.shift(1).fillna(0.0)


# --------------------------------------------------------------------------- #
# RSI
# --------------------------------------------------------------------------- #
def rsi_signal(close: pd.Series, period: int = 14, low: float = 30.0, high: float = 70.0) -> pd.Series:
    """Long when RSI crosses up through ``low`` (oversold rebound); flat when it
    crosses up through ``high`` (overbought). Held in between."""
    r = rsi(close, period)
    entry = (r > low) & (r.shift(1) <= low)
    exit_ = (r > high) & (r.shift(1) <= high)
    signal = pd.Series(np.nan, index=close.index)
    signal[entry] = 1.0
    signal[exit_] = 0.0
    return _hold(signal, r.notna())


# --------------------------------------------------------------------------- #
# MACD
# --------------------------------------------------------------------------- #
def macd_signal(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Long while the MACD line is above its signal line; flat otherwise."""
    macd_line, signal_line, _ = macd(close, fast, slow, signal)
    long = (macd_line > signal_line).astype("float")
    long[macd_line.isna() | signal_line.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


# --------------------------------------------------------------------------- #
# Divergences (price vs. an oscillator, over a fixed lookback)
# --------------------------------------------------------------------------- #
def divergence_signal(price: pd.Series, oscillator: pd.Series, lookback: int) -> pd.Series:
    """Long/flat from regular price-vs-oscillator divergences.

    Bullish divergence (go long): price is *below* its value ``lookback`` bars
    ago while the oscillator is *above* — momentum improving under a lower price.
    Bearish divergence (go flat): price higher but oscillator lower. The position
    is held between signals. Comparing against a fixed lag avoids look-ahead.
    """
    ref_valid = oscillator.notna() & oscillator.shift(lookback).notna()
    bull = (price < price.shift(lookback)) & (oscillator > oscillator.shift(lookback))
    bear = (price > price.shift(lookback)) & (oscillator < oscillator.shift(lookback))
    signal = pd.Series(np.nan, index=price.index)
    signal[bull] = 1.0
    signal[bear] = 0.0
    return _hold(signal, ref_valid)


def rsi_divergence_signal(close: pd.Series, period: int = 14, lookback: int = 20) -> pd.Series:
    """Divergence of price against RSI."""
    return divergence_signal(close, rsi(close, period), lookback)


def macd_divergence_signal(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9, lookback: int = 20
) -> pd.Series:
    """Divergence of price against the MACD line."""
    macd_line, _, _ = macd(close, fast, slow, signal)
    return divergence_signal(close, macd_line, lookback)


# --------------------------------------------------------------------------- #
# Trend-following (close-based, so they also run on the synthetic families)
# --------------------------------------------------------------------------- #
def donchian_signal(close: pd.Series, entry: int = 20, exit_len: int = 10) -> pd.Series:
    """Channel breakout: long on a new ``entry``-bar high, flat on an
    ``exit_len``-bar low (classic turtle-style breakout)."""
    upper = close.rolling(entry).max().shift(1)     # prior N-bar high
    lower = close.rolling(exit_len).min().shift(1)  # prior M-bar low
    signal = pd.Series(np.nan, index=close.index)
    signal[close > upper] = 1.0
    signal[close < lower] = 0.0
    return _hold(signal, upper.notna())


def sma_trend_signal(close: pd.Series, window: int = 50) -> pd.Series:
    """Long while price is above its ``window``-bar simple moving average."""
    ma = moving_average(close, window, "sma")
    long = (close > ma).astype("float")
    long[ma.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def triple_ma_signal(close: pd.Series, fast: int = 8, mid: int = 21, slow: int = 55, kind: str = "ema") -> pd.Series:
    """Long when the three moving averages are stacked bullishly (fast > mid > slow)."""
    f = moving_average(close, fast, kind)
    m = moving_average(close, mid, kind)
    s = moving_average(close, slow, kind)
    long = ((f > m) & (m > s)).astype("float")
    long[f.isna() | m.isna() | s.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def linreg_signal(close: pd.Series, window: int = 20) -> pd.Series:
    """Long while the rolling regression slope is positive (price trending up)."""
    slope = linreg_slope(close, window)
    long = (slope > 0).astype("float")
    long[slope.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def aroon_signal(close: pd.Series, window: int = 25) -> pd.Series:
    """Aroon on close: long while Aroon-Up (bars since the window high) exceeds
    Aroon-Down (bars since the window low)."""
    since_high = close.rolling(window + 1).apply(np.argmax, raw=True)
    since_low = close.rolling(window + 1).apply(np.argmin, raw=True)
    up = 100.0 * since_high / window
    down = 100.0 * since_low / window
    long = (up > down).astype("float")
    long[up.isna() | down.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def trix(close: pd.Series, window: int = 15) -> pd.Series:
    """Absolute change of a triple-smoothed EMA (kept absolute so it survives
    spreads that cross zero, where a percentage TRIX would blow up)."""
    e1 = close.ewm(span=window, adjust=False, min_periods=window).mean()
    e2 = e1.ewm(span=window, adjust=False).mean()
    e3 = e2.ewm(span=window, adjust=False).mean()
    return e3.diff()


def trix_signal(close: pd.Series, window: int = 15, signal: int = 9) -> pd.Series:
    """Long while the TRIX line is above its signal EMA."""
    t = trix(close, window)
    sig = t.ewm(span=signal, adjust=False).mean()
    long = (t > sig).astype("float")
    long[t.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


# --------------------------------------------------------------------------- #
# Momentum (close-based)
# --------------------------------------------------------------------------- #
def momentum_signal(close: pd.Series, window: int = 20) -> pd.Series:
    """Absolute momentum: long while price is higher than ``window`` bars ago."""
    mom = close - close.shift(window)
    long = (mom > 0).astype("float")
    long[mom.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def stochastic_signal(close: pd.Series, k: int = 14, d: int = 3) -> pd.Series:
    """Close-based stochastic: long while %K is above its %D average."""
    lo = close.rolling(k).min()
    hi = close.rolling(k).max()
    span = (hi - lo).replace(0.0, np.nan)
    percent_k = 100.0 * (close - lo) / span
    percent_d = percent_k.rolling(d).mean()
    long = (percent_k > percent_d).astype("float")
    long[percent_k.isna() | percent_d.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def cci_signal(close: pd.Series, window: int = 20) -> pd.Series:
    """Close-based Commodity Channel Index: long while CCI is positive."""
    ma = close.rolling(window).mean()
    mad = (close - ma).abs().rolling(window).mean()
    cci = (close - ma) / (0.015 * mad.replace(0.0, np.nan))
    long = (cci > 0).astype("float")
    long[cci.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def awesome_signal(close: pd.Series, fast: int = 5, slow: int = 34) -> pd.Series:
    """Close-based Awesome Oscillator: long while SMA(fast) - SMA(slow) is positive."""
    ao = close.rolling(fast).mean() - close.rolling(slow).mean()
    long = (ao > 0).astype("float")
    long[ao.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


# --------------------------------------------------------------------------- #
# Adaptive / smoothed trend
# --------------------------------------------------------------------------- #
def wma(series: pd.Series, window: int) -> pd.Series:
    """Linearly weighted moving average (most recent bar weighted highest)."""
    w_sum = window * (window + 1) / 2.0
    num = sum((j + 1) * series.shift(window - 1 - j) for j in range(window))
    return num / w_sum


def kama(close: pd.Series, er_window: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average: smoothing speeds up in trends, slows in noise."""
    er = efficiency_ratio(close, er_window)
    sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
    vals = close.to_numpy(dtype=float)
    scv = sc.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    prev = np.nan
    for i in range(len(vals)):
        if np.isnan(scv[i]) or np.isnan(vals[i]):
            continue
        prev = vals[i] if np.isnan(prev) else prev + scv[i] * (vals[i] - prev)
        out[i] = prev
    return pd.Series(out, index=close.index)


def kama_signal(close: pd.Series, er_window: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Long while price is above its adaptive moving average."""
    k = kama(close, er_window, fast, slow)
    long = (close > k).astype("float")
    long[k.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def dema(close: pd.Series, window: int) -> pd.Series:
    """Double exponential moving average (less lag than a plain EMA)."""
    e1 = close.ewm(span=window, adjust=False, min_periods=window).mean()
    e2 = e1.ewm(span=window, adjust=False).mean()
    return 2 * e1 - e2


def dema_cross_signal(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Long while the fast DEMA is above the slow DEMA."""
    long = (dema(close, fast) > dema(close, slow)).astype("float")
    long[dema(close, fast).isna() | dema(close, slow).isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def hma(close: pd.Series, window: int = 20) -> pd.Series:
    """Hull moving average: fast and smooth, minimal lag."""
    half = max(1, window // 2)
    root = max(1, int(window ** 0.5))
    return wma(2 * wma(close, half) - wma(close, window), root)


def hma_signal(close: pd.Series, window: int = 20) -> pd.Series:
    """Long while price is above its Hull moving average."""
    h = hma(close, window)
    long = (close > h).astype("float")
    long[h.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


# --------------------------------------------------------------------------- #
# More momentum (close-based, kept absolute to survive zero-crossing spreads)
# --------------------------------------------------------------------------- #
def coppock(close: pd.Series, roc_long: int = 14, roc_short: int = 11, smooth: int = 10) -> pd.Series:
    """Coppock curve on absolute momentum (sum of two look-backs, WMA-smoothed)."""
    mom = (close - close.shift(roc_long)) + (close - close.shift(roc_short))
    return wma(mom, smooth)


def coppock_signal(close: pd.Series, roc_long: int = 14, roc_short: int = 11, smooth: int = 10) -> pd.Series:
    """Long while the Coppock curve is positive."""
    c = coppock(close, roc_long, roc_short, smooth)
    long = (c > 0).astype("float")
    long[c.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def tsi(close: pd.Series, long_window: int = 25, short_window: int = 13) -> pd.Series:
    """True Strength Index: double-smoothed momentum, scaled to \u00b1100."""
    m = close.diff()
    m_smooth = m.ewm(span=long_window, adjust=False).mean().ewm(span=short_window, adjust=False).mean()
    a_smooth = m.abs().ewm(span=long_window, adjust=False).mean().ewm(span=short_window, adjust=False).mean()
    return 100.0 * m_smooth / a_smooth.replace(0.0, np.nan)


def tsi_signal(close: pd.Series, long_window: int = 25, short_window: int = 13, signal: int = 7) -> pd.Series:
    """Long while the TSI line is above its signal EMA."""
    t = tsi(close, long_window, short_window)
    sig = t.ewm(span=signal, adjust=False).mean()
    long = (t > sig).astype("float")
    long[t.isna() | sig.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def cmo(close: pd.Series, window: int = 20) -> pd.Series:
    """Chande Momentum Oscillator in [-100, 100]."""
    d = close.diff()
    up = d.clip(lower=0.0).rolling(window).sum()
    down = (-d.clip(upper=0.0)).rolling(window).sum()
    return 100.0 * (up - down) / (up + down).replace(0.0, np.nan)


def cmo_signal(close: pd.Series, window: int = 20) -> pd.Series:
    """Long while the Chande Momentum Oscillator is positive."""
    c = cmo(close, window)
    long = (c > 0).astype("float")
    long[c.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


# --------------------------------------------------------------------------- #
# Mean-reversion (well suited to the stationary calendar spreads)
# --------------------------------------------------------------------------- #
def bollinger_signal(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Long when price closes below the lower Bollinger band; flat back at the mean."""
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    lower = mean - num_std * std
    signal = pd.Series(np.nan, index=close.index)
    signal[close < lower] = 1.0
    signal[close >= mean] = 0.0
    return _hold(signal, mean.notna() & std.notna())


def zscore_signal(close: pd.Series, window: int = 20, entry: float = 2.0, exit_z: float = 0.0) -> pd.Series:
    """Long when the rolling z-score drops below ``-entry``; flat when it recovers to ``-exit_z``."""
    mean = close.rolling(window).mean()
    std = close.rolling(window).std().replace(0.0, np.nan)
    z = (close - mean) / std
    signal = pd.Series(np.nan, index=close.index)
    signal[z < -entry] = 1.0
    signal[z >= -exit_z] = 0.0
    return _hold(signal, z.notna())


# --------------------------------------------------------------------------- #
# Regime-switching: trend-follow when trending, mean-revert when ranging
# --------------------------------------------------------------------------- #
def regime_switch_signal(
    close: pd.Series,
    er_window: int = 20,
    er_threshold: float = 0.35,
    trend_fast: int = 20,
    trend_slow: int = 50,
    z_window: int = 20,
    kind: str = "ema",
) -> pd.Series:
    """Route between a trend rule and a mean-reversion rule using the Efficiency Ratio.

    - Trending regime (ER >= threshold): long while the fast MA is above the slow MA.
    - Ranging regime (ER < threshold): long while price is below its mean (z < 0),
      betting on reversion back up.
    """
    er = efficiency_ratio(close, er_window)
    fast_ma = moving_average(close, trend_fast, kind)
    slow_ma = moving_average(close, trend_slow, kind)
    mean = close.rolling(z_window).mean()
    std = close.rolling(z_window).std().replace(0.0, np.nan)
    z = (close - mean) / std

    trend_long = (fast_ma > slow_ma).to_numpy()
    mr_long = (z < 0).to_numpy()
    is_trend = (er >= er_threshold).to_numpy()

    long = pd.Series(np.where(is_trend, trend_long, mr_long).astype("float"), index=close.index)
    valid = er.notna() & fast_ma.notna() & slow_ma.notna() & z.notna()
    long[~valid] = float("nan")
    return long.shift(1).fillna(0.0)


# --------------------------------------------------------------------------- #
# Directional / volatility trend (close-to-close true range, so they run on
# every family including the synthetic spreads that have no high/low)
# --------------------------------------------------------------------------- #
def _atr_cc(close: pd.Series, window: int) -> pd.Series:
    """Close-to-close 'ATR': Wilder-smoothed absolute daily change."""
    return close.diff().abs().ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def adx_signal(close: pd.Series, window: int = 14, threshold: float = 25.0) -> pd.Series:
    """ADX/DMI (close-based): long while +DI > -DI and the trend is strong (ADX > threshold)."""
    up = close.diff().clip(lower=0.0)
    down = (-close.diff()).clip(lower=0.0)
    atr = _atr_cc(close, window)
    plus_di = 100.0 * up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * down.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    long = ((plus_di > minus_di) & (adx > threshold)).astype("float")
    long[plus_di.isna() | minus_di.isna() | adx.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def supertrend_signal(close: pd.Series, window: int = 10, mult: float = 3.0) -> pd.Series:
    """Supertrend (close-based ATR): long while the trend flag is up."""
    atr = _atr_cc(close, window)
    basic_ub = (close + mult * atr).to_numpy()
    basic_lb = (close - mult * atr).to_numpy()
    c = close.to_numpy(dtype=float)
    n = len(c)
    fub = np.full(n, np.nan)
    flb = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(basic_ub[i]):
            continue
        if i == 0 or np.isnan(fub[i - 1]):
            fub[i], flb[i], direction[i] = basic_ub[i], basic_lb[i], 1.0
            continue
        fub[i] = basic_ub[i] if (basic_ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = basic_lb[i] if (basic_lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if direction[i - 1] == 1.0:
            direction[i] = -1.0 if c[i] < flb[i] else 1.0
        else:
            direction[i] = 1.0 if c[i] > fub[i] else -1.0
    long = pd.Series(np.where(direction == 1.0, 1.0, 0.0), index=close.index)
    long[np.isnan(direction)] = float("nan")
    return long.shift(1).fillna(0.0)


def keltner_breakout_signal(close: pd.Series, window: int = 20, mult: float = 2.0, atr_window: int = 10) -> pd.Series:
    """Keltner channel breakout (close-based): long above the upper channel, flat back at the mid."""
    mid = moving_average(close, window, "ema")
    atr = _atr_cc(close, atr_window)
    upper = mid + mult * atr
    signal = pd.Series(np.nan, index=close.index)
    signal[close > upper] = 1.0
    signal[close < mid] = 0.0
    return _hold(signal, mid.notna() & atr.notna())


def bollinger_breakout_signal(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Bollinger breakout (momentum): long above the upper band, flat back at the mean."""
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mean + num_std * std
    signal = pd.Series(np.nan, index=close.index)
    signal[close > upper] = 1.0
    signal[close < mean] = 0.0
    return _hold(signal, mean.notna() & std.notna())


# --------------------------------------------------------------------------- #
# More smoothed-trend crossovers
# --------------------------------------------------------------------------- #
def tema(close: pd.Series, window: int) -> pd.Series:
    """Triple exponential moving average (very low lag)."""
    e1 = close.ewm(span=window, adjust=False, min_periods=window).mean()
    e2 = e1.ewm(span=window, adjust=False).mean()
    e3 = e2.ewm(span=window, adjust=False).mean()
    return 3 * e1 - 3 * e2 + e3


def tema_cross_signal(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Long while the fast TEMA is above the slow TEMA."""
    tf, ts = tema(close, fast), tema(close, slow)
    long = (tf > ts).astype("float")
    long[tf.isna() | ts.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def zlema(close: pd.Series, window: int) -> pd.Series:
    """Zero-lag EMA: EMA of a de-lagged price."""
    lag = (window - 1) // 2
    delagged = close + (close - close.shift(lag))
    return delagged.ewm(span=window, adjust=False, min_periods=window).mean()


def zlema_cross_signal(close: pd.Series, fast: int = 9, slow: int = 21) -> pd.Series:
    """Long while the fast zero-lag EMA is above the slow one."""
    zf, zs = zlema(close, fast), zlema(close, slow)
    long = (zf > zs).astype("float")
    long[zf.isna() | zs.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def ichimoku_signal(close: pd.Series, tenkan: int = 9, kijun: int = 26) -> pd.Series:
    """Ichimoku TK cross (close-based): long while the Tenkan line is above the Kijun line."""
    tenkan_line = (close.rolling(tenkan).max() + close.rolling(tenkan).min()) / 2
    kijun_line = (close.rolling(kijun).max() + close.rolling(kijun).min()) / 2
    long = (tenkan_line > kijun_line).astype("float")
    long[tenkan_line.isna() | kijun_line.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def gmma_signal(close: pd.Series) -> pd.Series:
    """Guppy Multiple MA: long while the short EMA group is, on average, above the long group."""
    short = [3, 5, 8, 10, 12, 15]
    long_group = [30, 35, 40, 45, 50, 60]
    avg_short = sum(moving_average(close, w, "ema") for w in short) / len(short)
    avg_long = sum(moving_average(close, w, "ema") for w in long_group) / len(long_group)
    long = (avg_short > avg_long).astype("float")
    long[avg_short.isna() | avg_long.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


# --------------------------------------------------------------------------- #
# More oscillators (close-based)
# --------------------------------------------------------------------------- #
def stc(close: pd.Series, fast: int = 23, slow: int = 50, cycle: int = 10) -> pd.Series:
    """Schaff Trend Cycle (0\u2013100): a stochastic applied twice over the MACD line."""
    macd_line = (
        close.ewm(span=fast, adjust=False, min_periods=fast).mean()
        - close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    )

    def _stoch(series: pd.Series) -> pd.Series:
        lo = series.rolling(cycle).min()
        hi = series.rolling(cycle).max()
        return 100.0 * (series - lo) / (hi - lo).replace(0.0, np.nan)

    k1 = _stoch(macd_line)
    d1 = k1.ewm(span=cycle, adjust=False).mean()
    k2 = _stoch(d1)
    return k2.ewm(span=cycle, adjust=False).mean()


def stc_signal(close: pd.Series, fast: int = 23, slow: int = 50, cycle: int = 10) -> pd.Series:
    """Long while the Schaff Trend Cycle is above 50."""
    s = stc(close, fast, slow, cycle)
    long = (s > 50).astype("float")
    long[s.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def stoch_rsi_signal(close: pd.Series, rsi_period: int = 14, stoch: int = 14, k: int = 3, d: int = 3) -> pd.Series:
    """Stochastic RSI: long while %K is above %D."""
    r = rsi(close, rsi_period)
    lo = r.rolling(stoch).min()
    hi = r.rolling(stoch).max()
    stoch_rsi = (r - lo) / (hi - lo).replace(0.0, np.nan)
    percent_k = stoch_rsi.rolling(k).mean()
    percent_d = percent_k.rolling(d).mean()
    long = (percent_k > percent_d).astype("float")
    long[percent_k.isna() | percent_d.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def williams_r_signal(close: pd.Series, window: int = 14) -> pd.Series:
    """Williams %R (close-based): long while %R is in the upper half (above -50)."""
    hi = close.rolling(window).max()
    lo = close.rolling(window).min()
    percent_r = -100.0 * (hi - close) / (hi - lo).replace(0.0, np.nan)
    long = (percent_r > -50.0).astype("float")
    long[percent_r.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def fisher_signal(close: pd.Series, window: int = 9) -> pd.Series:
    """Fisher Transform: long while the Fisher line is above its previous value (trigger)."""
    lo = close.rolling(window).min()
    hi = close.rolling(window).max()
    raw = 2.0 * (close - lo) / (hi - lo).replace(0.0, np.nan) - 1.0
    value = raw.clip(-0.999, 0.999)
    fisher = 0.5 * np.log((1 + value) / (1 - value))
    long = (fisher > fisher.shift(1)).astype("float")
    long[fisher.isna() | fisher.shift(1).isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def roc_signal(close: pd.Series, window: int = 12) -> pd.Series:
    """Rate of change (absolute): long while price is higher than ``window`` bars ago."""
    roc = close - close.shift(window)
    long = (roc > 0).astype("float")
    long[roc.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def rsi_trend_signal(close: pd.Series, period: int = 14, level: float = 50.0) -> pd.Series:
    """RSI as a trend filter: long while RSI is above ``level`` (default 50)."""
    r = rsi(close, period)
    long = (r > level).astype("float")
    long[r.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


def kst(close: pd.Series) -> pd.Series:
    """Know Sure Thing: weighted sum of four smoothed (absolute) momenta."""
    def rcma(roc_window: int, smooth: int) -> pd.Series:
        return (close - close.shift(roc_window)).rolling(smooth).mean()

    return 1 * rcma(10, 10) + 2 * rcma(15, 10) + 3 * rcma(20, 10) + 4 * rcma(30, 15)


def kst_signal(close: pd.Series, signal: int = 9) -> pd.Series:
    """Long while the KST line is above its signal moving average."""
    k = kst(close)
    sig = k.rolling(signal).mean()
    long = (k > sig).astype("float")
    long[k.isna() | sig.isna()] = float("nan")
    return long.shift(1).fillna(0.0)


# --------------------------------------------------------------------------- #
# Mean-reversion "family": alternatives to the RSI / Bollinger / Z-score /
# divergence rules that scored best. All long/flat, close-based.
# --------------------------------------------------------------------------- #
def cci(close: pd.Series, window: int = 20) -> pd.Series:
    """Close-based Commodity Channel Index (raw oscillator)."""
    ma = close.rolling(window).mean()
    mad = (close - ma).abs().rolling(window).mean()
    return (close - ma) / (0.015 * mad.replace(0.0, np.nan))


def _stoch_k(close: pd.Series, k: int = 14) -> pd.Series:
    lo = close.rolling(k).min()
    hi = close.rolling(k).max()
    return 100.0 * (close - lo) / (hi - lo).replace(0.0, np.nan)


def _reversion_from_oscillator(osc: pd.Series, low: float, high: float) -> pd.Series:
    """Long when ``osc`` crosses up through ``low`` (oversold rebound); flat when
    it crosses up through ``high`` (overbought). Held in between."""
    entry = (osc > low) & (osc.shift(1) <= low)
    exit_ = (osc > high) & (osc.shift(1) <= high)
    signal = pd.Series(np.nan, index=osc.index)
    signal[entry] = 1.0
    signal[exit_] = 0.0
    return _hold(signal, osc.notna())


def stoch_reversion_signal(close: pd.Series, k: int = 14, d: int = 3, low: float = 20.0, high: float = 80.0) -> pd.Series:
    """Stochastic mean-reversion: long as %D rebounds out of oversold, flat on overbought."""
    percent_d = _stoch_k(close, k).rolling(d).mean()
    return _reversion_from_oscillator(percent_d, low, high)


def cci_reversion_signal(close: pd.Series, window: int = 20, low: float = -100.0, high: float = 100.0) -> pd.Series:
    """CCI mean-reversion: long as CCI rebounds up through the oversold band, flat on overbought."""
    return _reversion_from_oscillator(cci(close, window), low, high)


def williams_reversion_signal(close: pd.Series, window: int = 14, low: float = -80.0, high: float = -20.0) -> pd.Series:
    """Williams %R mean-reversion: long as %R rebounds out of oversold (-80), flat near -20."""
    hi = close.rolling(window).max()
    lo = close.rolling(window).min()
    percent_r = -100.0 * (hi - close) / (hi - lo).replace(0.0, np.nan)
    return _reversion_from_oscillator(percent_r, low, high)


def keltner_reversion_signal(close: pd.Series, window: int = 20, mult: float = 2.0, atr_window: int = 10) -> pd.Series:
    """Keltner mean-reversion: long when price closes below the lower channel, flat back at the mid."""
    mid = moving_average(close, window, "ema")
    atr = _atr_cc(close, atr_window)
    lower = mid - mult * atr
    signal = pd.Series(np.nan, index=close.index)
    signal[close < lower] = 1.0
    signal[close >= mid] = 0.0
    return _hold(signal, mid.notna() & atr.notna())


def rsi_bollinger_signal(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
    rsi_period: int = 14,
    rsi_low: float = 35.0,
) -> pd.Series:
    """Confluence reversion: long only when price is below the lower Bollinger band
    *and* RSI is oversold; flat back at the mean."""
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    lower = mean - num_std * std
    r = rsi(close, rsi_period)
    signal = pd.Series(np.nan, index=close.index)
    signal[(close < lower) & (r < rsi_low)] = 1.0
    signal[close >= mean] = 0.0
    return _hold(signal, mean.notna() & std.notna() & r.notna())


def cci_divergence_signal(close: pd.Series, window: int = 20, lookback: int = 20) -> pd.Series:
    """Divergence of price against the CCI oscillator."""
    return divergence_signal(close, cci(close, window), lookback)


def stoch_divergence_signal(close: pd.Series, k: int = 14, lookback: int = 20) -> pd.Series:
    """Divergence of price against the stochastic %K oscillator."""
    return divergence_signal(close, _stoch_k(close, k), lookback)
