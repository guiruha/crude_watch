"""Continuous feature matrix, as-of ``t`` (the shared half of the harness).

Unlike the long/flat *signals* in ``backtest.indicators`` (which emit a 0/1
position shifted one bar to simulate trading), these are the **raw continuous
values** of each indicator, known at the close of ``t``. They feed both:

* the backtest — cross a feature at ``t`` with a forward outcome (``targets``)
  strictly after ``t`` to measure predictive power; and
* live scoring — compute the same features on today's bar to produce the score.

Look-ahead guarantee: every feature uses only information up to and including
``t`` (rolling / EWM windows, never ``shift(-k)``). No feature is shifted; the
temporal separation lives entirely in the target being strictly future. This is
verified by a test that appends future bars and checks past feature values do
not change.

Feature set is deliberately focused on the reversion / level / regime edge the
current backtest points to (Bloque D/E), plus a couple of direction/quality
primitives (Bloque B/C). Redundant look-alikes are left out on purpose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crudewatch.indicators import (
    efficiency_ratio,
    linreg_slope,
    macd,
    moving_average,
    rsi,
)


def _atr_cc(close: pd.Series, window: int) -> pd.Series:
    """Close-to-close 'ATR': Wilder-smoothed absolute daily change."""
    return close.diff().abs().ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def zscore(close: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: standard deviations of ``close`` from its window mean."""
    mean = close.rolling(window).mean()
    std = close.rolling(window).std().replace(0.0, np.nan)
    return (close - mean) / std


def bollinger_pctb(close: pd.Series, window: int, num_std: float) -> pd.Series:
    """Bollinger %B: 0 at the lower band, 1 at the upper (below/above go out of range)."""
    mean = close.rolling(window).mean()
    std = close.rolling(window).std().replace(0.0, np.nan)
    upper = mean + num_std * std
    lower = mean - num_std * std
    return (close - lower) / (upper - lower)


def keltner_distance(close: pd.Series, window: int = 20, atr_window: int = 10) -> pd.Series:
    """Distance from the Keltner mid (EMA) in ATR units; negative = below the mid."""
    mid = moving_average(close, window, "ema")
    atr = _atr_cc(close, atr_window)
    return (close - mid) / atr


def slope_normalized(close: pd.Series, window: int = 20, atr_window: int = 10) -> pd.Series:
    """Regression slope over ``window`` normalised by ATR (a vol-adjusted trend gradient)."""
    return linreg_slope(close, window) / _atr_cc(close, atr_window)


def macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram (line minus signal); positive = bullish momentum."""
    _, _, hist = macd(close, fast, slow, signal)
    return hist


def _roll_z(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score of a series over ``window`` (as-of t)."""
    mean = s.rolling(window).mean()
    std = s.rolling(window).std().replace(0.0, np.nan)
    return (s - mean) / std


def divergence(
    close: pd.Series, indicator: pd.Series, change_window: int = 14, norm_window: int = 60
) -> pd.Series:
    """Indicator-vs-price divergence: the recent move of the indicator minus that
    of price, both z-scored over ``norm_window`` so they are comparable.

    Positive = the indicator is stronger than price warrants (bullish divergence,
    e.g. price fell but momentum held); negative = bearish divergence. Uses only
    past changes (``diff`` / rolling), so it is look-ahead free.
    """
    pr_chg = close.diff(change_window)
    ind_chg = indicator.diff(change_window)
    return _roll_z(ind_chg, norm_window) - _roll_z(pr_chg, norm_window)


def momentum_deceleration(
    close: pd.Series, mom_window: int = 10, accel_window: int = 5, atr_window: int = 10
) -> pd.Series:
    """Change in momentum (a discrete 2nd derivative) normalised by ATR.

    ``mom = close.diff(mom_window)``; the feature is how much that momentum has
    changed over the last ``accel_window`` bars. Negative = an up-move losing
    steam (deceleration), a classic exhaustion / reversion precursor.
    """
    mom = close.diff(mom_window)
    return (mom - mom.shift(accel_window)) / _atr_cc(close, atr_window)


def er_change(close: pd.Series, window: int = 20, change_window: int = 5) -> pd.Series:
    """Change in Efficiency Ratio over ``change_window``; negative = trend quality
    decaying (the move is turning choppy), which favours reversion."""
    return efficiency_ratio(close, window).diff(change_window)


def vol_ratio(close: pd.Series, short: int = 10, long: int = 50) -> pd.Series:
    """Short-vs-long realized-vol ratio (>1 = volatility expansion, <1 = contraction)."""
    d = close.diff()
    short_v = d.rolling(short).std()
    long_v = d.rolling(long).std().replace(0.0, np.nan)
    return short_v / long_v


def ema_alignment(close: pd.Series, atr_window: int = 10) -> pd.Series:
    """Bullish EMA stack: mean distance of close above EMA 20/50/100 in ATR units."""
    atr = _atr_cc(close, atr_window)
    parts = [(close - moving_average(close, n, "ema")) / atr for n in (20, 50, 100)]
    return sum(parts) / len(parts)


def momentum(close: pd.Series, window: int, atr_window: int = 10) -> pd.Series:
    """Vol-normalised momentum: close.diff(window) / ATR (+ = up-move)."""
    return close.diff(window) / _atr_cc(close, atr_window)


def variance_ratio(close: pd.Series, q: int = 5, window: int = 60) -> pd.Series:
    """Lo-MacKinlay variance ratio: Var(q-bar change) / (q * Var(1-bar change)),
    rolling over ``window``. >1 = trending/persistent, <1 = mean-reverting."""
    d1 = close.diff()
    dq = close.diff(q)
    var1 = d1.rolling(window).var().replace(0.0, np.nan)
    varq = dq.rolling(window).var()
    return varq / (q * var1)


def _lag1_autocorr(x: np.ndarray) -> float:
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if denom == 0.0:
        return np.nan
    return float(np.sum(x[1:] * x[:-1]) / denom)


def autocorr_1(close: pd.Series, window: int = 20) -> pd.Series:
    """Lag-1 autocorrelation of daily changes over ``window`` (+ = persistence)."""
    return close.diff().rolling(window).apply(_lag1_autocorr, raw=True)


def direction_persistence(close: pd.Series, window: int = 20) -> pd.Series:
    """One-sidedness of the path over ``window``: ``|2·frac_up − 1|`` in ``[0, 1]``.

    0 = as many up days as down days (choppy / range); 1 = every session in the
    same direction (a very clean, persistent trend). Direction-agnostic quality
    measure (Bloque C), distinct from the regime/persistence metrics of Bloque A.
    Uses only past changes, so it is look-ahead free.
    """
    up = (close.diff() > 0).astype(float)
    frac_up = up.rolling(window).mean()
    return (2.0 * frac_up - 1.0).abs()


def r2_trend(close: pd.Series, window: int = 20) -> pd.Series:
    """R^2 of a linear fit of close over ``window`` (trend cleanliness, 0..1)."""
    def _r2(y: np.ndarray) -> float:
        if np.std(y) == 0.0:
            return np.nan
        x = np.arange(len(y), dtype=float)
        r = np.corrcoef(x, y)[0, 1]
        return float(r * r)
    return close.rolling(window).apply(_r2, raw=True)


# name -> callable(close) -> continuous feature series (all as-of t).
FEATURES: dict[str, object] = {
    # Level / extension (Bloque D) and reversion (Bloque E).
    "z_10": lambda c: zscore(c, 10),
    "z_20": lambda c: zscore(c, 20),
    "z_50": lambda c: zscore(c, 50),
    "pctb_20_2": lambda c: bollinger_pctb(c, 20, 2.0),
    "pctb_10_1_5": lambda c: bollinger_pctb(c, 10, 1.5),
    "keltner_dist_20": lambda c: keltner_distance(c, 20, 10),
    "rsi_2": lambda c: rsi(c, 2),
    "rsi_14": lambda c: rsi(c, 14),
    # Exhaustion / reversion precursors (Bloque E): divergences, momentum
    # deceleration, efficiency decay and volatility expansion.
    "rsi_div_14": lambda c: divergence(c, rsi(c, 14), 14, 60),
    "macd_div": lambda c: divergence(c, macd(c)[0], 14, 60),
    "mom_decel_10": lambda c: momentum_deceleration(c, 10, 5, 10),
    "er_drop_20": lambda c: er_change(c, 20, 5),
    "vol_ratio": lambda c: vol_ratio(c, 10, 50),
    # Regime (Bloque A) and direction / quality (Bloque B/C).
    "er_20": lambda c: efficiency_ratio(c, 20),
    "slope_20": lambda c: slope_normalized(c, 20, 10),
    "macd_hist": lambda c: macd_histogram(c),
    # Direction / regime enrichment (Bloque A/B).
    "ema_align": lambda c: ema_alignment(c),
    "mom_5": lambda c: momentum(c, 5),
    "mom_10": lambda c: momentum(c, 10),
    "mom_20": lambda c: momentum(c, 20),
    "variance_ratio_5": lambda c: variance_ratio(c, 5, 60),
    "autocorr_20": lambda c: autocorr_1(c, 20),
    # Trend quality (Bloque C): linearity and directional persistence.
    "r2_20": lambda c: r2_trend(c, 20),
    "dir_persistence_20": lambda c: direction_persistence(c, 20),
}

FEATURE_NAMES: list[str] = list(FEATURES)


def _contract_features(close: pd.Series) -> pd.DataFrame:
    """All continuous features for one (date-sorted) contract, sharing its index."""
    return pd.DataFrame({name: fn(close) for name, fn in FEATURES.items()}, index=close.index)


def add_features(
    frame: pd.DataFrame,
    price_col: str = "close",
    contract_col: str = "contract",
    date_col: str = "date",
) -> pd.DataFrame:
    """Return ``frame`` (sorted by contract, date) with a column per continuous feature.

    Features are computed independently per contract (warmup rows are ``NaN``),
    so a short contract simply yields fewer usable feature rows. No feature uses
    information after ``t``.
    """
    for col in (price_col, contract_col, date_col):
        if col not in frame.columns:
            raise KeyError(f"frame is missing required column {col!r}")

    ordered = frame.sort_values([contract_col, date_col])
    parts = [
        _contract_features(group[price_col].astype(float))
        for _, group in ordered.groupby(contract_col, sort=False)
    ]
    feats = pd.concat(parts).reindex(ordered.index)
    return pd.concat([ordered, feats], axis=1)
