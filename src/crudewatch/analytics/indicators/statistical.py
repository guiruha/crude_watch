"""QUANT statistical layer — the mean-reversion diagnostics (Phase 5).

These are the tools that lift the platform above a retail dashboard: formal
time-series statistics on the spread. Unlike the directional indicators, most of
these are *diagnostic* — they tell you whether the mean-reversion premise holds
(and how fast it reverts), which the composite engine uses to calibrate
confidence rather than as a directional vote:

* **Hurst exponent** — H < 0.5 anti-persistent (mean-reverting), > 0.5 trending.
* **Half-life** (Ornstein-Uhlenbeck) — days for a deviation to decay by half.
* **ADF p-value** — probability the series has a unit root (non-stationary).
* **Variance ratio** — < 1 mean-reverting, > 1 trending.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def hurst_exponent(close: pd.Series, max_lag: int = 20) -> float | None:
    """Hurst exponent via the aggregated variance of lagged differences.

    Fits ``log Var(Δ_lag) ≈ 2H · log(lag)`` over ``lag = 2..max_lag``. Returns
    ``None`` when there is not enough data.
    """
    x = close.dropna().to_numpy()
    if x.size < max_lag * 2:
        return None
    lags = np.arange(2, max_lag + 1)
    tau = []
    for lag in lags:
        diff = x[lag:] - x[:-lag]
        std = diff.std()
        tau.append(std if std > 0 else np.nan)
    tau = np.asarray(tau)
    ok = ~np.isnan(tau) & (tau > 0)
    if ok.sum() < 3:
        return None
    slope = np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)[0]
    return float(slope)  # slope == H for this estimator


def half_life(close: pd.Series) -> float | None:
    """Ornstein-Uhlenbeck half-life: OLS of Δy on lagged y, ``-ln2 / b``.

    Returns ``None`` for a non-mean-reverting fit (b >= 0) or too little data.
    """
    y = close.dropna()
    if len(y) < 20:
        return None
    lagged = y.shift(1)
    delta = y - lagged
    frame = pd.concat([delta, lagged], axis=1).dropna()
    frame.columns = ["delta", "lag"]
    x = frame["lag"].to_numpy()
    d = frame["delta"].to_numpy()
    b, _ = np.polyfit(x, d, 1)
    if b >= 0:
        return None
    return float(-np.log(2) / b)


def variance_ratio(close: pd.Series, lag: int = 5) -> float | None:
    """Lo-MacKinlay variance ratio: Var(k-step) / (k · Var(1-step)).

    ~1 random walk, < 1 mean-reverting, > 1 trending.
    """
    x = close.dropna().to_numpy()
    if x.size < lag * 3:
        return None
    d1 = np.diff(x)
    dk = x[lag:] - x[:-lag]
    v1 = d1.var(ddof=1)
    vk = dk.var(ddof=1)
    if v1 == 0:
        return None
    return float(vk / (lag * v1))


def adf_pvalue(close: pd.Series) -> float | None:
    """Augmented Dickey-Fuller p-value (stationarity). Needs statsmodels.

    Low p (< ~0.1) supports stationarity / mean-reversion. Returns ``None`` if
    statsmodels is unavailable or the series is too short/degenerate.
    """
    y = close.dropna()
    if len(y) < 20 or y.nunique() < 3:
        return None
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return None
    try:
        return float(adfuller(y.to_numpy(), autolag="AIC")[1])
    except (ValueError, np.linalg.LinAlgError):
        return None


@dataclass(frozen=True)
class RegimeDiagnostics:
    hurst: float | None
    half_life: float | None
    adf_pvalue: float | None
    variance_ratio: float | None

    @property
    def is_mean_reverting(self) -> bool:
        """Majority vote of the available diagnostics toward mean-reversion."""
        votes = []
        if self.hurst is not None:
            votes.append(self.hurst < 0.5)
        if self.variance_ratio is not None:
            votes.append(self.variance_ratio < 1.0)
        if self.adf_pvalue is not None:
            votes.append(self.adf_pvalue < 0.10)
        if not votes:
            return False
        return sum(votes) > len(votes) / 2


def regime_diagnostics(close: pd.Series) -> RegimeDiagnostics:
    """Bundle the four mean-reversion diagnostics for a series."""
    return RegimeDiagnostics(
        hurst=hurst_exponent(close),
        half_life=half_life(close),
        adf_pvalue=adf_pvalue(close),
        variance_ratio=variance_ratio(close),
    )


__all__ = [
    "hurst_exponent", "half_life", "variance_ratio", "adf_pvalue",
    "RegimeDiagnostics", "regime_diagnostics",
]
