"""Indicator library (Phase 2+).

Each indicator will be a pure function of an OHLCV-lite series plus parameters,
returning either a series/frame (for overlays) or an
:class:`IndicatorSignal` (a bias vote consumed by the composite scoring engine).
Grouped by the catalogue's categories: trend, momentum, volatility, volume,
statistical/mean-reversion, spread-specific and seasonality.

Only the shared vote type ships in the skeleton; the concrete indicators land in
Phases 2 and 5.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Bias(IntEnum):
    """A single indicator's directional vote."""
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1


@dataclass(frozen=True)
class IndicatorSignal:
    """The standard output every scoring-eligible indicator returns."""
    name: str
    bias: Bias
    value: float | None = None   # the raw reading (e.g. the RSI level)
    note: str = ""               # short human-readable context


# Re-export the concrete indicators once the classes above are defined (the
# submodules import Bias/IndicatorSignal from here, so these imports must trail).
from crudewatch.analytics.indicators.momentum import (  # noqa: E402
    adaptive_rsi_bands,
    macd,
    macd_divergence,
    macd_signal,
    rsi,
    rsi_signal,
)
from crudewatch.analytics.indicators.panel import (  # noqa: E402
    FAMILIES,
    FamilyVerdict,
    all_signals,
    signal_panel,
)
from crudewatch.analytics.indicators.trend import (  # noqa: E402
    adx_signal,
    dmi_adx,
    ema,
    ma_ribbon,
    ma_signal,
    sma,
    supertrend,
    supertrend_signal,
)
from crudewatch.analytics.indicators.statistical import (  # noqa: E402
    RegimeDiagnostics,
    adf_pvalue,
    half_life,
    hurst_exponent,
    regime_diagnostics,
    variance_ratio,
)
from crudewatch.analytics.indicators.volatility import (  # noqa: E402
    atr,
    bollinger,
    bollinger_signal,
    true_range,
    zscore,
    zscore_dual,
    zscore_signal,
)

__all__ = [
    "Bias", "IndicatorSignal",
    # volatility / statistical
    "true_range", "atr", "bollinger", "zscore", "zscore_dual",
    "zscore_signal", "bollinger_signal",
    # trend
    "ema", "sma", "ma_ribbon", "ma_signal",
    "dmi_adx", "adx_signal", "supertrend", "supertrend_signal",
    # momentum
    "rsi", "adaptive_rsi_bands", "rsi_signal", "macd", "macd_divergence", "macd_signal",
    # statistical / QUANT
    "hurst_exponent", "half_life", "variance_ratio", "adf_pvalue",
    "RegimeDiagnostics", "regime_diagnostics",
    # panel
    "FAMILIES", "FamilyVerdict", "signal_panel", "all_signals",
]
