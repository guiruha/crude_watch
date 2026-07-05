"""Signal-panel aggregation: group indicator votes into family verdicts.

Chapter 1's header traffic-light reads off :func:`signal_panel`, which runs the
CORE indicators, buckets them by family (Trend, Momentum, Volatility,
Statistical) and reduces each family to a net bias plus a conviction (share of
non-neutral members that agree with the net direction).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crudewatch.analytics.indicators import Bias, IndicatorSignal
from crudewatch.analytics.indicators.momentum import macd_signal, rsi_signal
from crudewatch.analytics.indicators.trend import adx_signal, ma_signal, supertrend_signal
from crudewatch.analytics.indicators.volatility import bollinger_signal, zscore_signal

# Family -> the signal callables it owns.
FAMILIES: dict[str, tuple] = {
    "Trend": (ma_signal, adx_signal, supertrend_signal),
    "Momentum": (macd_signal, rsi_signal),
    "Volatility": (bollinger_signal,),
    "Statistical": (zscore_signal,),
}


@dataclass(frozen=True)
class FamilyVerdict:
    family: str
    bias: Bias
    conviction: float               # 0..1 agreement among non-neutral members
    signals: list[IndicatorSignal]

    @property
    def label(self) -> str:
        return {Bias.BULLISH: "Bullish", Bias.BEARISH: "Bearish", Bias.NEUTRAL: "Neutral"}[self.bias]


def _reduce(signals: list[IndicatorSignal]) -> tuple[Bias, float]:
    votes = [s.bias for s in signals]
    net = sum(int(v) for v in votes)
    bias = Bias.BULLISH if net > 0 else Bias.BEARISH if net < 0 else Bias.NEUTRAL
    active = [v for v in votes if v != Bias.NEUTRAL]
    if not active or bias == Bias.NEUTRAL:
        return bias, 0.0
    agree = sum(1 for v in active if v == bias)
    return bias, agree / len(active)


def signal_panel(close: pd.Series) -> list[FamilyVerdict]:
    """Full family-grouped verdict list for the traffic-light header."""
    out: list[FamilyVerdict] = []
    for family, funcs in FAMILIES.items():
        signals = [f(close) for f in funcs]
        bias, conviction = _reduce(signals)
        out.append(FamilyVerdict(family, bias, conviction, signals))
    return out


def all_signals(close: pd.Series) -> list[IndicatorSignal]:
    """Flat list of every CORE indicator signal (for the composite engine / tables)."""
    return [f(close) for funcs in FAMILIES.values() for f in funcs]


__all__ = ["FAMILIES", "FamilyVerdict", "signal_panel", "all_signals"]
