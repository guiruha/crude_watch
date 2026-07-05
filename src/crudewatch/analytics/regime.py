"""Regime detection, multilayer Bollinger bands and dynamic TP/SL (Chapter 3).

The spread's structure sets the strategy mode:

* **Contango** (upward curve → the near-minus-far spread sits below zero) →
  **mean-reversion**: fade band touches back toward the mean.
* **Backwardation** (spread above zero) → **breakout/momentum**: ride extensions.
* **Butterflies** measure curvature, not direction, so they run mean-reversion
  regardless of the outright curve's regime.

Take-profit is not a fixed price but a band that moves with volatility and mode:
mean-reversion targets the SMA20 with the far band as stop; breakout targets the
±3σ band with the ±2σ band as stop (and an ATR trailing stop in the app).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from crudewatch.analytics.indicators.volatility import atr
from crudewatch.analytics.scoring import Regime
from crudewatch.analytics.structures import Level, Structure


class StrategyMode(str, Enum):
    MEAN_REVERSION = "Mean reversion"
    BREAKOUT = "Breakout"


def detect_regime(close: pd.Series, window: int = 20) -> Regime:
    """Contango vs. backwardation from the recent spread level.

    The series is ``near - far``: a negative level means the deferred leg trades
    above the near one (upward/contango curve); a positive level is
    backwardation.
    """
    level = close.tail(window).mean()
    return Regime.BACKWARDATION if level > 0 else Regime.CONTANGO


def strategy_mode(regime: Regime, level: Level) -> StrategyMode:
    """Map regime + structure tier to the operating mode."""
    if level == Level.BUTTERFLY:
        return StrategyMode.MEAN_REVERSION
    return StrategyMode.MEAN_REVERSION if regime == Regime.CONTANGO else StrategyMode.BREAKOUT


def bollinger_layers(close: pd.Series, period: int = 20) -> pd.DataFrame:
    """SMA20 with ±1σ/±2σ/±3σ bands, plus %B and bandwidth."""
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    width = (2 * std).replace(0, np.nan)
    out = {"mid": mid}
    for k in (1, 2, 3):
        out[f"u{k}"] = mid + k * std
        out[f"l{k}"] = mid - k * std
    frame = pd.DataFrame(out)
    frame["pctb"] = (close - frame["l2"]) / (frame["u2"] - frame["l2"]).replace(0, np.nan)
    frame["bandwidth"] = width / mid.replace(0, np.nan)
    return frame


@dataclass(frozen=True)
class RiskMetrics:
    pctb: float | None
    bandwidth: float | None
    atr: float | None
    atr_percentile: float | None    # 0..100, current ATR vs its own history


def risk_metrics(close: pd.Series, period: int = 20, atr_period: int = 14) -> RiskMetrics:
    if len(close) < period:
        return RiskMetrics(None, None, None, None)
    last = bollinger_layers(close, period).iloc[-1]
    a = atr(close, atr_period)
    atr_now = a.iloc[-1]
    atr_pct = float((a.dropna() <= atr_now).mean() * 100) if a.notna().any() else None
    return RiskMetrics(
        pctb=None if pd.isna(last["pctb"]) else float(last["pctb"]),
        bandwidth=None if pd.isna(last["bandwidth"]) else float(last["bandwidth"]),
        atr=None if pd.isna(atr_now) else float(atr_now),
        atr_percentile=atr_pct,
    )


@dataclass(frozen=True)
class TradeLevels:
    mode: StrategyMode
    side: str                 # "Long" | "Short" | "None"
    entry: float | None
    take_profit: float | None
    stop_loss: float | None
    note: str = ""

    @property
    def risk_reward(self) -> float | None:
        if self.entry is None or self.take_profit is None or self.stop_loss is None:
            return None
        risk = abs(self.entry - self.stop_loss)
        if risk == 0:
            return None
        return abs(self.take_profit - self.entry) / risk


def trade_levels(close: pd.Series, mode: StrategyMode, period: int = 20) -> TradeLevels:
    """Suggested entry / TP / SL for entering *now*, given the mode and bands.

    Mean reversion: fade an outer-band (±2σ) touch back to the SMA20, stop at ±3σ.
    Breakout: enter on a ±2σ break targeting ±3σ, stop back at ±2σ.
    Returns a ``side="None"`` plan when the price sits inside the bands.
    """
    if len(close) < period:
        return TradeLevels(mode, "None", None, None, None, "insufficient history")
    bb = bollinger_layers(close, period).iloc[-1]
    price = float(close.iloc[-1])
    if bb[["u2", "l2", "u3", "l3", "mid"]].isna().any():
        return TradeLevels(mode, "None", None, None, None, "insufficient history")

    if mode == StrategyMode.MEAN_REVERSION:
        if price >= bb["u2"]:
            return TradeLevels(mode, "Short", price, float(bb["mid"]), float(bb["u3"]), "fade upper band to mean")
        if price <= bb["l2"]:
            return TradeLevels(mode, "Long", price, float(bb["mid"]), float(bb["l3"]), "fade lower band to mean")
        return TradeLevels(mode, "None", None, None, None, "inside bands — no mean-reversion setup")

    # Breakout / momentum
    if price >= bb["u2"]:
        return TradeLevels(mode, "Long", price, float(bb["u3"]), float(bb["u2"]), "upside breakout, trail with ATR")
    if price <= bb["l2"]:
        return TradeLevels(mode, "Short", price, float(bb["l3"]), float(bb["l2"]), "downside breakout, trail with ATR")
    return TradeLevels(mode, "None", None, None, None, "inside bands — no breakout")


__all__ = [
    "StrategyMode", "Regime",
    "detect_regime", "strategy_mode", "bollinger_layers",
    "RiskMetrics", "risk_metrics", "TradeLevels", "trade_levels",
]
