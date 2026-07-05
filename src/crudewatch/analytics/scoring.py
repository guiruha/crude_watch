"""Composite scoring engine (Phase 5).

Aggregates every indicator's :class:`~crudewatch.analytics.indicators.IndicatorSignal`
into one regime-weighted score in [-100, +100]. In contango, mean-reversion
indicators (Z-Score, Hurst, adaptive RSI) carry more weight; in backwardation,
trend/momentum (ADX, Supertrend, MACD) dominate. Weights will be versioned and
auditable.

The skeleton fixes the public API and the score->reading buckets so the app can
render the gauge/table shell now and drop in real votes later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from crudewatch.analytics.indicators import Bias, IndicatorSignal


class Regime(str, Enum):
    CONTANGO = "Contango"
    BACKWARDATION = "Backwardation"


# Indicator -> category (matches the panel families).
CATEGORY_OF: dict[str, str] = {
    "MA ribbon": "trend",
    "ADX/DMI": "trend",
    "Supertrend": "trend",
    "MACD": "momentum",
    "RSI (adaptive)": "momentum",
    "Bollinger %B": "volatility",
    "Z-Score": "statistical",
}

# Regime-dependent category weights. In contango the spread mean-reverts, so the
# statistical/volatility read dominates; in backwardation it can extend, so
# trend/momentum lead. Weights are the auditable, versioned knobs of the engine.
REGIME_WEIGHTS: dict[Regime, dict[str, float]] = {
    Regime.CONTANGO: {"trend": 0.5, "momentum": 0.8, "volatility": 1.2, "statistical": 1.4},
    Regime.BACKWARDATION: {"trend": 1.3, "momentum": 1.2, "volatility": 0.8, "statistical": 0.6},
}

DEFAULT_WEIGHT = 1.0


def weight_for(name: str, regime: Regime) -> float:
    """Weight applied to an indicator's vote under a regime."""
    category = CATEGORY_OF.get(name, "trend")
    return REGIME_WEIGHTS[regime].get(category, DEFAULT_WEIGHT)


# Score bucket -> (reading label, colour token) as defined in the proposal.
SCORE_BANDS: list[tuple[float, float, str, str]] = [
    (60, 100, "Strong bullish bias", "strong_up"),
    (20, 60, "Moderate bullish bias", "up"),
    (-20, 20, "Neutral / no clear edge", "neutral"),
    (-60, -20, "Moderate bearish bias", "down"),
    (-100, -60, "Strong bearish bias", "strong_down"),
]


@dataclass(frozen=True)
class ScoreContribution:
    name: str
    category: str
    bias: Bias
    weight: float

    @property
    def contribution(self) -> float:
        return self.weight * int(self.bias)


@dataclass(frozen=True)
class CompositeScore:
    value: float                                  # -100..+100
    regime: Regime
    reading: str
    color_token: str
    contributions: list[ScoreContribution] = field(default_factory=list)


def reading_for(value: float) -> tuple[str, str]:
    """Map a score to its (reading, colour token)."""
    for lo, hi, label, token in SCORE_BANDS:
        if lo <= value <= hi:
            return label, token
    return "Neutral / no clear edge", "neutral"


def composite_score(signals: list[IndicatorSignal], regime: Regime) -> CompositeScore:
    """Regime-weighted aggregation of indicator votes into a [-100, +100] score.

    Each vote (``+1`` / ``-1`` / ``0``) is weighted by its category weight for the
    regime; the score is the weighted vote sum scaled by the total weight, so it
    stays bounded and neutral votes simply dampen conviction.
    """
    contributions = [
        ScoreContribution(s.name, CATEGORY_OF.get(s.name, "trend"), s.bias, weight_for(s.name, regime))
        for s in signals
    ]
    total_weight = sum(c.weight for c in contributions)
    value = 0.0 if total_weight == 0 else 100.0 * sum(c.contribution for c in contributions) / total_weight
    reading, token = reading_for(value)
    return CompositeScore(round(value, 1), regime, reading, token, contributions)


__all__ = [
    "Regime", "CompositeScore", "ScoreContribution", "SCORE_BANDS",
    "CATEGORY_OF", "REGIME_WEIGHTS", "weight_for", "reading_for", "composite_score",
]
