"""Opportunity Score engine: block-decomposed, family-calibrated snapshots."""
from crudewatch.scoring.analogues import analogous_outcomes
from crudewatch.scoring.backtest import BacktestResult, backtest_contract
from crudewatch.scoring.blocks import (
    FamilyCalibrator,
    fit_calibrator,
    percentile,
    signed_pct,
)
from crudewatch.scoring.score import (
    FAMILY_WEIGHTS,
    WEIGHTS,
    BlockScores,
    InstrumentScore,
    score_family,
    score_instrument,
    weights_for,
)
from crudewatch.scoring.weight_search import (
    WalkForwardResult,
    WeightSearchResult,
    precompute_family,
    search_weights,
    walk_forward_weights,
)

__all__ = [
    "BlockScores",
    "InstrumentScore",
    "FamilyCalibrator",
    "FAMILY_WEIGHTS",
    "WEIGHTS",
    "weights_for",
    "BacktestResult",
    "analogous_outcomes",
    "backtest_contract",
    "fit_calibrator",
    "percentile",
    "score_family",
    "score_instrument",
    "signed_pct",
    "WalkForwardResult",
    "WeightSearchResult",
    "precompute_family",
    "search_weights",
    "walk_forward_weights",
]
