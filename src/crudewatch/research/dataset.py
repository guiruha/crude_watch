"""Shared dataset enrichment for the feature/scoring pipeline.

This is the app-facing half of the old research layer: the one-shot enrichment
(:func:`build_dataset`) that turns a raw contract-family frame into the
lifecycle + feature + level-panel + forward-outcome matrix the Opportunity Score
consumes, plus the two small constants both the live scorer and the offline
backtesting engine share (the per-family cost stub and the efficiency-ratio
regime terciles). The heavy walk-forward evaluation machinery lives outside
``src`` in the ``backtesting`` package.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crudewatch.research.features import add_features
from crudewatch.research.lifecycle import add_lifecycle
from crudewatch.research.panel import add_level_panel
from crudewatch.research.targets import add_forward_returns

# Round-trip transaction cost in PRICE POINTS, per family. These are explicit
# conservative assumptions until measured bid/offer + slippage are available.
COST_STUB_POINTS: dict[str, float] = {
    "outrights": 0.02,
    "calendars": 0.04,
    "cracks": 0.03,
    "brent_wti": 0.02,
    "quarterly": 0.01,
    "semestral": 0.015,
    "yearly": 0.02,
    "flies": 0.02,
}


def build_dataset(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    """One-shot enrichment: lifecycle -> features -> level panel -> forward outcomes."""
    enriched = add_level_panel(add_features(add_lifecycle(frame, family)))
    return add_forward_returns(enriched)


def regime_thresholds(train_er: np.ndarray, lo_q: float = 1 / 3, hi_q: float = 2 / 3):
    """Bottom/top tercile cut-offs of the (training) efficiency ratio.

    Convention: ``er <= lo`` is the range (mean-reverting) regime, ``er >= hi``
    the trend (directional) regime, and the middle tercile a transition/dead
    zone. Shared by the live scorer (regime label) and the offline regime-gated
    backtest, so both use identical cut-offs.
    """
    return np.quantile(train_er, [lo_q, hi_q])
