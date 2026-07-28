"""Feature / dataset pipeline for the scoring model (app-facing).

This subpackage holds the shared, look-ahead-free enrichment the live
Opportunity Score depends on: contract *lifecycle* (real WTI expiry dates,
days-to-expiry, seasonal slot and vintage), the continuous *feature* matrix, the
analogous-contract *level panel*, the forward-return *targets* and the one-shot
:func:`build_dataset` that stitches them together.

The heavier, offline walk-forward evaluation / regime-gated backtest / strategy
simulation and their HTML reports live outside ``src`` in the ``backtesting``
package; only the pieces the Streamlit app runs stay here.
"""
from crudewatch.research.lifecycle import (
    FAMILY_LIFECYCLE,
    NYMEXEnergyCalendar,
    add_lifecycle,
    wti_last_trading_day,
)
from crudewatch.research.targets import HORIZONS, add_forward_returns
from crudewatch.research.features import FEATURE_NAMES, FEATURES, add_features
from crudewatch.research.panel import PANEL_FEATURES, add_level_panel
from crudewatch.research.dataset import (
    COST_STUB_POINTS,
    build_dataset,
    regime_thresholds,
)

__all__ = [
    "FAMILY_LIFECYCLE",
    "NYMEXEnergyCalendar",
    "add_lifecycle",
    "wti_last_trading_day",
    "HORIZONS",
    "add_forward_returns",
    "FEATURES",
    "FEATURE_NAMES",
    "add_features",
    "PANEL_FEATURES",
    "add_level_panel",
    "COST_STUB_POINTS",
    "build_dataset",
    "regime_thresholds",
]
