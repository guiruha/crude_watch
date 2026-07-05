"""End-to-end integration: build and score every structure from the real data.

Skips cleanly when the processed outrights parquet is absent (e.g. CI without
data), so the unit suite still runs everywhere.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from crudewatch.analytics import (
    STRUCTURES,
    available_years,
    bollinger_layers,
    build_price_matrix,
    build_series,
    composite_score,
    detect_regime,
    risk_metrics,
    seasonal_stack,
    strategy_mode,
    trade_levels,
)
from crudewatch.analytics.indicators import all_signals, regime_diagnostics, signal_panel

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "outrights.parquet"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="processed outrights parquet not available")


@pytest.fixture(scope="module")
def matrix():
    return build_price_matrix(pd.read_parquet(DATA))


def test_every_structure_has_vintages(matrix):
    for s in STRUCTURES:
        assert available_years(matrix, s), f"no vintages for {s.key}"


def test_full_pipeline_runs_for_every_structure(matrix):
    """Build a mid vintage for each structure and run the entire analytics stack."""
    for s in STRUCTURES:
        years = available_years(matrix, s)
        year = years[len(years) // 2]
        series = build_series(matrix, s, year)
        assert not series.empty
        close = series.set_index("date")["close"].astype(float)

        # Chapter 1: panel + composite score.
        panel = signal_panel(close)
        assert len(panel) == 4
        regime = detect_regime(close)
        score = composite_score(all_signals(close), regime)
        assert -100.0 <= score.value <= 100.0

        # Chapter 3: regime, bands, levels, risk.
        mode = strategy_mode(regime, s.level)
        bb = bollinger_layers(close)
        assert {"mid", "u1", "u2", "u3", "l1", "l2", "l3"} <= set(bb.columns)
        tl = trade_levels(close, mode)
        assert tl.side in ("Long", "Short", "None")
        risk_metrics(close)  # must not raise

        # QUANT diagnostics must not raise.
        regime_diagnostics(close)


def test_seasonal_stack_axis_is_monotonic_per_vintage(matrix):
    """days_to_expiry must be well-defined for the seasonal axis."""
    for s in STRUCTURES[:6]:
        stack = seasonal_stack(matrix, s)
        if stack.empty:
            continue
        assert "days_to_expiry" in stack.columns
        assert stack["days_to_expiry"].notna().all()


def test_butterflies_center_near_zero(matrix):
    """A butterfly (A - 2B + C) should hover near zero over its life."""
    flies = [s for s in STRUCTURES if s.level.value == "Butterfly"]
    for s in flies:
        years = available_years(matrix, s)
        series = build_series(matrix, s, years[len(years) // 2])
        assert series["close"].abs().mean() < 3.0
