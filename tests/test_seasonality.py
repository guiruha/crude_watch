"""Unit tests for the seasonality *context* layer (v1 spec)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crudewatch.analytics.seasonality import (
    MATURITY_BUCKETS,
    MAX_DAYS_TO_EXPIRY,
    alignment,
    alignment_backtest,
    bucket_bounds,
    buffer_cone,
    cap_to_year,
    current_percentile,
    forward_tendency,
    lifecycle_phase,
    monthly_heatmap,
    seasonal_bias,
)


def _stack(vintages: int = 4, per_year: int = 60, slope: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Synthetic stack: `vintages` years, each a noisy line over the same dates."""
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(vintages):
        year = 2016 + i
        dates = pd.date_range(f"{year}-01-01", periods=per_year, freq="D")
        close = slope * np.arange(per_year) + rng.normal(0, 0.5, per_year)
        parts.append(pd.DataFrame({
            "vintage": year,
            "date": dates,
            "day_of_year": dates.dayofyear,
            "days_to_expiry": (dates.max() - dates).days,
            "close": close,
        }))
    return pd.concat(parts, ignore_index=True)


# -- cone / heatmap / percentile ----------------------------------------------

def test_buffer_cone_columns_and_ordering():
    cone = buffer_cone(_stack(), buffer_days=5)
    assert list(cone.columns) == ["mean", "std", "lower", "upper", "median", "n"]
    assert (cone["upper"] >= cone["mean"]).all()
    assert (cone["lower"] <= cone["mean"]).all()
    assert (cone["n"] > 0).all()
    assert cone.index.is_monotonic_increasing


def test_buffer_cone_wider_window_more_samples():
    stack = _stack()
    narrow = buffer_cone(stack, buffer_days=2)["n"].mean()
    wide = buffer_cone(stack, buffer_days=10)["n"].mean()
    assert wide > narrow


def test_buffer_cone_exclude_vintage():
    stack = _stack()
    full = buffer_cone(stack, buffer_days=5)["n"].mean()
    excl = buffer_cone(stack, buffer_days=5, exclude_vintage=2016)["n"].mean()
    assert excl < full


def test_current_percentile_bounds():
    p = current_percentile(_stack(seed=1), current_vintage=2019, buffer_days=5)
    assert p is None or 0.0 <= p <= 100.0


def test_current_percentile_high_when_active_is_richest():
    stack = _stack(seed=2)
    mask = stack["vintage"] == 2019
    stack.loc[stack[mask].index[-1], "close"] = 100.0
    p = current_percentile(stack, current_vintage=2019, buffer_days=7)
    assert p == pytest.approx(100.0)


def test_monthly_heatmap_shape():
    hm = monthly_heatmap(_stack(vintages=3, per_year=120))
    assert hm.index.name == "month"
    assert set(hm.columns) <= {2016, 2017, 2018}


# -- seasonal bias & consistency ----------------------------------------------

def test_seasonal_bias_counts_consistent():
    b = seasonal_bias(_stack(), current_vintage=2019, buffer_days=5, horizon=10)
    assert b.up + b.down == b.total
    assert 0.0 <= b.hit_rate <= 1.0
    assert b.direction in ("Bullish", "Bearish", "Neutral")
    assert b.consistency in ("Strong", "Medium", "Weak", "Mixed")


def test_seasonal_bias_detects_uptrend_and_strong_consistency():
    # Every vintage strongly rising; active vintage only partway through.
    stack = _stack(vintages=12, slope=1.0, seed=3)
    active = stack["vintage"] == 2019
    stack = stack[~active | (stack["day_of_year"] <= 20)].reset_index(drop=True)
    b = seasonal_bias(stack, current_vintage=2019, buffer_days=5, horizon=15)
    assert b.direction == "Bullish"
    assert b.up >= b.down
    assert b.total > 0
    assert b.consistency == "Strong"       # unanimous rise -> hit_rate 1.0


def test_consistency_mixed_on_small_sample():
    b = seasonal_bias(_stack(vintages=2), current_vintage=2017, buffer_days=5, horizon=10)
    assert b.consistency == "Mixed"        # total < 5 always reads Mixed


def test_forward_tendency_covers_all_horizons():
    t = forward_tendency(_stack(), current_vintage=2019, buffer_days=5)
    assert set(t) == {10, 20, 30}
    assert all(v.direction in ("Bullish", "Bearish", "Neutral") for v in t.values())


# -- alignment ----------------------------------------------------------------

@pytest.mark.parametrize("seasonal,trend,expected", [
    ("Bullish", "Bullish", "Tailwind"),
    ("Bearish", "Bearish", "Tailwind"),
    ("Bullish", "Bearish", "Headwind"),
    ("Bearish", "Bullish", "Headwind"),
    ("Neutral", "Bullish", "Mixed"),
    ("Bullish", "Neutral", "Mixed"),
    ("Bearish", "Down", "Tailwind"),      # trend vocabulary normalised
    ("Bullish", "Down", "Headwind"),
])
def test_alignment_matrix(seasonal, trend, expected):
    assert alignment(seasonal, trend) == expected


# -- lifecycle phase ----------------------------------------------------------

def test_lifecycle_expiry_risk_near_expiry():
    stack = _stack(per_year=120)
    # Trim the active vintage so its latest row is a few days from expiry.
    active = stack["vintage"] == 2019
    keep = ~active | (stack["days_to_expiry"] >= 5)
    stack = stack[keep].reset_index(drop=True)
    life = lifecycle_phase(stack, current_vintage=2019, expiry_days=15)
    assert life.phase == "Expiry-risk"
    assert 0.0 <= life.elapsed <= 1.0


def test_lifecycle_quiet_early_in_life():
    stack = _stack(per_year=200)
    active = stack["vintage"] == 2019
    # Keep only the earliest part of the active life (largest days_to_expiry).
    thresh = stack.loc[active, "days_to_expiry"].max() - 10
    stack = stack[~active | (stack["days_to_expiry"] >= thresh)].reset_index(drop=True)
    life = lifecycle_phase(stack, current_vintage=2019)
    assert life.phase == "Quiet"


# -- validation backtest ------------------------------------------------------

def test_alignment_backtest_shape_and_bounds():
    bt = alignment_backtest(_stack(vintages=6, per_year=120, slope=0.3, seed=5), horizon=15)
    assert list(bt.columns) == ["alignment", "n", "avg_continuation", "hit_rate", "median_return"]
    assert set(bt["alignment"]) == {"Tailwind", "Headwind", "Mixed"}
    valid = bt["hit_rate"].dropna()
    assert ((valid >= 0) & (valid <= 1)).all()


# -- one-year cap -------------------------------------------------------------

def _long_life_stack(vintages: int = 4, per_year: int = 600, seed: int = 7) -> pd.DataFrame:
    """A stack whose life exceeds one year (days_to_expiry up to ~599)."""
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(vintages):
        year = 2016 + i
        dates = pd.date_range(f"{year}-01-01", periods=per_year, freq="D")
        parts.append(pd.DataFrame({
            "vintage": year,
            "date": dates,
            "day_of_year": dates.dayofyear,
            "days_to_expiry": (dates.max() - dates).days,
            "close": rng.normal(0, 0.5, per_year),
        }))
    return pd.concat(parts, ignore_index=True)


def test_cap_to_year_drops_older_than_365():
    stack = _long_life_stack()
    assert stack["days_to_expiry"].max() > MAX_DAYS_TO_EXPIRY
    capped = cap_to_year(stack)
    assert capped["days_to_expiry"].max() <= MAX_DAYS_TO_EXPIRY
    assert not capped.empty


def test_buffer_cone_respects_year_cap():
    cone = buffer_cone(_long_life_stack(), buffer_days=3)
    assert cone.index.max() <= MAX_DAYS_TO_EXPIRY


# -- maturity buckets ---------------------------------------------------------

def test_bucket_bounds_all_is_full_range():
    lo, hi = bucket_bounds("All")
    assert lo == 0 and hi > MAX_DAYS_TO_EXPIRY


def test_bucket_bounds_known_windows():
    assert bucket_bounds("6–1m") == (30, 183)
    assert bucket_bounds("1–0m") == (0, 30)
    assert set(MATURITY_BUCKETS) == {"All", "12–10m", "10–6m", "6–1m", "1–0m"}


def test_cap_to_year_min_dte_window():
    stack = _long_life_stack()
    win = cap_to_year(stack, max_dte=183, min_dte=30)
    assert win["days_to_expiry"].min() >= 30
    assert win["days_to_expiry"].max() <= 183
    assert not win.empty


def test_buffer_cone_restricted_to_bucket_window():
    lo, hi = bucket_bounds("6–1m")
    cone = buffer_cone(_long_life_stack(), buffer_days=3, max_dte=hi, min_dte=lo)
    assert cone.index.min() >= lo
    assert cone.index.max() <= hi


# -- safety -------------------------------------------------------------------

def test_empty_stack_is_safe():
    empty = pd.DataFrame(columns=["vintage", "date", "day_of_year", "days_to_expiry", "close"])
    assert buffer_cone(empty).empty
    assert current_percentile(empty, 2020) is None
    assert monthly_heatmap(empty).empty
    assert seasonal_bias(empty, 2020).total == 0
    assert lifecycle_phase(empty, 2020).phase == "Quiet"
    assert alignment_backtest(empty).empty
