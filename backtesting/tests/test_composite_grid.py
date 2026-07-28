"""Tests for the composite reversion signal (WS-C2) and conditional grid (WS-G3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.research.composite import evaluate_composite_family
from backtesting.research.diagnostics import conditional_grid

RNG = np.random.default_rng(11)


def _composite_frame() -> pd.DataFrame:
    """Two reversion features (z_20, rsi_2) that both predict a mean-reverting fwd."""
    rows = []
    for v in range(2010, 2019):  # 9 vintages -> several OOS folds
        per = 60
        z = RNG.normal(0, 1, per)
        rsi = 50 + RNG.normal(0, 12, per)
        # Cheap (low z / low rsi) -> positive forward move (reversion up).
        fwd = -(z + (rsi - 50) / 12.0) + RNG.normal(0, 0.3, per)
        rows.append(pd.DataFrame({
            "vintage": [v] * per,
            "contract": [f"{v}"] * per,
            "date": pd.date_range("2010-01-01", periods=per, freq="B"),
            "z_20": z,
            "rsi_2": rsi,
            "fwd_10": fwd,
            "mfe_10": np.abs(fwd),
            "mae_10": -np.abs(fwd),
        }))
    return pd.concat(rows, ignore_index=True)


def test_composite_combines_features_and_predicts_oos():
    data = _composite_frame()
    out = evaluate_composite_family(data, "fam", (10,), min_fold_rows=20)
    assert not out.empty
    row = out.iloc[0]
    assert row["feature"] == "composite"
    assert int(row["n_features"]) == 2            # both z_20 and rsi_2 combined
    # Oriented composite predicts the reversion target OOS with a stable, positive IC.
    assert row["ic_mean"] > 0.2
    assert row["ic_t"] > 2.0
    assert int(row["n_trades"]) > 0


def test_composite_needs_at_least_two_features():
    data = _composite_frame().drop(columns="rsi_2")
    # Only z_20 present among candidates -> fewer than 2 -> empty result.
    out = evaluate_composite_family(data, "fam", (10,), features=["z_20", "rsi_2"], min_fold_rows=20)
    assert out.empty


def test_conditional_grid_shape_and_reversion_gradient():
    n = 600
    z = RNG.normal(0, 1, n)
    conf = RNG.normal(0, 1, n)
    fwd = -1.5 * z + RNG.normal(0, 0.4, n)  # low z (cheap) -> positive fwd
    data = pd.DataFrame({"z_20": z, "level_z": conf, "fwd_10": fwd})

    grid = conditional_grid(data, "fam", "z_20", "level_z", 10, p_buckets=5, c_buckets=3)
    assert len(grid) == 15  # 5 primary x 3 confirmator cells
    assert set(grid["p_bucket"]) == {1, 2, 3, 4, 5}

    # Cheap primary bucket (P1) reverts up; dear bucket (P5) reverts down.
    p1 = grid[grid["p_bucket"] == 1]["mean_fwd"].mean()
    p5 = grid[grid["p_bucket"] == 5]["mean_fwd"].mean()
    assert p1 > p5


def test_conditional_grid_rejects_bad_inputs():
    data = pd.DataFrame({"z_20": RNG.normal(0, 1, 50), "fwd_10": RNG.normal(0, 1, 50)})
    # Missing confirmator column -> empty; too few rows also -> empty.
    assert conditional_grid(data, "fam", "z_20", "level_z", 10).empty
