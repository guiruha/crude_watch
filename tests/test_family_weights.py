from __future__ import annotations

import numpy as np
import pandas as pd

from crudewatch.scoring.blocks import fit_calibrator
from crudewatch.scoring.score import BlockScores, compute_opportunity, weights_for, WEIGHTS


def _row(**kw):
    base = dict(er_20=0.1, level_pct=0.9, slope_20=0.0, mom_decel_10=0.0, vol_ratio=1.0)
    base.update(kw)
    return pd.Series(base)


def _blocks(regime="range", level=60.0, direction=0.0):
    return BlockScores(
        regime=regime, trendiness=50.0, direction=direction, strength=40.0,
        level=level, p_reversion=0.7, p_continuation=0.6,
    )


def test_weights_for_defaults_to_equal_when_no_override():
    assert weights_for("no_such_family") is WEIGHTS

def test_compute_opportunity_none_equals_explicit_default():
    b, r = _blocks(), _row()
    cal = fit_calibrator(pd.DataFrame({"er_20": [0.1, 0.2], "date": pd.to_datetime(["2020-01-01", "2020-01-02"])}), "flies")
    assert compute_opportunity(b, r, cal, None) == compute_opportunity(b, r, cal, WEIGHTS)
