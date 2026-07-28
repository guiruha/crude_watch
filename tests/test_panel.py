"""Tests for the analogous-contract level panel (Bloque D)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from crudewatch.research.panel import add_level_panel


def _panel_frame() -> pd.DataFrame:
    """One December slot; each vintage sits in the same life-phase bin with a
    steadily rising close (10, 11, ... per vintage)."""
    rows = []
    for v in range(2010, 2017):  # 7 vintages
        rows.append({"contract": f"CLZ{v}", "date": pd.Timestamp(f"{v}-06-01"),
                     "close": float(v - 2000), "slot": "Z", "vintage": v, "dte": 100})
    return pd.DataFrame(rows)


def test_first_vintages_have_no_reference():
    out = add_level_panel(_panel_frame(), dte_bin_days=21, min_prior=3)
    # 2010, 2011, 2012 have < 3 prior vintages -> NaN.
    early = out[out["vintage"].isin([2010, 2011, 2012])]
    assert early["level_pct"].isna().all()
    assert early["level_z"].isna().all()


def test_rising_slot_is_expensive_versus_prior_vintages():
    out = add_level_panel(_panel_frame(), dte_bin_days=21, min_prior=3)
    # 2016 close=16 vs priors [10,11,12,13,14,15] -> most expensive ever seen.
    row = out[out["vintage"] == 2016].iloc[0]
    assert row["level_pct"] == 1.0
    assert row["level_z"] > 0
    # A cheap synthetic point would rank at the bottom.
    assert out[out["vintage"] == 2013].iloc[0]["level_pct"] == 1.0  # 13 > [10,11,12]


def test_point_in_time_ignores_later_vintages():
    df = _panel_frame()
    out_full = add_level_panel(df, min_prior=3)
    # Dropping the future vintages must not change 2015's level (only priors used).
    out_trunc = add_level_panel(df[df["vintage"] <= 2015], min_prior=3)
    a = out_full[out_full["vintage"] == 2015].iloc[0]
    b = out_trunc[out_trunc["vintage"] == 2015].iloc[0]
    assert a["level_pct"] == b["level_pct"]
    assert np.isclose(a["level_z"], b["level_z"])


def test_missing_columns_raise():
    try:
        add_level_panel(pd.DataFrame({"close": [1.0]}))
    except KeyError as exc:
        assert "panel columns" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError for missing lifecycle columns")
