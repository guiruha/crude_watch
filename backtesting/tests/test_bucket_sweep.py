"""Tests for the offline indicator bucket sweep.

The load-bearing property under test is that nothing which decides a row's
bucket may depend on that row's date or any later date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.research.bucket_sweep import BUCKET_LABELS, MISSING_CODE, expanding_cutoffs


def _dated(values, contract="A", start="2015-01-01"):
    """One row per business day, one contract, with the given indicator values."""
    return pd.DataFrame(
        {
            "date": pd.bdate_range(start, periods=len(values)),
            "contract": contract,
            "ind": np.asarray(values, dtype=float),
        }
    )


def test_cutoffs_use_only_strictly_prior_dates():
    """Edges applied on date d are the quantiles of the rows before d, nothing else."""
    values = [5, 1, 9, 3, 7, 2, 8, 4, 6, 0]
    panel = _dated(values)

    cuts = expanding_cutoffs(panel, ["ind"], n_buckets=3, min_history=3)
    on_date_3 = cuts[cuts["date"] == panel["date"].iloc[3]].sort_values("edge_index")

    # Prior rows are [5, 1, 9]; linear-interpolated 1/3 and 2/3 quantiles.
    assert on_date_3["value"].tolist() == pytest.approx([3.666667, 6.333333], abs=1e-5)


def test_cutoffs_are_nan_inside_the_warmup():
    panel = _dated([5, 1, 9, 3, 7, 2, 8, 4, 6, 0])

    cuts = expanding_cutoffs(panel, ["ind"], n_buckets=3, min_history=4)

    early = cuts[cuts["date"].isin(panel["date"].iloc[:4])]
    assert early["value"].isna().all()
    late = cuts[cuts["date"] == panel["date"].iloc[4]]
    assert late["value"].notna().all()


def test_same_date_rows_do_not_influence_each_other():
    """Two contracts share every date; changing one must not move same-date edges,
    but it must still move edges on strictly later dates once that row becomes
    prior history — otherwise this test would pass even under partial leakage.
    """
    a = _dated([5, 1, 9, 3, 7, 2, 8, 4, 6, 0], contract="A")
    b = _dated([4, 2, 8, 1, 6, 3, 9, 5, 7, 1], contract="B")
    panel = pd.concat([a, b], ignore_index=True)

    base = expanding_cutoffs(panel, ["ind"], n_buckets=3, min_history=4)

    bumped = panel.copy()
    mid_date = panel["date"].sort_values().unique()[len(panel["date"].unique()) // 2]
    bumped.loc[(bumped["contract"] == "B") & (bumped["date"] == mid_date), "ind"] = 999.0
    after = expanding_cutoffs(bumped, ["ind"], n_buckets=3, min_history=4)

    # Same-day rows must not influence their own date's edges.
    same_day_base = base[base["date"] == mid_date].reset_index(drop=True)
    same_day_after = after[after["date"] == mid_date].reset_index(drop=True)
    pd.testing.assert_frame_equal(same_day_base, same_day_after)

    # But the mutated row must enter the prior-history pool for later dates,
    # proving the test is actually sensitive to the data.
    later_base = base[base["date"] > mid_date].reset_index(drop=True)
    later_after = after[after["date"] > mid_date].reset_index(drop=True)
    assert not later_base["value"].equals(later_after["value"])


def test_empty_panel_returns_empty_frame():
    """An empty panel must not crash; it should yield an empty tidy frame."""
    panel = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "ind": pd.Series(dtype=float)})

    cuts = expanding_cutoffs(panel, ["ind"], n_buckets=3, min_history=4)

    assert list(cuts.columns) == ["date", "indicator", "edge_index", "value"]
    assert len(cuts) == 0


def test_module_constants():
    assert BUCKET_LABELS == ("low", "mid", "high")
    assert MISSING_CODE == -1
