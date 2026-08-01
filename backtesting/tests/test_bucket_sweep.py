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
    """Two contracts share every date; changing one must not move the other's edges."""
    a = _dated([5, 1, 9, 3, 7, 2, 8, 4, 6, 0], contract="A")
    b = _dated([4, 2, 8, 1, 6, 3, 9, 5, 7, 1], contract="B")
    panel = pd.concat([a, b], ignore_index=True)

    base = expanding_cutoffs(panel, ["ind"], n_buckets=3, min_history=4)

    bumped = panel.copy()
    last_date = panel["date"].max()
    bumped.loc[(bumped["contract"] == "B") & (bumped["date"] == last_date), "ind"] = 999.0
    after = expanding_cutoffs(bumped, ["ind"], n_buckets=3, min_history=4)

    pd.testing.assert_frame_equal(base, after)


def test_module_constants():
    assert BUCKET_LABELS == ("low", "mid", "high")
    assert MISSING_CODE == -1
