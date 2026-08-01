"""Tests for the offline indicator bucket sweep.

The load-bearing property under test is that nothing which decides a row's
bucket may depend on that row's date or any later date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.research.bucket_sweep import BUCKET_LABELS, MISSING_CODE, expanding_cutoffs
from backtesting.research.bucket_sweep import bucketize


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


def test_bucket_codes_ignore_future_bars():
    """THE binding test: appending future history must not re-bucket earlier rows.

    This is what separates point-in-time cutoffs from a full-sample
    `.quantile()`. If it ever fails, look-ahead has been reintroduced.
    """
    rng = np.random.default_rng(3)
    past = _dated(rng.normal(size=200))
    codes_past, _ = bucketize(past, ["ind"], n_buckets=3, min_history=20)

    future = _dated(rng.normal(size=80) + 25.0, start="2016-06-01")  # a regime shift
    extended = pd.concat([past, future], ignore_index=True)
    codes_extended, _ = bucketize(extended, ["ind"], n_buckets=3, min_history=20)

    original = codes_past["ind"].to_numpy()
    recomputed = codes_extended["ind"].to_numpy()[: len(past)]
    np.testing.assert_array_equal(original, recomputed)


def test_bucketize_marks_the_warmup_as_missing():
    panel = _dated([5, 1, 9, 3, 7, 2, 8, 4, 6, 0])

    codes, _ = bucketize(panel, ["ind"], n_buckets=3, min_history=4)

    assert (codes["ind"].to_numpy()[:4] == MISSING_CODE).all()
    assert (codes["ind"].to_numpy()[4:] != MISSING_CODE).all()


def test_bucketize_assigns_codes_against_prior_date_edges():
    panel = _dated([5, 1, 9, 3, 7, 2, 8, 4, 6, 0])

    codes, _ = bucketize(panel, ["ind"], n_buckets=3, min_history=3)

    # Row 3 has value 3.0 against edges 3.666667 / 6.333333 -> below both -> low.
    assert codes["ind"].iloc[3] == 0
    # Row 6 has value 8.0; prior [5,1,9,3,7,2] -> edges 2.666667 / 6.333333 -> high.
    assert codes["ind"].iloc[6] == 2


def test_bucketize_is_column_selective():
    """bucketize only reads the columns named in `indicators`; other columns —
    including forward-outcome columns — cannot influence the result no matter
    their contents, because they are never looked up by name.
    """
    rng = np.random.default_rng(1)
    panel = _dated(rng.normal(size=200))
    panel["fwd_1"] = rng.normal(size=200) * 1000.0
    codes_a, _ = bucketize(panel, ["ind"], n_buckets=3, min_history=20)

    perturbed = panel.copy()
    perturbed["fwd_1"] = rng.normal(size=200) * 1000.0
    codes_b, _ = bucketize(perturbed, ["ind"], n_buckets=3, min_history=20)

    pd.testing.assert_frame_equal(codes_a, codes_b)


def test_bucketize_marks_nan_indicator_value_as_missing():
    """A NaN reading in the row's own indicator value must yield MISSING_CODE,
    not a fabricated bucket. `value >= edge` is False for every edge when
    value is NaN, so a naive implementation that only checks the edges for
    NaN accumulates code 0 ("low") for a NaN reading instead of MISSING_CODE.
    """
    rng = np.random.default_rng(7)
    values = rng.normal(size=30)
    values[25] = np.nan  # well past warmup (min_history=20)
    panel = _dated(values)

    codes, _ = bucketize(panel, ["ind"], n_buckets=3, min_history=20)

    assert codes["ind"].iloc[25] == MISSING_CODE


def test_bucketize_marks_degenerate_edges_as_missing():
    """A constant prior window gives equal edges, which is not a real split."""
    panel = _dated([2.0] * 6 + [1.0, 5.0, 3.0, 4.0])

    codes, _ = bucketize(panel, ["ind"], n_buckets=3, min_history=3)

    # Rows 3-5 look back on a constant window -> both edges are 2.0 -> missing.
    assert (codes["ind"].iloc[3:6] == MISSING_CODE).all()


from backtesting.research.bucket_sweep import cell_stats, decode_cell


def _tiny_data():
    """Six rows, one horizon. Cell 0 = [1, 2, 3]; cell 1 = [-1, -2, 0.5]."""
    fwd = pd.Series([1.0, 2.0, 3.0, -1.0, -2.0, 0.5])
    return pd.DataFrame({"fwd_1": fwd, "hit_1": (fwd > 0).astype(float)})


def test_cell_stats_match_hand_computed_values():
    out = cell_stats(
        _tiny_data(), pd.Series([0, 0, 0, 1, 1, 1]), horizons=[1], min_samples=3
    ).set_index("cell")

    low = out.loc[0]
    assert low["n"] == 3
    assert low["mean"] == pytest.approx(2.0)
    assert low["median"] == pytest.approx(2.0)
    assert low["std"] == pytest.approx(1.0)          # sample std of 1, 2, 3
    assert low["hit_rate"] == pytest.approx(1.0)
    assert low["t_stat"] == pytest.approx(2.0 / (1.0 / np.sqrt(3)))

    mid = out.loc[1]
    assert mid["n"] == 3
    assert mid["mean"] == pytest.approx(-0.8333333, abs=1e-6)
    assert mid["median"] == pytest.approx(-1.0)
    assert mid["std"] == pytest.approx(1.2583057, abs=1e-6)
    assert mid["hit_rate"] == pytest.approx(1 / 3)


def test_cell_stats_drops_cells_below_min_samples():
    """A thin cell is absent from the output entirely, not present with NaN."""
    out = cell_stats(
        _tiny_data(), pd.Series([0, 0, 0, 0, 0, 1]), horizons=[1], min_samples=3
    )

    assert out["cell"].tolist() == [0]


def test_decode_cell_maps_joint_code_to_labels():
    # cell = b0 + 3*b1 + 9*b2 with b0=2 (high), b1=0 (low), b2=1 (mid)
    assert decode_cell(2 + 3 * 0 + 9 * 1, k=3, n_buckets=3) == "high|low|mid"
    assert decode_cell(0, k=1, n_buckets=3) == "low"
