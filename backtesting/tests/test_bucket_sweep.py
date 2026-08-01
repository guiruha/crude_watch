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
from backtesting.research.bucket_sweep import cell_stats, decode_cell


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
    # Independently computed literal (not the formula re-expressed): mean
    # -0.8333333, sample std 1.2583057, n=3 -> t_stat -1.1470787.
    assert mid["t_stat"] == pytest.approx(-1.1470787, abs=1e-6)


def test_cell_stats_drops_cells_below_min_samples():
    """A thin cell is absent from the output entirely, not present with NaN."""
    out = cell_stats(
        _tiny_data(), pd.Series([0, 0, 0, 0, 0, 1]), horizons=[1], min_samples=3
    )

    assert out["cell"].tolist() == [0]


def test_cell_stats_empty_and_nonempty_dtypes_match():
    """The empty-result fallback must type columns exactly like the non-empty
    path, so pd.concat([empty, nonempty]) does not upcast int64 columns
    (cell, horizon, n) to float64 -- which breaks decode_cell downstream.
    """
    empty = cell_stats(
        _tiny_data(), pd.Series([0, 0, 0, 1, 1, 1]), horizons=[1], min_samples=10
    )
    nonempty = cell_stats(
        _tiny_data(), pd.Series([0, 0, 0, 1, 1, 1]), horizons=[1], min_samples=3
    )

    assert len(empty) == 0
    assert len(nonempty) > 0
    pd.testing.assert_series_equal(empty.dtypes, nonempty.dtypes)


def test_decode_cell_maps_joint_code_to_labels():
    # cell = b0 + 3*b1 + 9*b2 with b0=2 (high), b1=0 (low), b2=1 (mid)
    assert decode_cell(2 + 3 * 0 + 9 * 1, k=3, n_buckets=3) == "high|low|mid"
    assert decode_cell(0, k=1, n_buckets=3) == "low"


from crudewatch.research.features import FEATURE_NAMES

from backtesting.research.bucket_sweep import (
    INDICATOR_THEME,
    THEMES,
    count_theme_combinations,
    theme_combinations,
)


def test_themes_partition_every_indicator_exactly_once():
    """Adding an indicator to FEATURES without a theme must fail here, loudly."""
    assigned = [name for names in THEMES.values() for name in names]

    assert len(assigned) == len(set(assigned)), "an indicator appears in two themes"
    assert set(assigned) == set(FEATURE_NAMES)
    assert len(THEMES) == 7
    assert INDICATOR_THEME["z_20"] == "level"
    assert INDICATOR_THEME["mom_10"] == "direction"
    assert INDICATOR_THEME["vol_ratio"] == "volatility"


def test_no_combination_mixes_two_indicators_of_one_theme():
    for combo in theme_combinations(FEATURE_NAMES, max_k=4):
        themes = [INDICATOR_THEME[name] for name in combo]
        assert len(themes) == len(set(themes)), f"{combo} repeats a theme"


def test_combination_count_matches_symmetric_polynomial():
    """Theme sizes [6, 6, 4, 3, 2, 2, 1] -> e1..e4 = 24, 235, 1212, 3544."""
    per_k = {k: 0 for k in range(1, 5)}
    for combo in theme_combinations(FEATURE_NAMES, max_k=4):
        per_k[len(combo)] += 1

    assert per_k == {1: 24, 2: 235, 3: 1212, 4: 3544}
    assert sum(per_k.values()) == 5015
    assert count_theme_combinations(FEATURE_NAMES, max_k=4) == 5015


def test_combinations_are_unordered_and_unique():
    small = ["a1", "a2", "b1", "c1"]
    theme_of = {"a1": "A", "a2": "A", "b1": "B", "c1": "C"}

    combos = list(theme_combinations(small, max_k=3, theme_of=theme_of))

    # (1 + 2x)(1 + x)(1 + x) = 1 + 4x + 5x^2 + 2x^3 -> e1=4, e2=5, e3=2
    assert len(combos) == 11
    assert len(set(combos)) == len(combos)
    assert ("a1", "a2") not in set(combos)      # same theme, never emitted
    assert ("a1", "b1") in set(combos)
    assert ("b1", "a1") not in set(combos)      # unordered: only one orientation
    assert count_theme_combinations(small, max_k=3, theme_of=theme_of) == 11


from backtesting.research.bucket_sweep import RESULT_COLUMNS, sweep

_SMALL_THEMES = {"a1": "A", "a2": "A", "b1": "B", "c1": "C"}


def test_sweep_covers_every_cross_theme_combination_exactly_once():
    """Theme sizes [2, 1, 1] at max_k=3 -> 4 + 5 + 2 = 11 combinations."""
    rng = np.random.default_rng(0)
    names = list(_SMALL_THEMES)
    codes = pd.DataFrame({n: np.zeros(50, dtype=np.int8) for n in names})
    forwards = pd.DataFrame({"fwd_1": rng.normal(size=50)})

    out = sweep(codes, forwards, max_k=3, min_samples=1, n_buckets=3,
                theme_of=_SMALL_THEMES)

    assert len(out) == 11        # all codes are 0 -> one cell per combination
    assert out["indicators"].nunique() == 11
    assert not out["indicators"].duplicated().any()
    emitted = set(out["indicators"])
    assert "a1|a2" not in emitted   # same theme, never paired
    assert "a1|b1" in emitted
    assert "b1|a1" not in emitted   # unordered: one orientation only


def test_sweep_labels_themes_alongside_indicators():
    codes = pd.DataFrame({"a1": np.zeros(6, dtype=np.int8),
                          "b1": np.zeros(6, dtype=np.int8)})
    forwards = pd.DataFrame({"fwd_1": [1.0, 2.0, 3.0, -1.0, -2.0, 0.5]})

    out = sweep(codes, forwards, max_k=2, min_samples=3, n_buckets=3,
                theme_of=_SMALL_THEMES)

    pair = out[out["indicators"] == "a1|b1"]
    assert pair["themes"].iloc[0] == "A|B"


def test_sweep_returns_expected_columns_and_k():
    codes = pd.DataFrame({"a1": np.array([0, 0, 0, 1, 1, 1], dtype=np.int8),
                          "b1": np.zeros(6, dtype=np.int8)})
    forwards = pd.DataFrame({"fwd_1": [1.0, 2.0, 3.0, -1.0, -2.0, 0.5]})

    out = sweep(codes, forwards, max_k=2, min_samples=3, n_buckets=3,
                theme_of=_SMALL_THEMES)

    assert list(out.columns) == list(RESULT_COLUMNS)
    assert set(out["k"]) == {1, 2}
    single = out[(out["indicators"] == "a1") & (out["buckets"] == "low")]
    assert single["mean"].iloc[0] == pytest.approx(2.0)


def test_sweep_reports_progress():
    codes = pd.DataFrame({"a1": np.zeros(10, dtype=np.int8),
                          "b1": np.zeros(10, dtype=np.int8)})
    forwards = pd.DataFrame({"fwd_1": np.arange(10, dtype=float)})
    seen: list[tuple[int, int]] = []

    sweep(codes, forwards, max_k=2, min_samples=1, n_buckets=3,
          theme_of=_SMALL_THEMES,
          progress=lambda done, total: seen.append((done, total)))

    assert seen == [(1, 3), (2, 3), (3, 3)]  # a1, b1, then (a1, b1)


def test_sweep_reports_progress_even_when_no_combination_survives():
    """Progress must still advance once per combination when every combination
    yields zero surviving cells, otherwise a long run's progress bar stalls.
    """
    codes = pd.DataFrame({"a1": np.zeros(10, dtype=np.int8),
                          "b1": np.zeros(10, dtype=np.int8)})
    forwards = pd.DataFrame({"fwd_1": np.arange(10, dtype=float)})
    seen: list[tuple[int, int]] = []

    out = sweep(codes, forwards, max_k=2, min_samples=1000, n_buckets=3,
                theme_of=_SMALL_THEMES,
                progress=lambda done, total: seen.append((done, total)))

    assert out.empty
    assert seen == [(1, 3), (2, 3), (3, 3)]  # a1, b1, then (a1, b1)


def test_sweep_excludes_missing_code_rows_instead_of_fabricating_a_high_bucket():
    """-1 % 3 == 2 in Python, so a naive mixed-radix decode would label
    MISSING_CODE rows "high" and pool them into a real cell. `sweep` must
    drop any row where the current combination's indicator is MISSING_CODE
    before grouping, regardless of what run_family does upstream.
    """
    codes = pd.DataFrame({
        "a1": np.array([MISSING_CODE, MISSING_CODE, 0, 0, 0, 0], dtype=np.int8),
        "b1": np.zeros(6, dtype=np.int8),
    })
    forwards = pd.DataFrame({"fwd_1": [1.0, 2.0, 3.0, -1.0, -2.0, 0.5]})

    out = sweep(codes, forwards, max_k=1, min_samples=1, n_buckets=3,
                theme_of=_SMALL_THEMES)

    a1 = out[out["indicators"] == "a1"]
    assert (a1["n"] == 4).all()          # only the 4 non-missing rows
    assert set(a1["buckets"]) == {"low"}  # never a fabricated "high"


def test_sweep_empty_and_nonempty_dtypes_match():
    """The empty-result fallback must type columns exactly like the non-empty
    path (int64 k/horizon/n, object themes/indicators/buckets, float64 for the
    stats), so pd.concat-ing per-family results does not upcast int columns
    to float64 for the whole pooled table.
    """
    codes = pd.DataFrame({"a1": np.array([0, 0, 0, 1, 1, 1], dtype=np.int8),
                          "b1": np.zeros(6, dtype=np.int8)})
    forwards = pd.DataFrame({"fwd_1": [1.0, 2.0, 3.0, -1.0, -2.0, 0.5]})

    empty = sweep(codes, forwards, max_k=2, min_samples=1000, n_buckets=3,
                  theme_of=_SMALL_THEMES)
    nonempty = sweep(codes, forwards, max_k=2, min_samples=3, n_buckets=3,
                     theme_of=_SMALL_THEMES)

    assert len(empty) == 0
    assert len(nonempty) > 0
    pd.testing.assert_series_equal(empty.dtypes, nonempty.dtypes)


from backtesting.research.bucket_sweep import HORIZONS, InsufficientData, run_family


def _synthetic_frame(n_contracts: int = 6, n_rows: int = 400) -> pd.DataFrame:
    """A price panel shaped like a family frame: date, contract, open, close."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2015-01-01", periods=n_rows)
    parts = []
    for c in range(n_contracts):
        close = 50.0 + np.cumsum(rng.normal(0, 0.5, size=n_rows))
        parts.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "contract": f"C{c}",
                    "open": close + rng.normal(0, 0.05, size=n_rows),
                    "close": close,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_horizons_match_the_spec():
    assert HORIZONS == (1, 2, 3, 5, 10, 15, 20)


def test_run_family_produces_labelled_results():
    results, cutoffs = run_family(
        "synthetic", _synthetic_frame(),
        max_k=2, min_samples=5, n_buckets=3, min_history=200,
    )

    assert not results.empty
    assert results.columns[0] == "family"
    assert set(results["family"]) == {"synthetic"}
    assert set(results["horizon"]) <= set(HORIZONS)
    assert set(results["k"]) == {1, 2}
    assert (results["n"] >= 5).all()
    assert cutoffs.columns[0] == "family"
    # Every k=1 combination names exactly one indicator.
    singles = results[results["k"] == 1]
    assert not singles["indicators"].str.contains(r"\|").any()
    # No MISSING_CODE ever reaches a bucket label.
    assert set(results["buckets"].str.split("|").explode()) <= set(BUCKET_LABELS)
    # Themes are real and never repeat inside a combination.
    assert set(results["themes"].str.split("|").explode()) <= set(THEMES)
    repeated = results["themes"].str.split("|").apply(lambda t: len(t) != len(set(t)))
    assert not repeated.any()


def test_run_family_raises_on_a_thin_panel():
    with pytest.raises(InsufficientData):
        run_family("thin", _synthetic_frame(n_contracts=1, n_rows=130),
                   max_k=4, min_samples=30, n_buckets=3, min_history=2000)


def test_run_family_raises_when_codeable_rows_are_too_few_for_the_cell_grid():
    """The SECOND InsufficientData guard: enough rows to clear min_history and
    get coded, but too few surviving rows to satisfy
    ``min_samples * n_buckets**max_k``. Distinct from the thin-panel test
    above, which trips the FIRST guard (fewer than min_history rows at all).
    """
    # min_history=300 is comfortably cleared (~601 feature-complete rows out
    # of 700), so the FIRST guard passes; but only ~300 rows are left
    # codeable after the min_history warmup, well short of
    # min_samples=30 * 3**4 = 2430, so the SECOND guard trips.
    with pytest.raises(InsufficientData, match="codeable rows after warmup"):
        run_family("thin_cells", _synthetic_frame(n_contracts=1, n_rows=700),
                   max_k=4, min_samples=30, n_buckets=3, min_history=300)


def _staggered_frame(n_contracts: int = 6, n_rows: int = 400,
                     short_tail_at: int = 300, extra_rows: int = 0) -> pd.DataFrame:
    """Like `_synthetic_frame`, but contract C0 ends at ``short_tail_at +
    extra_rows`` while every OTHER contract always runs the full ``n_rows``.
    This is the seam `_synthetic_frame` hides: with identical date ranges for
    every contract, a leak that depends on ONE contract's end date shifting
    (which changes whether ITS OWN near-tail rows have a valid forward
    column, and therefore whether they enter the pooled quantile pool used
    for OTHER contracts' same-date cutoffs) can't show up.
    """
    dates = pd.bdate_range("2015-01-01", periods=n_rows)
    parts = []
    for c in range(n_contracts):
        # A per-contract RNG stream, independent of every other contract's
        # length, so extending C0's tail cannot desync C1..Cn's draws --
        # otherwise a difference in "later" contract data would be a test
        # artifact, not the real thing Finding 1 is about.
        rng = np.random.default_rng(1000 + c)
        length = short_tail_at + extra_rows if c == 0 else n_rows
        close = 50.0 + np.cumsum(rng.normal(0, 0.5, size=length))
        parts.append(
            pd.DataFrame(
                {
                    "date": dates[:length],
                    "contract": f"C{c}",
                    "open": close + rng.normal(0, 0.05, size=length),
                    "close": close,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_run_family_extending_one_contracts_tail_does_not_change_earlier_buckets():
    """Look-ahead regression guard with staggered contract end dates.

    C0 originally ends at bar 300; every other contract runs the full 400
    bars throughout. Extending ONLY C0's own tail (so its near-300 rows gain
    a valid forward column they didn't have before) must not change any
    cutoff or bucket-derived result dated at or before bar 300 for the OTHER
    contracts -- exactly the leak Finding 1 describes: dropping forward-NaN
    rows before bucketizing lets a date's quantile pool depend on whether a
    DIFFERENT contract's row (dated even earlier) happened to have data far
    enough in the future to keep a valid forward column.
    """
    base = _staggered_frame(short_tail_at=300, n_rows=400, extra_rows=0)
    extended = _staggered_frame(short_tail_at=300, n_rows=400, extra_rows=25)

    _, cutoffs_base = run_family(
        "stagger", base, max_k=1, min_samples=5, n_buckets=3, min_history=150,
    )
    _, cutoffs_ext = run_family(
        "stagger", extended, max_k=1, min_samples=5, n_buckets=3, min_history=150,
    )

    cutoff_date_max = base[base["contract"] == "C0"]["date"].max()
    cuts_base = cutoffs_base[cutoffs_base["date"] <= cutoff_date_max].reset_index(drop=True)
    cuts_ext = cutoffs_ext[cutoffs_ext["date"] <= cutoff_date_max].reset_index(drop=True)
    pd.testing.assert_frame_equal(cuts_base, cuts_ext)


def test_run_family_stats_are_paired_with_the_right_rows():
    """A regression guard against silent bucket/outcome misalignment.

    If a row's bucket code were ever paired with a *different* row's forward
    outcome -- a stray sort, reset_index, or reindex anywhere in the chain --
    every downstream statistic would still look structurally valid (real
    labels, plausible n, sensible themes) while being quietly wrong. Structural
    checks like ``test_run_family_produces_labelled_results`` cannot see this.

    This test recomputes one k=1 cell's ``n`` and ``mean`` completely
    independently: rebuild the feature/outcome panel by hand, use the
    `cutoffs` frame `run_family` returned to derive that indicator's per-date
    edges (exactly as `bucketize` documents: a row's code is the count of its
    own date's edges its value is `>=`, edges coming from strictly prior
    dates), select the rows landing in the reported bucket, and average their
    `fwd_h` directly -- with no dependency on `bucketize`'s internal row order.
    """
    from crudewatch.research.features import FEATURE_NAMES, add_features
    from crudewatch.research.targets import add_forward_returns

    frame = _synthetic_frame()
    results, cutoffs = run_family(
        "synthetic", frame, max_k=1, min_samples=5, n_buckets=3, min_history=200,
    )

    singles = results[results["k"] == 1]
    assert not singles.empty
    row = singles.iloc[0]
    indicator = str(row["indicators"])
    bucket = str(row["buckets"])
    horizon = int(row["horizon"])

    # Rebuild the exact complete-case panel run_family builds internally.
    featured = add_features(frame)
    full = add_forward_returns(featured, horizons=HORIZONS)
    fwd_cols = [f"fwd_{h}" for h in HORIZONS]
    panel = full.dropna(subset=list(FEATURE_NAMES) + fwd_cols)

    # This indicator's per-date edges, taken from run_family's own cutoffs
    # output -- family, date, indicator, edge_index, value.
    ind_cutoffs = cutoffs[cutoffs["indicator"] == indicator]
    edges = ind_cutoffs.pivot(index="date", columns="edge_index", values="value")
    edges.columns = [f"edge_{j}" for j in edges.columns]
    edge_cols = sorted(edges.columns, key=lambda c: int(c.split("_")[1]))

    merged = panel.merge(edges, left_on="date", right_index=True, how="left")

    usable = merged[indicator].notna()
    for c in edge_cols:
        usable &= merged[c].notna()
    for prev, cur in zip(edge_cols, edge_cols[1:]):
        usable &= merged[cur] > merged[prev]  # strictly increasing, else no real split

    code = pd.Series(0, index=merged.index, dtype=int)
    for c in edge_cols:
        code = code + (merged[indicator] >= merged[c]).astype(int)

    bucket_index = list(BUCKET_LABELS).index(bucket)
    in_cell = usable & (code == bucket_index)

    recomputed = merged.loc[in_cell, f"fwd_{horizon}"]

    assert len(recomputed) == int(row["n"])
    assert recomputed.mean() == pytest.approx(float(row["mean"]))
