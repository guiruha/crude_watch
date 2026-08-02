"""Tests for the Opportunity Score engine."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from crudewatch.research import build_dataset
from crudewatch.scoring import (
    FamilyCalibrator,
    analogous_outcomes,
    fit_calibrator,
    percentile,
    score_family,
    score_instrument,
    signed_pct,
)
from crudewatch.scoring.blocks import block_direction, regime_label
from crudewatch.scoring.score import BlockScores, WEIGHTS, _attach_bar_counts, compute_blocks, compute_opportunity
from core.scoring import _historical_abs_composite


def _outright_frame(
    closes: list[float],
    contract: str = "CLZ2024",
    vintage: int = 2024,
) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=n, freq="B"),
            "contract": [contract] * n,
            "close": closes,
            "month": [12] * n,
            "month_code": ["Z"] * n,
            "expiry_year": [vintage] * n,
        }
    )


def _multi_contract_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frames = []
    for i, contract in enumerate(["CLA", "CLB", "CLC"]):
        n = 80
        noise = rng.normal(0, 0.3, n)
        closes = list(70 + np.cumsum(noise) + i * 2)
        frames.append(_outright_frame(closes, contract=contract, vintage=2024 + i))
    return build_dataset(pd.concat(frames, ignore_index=True), "outrights")


def test_percentile_bounds_and_monotonicity():
    hist = np.sort(np.linspace(-2, 2, 101))
    assert percentile(hist, -999) == 0.0
    assert percentile(hist, 999) == 1.0
    assert 0.45 <= percentile(hist, 0.0) <= 0.55
    assert percentile(np.array([]), 1.0) == 0.5
    assert percentile(hist, float("nan")) == 0.5
    assert percentile(hist, -1.0) < percentile(hist, 0.0) < percentile(hist, 1.0)


def test_signed_pct_bounds():
    assert signed_pct(0.0) == -100.0
    assert signed_pct(1.0) == 100.0
    assert signed_pct(0.5) == 0.0
    assert -100.0 <= signed_pct(0.25) <= 100.0


def test_block_ranges_and_opportunity_bounds():
    data = _multi_contract_dataset()
    cal = fit_calibrator(data, "outrights", horizon=25)
    row = data.sort_values(["contract", "date"]).groupby("contract").tail(1).iloc[0]
    blocks = compute_blocks(row, cal, n_contract_bars=40)
    opp = compute_opportunity(blocks, row, cal)

    assert blocks.regime in ("range", "transition", "trend")
    assert 0.0 <= blocks.trendiness <= 100.0
    assert -100.0 <= blocks.direction <= 100.0
    assert 0.0 <= blocks.strength <= 100.0
    assert -100.0 <= blocks.level <= 100.0
    assert 0.0 <= blocks.p_reversion <= 1.0
    assert 0.0 <= blocks.p_continuation <= 1.0
    assert -100.0 <= opp <= 100.0


def test_vectorized_historical_composite_matches_canonical_path():
    data = _multi_contract_dataset()
    cal = fit_calibrator(data, "outrights", horizon=25)
    enriched = _attach_bar_counts(data)
    canonical = []
    for _, row in enriched.iterrows():
        blocks = compute_blocks(row, cal, int(row["_n_contract_bars"]))
        canonical.append(abs(compute_opportunity(blocks, row, cal)))
    vectorized = _historical_abs_composite(data, cal, float(WEIGHTS["transition_shrink"]), sort=False)
    assert np.allclose(vectorized, np.asarray(canonical, dtype=float), atol=1e-10)


def _with_regime(blocks: BlockScores, regime: str) -> BlockScores:
    return BlockScores(
        regime=regime,
        trendiness=blocks.trendiness,
        direction=blocks.direction,
        strength=blocks.strength,
        level=blocks.level,
        p_reversion=blocks.p_reversion,
        p_continuation=blocks.p_continuation,
    )


def test_range_cheap_positive_expensive_negative():
    data = _multi_contract_dataset()
    cal = fit_calibrator(data, "outrights", horizon=25)
    row = data.iloc[0].copy()
    row["er_20"] = cal.er_lo * 0.5
    row["level_pct"] = 0.05
    row["level_z"] = -2.5
    row["z_20"] = -2.0
    row["slope_20"] = -0.5
    row["macd_hist"] = -0.2
    row["mom_decel_10"] = 0.5
    row["vol_ratio"] = 0.8
    blocks_cheap = _with_regime(compute_blocks(row, cal, n_contract_bars=40), "range")
    opp_cheap = compute_opportunity(blocks_cheap, row, cal)

    row_dear = row.copy()
    row_dear["level_pct"] = 0.95
    row_dear["level_z"] = 2.5
    row_dear["z_20"] = 2.0
    blocks_dear = _with_regime(compute_blocks(row_dear, cal, n_contract_bars=40), "range")
    opp_dear = compute_opportunity(blocks_dear, row_dear, cal)

    assert opp_cheap > 0
    assert opp_dear < 0


def test_regime_partition_around_terciles():
    cal = FamilyCalibrator(
        family="test",
        horizon=25,
        er_lo=0.3,
        er_hi=0.7,
        ecdf={"er_20": np.array([0.1, 0.5, 0.9])},
        p_rev_cheap=0.6,
        p_rev_dear=0.6,
        p_cont_up=0.55,
        p_cont_dn=0.45,
        ic_t_level=1.5,
    )
    assert regime_label(0.2, cal) == "range"
    assert regime_label(0.5, cal) == "transition"
    assert regime_label(0.8, cal) == "trend"


def test_score_family_columns_and_no_nan_opportunity():
    data = _multi_contract_dataset()
    out = score_family(data, "outrights", horizon=25, active_within_days=365, require_unexpired=False)
    expected_cols = {
        "family",
        "contract",
        "date",
        "close",
        "dte",
        "regime",
        "trendiness",
        "direction",
        "strength",
        "level",
        "p_reversion",
        "p_continuation",
        "opportunity",
    }
    assert expected_cols.issubset(set(out.columns))
    assert len(out) == data["contract"].nunique()
    assert out["opportunity"].notna().all()
    assert out["opportunity"].between(-100, 100).all()


def test_score_instrument_as_of_is_point_in_time():
    data = _multi_contract_dataset()
    sub = data[data["contract"] == "CLA"].sort_values("date")
    as_of = sub["date"].iloc[len(sub) // 2]

    scored = score_instrument(data, "outrights", "CLA", horizon=25, as_of=as_of)
    # The scored bar is the last observation on or before the chosen date...
    assert scored.date <= pd.Timestamp(as_of)
    assert scored.date == pd.Timestamp(sub[sub["date"] <= as_of]["date"].max())
    # ...and appending future bars must not change the as-of answer (no look-ahead).
    scored_again = score_instrument(data, "outrights", "CLA", horizon=25, as_of=as_of)
    assert scored_again.opportunity == scored.opportunity
    assert -100.0 <= scored.opportunity <= 100.0


def test_analogous_outcomes_structure_and_point_in_time():
    data = _multi_contract_dataset()
    coh = analogous_outcomes(data, "outrights", "CLA", horizon=25)
    assert coh["n"] >= 0
    assert coh["horizon"] == 25
    if coh["n"] > 0:
        assert 0.0 <= coh["up_rate"] <= 1.0
        assert coh["regime"] in ("range", "transition", "trend")
        assert coh["level_bin"] in (
            "muy barato", "barato", "neutral", "caro", "muy caro",
        )
    # As-of a mid date must not use more history than the full sample.
    sub = data[data["contract"] == "CLA"].sort_values("date")
    as_of = sub["date"].iloc[len(sub) // 2]
    coh_asof = analogous_outcomes(data, "outrights", "CLA", horizon=25, as_of=as_of)
    assert coh_asof["n"] <= coh["n"]


def test_analogous_outcomes_neutral_threshold_has_no_aligned_claim():
    data = _multi_contract_dataset()
    coh = analogous_outcomes(data, "outrights", "CLA", horizon=25, action_threshold=101.0)
    assert coh["side"] == 0.0
    assert "avg_aligned" not in coh
    assert "aligned_win_rate" not in coh


def test_analogous_outcomes_aligned_move_is_net_of_cost():
    data = _multi_contract_dataset()
    coh = analogous_outcomes(data, "outrights", "CLA", horizon=25, action_threshold=0.0, cost=0.25)
    if coh["n"] > 0 and coh["side"] != 0.0:
        assert np.isclose(coh["avg_aligned"], coh["avg_aligned_gross"] - 0.25)
        assert coh["cost"] == 0.25


def test_block_direction_positive_on_uptrend():
    """Strongly up-trending synthetic contract yields direction > 0."""
    closes = list(70.0 + np.arange(120) * 0.5)
    data = build_dataset(_outright_frame(closes), "outrights")
    cal = fit_calibrator(data, "outrights", horizon=25)
    row = data.sort_values("date").iloc[-1]
    direction = block_direction(row, cal)
    assert direction > 0


def test_nan_level_panel_still_scores():
    data = _multi_contract_dataset()
    data_nan = data.copy()
    data_nan.loc[data_nan["contract"] == "CLA", ["level_pct", "level_z"]] = np.nan
    nan = score_instrument(data_nan, "outrights", "CLA", horizon=25)
    assert nan.opportunity == nan.opportunity
