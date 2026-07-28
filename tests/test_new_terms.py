import numpy as np
import pandas as pd

from crudewatch.scoring.blocks import ECDF_FEATURES, fit_calibrator
from crudewatch.scoring.score import (
    RANGE_TERM_KEYS, TREND_TERM_KEYS, RANGE_TERMS, TREND_TERMS, compute_blocks,
)
from test_backtest import _synthetic_family


def _setup(seed=0):
    df = _synthetic_family(seed)
    df["date"] = pd.to_datetime(df["date"])
    cal = fit_calibrator(df, "outrights", 25)
    sub = df[df["contract"] == "A"].sort_values("date").reset_index(drop=True)
    return df, cal, sub


def _setup_with_injected(seed=0):
    """Copy of _synthetic_family with locally injected ECDF columns (fixture untouched)."""
    df = _synthetic_family(seed).copy()
    df["date"] = pd.to_datetime(df["date"])
    n = len(df)
    df["macd_div"] = np.linspace(-10.0, 10.0, n)
    df["er_drop_20"] = np.linspace(-5.0, 5.0, n)
    cal = fit_calibrator(df, "outrights", 25)
    sub = df[df["contract"] == "A"].sort_values("date").reset_index(drop=True)
    return df, cal, sub


def _macd_div_extremes(cal):
    ecdf = cal.ecdf["macd_div"]
    return float(ecdf[0]), float(ecdf[-1])


def _er_drop_extremes(cal):
    ecdf = cal.ecdf["neg_er_drop_20"]
    return float(-ecdf[-1]), float(-ecdf[0])


def _autocorr_extremes(cal):
    ecdf = cal.ecdf["autocorr_20"]
    return float(ecdf[0]), float(ecdf[-1])


def _ema_align_extremes(cal):
    ecdf = cal.ecdf["ema_align"]
    return float(ecdf[0]), float(ecdf[-1])


def test_ecdf_features_extended():
    for feat in ("macd_div", "rsi_div_14"):
        assert feat in ECDF_FEATURES


def test_new_keys_present_in_canonical_order():
    assert RANGE_TERM_KEYS == (
        "rev_term", "lvl_term", "timing_term", "vol_term",
        "macd_div_term", "rsi_div_term", "mom_decel_term", "er_drop_term", "autocorr_term",
    )
    assert TREND_TERM_KEYS == (
        "dir_term", "qual_term", "cont_term", "ext_low",
        "ema_align_term", "mom10_term", "r2_term", "dirpers_term",
    )


def test_all_terms_unit_interval_and_nan_safe():
    _, cal, sub = _setup()
    for i, row in sub.iterrows():
        b = compute_blocks(row, cal, i + 1)
        for k in RANGE_TERM_KEYS:
            v = RANGE_TERMS[k](b, row, cal, b.level, b.direction)
            assert 0.0 <= v <= 1.0
        for k in TREND_TERM_KEYS:
            v = TREND_TERMS[k](b, row, cal, b.level, b.direction)
            assert 0.0 <= v <= 1.0


def test_macd_div_range_directional_gating():
    _, cal, sub = _setup_with_injected(0)
    b = compute_blocks(sub.iloc[10], cal, 11)
    row = sub.iloc[10].copy()
    bearish, bullish = _macd_div_extremes(cal)

    row["macd_div"] = bullish
    up_bull = RANGE_TERMS["macd_div_term"](b, row, cal, -50.0, b.direction)
    dn_bull = RANGE_TERMS["macd_div_term"](b, row, cal, +50.0, b.direction)
    assert up_bull > 0.5
    assert dn_bull == 0.0

    row["macd_div"] = bearish
    up_bear = RANGE_TERMS["macd_div_term"](b, row, cal, -50.0, b.direction)
    dn_bear = RANGE_TERMS["macd_div_term"](b, row, cal, +50.0, b.direction)
    assert up_bear == 0.0
    assert dn_bear > 0.5


def test_ema_align_trend_directional_gating():
    _, cal, sub = _setup(0)
    b = compute_blocks(sub.iloc[10], cal, 11)
    row = sub.iloc[10].copy()
    bearish, bullish = _ema_align_extremes(cal)

    row["ema_align"] = bullish
    up_bull = TREND_TERMS["ema_align_term"](b, row, cal, b.level, +50.0)
    dn_bull = TREND_TERMS["ema_align_term"](b, row, cal, b.level, -50.0)
    assert up_bull > 0.5
    assert dn_bull == 0.0

    row["ema_align"] = bearish
    up_bear = TREND_TERMS["ema_align_term"](b, row, cal, b.level, +50.0)
    dn_bear = TREND_TERMS["ema_align_term"](b, row, cal, b.level, -50.0)
    assert up_bear == 0.0
    assert dn_bear > 0.5


def test_er_drop_term_polarity():
    _, cal, sub = _setup_with_injected(0)
    b = compute_blocks(sub.iloc[10], cal, 11)
    neg_val, pos_val = _er_drop_extremes(cal)

    row_neg = sub.iloc[10].copy()
    row_neg["er_drop_20"] = neg_val
    row_pos = sub.iloc[10].copy()
    row_pos["er_drop_20"] = pos_val

    term_neg = RANGE_TERMS["er_drop_term"](b, row_neg, cal, b.level, b.direction)
    term_pos = RANGE_TERMS["er_drop_term"](b, row_pos, cal, b.level, b.direction)
    assert term_neg > term_pos
    assert term_neg > 0.5
    assert term_pos < 0.5


def test_autocorr_term_polarity():
    _, cal, sub = _setup(0)
    b = compute_blocks(sub.iloc[10], cal, 11)
    low, high = _autocorr_extremes(cal)

    row_low = sub.iloc[10].copy()
    row_low["autocorr_20"] = low
    row_high = sub.iloc[10].copy()
    row_high["autocorr_20"] = high

    term_low = RANGE_TERMS["autocorr_term"](b, row_low, cal, b.level, b.direction)
    term_high = RANGE_TERMS["autocorr_term"](b, row_high, cal, b.level, b.direction)
    assert term_low > term_high
    assert term_low > 0.5
    assert term_high < 0.5


def test_nan_term_is_zero():
    _, cal, sub = _setup()
    b = compute_blocks(sub.iloc[5], cal, 6)
    row = sub.iloc[5].copy()
    row["r2_20"] = np.nan
    assert TREND_TERMS["r2_term"](b, row, cal, b.level, b.direction) == 0.0

    _, cal_inj, sub_inj = _setup_with_injected(0)
    b_inj = compute_blocks(sub_inj.iloc[5], cal_inj, 6)
    row_macd = sub_inj.iloc[5].copy()
    row_macd["macd_div"] = np.nan
    assert RANGE_TERMS["macd_div_term"](b_inj, row_macd, cal_inj, -50.0, b_inj.direction) == 0.0

    row_er = sub_inj.iloc[5].copy()
    row_er["er_drop_20"] = np.nan
    assert RANGE_TERMS["er_drop_term"](b_inj, row_er, cal_inj, b_inj.level, b_inj.direction) == 0.0
