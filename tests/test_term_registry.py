import numpy as np
import pandas as pd
import pytest

from crudewatch.scoring.blocks import fit_calibrator
from crudewatch.scoring.score import (
    RANGE_TERM_KEYS, TREND_TERM_KEYS, RANGE_TERMS, TREND_TERMS,
    compute_blocks, compute_opportunity,
)
from test_backtest import _synthetic_family


def _cal_and_rows(seed=0):
    df = _synthetic_family(seed)
    df["date"] = pd.to_datetime(df["date"])
    cal = fit_calibrator(df, "outrights", 25)
    sub = df[df["contract"] == "A"].sort_values("date").reset_index(drop=True)
    return df, cal, sub


def test_registry_lists_the_eight_current_terms():
    assert RANGE_TERM_KEYS[:4] == ("rev_term", "lvl_term", "timing_term", "vol_term")
    assert TREND_TERM_KEYS[:4] == ("dir_term", "qual_term", "cont_term", "ext_low")
    assert set(RANGE_TERMS) >= set(RANGE_TERM_KEYS[:4])
    assert set(TREND_TERMS) >= set(TREND_TERM_KEYS[:4])


def test_each_term_in_unit_interval():
    _, cal, sub = _cal_and_rows()
    for i, row in sub.iterrows():
        blocks = compute_blocks(row, cal, i + 1)
        for k in RANGE_TERM_KEYS:
            v = RANGE_TERMS[k](blocks, row, cal, blocks.level, blocks.direction)
            assert 0.0 <= v <= 1.0 or v != v
        for k in TREND_TERM_KEYS:
            v = TREND_TERMS[k](blocks, row, cal, blocks.level, blocks.direction)
            assert 0.0 <= v <= 1.0 or v != v


def test_compute_opportunity_equal_weight_golden():
    # Golden values for the live equal-weight term set on this seed.
    _, cal, sub = _cal_and_rows(seed=0)
    opp = [
        compute_opportunity(compute_blocks(r, cal, i + 1), r, cal)
        for i, r in sub.iterrows()
    ]
    expected = [
        -37.9888683259, -16.5804908547, 25.2279753458, -9.2992302114,
        -32.3119228511, 17.6960238382, -26.6857135587, -52.4026426214,
        38.5915387714, -37.1972528567, -16.3363742608, -28.170359336,
        -7.925483064, -21.5743044172, 11.5194969661, -16.8351928964,
        2.0522790883, 26.638551878, -25.2758532368, -16.0965095928,
        -43.4520103309, 6.8024163958, -11.2238244677, -37.7679473917,
        -11.3170657009, 35.2720503762, 13.6259213083, 30.7813052206,
        -32.3490914714, 14.6194702606, 30.4781410109, 42.3669251992,
        -35.8496332624, 32.9266855663, -37.751504331, -22.0225783557,
        -23.6856296661, -30.6682375659, 62.1991596176, 32.4482228946,
        -9.7086767706, 49.6298209274, 36.5861681554, -39.7145184953,
        5.7322928843, 14.4257251749, -15.8550861858, 29.1585634608,
        26.7400549286, -13.5466310896, 9.2142092251, 24.9906412732,
        -17.4747748553, 8.141181918, 31.4084578668, 9.7051723965,
        33.5824744072, 32.3589135046, -39.5793812126, 37.616139124,
    ]
    assert len(opp) == len(sub)
    assert np.allclose(opp, expected, atol=1e-9)
