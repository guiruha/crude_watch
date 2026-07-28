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


def test_compute_opportunity_unchanged_default_weights():
    # Golden values captured from the pre-refactor implementation on this seed.
    _, cal, sub = _cal_and_rows(seed=0)
    opp = [
        compute_opportunity(compute_blocks(r, cal, i + 1), r, cal)
        for i, r in sub.iterrows()
    ]
    expected = [
        -48.6860699852, -9.202648376, 32.1226173582, -14.3484604228,
        -40.0405123688, 25.3920476765, -41.7047604508, -38.5629828636,
        45.0997442094, -45.2278390467, -18.5894151882, -35.0373457727,
        -8.4342994613, -29.0536713537, 12.5389939322, -23.5870524594,
        1.8545581765, 37.3426254908, -31.7082649013, -20.1096858522,
        -34.7278418577, 3.2714994582, -15.2809822688, -30.0387451931,
        -12.4674647352, 50.8351776335, 17.4185092834, 31.9289378527,
        -42.8231829428, 17.9056071879, 36.4509635527, 40.9986533608,
        -41.2825998582, 33.7806715486, -32.8477469269, -24.5478525943,
        -28.4129259988, -30.8276802433, 42.0240616893, 32.6490606626,
        -11.584020208, 52.3846418547, 38.6092772124, -42.9326816227,
        3.1312524353, 14.6014503498, -16.543505705, 32.8013376493,
        32.4384431904, -14.9265955126, 10.5117517836, 38.7312825464,
        -17.6162163773, 12.9490305027, 38.3158660922, 9.4103447929,
        35.1862576161, 41.3844936759, -42.5925004151, 53.982278248,
    ]
    assert len(opp) == len(sub)
    assert np.allclose(opp, expected, atol=1e-9)
