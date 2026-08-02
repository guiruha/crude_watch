from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from screens.component import COMPONENTS, FEATURE_CTX, FEATURE_ICON, _reading
from core.indicator_buckets import (
    indicator_bucket_combinations_cached,
    indicator_bucket_outcomes_cached,
)


def test_all_reversion_indicators_are_visible_in_dashboard():
    visible = {name for name, _ in COMPONENTS["p_reversion"]["features"]}
    expected = {
        "rsi_2",
        "rsi_14",
        "pctb_20_2",
        "pctb_10_1_5",
        "rsi_div_14",
        "macd_div",
        "mom_decel_10",
        "er_drop_20",
        "vol_ratio",
    }
    assert expected <= visible


def test_visible_reversion_indicators_have_context_and_icons():
    for name, _ in COMPONENTS["p_reversion"]["features"]:
        assert name in FEATURE_CTX
        assert name in FEATURE_ICON


def test_regime_indicator_readings_use_distinct_vocabulary():
    assert _reading("er_20", 0.5399) == "Recorrido eficiente"
    assert _reading("variance_ratio_5", 1.1201) == "Persistencia leve"
    assert _reading("autocorr_20", -0.2720) == "Reversión de corto plazo"


def test_pm_bucket_outcomes_use_positional_indices_after_date_filter():
    buckets = indicator_bucket_outcomes_cached("outrights", "CLU2026", "2026-07-02", (25,))
    combos = indicator_bucket_combinations_cached(
        "outrights",
        "CLU2026",
        "2026-07-02",
        (25,),
        focus_horizon=25,
        max_evolution_pairs=4,
        combo_size=2,
    )

    assert not buckets.empty
    assert not combos.empty
