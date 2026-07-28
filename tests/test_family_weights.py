from __future__ import annotations

import json

import numpy as np
import pandas as pd

from crudewatch.scoring import score as S
from crudewatch.scoring.blocks import fit_calibrator
from crudewatch.scoring.score import BlockScores, compute_opportunity, weights_for, WEIGHTS


def _row(**kw):
    base = dict(er_20=0.1, level_pct=0.9, slope_20=0.0, mom_decel_10=0.0, vol_ratio=1.0)
    base.update(kw)
    return pd.Series(base)


def _blocks(regime="range", level=60.0, direction=0.0):
    return BlockScores(
        regime=regime, trendiness=50.0, direction=direction, strength=40.0,
        level=level, p_reversion=0.7, p_continuation=0.6, confidence=50.0,
    )


def test_weights_for_defaults_to_equal_when_no_override():
    assert weights_for("no_such_family") is WEIGHTS


def test_weights_for_uses_override(monkeypatch):
    custom = {
        "range": {"rev_term": 0.7, "lvl_term": 0.1, "timing_term": 0.1, "vol_term": 0.1},
        "trend": {"dir_term": 0.25, "qual_term": 0.25, "cont_term": 0.25, "ext_low": 0.25},
        "transition_shrink": 0.4,
    }
    monkeypatch.setattr(S, "FAMILY_WEIGHTS", {"flies": custom})
    assert weights_for("flies")["range"]["rev_term"] == 0.7


def test_compute_opportunity_none_equals_explicit_default():
    b, r = _blocks(), _row()
    cal = fit_calibrator(pd.DataFrame({"er_20": [0.1, 0.2], "date": pd.to_datetime(["2020-01-01", "2020-01-02"])}), "flies")
    assert compute_opportunity(b, r, cal, None) == compute_opportunity(b, r, cal, WEIGHTS)


def test_load_family_weights_skips_malformed_entries(tmp_path, monkeypatch):
    path = tmp_path / "family_weights.json"
    path.write_text(json.dumps({
        "good": {
            "range": {"rev_term": 0.5, "lvl_term": 0.5, "timing_term": 0.0, "vol_term": 0.0},
            "trend": {"dir_term": 0.25, "qual_term": 0.25, "cont_term": 0.25, "ext_low": 0.25},
        },
        "bad": "skip-me",
        "incomplete": {"range": {"rev_term": 1.0}},
    }))
    monkeypatch.setattr(S, "_WEIGHTS_PATH", path)
    loaded = S._load_family_weights()
    assert loaded == {
        "good": {
            "range": {"rev_term": 0.5, "lvl_term": 0.5, "timing_term": 0.0, "vol_term": 0.0},
            "trend": {"dir_term": 0.25, "qual_term": 0.25, "cont_term": 0.25, "ext_low": 0.25},
            "transition_shrink": 0.4,
        },
    }


def test_load_family_weights_non_dict_top_level_returns_empty(tmp_path, monkeypatch):
    path = tmp_path / "family_weights.json"
    path.write_text(json.dumps(["not", "a", "dict"]))
    monkeypatch.setattr(S, "_WEIGHTS_PATH", path)
    assert S._load_family_weights() == {}
