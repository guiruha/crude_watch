"""Tests for robustness diagnostics: redundancy (WS6) + subgroups (WS7)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.research.diagnostics import (
    cluster_features,
    feature_correlation,
    incremental_ic,
    redundancy_report,
    subgroup_report,
)

RNG = np.random.default_rng(42)


def _base_frame(n: int = 400) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2016-01-01", periods=n, freq="B"),
        "contract": ["A"] * n,
        "close": np.cumsum(RNG.normal(0, 1, n)) + 100.0,
    })


# --------------------------------------------------------------------------- #
# WS6 - redundancy
# --------------------------------------------------------------------------- #
def test_identical_features_cluster_and_are_redundant():
    n = 400
    x = RNG.normal(0, 1, n)
    data = _base_frame(n)
    data["a"] = x
    data["b"] = x  # exact copy of a
    data["c"] = RNG.normal(0, 1, n)  # independent
    data["fwd_10"] = x + RNG.normal(0, 0.5, n)  # signal driven by a/b

    corr = feature_correlation(data, ["a", "b", "c"])
    assert corr.loc["a", "b"] > 0.99
    assert abs(corr.loc["a", "c"]) < 0.3

    clusters = cluster_features(corr, threshold=0.8)
    ab = next(cl for cl in clusters if "a" in cl)
    assert set(ab) == {"a", "b"}  # a & b together, c apart
    assert ["c"] in clusters

    # 'a' is the representative (stronger |t|); 'b' is a perfect copy -> redundant.
    strength = {"a": 5.0, "b": 4.9, "c": 4.0}
    rep = redundancy_report(data, "fam", ["a", "b", "c"], 10, strength, threshold=0.8)
    b_row = rep.set_index("feature").loc["b"]
    assert b_row["representative"] == "a"
    assert abs(b_row["ic_incremental"]) < 1e-6
    assert b_row["verdict"] == "redundante"
    assert rep.set_index("feature").loc["a", "verdict"] == "representante"


def test_incremental_ic_recovers_independent_signal():
    n = 500
    rep = RNG.normal(0, 1, n)
    extra = RNG.normal(0, 1, n)  # independent component
    data = _base_frame(n)
    data["rep"] = rep
    data["feat"] = rep + extra  # correlated with rep but carries 'extra'
    data["fwd_10"] = extra  # target driven ONLY by the independent part

    inc = incremental_ic(data, "feat", "rep", "fwd_10")
    # Residual of feat on rep is ~extra, which drives the target -> strong IC.
    assert abs(inc) > 0.3


# --------------------------------------------------------------------------- #
# WS7 - subgroups
# --------------------------------------------------------------------------- #
def test_subgroup_detects_concentrated_edge():
    n = 1600  # ~6 business years -> straddles the 2020 cut
    data = _base_frame(n)
    data["date"] = pd.date_range("2016-01-01", periods=n, freq="B")
    data["dte"] = np.tile(np.arange(200, 0, -1), 9)[:n]
    feat = RNG.normal(0, 1, n)
    data["feat"] = feat
    # Signal exists only for dates >= 2020; earlier is pure noise.
    is_recent = (data["date"].dt.year >= 2020).to_numpy()
    target = np.where(is_recent, feat, RNG.normal(0, 1, n)) + RNG.normal(0, 0.3, n)
    data["fwd_10"] = target

    rep = subgroup_report(data, "fam", "feat", 10)
    era = rep[rep["dimension"] == "era"].set_index("group")["ic"]
    assert era.loc["\u22652020"] > 0.4  # strong where the edge lives
    assert abs(era.loc["<2020"]) < 0.2  # ~flat where it doesn't

    # All four dimensions present, month has 12 buckets.
    assert set(rep["dimension"]) == {"era", "vol", "fase", "mes"}
    assert (rep["dimension"] == "mes").sum() == 12
