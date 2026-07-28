# tests/test_weight_search.py
from __future__ import annotations

import numpy as np
import pandas as pd

from crudewatch.scoring.score import (
    RANGE_TERM_KEYS,
    TREND_TERM_KEYS,
    WEIGHTS,
    compute_blocks,
    compute_opportunity,
)
from crudewatch.scoring.blocks import fit_calibrator
from crudewatch.scoring.backtest import simulate
from crudewatch.scoring.weight_search import (
    PrecomputedContract,
    expanding_vintage_splits,
    fast_pnl,
    objective,
    opportunity_from_precomputed,
    pooled_sharpe,
    precompute_contract,
    precompute_family,
    search_weights,
    simplex_samples,
    walk_forward_weights,
)
from test_backtest import _synthetic_family

_EQ_RANGE = np.zeros(len(RANGE_TERM_KEYS))
_EQ_RANGE[:4] = 0.25
_EQ_TREND = np.zeros(len(TREND_TERM_KEYS))
_EQ_TREND[:4] = 0.25


def test_precompute_matches_compute_opportunity_equal_weights():
    data = _synthetic_family(seed=1)
    pc = precompute_contract(data, "outrights", "A", horizon=25)
    got = opportunity_from_precomputed(pc, _EQ_RANGE, _EQ_TREND)
    # Reference: the real scorer, bar by bar, point-in-time.
    df = data[data["contract"] == "A"].sort_values("date").reset_index(drop=True)
    dates_all = pd.to_datetime(data["date"]).to_numpy()
    ref = []
    for i, row in df.iterrows():
        d = pd.to_datetime(row["date"])
        win = data[dates_all <= np.datetime64(d)]
        cal = fit_calibrator(win, "outrights", 25, outcome_asof=d)
        blocks = compute_blocks(row, cal, i + 1)
        ref.append(compute_opportunity(blocks, row, cal, WEIGHTS))
    assert np.allclose(got, np.array(ref), atol=1e-9, equal_nan=True)


def test_precompute_matches_for_random_weights():
    data = _synthetic_family(seed=2)
    pc = precompute_contract(data, "outrights", "A", horizon=25)
    rng = np.random.default_rng(0)
    for _ in range(5):
        wr = rng.dirichlet(np.ones(len(RANGE_TERM_KEYS)))
        wt = rng.dirichlet(np.ones(len(TREND_TERM_KEYS)))
        got = opportunity_from_precomputed(pc, wr, wt)
        df = data[data["contract"] == "A"].sort_values("date").reset_index(drop=True)
        dates_all = pd.to_datetime(data["date"]).to_numpy()
        ref = []
        for i, row in df.iterrows():
            d = pd.to_datetime(row["date"])
            win = data[dates_all <= np.datetime64(d)]
            cal = fit_calibrator(win, "outrights", 25, outcome_asof=d)
            blocks = compute_blocks(row, cal, i + 1)
            w = {
                "range": dict(zip(RANGE_TERM_KEYS, wr)),
                "trend": dict(zip(TREND_TERM_KEYS, wt)),
                "transition_shrink": 0.4,
            }
            ref.append(compute_opportunity(blocks, row, cal, w))
        assert np.allclose(got, np.array(ref), atol=1e-9, equal_nan=True)


def test_precompute_is_point_in_time():
    base = _synthetic_family(seed=3)
    pc_base = precompute_contract(base, "outrights", "A", horizon=25)
    future = base[base["contract"] == "B"].copy()
    future["contract"] = "C"
    future["date"] = pd.to_datetime(future["date"]) + pd.Timedelta(days=400)
    ext = pd.concat([base, future], ignore_index=True)
    pc_ext = precompute_contract(ext, "outrights", "A", horizon=25)
    assert np.allclose(
        opportunity_from_precomputed(pc_base, _EQ_RANGE, _EQ_TREND),
        opportunity_from_precomputed(pc_ext, _EQ_RANGE, _EQ_TREND),
        equal_nan=True,
    )


def test_simplex_samples_valid():
    s = simplex_samples(50, dim=4, seed=0)
    assert s.shape == (55, 4)                       # equal + 4 one-hots + 50
    assert np.all(s >= -1e-12)
    assert np.allclose(s.sum(axis=1), 1.0)
    assert np.allclose(s[0], 0.25)                  # equal-weight first
    assert np.allclose(s[1:5], np.eye(4))           # one-hots


def test_fast_pnl_matches_simulate():
    # Build a score_df directly and compare fast_pnl to simulate's equity increments.
    n = 6
    sdf = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "close": [100, 101, 103, 104, 102, 101.0],
        "open": [100, 101, 102, 103, 104, 105.0],
        "high": [100, 101, 103, 105, 104, 105.0],
        "low": [100, 100, 101, 102, 101, 100.0],
        "opportunity": [0, 60, 30, 10, 0, 0.0],
        "regime": ["range"] * n,
    })
    _, equity, _, _ = simulate(sdf, cost=0.10, enter_at=50.0, exit_at=20.0)
    # A tiny PrecomputedContract whose opportunity_from_precomputed returns sdf["opportunity"].
    opp = sdf["opportunity"].to_numpy()
    pc = PrecomputedContract(
        contract="X", vintage=0, date=sdf["date"].to_numpy(),
        close=sdf["close"].to_numpy(), open=sdf["open"].to_numpy(),
        high=sdf["high"].to_numpy(), low=sdf["low"].to_numpy(),
        regime_code=np.zeros(n, np.int8),
        m_range=np.where(opp >= 0, 1.0, -1.0) * 0.0,  # placeholder, unused below
        range_terms=np.zeros((n, len(RANGE_TERM_KEYS))),
        m_trend=np.zeros(n),
        trend_terms=np.zeros((n, len(TREND_TERM_KEYS))),
    )
    # Monkeypatch-free: call fast_pnl on a pc whose score we override by injecting opp.
    net = fast_pnl(pc, np.zeros(len(RANGE_TERM_KEYS)), np.zeros(len(TREND_TERM_KEYS)), cost=0.10, opp_override=opp)
    sim_daily = np.diff(equity.to_numpy(), prepend=0.0)
    assert np.allclose(net, sim_daily, atol=1e-9)
    assert abs(net.sum() - equity.iloc[-1]) < 1e-9


def test_pooled_sharpe_handles_degenerate():
    assert np.isnan(pooled_sharpe([np.array([1.0])]))       # <2 points
    assert np.isnan(pooled_sharpe([np.zeros(5)]))           # std == 0
    val = pooled_sharpe([np.array([1.0, -0.5, 0.3, 0.2])])
    assert np.isfinite(val)


def test_precompute_family_filters_short_contracts():
    data = _synthetic_family(seed=1)              # A and B, 60 bars each
    pcs = precompute_family(data, "outrights", horizon=25, min_bars=60)
    assert {p.contract for p in pcs} == {"A", "B"}
    pcs2 = precompute_family(data, "outrights", horizon=25, min_bars=1000)
    assert pcs2 == []


def test_search_weights_beats_or_ties_equal_in_sample():
    data = _synthetic_family(seed=7)
    pcs = precompute_family(data, "outrights", horizon=25, min_bars=60)
    res = search_weights(pcs, cost=0.02, dirichlet_n=200, sparse_n=200, seed=0)
    # Equal-weight is always a candidate, so the best is never worse than equal
    # (treating NaN as -inf).
    best = res.sharpe if res.sharpe == res.sharpe else -np.inf
    eq = res.equal_sharpe if res.equal_sharpe == res.equal_sharpe else -np.inf
    assert best >= eq - 1e-9
    assert np.isclose(res.w_range.sum(), 1.0) and np.isclose(res.w_trend.sum(), 1.0)


def test_expanding_vintage_splits():
    assert expanding_vintage_splits([2020, 2021, 2022, 2023], min_train=2) == [
        ([2020, 2021], 2022), ([2020, 2021, 2022], 2023),
    ]
    assert expanding_vintage_splits([2020, 2021], min_train=3) == []


def test_walk_forward_runs_and_reports_both(monkeypatch):
    # Give contracts distinct vintages so splits exist.
    data = _synthetic_family(seed=4)
    data["vintage"] = np.where(data["contract"] == "A", 2020, 2021)
    pcs = precompute_family(data, "outrights", horizon=25, min_bars=60)
    res = walk_forward_weights(
        pcs, cost=0.02, dirichlet_n=50, sparse_n=50, seed=0, min_train=1,
    )
    assert res.n_splits >= 1
    assert isinstance(res.oos_sharpe_opt, float)
    assert isinstance(res.oos_sharpe_equal, float)


def test_walk_forward_selects_on_train_only(monkeypatch):
    import crudewatch.scoring.weight_search as ws

    df = _synthetic_family(4)
    df["date"] = pd.to_datetime(df["date"])
    df["vintage"] = np.where(df["contract"] == "A", 2020, 2021)
    pcs = precompute_family(df, "outrights", min_bars=60)
    seen = []
    real = ws.search_weights

    def spy(train_pcs, cost, **kw):
        seen.extend(pc.vintage for pc in train_pcs)
        return real(train_pcs, cost, **kw)

    monkeypatch.setattr(ws, "search_weights", spy)
    res = walk_forward_weights(
        pcs, cost=0.02, dirichlet_n=50, sparse_n=50, seed=0, min_train=1,
    )
    assert res.n_splits == 1
    assert seen == [2020] and 2021 not in seen


def test_precompute_full_term_parity_random_weights():
    df = _synthetic_family(seed=3)
    df["date"] = pd.to_datetime(df["date"])
    rng_cols = np.random.default_rng(9)
    df["macd_div"] = rng_cols.normal(0, 1, len(df))
    df["er_drop_20"] = rng_cols.normal(0, 1, len(df))
    pc = precompute_contract(df, "outrights", "A", 25)
    assert pc.range_terms.shape[1] == len(RANGE_TERM_KEYS)
    assert pc.trend_terms.shape[1] == len(TREND_TERM_KEYS)
    rng = np.random.default_rng(0)
    for _ in range(5):
        wr = rng.dirichlet(np.ones(len(RANGE_TERM_KEYS)))
        wt = rng.dirichlet(np.ones(len(TREND_TERM_KEYS)))
        weights = {
            "range": dict(zip(RANGE_TERM_KEYS, wr)),
            "trend": dict(zip(TREND_TERM_KEYS, wt)),
            "transition_shrink": 0.4,
        }
        dates_all = pd.to_datetime(df["date"]).to_numpy()
        csub = df[df["contract"] == "A"].sort_values("date").reset_index(drop=True)
        ref = []
        for i, row in csub.iterrows():
            d = row["date"]
            cal = fit_calibrator(df[dates_all <= np.datetime64(d)], "outrights", 25, outcome_asof=d)
            b = compute_blocks(row, cal, i + 1)
            ref.append(compute_opportunity(b, row, cal, weights))
        got = opportunity_from_precomputed(pc, wr, wt, 0.4)
        assert np.allclose(got, ref, atol=1e-9)


def test_sparse_simplex_samples_valid():
    import numpy as np
    from crudewatch.scoring.weight_search import sparse_simplex_samples
    s = sparse_simplex_samples(dim=9, n=200, seed=0, k_min=2, k_max=5)
    assert s.shape == (200, 9)
    assert np.all(s >= 0.0)
    assert np.allclose(s.sum(axis=1), 1.0)
    active = (s > 0).sum(axis=1)
    assert active.min() >= 2 and active.max() <= 5


def test_search_weights_full_dim_beats_or_ties_baseline():
    import numpy as np, pandas as pd
    from crudewatch.scoring.score import RANGE_TERM_KEYS, TREND_TERM_KEYS
    from crudewatch.scoring.weight_search import precompute_family, search_weights
    df = _synthetic_family(2)
    df["date"] = pd.to_datetime(df["date"])
    pcs = precompute_family(df, "outrights", min_bars=60)
    res = search_weights(pcs, cost=0.02, dirichlet_n=200, sparse_n=200, seed=0)
    assert len(res.w_range) == len(RANGE_TERM_KEYS)
    assert len(res.w_trend) == len(TREND_TERM_KEYS)
    assert np.isclose(res.w_range.sum(), 1.0) and np.isclose(res.w_trend.sum(), 1.0)
    a = res.sharpe if res.sharpe == res.sharpe else -np.inf
    e = res.equal_sharpe if res.equal_sharpe == res.equal_sharpe else -np.inf
    assert a >= e
    assert set(res.active_range) <= set(RANGE_TERM_KEYS)


def test_objective_batch_matches_scalar_objective():
    from crudewatch.scoring.weight_search import (
        precompute_family, objective, objective_batch,
    )
    from crudewatch.scoring.score import RANGE_TERM_KEYS, TREND_TERM_KEYS
    df = _synthetic_family(3)
    df["date"] = pd.to_datetime(df["date"])
    df["macd_div"] = np.random.default_rng(9).normal(0, 1, len(df))
    df["er_drop_20"] = np.random.default_rng(10).normal(0, 1, len(df))
    pcs = precompute_family(df, "outrights", min_bars=60)
    rng = np.random.default_rng(0)
    dr, dt = len(RANGE_TERM_KEYS), len(TREND_TERM_KEYS)
    wr = rng.dirichlet(np.ones(dr), size=7).T  # (dr, 7)
    wt = rng.dirichlet(np.ones(dt), size=7).T
    batch = objective_batch(pcs, wr, wt, cost=0.02)
    for i in range(7):
        scalar = objective(pcs, wr[:, i], wt[:, i], 0.02)
        if scalar != scalar:
            assert batch[i] != batch[i]
        else:
            assert np.isclose(batch[i], scalar, atol=1e-9)


def test_apply_shrinkage_convex_blend():
    from crudewatch.scoring.weight_search import apply_shrinkage, _EQUAL_RANGE
    w = np.zeros(len(_EQUAL_RANGE))
    w[4] = 1.0  # all mass on a new indicator term
    assert np.allclose(apply_shrinkage(w, _EQUAL_RANGE, 0.0), w)
    assert np.allclose(apply_shrinkage(w, _EQUAL_RANGE, 1.0), _EQUAL_RANGE)
    mid = apply_shrinkage(w, _EQUAL_RANGE, 0.5)
    assert np.isclose(mid.sum(), 1.0) and np.all(mid >= 0.0)
    assert np.allclose(mid, 0.5 * w + 0.5 * _EQUAL_RANGE)


def test_walk_forward_sweep_one_search_per_split_and_equal_matches():
    import crudewatch.scoring.weight_search as ws
    from crudewatch.scoring.weight_search import (
        precompute_family, walk_forward_sweep, walk_forward_weights,
    )
    df = _synthetic_family(4)
    df["date"] = pd.to_datetime(df["date"])
    df["vintage"] = np.where(df["contract"] == "A", 2020, 2021)
    pcs = precompute_family(df, "outrights", min_bars=60)

    calls = {"n": 0}
    real = ws.search_weights

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    import pytest
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ws, "search_weights", counting)
    try:
        res = ws.walk_forward_sweep(
            pcs, cost=0.02, lambdas=(0.3, 0.7, 0.95),
            dirichlet_n=50, sparse_n=50, seed=0, min_train=1,
        )
    finally:
        monkeypatch.undo()

    # Exactly one selection search per split, regardless of #lambdas.
    assert res.n_splits == 1
    assert calls["n"] == res.n_splits
    assert len(res.oos_sharpe_by_lambda) == 3
    # Equal-weight OOS baseline matches the standard walk-forward's equal series.
    ref = walk_forward_weights(pcs, cost=0.02, dirichlet_n=50, sparse_n=50, seed=0, min_train=1)
    assert (res.oos_sharpe_equal == ref.oos_sharpe_equal) or (
        res.oos_sharpe_equal != res.oos_sharpe_equal
        and ref.oos_sharpe_equal != ref.oos_sharpe_equal
    )
