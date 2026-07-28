# src/crudewatch/scoring/weight_search.py
"""Search per-family Opportunity-Score weights that maximise backtest Sharpe.

The 8 terms combined by ``compute_opportunity`` do not depend on the weights;
only their linear combination does. So we precompute, once per contract and
strictly point-in-time, the per-bar term values and regime sign, after which the
score for any weight vector is a matrix-vector product. See
``docs/superpowers/specs/2026-07-26-weight-search-optimization-design.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crudewatch.scoring.blocks import fit_calibrator
from crudewatch.scoring.backtest import _hysteresis
from crudewatch.scoring.score import (
    RANGE_TERM_KEYS,
    RANGE_TERMS,
    TREND_TERM_KEYS,
    TREND_TERMS,
    compute_blocks,
)
_REGIME_CODE = {"range": 0, "trend": 1, "transition": 2}

_N_RANGE = len(RANGE_TERM_KEYS)
_N_TREND = len(TREND_TERM_KEYS)
_EQUAL_RANGE = np.array([0.25, 0.25, 0.25, 0.25] + [0.0] * (_N_RANGE - 4))
_EQUAL_TREND = np.array([0.25, 0.25, 0.25, 0.25] + [0.0] * (_N_TREND - 4))


@dataclass(frozen=True)
class PrecomputedContract:
    contract: str
    vintage: int
    date: np.ndarray
    close: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    regime_code: np.ndarray
    m_range: np.ndarray
    range_terms: np.ndarray
    m_trend: np.ndarray
    trend_terms: np.ndarray


def _col(csub: pd.DataFrame, name: str) -> np.ndarray:
    if name in csub.columns:
        return csub[name].to_numpy(dtype=float)
    return np.full(len(csub), np.nan)


def precompute_contract(data, family, contract, horizon: int = 25, cal_cache=None) -> PrecomputedContract:
    df = data.reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])
    dates_all = df["date"].to_numpy()
    csub = df[df["contract"] == contract].sort_values("date").reset_index(drop=True)
    if csub.empty:
        raise KeyError(f"contract {contract!r} not found in data")
    n = len(csub)
    regime = np.zeros(n, dtype=np.int8)
    m_range = np.zeros(n)
    m_trend = np.zeros(n)
    range_terms = np.zeros((n, _N_RANGE))
    trend_terms = np.zeros((n, _N_TREND))
    for i, row in csub.iterrows():
        d = row["date"]
        # The point-in-time calibrator at date d is fit on all family rows <= d
        # and is therefore identical across contracts; memoise it by date so a
        # family precompute fits each unique vintage-date calibrator only once.
        key = np.datetime64(d)
        cal = None if cal_cache is None else cal_cache.get(key)
        if cal is None:
            cal = fit_calibrator(df[dates_all <= key], family, horizon, outcome_asof=d)
            if cal_cache is not None:
                cal_cache[key] = cal
        b = compute_blocks(row, cal, i + 1)
        regime[i] = _REGIME_CODE[b.regime]
        lvl = 0.0 if b.level != b.level else b.level
        dirn = 0.0 if b.direction != b.direction else b.direction
        m_range[i] = 100.0 if lvl < 0 else (-100.0 if lvl > 0 else 0.0)
        m_trend[i] = 100.0 if dirn > 0 else (-100.0 if dirn < 0 else 0.0)
        range_terms[i] = [RANGE_TERMS[k](b, row, cal, lvl, dirn) for k in RANGE_TERM_KEYS]
        trend_terms[i] = [TREND_TERMS[k](b, row, cal, lvl, dirn) for k in TREND_TERM_KEYS]
    vintage = int(csub["vintage"].iloc[0]) if "vintage" in csub.columns else 0
    return PrecomputedContract(
        contract=str(contract), vintage=vintage,
        date=csub["date"].to_numpy(),
        close=_col(csub, "close"), open=_col(csub, "open"),
        high=_col(csub, "high"), low=_col(csub, "low"),
        regime_code=regime, m_range=m_range, range_terms=range_terms,
        m_trend=m_trend, trend_terms=trend_terms,
    )


@dataclass(frozen=True)
class WeightSearchResult:
    w_range: np.ndarray
    w_trend: np.ndarray
    sharpe: float
    equal_sharpe: float
    active_range: tuple
    active_trend: tuple


def precompute_family(data, family, horizon: int = 25, min_bars: int = 60, progress: bool = False) -> list[PrecomputedContract]:
    counts = data.groupby("contract").size()
    keep = sorted(counts[counts >= min_bars].index)
    cal_cache: dict = {}
    out = []
    for j, c in enumerate(keep, 1):
        out.append(precompute_contract(data, family, c, horizon, cal_cache=cal_cache))
        if progress:
            print(f"    precompute {j}/{len(keep)} contracts (cache={len(cal_cache)} dates)", flush=True)
    return out


def _nan_to_neg_inf(x: float) -> float:
    return x if x == x else float("-inf")


def _candidates(dim, dirichlet_n, sparse_n, k_min, k_max, seed):
    dense = simplex_samples(dirichlet_n, dim, seed)
    sparse = sparse_simplex_samples(dim, sparse_n, seed + 7, k_min, k_max)
    return np.vstack([dense, sparse])


def search_weights(pcs, cost, dirichlet_n=1200, sparse_n=800, k_min=2, k_max=5, seed=0, chunk=4000):
    dr = len(RANGE_TERM_KEYS)
    dt = len(TREND_TERM_KEYS)
    cand_r = _candidates(dr, dirichlet_n, sparse_n, k_min, k_max, seed)
    cand_t = _candidates(dt, dirichlet_n, sparse_n, k_min, k_max, seed + 1)
    equal_sharpe = objective(pcs, _EQUAL_RANGE, _EQUAL_TREND, cost)
    best = _nan_to_neg_inf(equal_sharpe)
    best_r, best_t = _EQUAL_RANGE, _EQUAL_TREND
    m = min(len(cand_r), len(cand_t))
    for start in range(0, m, chunk):
        stop = min(start + chunk, m)
        w_r = cand_r[start:stop].T
        w_t = cand_t[start:stop].T
        sh = objective_batch(pcs, w_r, w_t, cost)
        sh = np.where(np.isnan(sh), -np.inf, sh)
        j = int(np.argmax(sh))
        if sh[j] > best:
            best = float(sh[j])
            best_r, best_t = cand_r[start + j], cand_t[start + j]
    active_r = tuple(k for k, w in zip(RANGE_TERM_KEYS, best_r) if w > 0)
    active_t = tuple(k for k, w in zip(TREND_TERM_KEYS, best_t) if w > 0)
    return WeightSearchResult(
        best_r, best_t,
        best if best != float("-inf") else float("nan"),
        equal_sharpe, active_r, active_t,
    )


def opportunity_from_precomputed(
    pc: PrecomputedContract, w_range, w_trend, transition_shrink: float = 0.4,
) -> np.ndarray:
    w_range = np.asarray(w_range, dtype=float)
    w_trend = np.asarray(w_trend, dtype=float)
    o_range = np.clip(pc.m_range * (pc.range_terms @ w_range), -100.0, 100.0)
    o_trend = np.clip(pc.m_trend * (pc.trend_terms @ w_trend), -100.0, 100.0)
    return np.where(
        pc.regime_code == 1,
        o_trend,
        np.where(pc.regime_code == 2, transition_shrink * o_range, o_range),
    )


def simplex_samples(n: int, dim: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.dirichlet(np.ones(dim), size=n)
    fixed = np.vstack([np.full(dim, 1.0 / dim), np.eye(dim)])
    return np.vstack([fixed, draws])


def sparse_simplex_samples(dim, n, seed, k_min, k_max):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, dim), dtype=float)
    k_max = min(k_max, dim)
    k_min = max(1, min(k_min, k_max))
    for i in range(n):
        k = int(rng.integers(k_min, k_max + 1))
        idx = rng.choice(dim, size=k, replace=False)
        out[i, idx] = rng.dirichlet(np.ones(k))
    return out


def fast_pnl(pc, w_range, w_trend, cost, enter=50.0, exit=20.0, opp_override=None) -> np.ndarray:
    """Daily net PnL reproducing ``backtest.simulate`` (open[t+1] fills, cost/2 per leg)."""
    opp = pc_opportunity if (pc_opportunity := opp_override) is not None else opportunity_from_precomputed(pc, w_range, w_trend)
    n = len(opp)
    if n < 2:
        return np.zeros(0)
    desired = _hysteresis(opp, enter, exit)
    effective = np.empty(n, dtype=int)
    effective[0] = 0
    effective[1:] = desired[:-1]
    prev_eff = np.empty(n, dtype=int)
    prev_eff[0] = 0
    prev_eff[1:] = effective[:-1]
    close = pc.close
    open_eff = np.where(np.isnan(pc.open), close, pc.open)
    close_prev = np.empty(n)
    close_prev[0] = close[0]
    close_prev[1:] = close[:-1]
    pnl = prev_eff * (open_eff - close_prev) + effective * (close - open_eff)
    pnl[0] = 0.0
    cost_daily = np.abs(effective - prev_eff) * (cost / 2.0)
    return pnl - cost_daily


def pooled_sharpe(pnl_arrays) -> float:
    pooled = np.concatenate([a for a in pnl_arrays if len(a)]) if pnl_arrays else np.array([])
    if len(pooled) < 2:
        return float("nan")
    sd = float(np.std(pooled, ddof=1))
    if not sd or sd <= 0:
        return float("nan")
    return float(np.mean(pooled) / sd * np.sqrt(252.0))


def objective(pcs, w_range, w_trend, cost) -> float:
    return pooled_sharpe([fast_pnl(pc, w_range, w_trend, cost) for pc in pcs])


def _hysteresis_matrix(opp: np.ndarray, enter_at: float, exit_at: float) -> np.ndarray:
    """Vectorised :func:`backtest._hysteresis` over a ``(n_bars, C)`` candidate grid.

    The state machine is inherently sequential in time, so we still loop over the
    ``n_bars`` rows, but every row updates all ``C`` candidate positions at once.
    Cost is O(n_bars) Python iterations regardless of ``C``.
    """
    n, c = opp.shape
    out = np.zeros((n, c), dtype=np.int8)
    s = np.zeros(c, dtype=np.int8)
    x_all = np.where(np.isnan(opp), 0.0, opp)
    for t in range(n):
        x = x_all[t]
        new0 = np.where(x >= enter_at, 1, np.where(x <= -enter_at, -1, 0))
        new1 = np.where(x <= -enter_at, -1, np.where(x < exit_at, 0, 1))
        newm1 = np.where(x >= enter_at, 1, np.where(x > -exit_at, 0, -1))
        s = np.where(s == 0, new0, np.where(s == 1, new1, newm1)).astype(np.int8)
        out[t] = s
    return out


def _opportunity_matrix(pc, w_range, w_trend, transition_shrink: float = 0.4) -> np.ndarray:
    """``(n_bars, C)`` opportunity for a ``(dim, C)`` weight grid (matches ``opportunity_from_precomputed``)."""
    o_range = np.clip(pc.m_range[:, None] * (pc.range_terms @ w_range), -100.0, 100.0)
    o_trend = np.clip(pc.m_trend[:, None] * (pc.trend_terms @ w_trend), -100.0, 100.0)
    rc = pc.regime_code[:, None]
    return np.where(rc == 1, o_trend, np.where(rc == 2, transition_shrink * o_range, o_range))


def _fast_pnl_matrix(pc, opp: np.ndarray, cost, enter=50.0, exit=20.0) -> np.ndarray:
    """``(n_bars, C)`` daily net PnL, column-for-column identical to :func:`fast_pnl`."""
    n, c = opp.shape
    desired = _hysteresis_matrix(opp, enter, exit)
    effective = np.empty((n, c), dtype=np.int64)
    effective[0] = 0
    effective[1:] = desired[:-1]
    prev_eff = np.empty((n, c), dtype=np.int64)
    prev_eff[0] = 0
    prev_eff[1:] = effective[:-1]
    close = pc.close
    open_eff = np.where(np.isnan(pc.open), close, pc.open)
    close_prev = np.empty(n)
    close_prev[0] = close[0]
    close_prev[1:] = close[:-1]
    pnl = prev_eff * (open_eff - close_prev)[:, None] + effective * (close - open_eff)[:, None]
    pnl[0] = 0.0
    cost_daily = np.abs(effective - prev_eff) * (cost / 2.0)
    return pnl - cost_daily


def objective_batch(pcs, w_range, w_trend, cost) -> np.ndarray:
    """Pooled Sharpe for every candidate weight column at once.

    ``w_range`` is ``(dim_range, C)`` and ``w_trend`` ``(dim_trend, C)``. Returns a
    length-``C`` array whose column ``i`` equals ``objective(pcs, w_range[:,i],
    w_trend[:,i], cost)`` to floating precision.
    """
    w_range = np.asarray(w_range, dtype=float)
    w_trend = np.asarray(w_trend, dtype=float)
    if w_range.ndim == 1:
        w_range = w_range[:, None]
    if w_trend.ndim == 1:
        w_trend = w_trend[:, None]
    c = w_range.shape[1]
    mats = []
    for pc in pcs:
        if len(pc.close) < 2:
            continue
        opp = _opportunity_matrix(pc, w_range, w_trend)
        mats.append(_fast_pnl_matrix(pc, opp, cost))
    if not mats:
        return np.full(c, np.nan)
    pooled = np.vstack(mats)
    if pooled.shape[0] < 2:
        return np.full(c, np.nan)
    sd = pooled.std(axis=0, ddof=1)
    mean = pooled.mean(axis=0)
    out = np.full(c, np.nan)
    good = sd > 0
    out[good] = mean[good] / sd[good] * np.sqrt(252.0)
    return out


@dataclass(frozen=True)
class WalkForwardResult:
    oos_sharpe_opt: float
    oos_sharpe_equal: float
    n_splits: int


def expanding_vintage_splits(vintages, min_train: int = 3):
    vs = sorted({int(v) for v in vintages})
    return [(vs[:i], vs[i]) for i in range(min_train, len(vs))]


def walk_forward_weights(pcs, cost, dirichlet_n=1200, sparse_n=800, k_min=2, k_max=5, seed=0, min_train=3):
    vintages = sorted({pc.vintage for pc in pcs})
    splits = expanding_vintage_splits(vintages, min_train=min_train)
    oos_opt, oos_eq = [], []
    used = 0
    for train_vs, test_v in splits:
        train = [pc for pc in pcs if pc.vintage in set(train_vs)]
        test = [pc for pc in pcs if pc.vintage == test_v]
        if not train or not test:
            continue
        sel = search_weights(
            train, cost,
            dirichlet_n=dirichlet_n, sparse_n=sparse_n,
            k_min=k_min, k_max=k_max, seed=seed,
        )
        for pc in test:
            oos_opt.append(fast_pnl(pc, sel.w_range, sel.w_trend, cost))
            oos_eq.append(fast_pnl(pc, _EQUAL_RANGE, _EQUAL_TREND, cost))
        used += 1
    return WalkForwardResult(pooled_sharpe(oos_opt), pooled_sharpe(oos_eq), used)


def apply_shrinkage(w, w_eq, lam: float) -> np.ndarray:
    """Convex blend of a weight vector toward the equal-weight baseline.

    ``lam=0`` returns ``w`` unchanged; ``lam=1`` returns ``w_eq``. Because both
    inputs are on the simplex (non-negative, sum to 1), any ``lam in [0,1]`` also
    lands on the simplex, so no renormalisation is needed.
    """
    w = np.asarray(w, dtype=float)
    w_eq = np.asarray(w_eq, dtype=float)
    return (1.0 - lam) * w + lam * w_eq


@dataclass(frozen=True)
class ShrinkSweepResult:
    lambdas: tuple
    oos_sharpe_by_lambda: tuple
    oos_sharpe_equal: float
    n_splits: int


def walk_forward_sweep(
    pcs, cost, lambdas=(0.3, 0.5, 0.7, 0.85, 0.95),
    dirichlet_n=1200, sparse_n=800, k_min=2, k_max=5, seed=0, min_train=3,
):
    """Honest nested walk-forward with a shrinkage sweep.

    Per split the config is *selected on train only* (once, the expensive step);
    the selected config is then shrunk toward equal-weight by each ``lambda`` and
    evaluated OOS on the held-out vintage. ``lambda`` does not affect selection,
    so all lambdas share a single search per split. Equal-weight OOS (the
    ``lambda=1`` limit) is reported once as the baseline.
    """
    lambdas = tuple(lambdas)
    vintages = sorted({pc.vintage for pc in pcs})
    splits = expanding_vintage_splits(vintages, min_train=min_train)
    oos_by_lam = {lam: [] for lam in lambdas}
    oos_eq = []
    used = 0
    for train_vs, test_v in splits:
        train = [pc for pc in pcs if pc.vintage in set(train_vs)]
        test = [pc for pc in pcs if pc.vintage == test_v]
        if not train or not test:
            continue
        sel = search_weights(
            train, cost,
            dirichlet_n=dirichlet_n, sparse_n=sparse_n,
            k_min=k_min, k_max=k_max, seed=seed,
        )
        for pc in test:
            oos_eq.append(fast_pnl(pc, _EQUAL_RANGE, _EQUAL_TREND, cost))
        for lam in lambdas:
            wr = apply_shrinkage(sel.w_range, _EQUAL_RANGE, lam)
            wt = apply_shrinkage(sel.w_trend, _EQUAL_TREND, lam)
            for pc in test:
                oos_by_lam[lam].append(fast_pnl(pc, wr, wt, cost))
        used += 1
    return ShrinkSweepResult(
        lambdas,
        tuple(pooled_sharpe(oos_by_lam[lam]) for lam in lambdas),
        pooled_sharpe(oos_eq),
        used,
    )
