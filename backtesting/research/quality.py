"""Direction and trend-quality diagnostics (WS5).

Two lenses that the plain IC / spread numbers hide:

* **Direction** (:func:`direction_breakdown`): a feature's traded edge split into
  its **long** and **short** legs. An edge that only works on one side (e.g. only
  buying the cheap bucket, never shorting the dear one) is operationally very
  different from a symmetric one, and the pooled win-rate masks it. Reuses the
  exact walk-forward trade machinery from :mod:`evaluate` (train-fixed buckets,
  train-sign side, non-overlapping trades, cost) and only records the side.

* **Trend quality** (:func:`trend_quality_gradient`): the project thesis is that
  reversion pays in choppy markets and continuation pays in clean trends. This
  slices the sample into **Efficiency-Ratio quintiles** and reports, per quintile,
  the IC of the best reversion feature and the best continuation feature. If the
  thesis holds, reversion IC gets more negative at low ER and continuation IC
  gets more positive at high ER. Descriptive (pooled), like the other robustness
  lenses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.research.evaluate import (
    TRADE_COLS,
    _accumulate_trades,
    _bucket_edges,
    walk_forward_splits,
)
from backtesting.research.regime import (
    CONTINUATION_CANDIDATES,
    REGIME_FEATURE,
    REVERSION_CANDIDATES,
)


def _leg_stats(pnls: np.ndarray, prefix: str) -> dict:
    """Count / win-rate / mean P&L for one side's trades."""
    if len(pnls) == 0:
        return {f"n_{prefix}": 0, f"win_{prefix}": np.nan, f"pnl_{prefix}": np.nan}
    return {
        f"n_{prefix}": int(len(pnls)),
        f"win_{prefix}": float((pnls > 0).mean()),
        f"pnl_{prefix}": float(pnls.mean()),
    }


def direction_breakdown(
    data: pd.DataFrame,
    feature: str,
    horizon: int,
    *,
    n_buckets: int = 5,
    min_train: int = 3,
    cost: float = 0.0,
    min_fold_rows: int | None = None,
) -> dict | None:
    """Split a feature's OOS trades into long vs short legs (WS5 direction).

    Same walk-forward as :func:`evaluate.evaluate_feature` but keeps the side of
    each trade, so we can report whether the edge is symmetric or one-sided.
    Returns ``None`` if the feature never trades.
    """
    target = f"fwd_{horizon}"
    mfe_col, mae_col = f"mfe_{horizon}", f"mae_{horizon}"
    need = {feature, target, *TRADE_COLS, mfe_col, mae_col, "vintage"}
    if not need.issubset(data.columns):
        missing = need - set(data.columns)
        raise KeyError(f"data is missing columns for direction breakdown: {missing}")

    cols = ["vintage", feature, target, *TRADE_COLS, mfe_col, mae_col]
    d = data[cols].dropna(subset=["vintage", feature, target])
    min_rows = min_fold_rows if min_fold_rows is not None else n_buckets * 4
    splits = walk_forward_splits(d["vintage"], min_train)

    pnls: list[float] = []
    favs: list[float] = []
    advs: list[float] = []
    sides: list[float] = []
    for train_vs, test_v in splits:
        train = d[d["vintage"].isin(train_vs)]
        test = d[d["vintage"] == test_v]
        if len(train) < min_rows or len(test) < min_rows:
            continue
        edges = _bucket_edges(train[feature].to_numpy(), n_buckets)
        if edges is None:
            continue
        train_ic = train[feature].corr(train[target], method="spearman")
        if np.isnan(train_ic) or train_ic == 0:
            continue
        _accumulate_trades(
            test, feature, edges, float(train_ic), n_buckets, horizon, cost,
            target, mfe_col, mae_col, pnls, favs, advs, sides,
        )

    if not pnls:
        return None
    arr = np.asarray(pnls, dtype=float)
    sd = np.asarray(sides, dtype=float)
    result = {
        "feature": feature,
        "horizon": horizon,
        "n_trades": int(len(arr)),
        "win_rate": float((arr > 0).mean()),
        "avg_pnl": float(arr.mean()),
    }
    result.update(_leg_stats(arr[sd > 0], "long"))
    result.update(_leg_stats(arr[sd < 0], "short"))
    return result


def _pooled_ic(data: pd.DataFrame, feature: str, target: str) -> float:
    sub = data[[feature, target]].dropna()
    if len(sub) < 30:
        return np.nan
    ic = sub[feature].corr(sub[target], method="spearman")
    return float(ic) if not np.isnan(ic) else np.nan


def _best_feature(data: pd.DataFrame, feats: list[str], target: str, prefer: str) -> str | None:
    """Best candidate by pooled IC of the desired sign (``neg``/``pos``)."""
    best, best_ic = None, 0.0
    for f in feats:
        if f not in data.columns:
            continue
        ic = _pooled_ic(data, f, target)
        if np.isnan(ic):
            continue
        if prefer == "neg" and ic < best_ic:
            best, best_ic = f, ic
        elif prefer == "pos" and ic > best_ic:
            best, best_ic = f, ic
    return best


def trend_quality_gradient(
    data: pd.DataFrame,
    family: str,
    horizon: int,
    *,
    regime_feature: str = REGIME_FEATURE,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """IC of the best reversion/continuation feature across ER quintiles (WS5).

    Tests the core thesis: reversion should strengthen (IC more negative) in low-ER
    (choppy) regimes and continuation should strengthen (IC more positive) in
    high-ER (clean-trend) regimes. Returns tidy rows
    ``family, kind, feature, er_bin, ic, n``.
    """
    target = f"fwd_{horizon}"
    cols = ["family", "kind", "feature", "er_bin", "ic", "n"]
    if regime_feature not in data.columns or target not in data.columns:
        return pd.DataFrame(columns=cols)

    d = data.dropna(subset=[regime_feature, target]).copy()
    if d.empty:
        return pd.DataFrame(columns=cols)
    try:
        d["er_bin"] = pd.qcut(d[regime_feature], n_quantiles, labels=range(1, n_quantiles + 1))
    except ValueError:
        ranked = d[regime_feature].rank(method="first")
        d["er_bin"] = pd.qcut(ranked, n_quantiles, labels=range(1, n_quantiles + 1))

    picks = {
        "reversion": _best_feature(d, REVERSION_CANDIDATES, target, "neg"),
        "continuation": _best_feature(d, CONTINUATION_CANDIDATES, target, "pos"),
    }
    rows: list[dict] = []
    for kind, feat in picks.items():
        if feat is None:
            continue
        for er_bin, gdf in d.groupby("er_bin", observed=True):
            sub = gdf[[feat, target]].dropna()
            ic = sub[feat].corr(sub[target], method="spearman") if len(sub) > 2 else np.nan
            rows.append({
                "family": family,
                "kind": kind,
                "feature": feat,
                "er_bin": int(er_bin),
                "ic": float(ic) if not np.isnan(ic) else np.nan,
                "n": int(len(sub)),
            })
    return pd.DataFrame(rows, columns=cols)
