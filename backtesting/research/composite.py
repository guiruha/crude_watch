"""Composite multi-indicator reversion signal, evaluated out-of-sample (Fase 1).

Instead of scoring one indicator at a time, this builds a single **combined**
reversion signal per family and backtests it with the same honest walk-forward
machinery as the individual features:

Per fold, on the TRAIN vintages only:
  * z-score each candidate reversion feature (train mean / std);
  * orient it by the sign of its train IC (so "cheap" always pushes the composite
    the same way);
  * average the oriented z-scores into one composite (equal weight — deliberately
    not fitted, to avoid overfitting the weights on a thin sample).

Then on the TEST fold the composite is recomputed with the frozen train
normalisation, bucketed with train edges, and scored (IC, bucket profile,
non-overlapping cost-adjusted trades) exactly like any single feature via
:func:`evaluate._fold_contribution` / :func:`evaluate._aggregate_result`. The
result is directly comparable to the best single indicator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.research.evaluate import (
    TRADE_COLS,
    _aggregate_result,
    _fold_contribution,
    walk_forward_splits,
)
from backtesting.research.regime import REVERSION_CANDIDATES


def _fold_composite(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, min_rows: int
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Equal-weight, sign-oriented, train-standardised composite for one fold.

    Weights (sign), means and stds are all fixed on ``train`` -> point-in-time.
    A feature missing on a row contributes 0 (neutral) after standardisation.
    """
    spec: dict[str, tuple[float, float, float]] = {}
    for f in features:
        sub = train[[f, target]].dropna()
        if len(sub) < min_rows:
            continue
        ic = sub[f].corr(sub[target], method="spearman")
        std = float(train[f].std())
        if np.isnan(ic) or ic == 0 or not std or std <= 0:
            continue
        spec[f] = (float(np.sign(ic)), float(train[f].mean()), std)
    if not spec:
        return None, None

    def build(df: pd.DataFrame) -> np.ndarray:
        acc = np.zeros(len(df))
        for f, (sgn, mean, std) in spec.items():
            z = np.nan_to_num(((df[f] - mean) / std).to_numpy(dtype=float), nan=0.0)
            acc = acc + sgn * z
        return acc / len(spec)

    return build(train), build(test)


def evaluate_composite_family(
    data: pd.DataFrame,
    family: str,
    horizons: tuple[int, ...],
    *,
    features: list[str] | None = None,
    n_buckets: int = 5,
    min_train: int = 3,
    cost: float = 0.0,
    min_fold_rows: int | None = None,
) -> pd.DataFrame:
    """Walk-forward stats for the composite reversion signal of one family.

    Returns one tidy row per horizon (``feature == "composite"``), sharing the
    schema of :func:`evaluate.evaluate_feature` so it drops straight into the
    report next to the single-indicator results.
    """
    feats = [f for f in (features or REVERSION_CANDIDATES) if f in data.columns]
    rows: list[dict] = []
    if len(feats) < 2:
        return pd.DataFrame(rows)

    for h in horizons:
        target, mfe_col, mae_col = f"fwd_{h}", f"mfe_{h}", f"mae_{h}"
        cols = list(dict.fromkeys(["vintage", target, *TRADE_COLS, mfe_col, mae_col, *feats]))
        d = data[[c for c in cols if c in data.columns]].dropna(subset=["vintage", target])
        n_vintages = int(d["vintage"].nunique())
        min_rows = min_fold_rows if min_fold_rows is not None else n_buckets * 4
        splits = walk_forward_splits(d["vintage"], min_train)

        ics: list[float] = []
        bucket_stack: list[np.ndarray] = []
        pnls: list[float] = []
        favs: list[float] = []
        advs: list[float] = []
        pooled: list[np.ndarray] = []
        n_obs = n_ext = n_rev = 0
        for train_vs, test_v in splits:
            train = d[d["vintage"].isin(train_vs)]
            test = d[d["vintage"] == test_v]
            if len(train) < min_rows or len(test) < min_rows:
                continue
            ctr, cte = _fold_composite(train, test, feats, target, min_rows)
            if ctr is None:
                continue
            train = train.assign(_composite=ctr)
            test = test.assign(_composite=cte)
            fc = _fold_contribution(
                train, test, "_composite", target, mfe_col, mae_col,
                n_buckets, h, cost, True, min_rows,
            )
            if fc is None:
                continue
            if fc["ic"] is not None:
                ics.append(fc["ic"])
            if fc["means"] is not None:
                bucket_stack.append(fc["means"])
                n_obs += fc["n_test"]
                n_ext += fc["n_ext"]
                n_rev += fc["n_rev"]
                pnls += fc["pnls"]
                favs += fc["favs"]
                advs += fc["advs"]
                pooled.append(test["_composite"].to_numpy())

        if pooled:
            edges_repr = np.quantile(np.concatenate(pooled), np.linspace(0.0, 1.0, n_buckets + 1))
        else:
            edges_repr = np.full(n_buckets + 1, np.nan)

        res = _aggregate_result(
            "composite", h, n_vintages, ics, bucket_stack,
            pnls, favs, advs, n_obs, n_buckets, cost, edges_repr,
            n_ext=n_ext, n_rev=n_rev,
        )
        if res is not None:
            rows.append({"family": family, "n_features": len(feats), **res})

    columns = [
        "family", "n_features", "feature", "horizon", "n_vintages", "n_folds", "n_obs_test",
        "ic_mean", "ic_std", "ic_t", "ic_ci_low", "ic_ci_high",
        "gross_spread", "net_spread", "monotonicity", "bucket_profile", "bucket_edges",
        "p_reversion", "p_continuation",
        "n_trades", "win_rate", "avg_pnl", "sharpe", "mfe_mean", "mae_mean",
        "confidence", "sign_consistency",
    ]
    return pd.DataFrame(rows, columns=columns)
