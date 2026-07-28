"""Regime-gated backtest: trade the right edge in the right market state.

The plain walk-forward in :mod:`evaluate` measures each feature everywhere. But
the whole premise of the project is that **reversion pays in a range and
continuation pays in a trend** — mixing both regimes dilutes each edge. This
module conditions the same honest, out-of-sample-by-vintage machinery on a
**regime label** derived from the Efficiency Ratio (``er_20``):

* ``range``  — ``er_20`` in the bottom tercile of the TRAIN window (choppy /
  mean-reverting); the natural home of the z-score / RSI / Bollinger features.
* ``trend``  — ``er_20`` in the top tercile (clean / directional); the home of
  slope / MACD continuation.
* the middle tercile is a **dead zone**: not traded, so we never operate on the
  noisy regime boundary.

The tercile cut-offs are computed on the TRAIN vintages of each fold only, so
the regime label is point-in-time (``er_20`` itself is an as-of-``t`` feature).

Two outputs, both reusing :func:`evaluate._fold_contribution` /
:func:`evaluate._aggregate_result` so buckets, costs and trade rules match the
rest of the harness exactly:

1. **Diagnostic** (:func:`evaluate_feature_regime`): every feature evaluated
   *inside* one regime — IC, buckets, trades, Sharpe, MFE/MAE — so you can see
   which indicator works where.
2. **Assembled strategy** (:func:`evaluate_gated_strategy`): per fold, pick the
   best reversion feature (most negative train IC) in the range regime and the
   best continuation feature (most positive train IC) in the trend regime, then
   trade each in its own regime on the test fold and pool the (non-overlapping,
   cost-adjusted) trades into one strategy.
"""
from __future__ import annotations

import warnings
from collections import Counter
from contextlib import contextmanager

import numpy as np
import pandas as pd

from crudewatch.research.dataset import regime_thresholds as _regime_thresholds
from backtesting.research.evaluate import (
    TRADE_COLS,
    _accumulate_trades,
    _aggregate_result,
    _bucket_edges,
    _fold_contribution,
    _trade_summary,
    walk_forward_splits,
)

REGIME_FEATURE = "er_20"
REGIMES: tuple[str, str] = ("range", "trend")

# Which edge is expected in each regime (candidates; the best per fold is chosen
# on TRAIN). Reversion goes long the cheap bucket in a range; continuation goes
# long the strong bucket in a trend.
REVERSION_CANDIDATES: list[str] = [
    "z_10", "z_20", "z_50", "pctb_20_2", "pctb_10_1_5",
    "keltner_dist_20", "rsi_2", "rsi_14", "level_z", "level_pct",
    "rsi_div_14", "macd_div", "mom_decel_10",
]
CONTINUATION_CANDIDATES: list[str] = ["slope_20", "macd_hist"]

REGIME_PREFER: dict[str, str] = {"range": "neg", "trend": "pos"}


@contextmanager
def _quiet_constant_corr():
    """Silence scipy's 'input array is constant' Spearman warning: a degenerate
    (flat) feature slice legitimately yields NaN IC, which we already handle."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="An input array is constant")
        yield


def _regime_mask(er_values: np.ndarray, lo: float, hi: float, regime: str) -> np.ndarray:
    """Boolean mask selecting the rows that belong to ``regime`` (dead zone excluded)."""
    if regime == "range":
        return er_values <= lo
    if regime == "trend":
        return er_values >= hi
    raise ValueError(f"unknown regime {regime!r}")


def _with_bar_pos(test: pd.DataFrame) -> pd.DataFrame:
    """Attach a per-contract bar index over the FULL fold (before regime filtering),
    so trade non-overlap stays measured in real bars even on a regime subset."""
    out = test.sort_values(list(TRADE_COLS)).copy()
    out["_pos"] = out.groupby("contract", sort=False).cumcount()
    return out


def evaluate_feature_regime(
    data: pd.DataFrame,
    feature: str,
    horizon: int,
    regime: str,
    *,
    regime_feature: str = REGIME_FEATURE,
    n_buckets: int = 5,
    min_train: int = 3,
    cost: float = 0.0,
    min_fold_rows: int | None = None,
) -> dict | None:
    """Walk-forward stats for one feature **restricted to one regime**.

    ``data`` must already be enriched (lifecycle + features + forward returns),
    i.e. carry ``vintage``, ``er_20``, the feature, ``fwd_h``/``mfe_h``/``mae_h``
    and ``contract``/``date``. Returns the usual tidy stats dict plus a
    ``regime`` key, or ``None`` if the regime has too little data to evaluate.
    """
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    target = f"fwd_{horizon}"
    mfe_col, mae_col = f"mfe_{horizon}", f"mae_{horizon}"
    cols = list(dict.fromkeys(
        ["vintage", feature, target, regime_feature, *TRADE_COLS, mfe_col, mae_col]
    ))
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise KeyError(f"data is missing columns for regime eval: {missing}")

    d = data[cols].dropna(subset=["vintage", feature, target, regime_feature])
    n_vintages = int(d["vintage"].nunique())
    min_rows = min_fold_rows if min_fold_rows is not None else n_buckets * 4
    splits = walk_forward_splits(d["vintage"], min_train)

    ics: list[float] = []
    bucket_stack: list[np.ndarray] = []
    pnls: list[float] = []
    favs: list[float] = []
    advs: list[float] = []
    regime_feature_vals: list[np.ndarray] = []
    n_obs_test = n_ext = n_rev = 0
    with _quiet_constant_corr():
        for train_vs, test_v in splits:
            train = d[d["vintage"].isin(train_vs)]
            test = d[d["vintage"] == test_v]
            if train.empty or test.empty:
                continue
            lo, hi = _regime_thresholds(train[regime_feature].to_numpy())
            test = _with_bar_pos(test)
            train_r = train[_regime_mask(train[regime_feature].to_numpy(), lo, hi, regime)]
            test_r = test[_regime_mask(test[regime_feature].to_numpy(), lo, hi, regime)]

            fc = _fold_contribution(
                train_r, test_r, feature, target, mfe_col, mae_col,
                n_buckets, horizon, cost, True, min_rows,
            )
            if fc is None:
                continue
            if fc["ic"] is not None:
                ics.append(fc["ic"])
            if fc["means"] is not None:
                bucket_stack.append(fc["means"])
                n_obs_test += fc["n_test"]
                n_ext += fc["n_ext"]
                n_rev += fc["n_rev"]
                pnls += fc["pnls"]
                favs += fc["favs"]
                advs += fc["advs"]
                regime_feature_vals.append(test_r[feature].to_numpy())

    if regime_feature_vals:
        pooled = np.concatenate(regime_feature_vals)
        edges_repr = np.quantile(pooled, np.linspace(0.0, 1.0, n_buckets + 1))
    else:
        edges_repr = np.full(n_buckets + 1, np.nan)

    res = _aggregate_result(
        feature, horizon, n_vintages, ics, bucket_stack,
        pnls, favs, advs, n_obs_test, n_buckets, cost, edges_repr,
        n_ext=n_ext, n_rev=n_rev,
    )
    if res is not None:
        res["regime"] = regime
    return res


def _pick_feature(
    train_slice: pd.DataFrame,
    feats: list[str],
    target: str,
    prefer: str,
    min_rows: int,
) -> tuple[str | None, float]:
    """Best feature in ``feats`` by train IC of the desired sign (none if no edge).

    ``prefer='neg'`` returns the most negative IC (reversion), ``'pos'`` the most
    positive (continuation). Only features whose IC has the desired sign qualify,
    so a regime with no usable edge that fold simply trades nothing.
    """
    best, best_ic = None, 0.0
    for f in feats:
        sub = train_slice[[f, target]].dropna()
        if len(sub) < min_rows:
            continue
        ic = sub[f].corr(sub[target], method="spearman")
        if np.isnan(ic):
            continue
        if prefer == "neg" and ic < best_ic:
            best, best_ic = f, float(ic)
        elif prefer == "pos" and ic > best_ic:
            best, best_ic = f, float(ic)
    return best, best_ic


CONFIRM_FEATURE = "level_z"


def _confirm_columns(
    train_r: pd.DataFrame,
    te: pd.DataFrame,
    confirm_feature: str,
    regime: str,
    q: float,
    min_rows: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-row long/short level-confirmation masks for a regime leg, or ``None``.

    Thresholds are the ``q`` / ``1-q`` quantiles of the confirmator on the (regime)
    TRAIN rows. Reversion (range) demands agreement at the value extreme
    ("double cheap / double expensive"); continuation (trend) demands the trade is
    NOT against the value extreme (anti-chasing: don't buy what's already dear nor
    short what's already cheap). Rows with a missing confirmator are not confirmed.
    """
    train_vals = train_r[confirm_feature].dropna().to_numpy()
    if len(train_vals) < min_rows:
        return None
    lo, hi = np.quantile(train_vals, [q, 1.0 - q])
    lvl = te[confirm_feature].to_numpy(dtype=float)
    valid = ~np.isnan(lvl)
    if regime == "range":       # value confirmation
        long_ok = valid & (lvl <= lo)
        short_ok = valid & (lvl >= hi)
    else:                        # trend: anti-chasing
        long_ok = valid & (lvl <= hi)
        short_ok = valid & (lvl >= lo)
    return long_ok, short_ok


def evaluate_gated_strategy(
    data: pd.DataFrame,
    horizon: int,
    *,
    regime_feature: str = REGIME_FEATURE,
    reversion: list[str] | None = None,
    continuation: list[str] | None = None,
    n_buckets: int = 5,
    min_train: int = 3,
    cost: float = 0.0,
    min_fold_rows: int | None = None,
    confirm: bool = False,
    confirm_feature: str = CONFIRM_FEATURE,
    confirm_q: float = 1 / 3,
) -> dict | None:
    """Assemble one gated strategy: reversion in range + continuation in trend.

    Per walk-forward fold: fix regime terciles on train ``er_20``; pick the best
    reversion feature (train IC most negative) among the range rows and the best
    continuation feature (train IC most positive) among the trend rows; then on
    the test fold trade each feature's extreme buckets within its own regime.
    Trades are pooled and summarised (``n_trades``, ``win_rate``, ``avg_pnl``,
    ``sharpe``, ``mfe_mean``, ``mae_mean``), and the features actually chosen per
    regime are reported. Returns ``None`` if no trades are produced.

    Non-overlap is enforced per contract within each regime leg (real bar
    spacing); a range and a trend trade on the same contract could in principle
    overlap, but the regimes are disjoint row sets so this is rare.

    With ``confirm=True`` a level confirmation (``confirm_feature``, default
    ``level_z``) gates each signal: reversion trades only fire on a value extreme
    of the same sign ("double cheap / double expensive"), continuation trades only
    fire when the price is NOT already at the opposing value extreme (anti-chasing).
    """
    reversion = reversion if reversion is not None else REVERSION_CANDIDATES
    continuation = continuation if continuation is not None else CONTINUATION_CANDIDATES
    target = f"fwd_{horizon}"
    mfe_col, mae_col = f"mfe_{horizon}", f"mae_{horizon}"

    rev = [f for f in reversion if f in data.columns]
    con = [f for f in continuation if f in data.columns]
    legs = {"range": (rev, REGIME_PREFER["range"]), "trend": (con, REGIME_PREFER["trend"])}

    base_cols = ["vintage", target, regime_feature, *TRADE_COLS, mfe_col, mae_col]
    cols = list(dict.fromkeys([*base_cols, *rev, *con]))
    use_confirm = confirm and confirm_feature in data.columns
    if use_confirm and confirm_feature not in cols:
        cols.append(confirm_feature)
    missing = [c for c in base_cols if c not in data.columns]
    if missing:
        raise KeyError(f"data is missing columns for gated strategy: {missing}")

    d = data[cols].dropna(subset=["vintage", target, regime_feature])
    n_vintages = int(d["vintage"].nunique())
    min_rows = min_fold_rows if min_fold_rows is not None else n_buckets * 4
    splits = walk_forward_splits(d["vintage"], min_train)

    pnls: list[float] = []
    favs: list[float] = []
    advs: list[float] = []
    trade_dates: list = []
    picks: dict[str, Counter] = {"range": Counter(), "trend": Counter()}
    n_obs_test = 0
    with _quiet_constant_corr():
        for train_vs, test_v in splits:
            train = d[d["vintage"].isin(train_vs)]
            test = d[d["vintage"] == test_v]
            if train.empty or test.empty:
                continue
            lo, hi = _regime_thresholds(train[regime_feature].to_numpy())
            test = _with_bar_pos(test)

            for regime, (feats, prefer) in legs.items():
                if not feats:
                    continue
                train_r = train[_regime_mask(train[regime_feature].to_numpy(), lo, hi, regime)]
                test_r = test[_regime_mask(test[regime_feature].to_numpy(), lo, hi, regime)]
                if len(train_r) < min_rows or test_r.empty:
                    continue
                feat, feat_ic = _pick_feature(train_r, feats, target, prefer, min_rows)
                if feat is None:
                    continue
                tr = train_r[[feat, target]].dropna()
                te = test_r.dropna(subset=[feat])
                if len(tr) < min_rows or te.empty:
                    continue
                edges = _bucket_edges(tr[feat].to_numpy(), n_buckets)
                if edges is None:
                    continue
                if use_confirm:
                    masks = _confirm_columns(train_r, te, confirm_feature, regime, confirm_q, min_rows)
                    if masks is not None:
                        te = te.assign(_confirm_long=masks[0], _confirm_short=masks[1])
                before = len(pnls)
                _accumulate_trades(
                    te, feat, edges, feat_ic, n_buckets, horizon, cost,
                    target, mfe_col, mae_col, pnls, favs, advs, dates=trade_dates,
                )
                if len(pnls) > before:
                    picks[regime][feat] += 1
                    n_obs_test += len(te)

    trades = _trade_summary(pnls, favs, advs, horizon, trade_dates)
    if trades["n_trades"] == 0:
        return None

    return {
        "horizon": horizon,
        "n_vintages": n_vintages,
        "n_obs_test": n_obs_test,
        "confirmed": bool(use_confirm),
        "range_feature": _top_pick(picks["range"]),
        "trend_feature": _top_pick(picks["trend"]),
        "range_trades": int(sum(picks["range"].values())),
        "trend_trades": int(sum(picks["trend"].values())),
        **trades,
    }


def _top_pick(counter: Counter) -> str | None:
    """Most frequently chosen feature across folds (``None`` if the leg never fired)."""
    return counter.most_common(1)[0][0] if counter else None


REGIME_LABELS3: tuple[str, str, str] = ("range", "dead", "trend")


def _label_regimes(er: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Vectorised range/dead/trend label from ER cut-offs."""
    out = np.full(len(er), "dead", dtype=object)
    out[er <= lo] = "range"
    out[er >= hi] = "trend"
    return out


def regime_profile(
    data: pd.DataFrame,
    family: str,
    horizon: int = 10,
    *,
    regime_feature: str = REGIME_FEATURE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptive anatomy of the ER regimes for one family (WS4).

    Returns two tidy frames:

    * **profile** — one row per regime (range / dead / trend) with its occupancy
      (share of bars), mean run length (consecutive bars in that regime within a
      contract), and the forward-return character at ``horizon``: mean, vol
      (std), typical magnitude (mean |fwd|) and the up-rate ``P(fwd>0)``.
    * **transitions** — the bar-to-bar 3x3 transition matrix (rows sum to 1),
      computed within each contract so no transition bleeds across contracts.

    The tercile cut-offs are taken over the whole family sample: this is a
    *description* of how the market behaves in each regime, not a traded signal,
    so a pooled (rather than walk-forward) split is the right lens here.
    """
    target = f"fwd_{horizon}"
    cols = [regime_feature, target, "contract", "date"]
    d = data[[c for c in cols if c in data.columns]].dropna(subset=[regime_feature]).copy()
    empty = pd.DataFrame(), pd.DataFrame()
    if d.empty:
        return empty
    d = d.sort_values(["contract", "date"])
    lo, hi = _regime_thresholds(d[regime_feature].to_numpy())
    d["regime"] = _label_regimes(d[regime_feature].to_numpy(), lo, hi)
    n = len(d)

    # Run lengths per regime: a new run starts whenever the label changes within
    # a contract (or the contract changes).
    change = (d["regime"] != d["regime"].shift()) | (d["contract"] != d["contract"].shift())
    run_id = change.cumsum()
    run_len = d.groupby(run_id)["regime"].transform("size")
    mean_run = d.assign(_rl=run_len).groupby("regime")["_rl"].mean()

    prof_rows = []
    for regime in REGIME_LABELS3:
        sub = d[d["regime"] == regime]
        fwd = sub[target].dropna().to_numpy() if target in sub.columns else np.array([])
        prof_rows.append({
            "family": family,
            "regime": regime,
            "occupancy": float(len(sub) / n) if n else np.nan,
            "mean_run": float(mean_run.get(regime, np.nan)),
            "mean_fwd": float(np.mean(fwd)) if len(fwd) else np.nan,
            "std_fwd": float(np.std(fwd, ddof=1)) if len(fwd) > 1 else np.nan,
            "abs_fwd": float(np.mean(np.abs(fwd))) if len(fwd) else np.nan,
            "up_rate": float((fwd > 0).mean()) if len(fwd) else np.nan,
            "n": int(len(sub)),
        })
    profile = pd.DataFrame(prof_rows, columns=[
        "family", "regime", "occupancy", "mean_run",
        "mean_fwd", "std_fwd", "abs_fwd", "up_rate", "n",
    ])

    # Transition matrix within contracts only.
    nxt = d["regime"].shift(-1)
    same_contract = d["contract"] == d["contract"].shift(-1)
    pairs = pd.DataFrame({"from": d["regime"], "to": nxt})[same_contract]
    trans_rows = []
    for frm in REGIME_LABELS3:
        block = pairs[pairs["from"] == frm]
        total = len(block)
        counts = block["to"].value_counts()
        for to in REGIME_LABELS3:
            trans_rows.append({
                "family": family,
                "from": frm,
                "to": to,
                "prob": float(counts.get(to, 0) / total) if total else np.nan,
                "n": int(total),
            })
    transitions = pd.DataFrame(trans_rows, columns=["family", "from", "to", "prob", "n"])
    return profile, transitions


def evaluate_regime_family(
    data: pd.DataFrame,
    family: str,
    features: list[str],
    horizon: int,
    *,
    n_buckets: int = 5,
    min_train: int = 3,
    cost: float = 0.0,
) -> pd.DataFrame:
    """Tidy per-feature diagnostic for a family: each feature x regime, one row."""
    rows: list[dict] = []
    for feature in features:
        if feature not in data.columns:
            continue
        for regime in REGIMES:
            res = evaluate_feature_regime(
                data, feature, horizon, regime,
                n_buckets=n_buckets, min_train=min_train, cost=cost,
            )
            if res is not None:
                rows.append({"family": family, **res})
    columns = [
        "family", "feature", "regime", "horizon", "n_vintages", "n_folds", "n_obs_test",
        "ic_mean", "ic_std", "ic_t", "ic_ci_low", "ic_ci_high",
        "gross_spread", "net_spread", "monotonicity", "bucket_profile", "bucket_edges",
        "p_reversion", "p_continuation",
        "n_trades", "win_rate", "avg_pnl", "sharpe", "mfe_mean", "mae_mean",
        "confidence", "sign_consistency",
    ]
    return pd.DataFrame(rows, columns=columns)
