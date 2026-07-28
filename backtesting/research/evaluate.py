"""Walk-forward evaluation: does a feature-as-of-t predict the forward outcome?

This is the "backtest-only" half of the harness. It never fits a live model; it
measures, **out-of-sample and by vintage**, whether each continuous feature
carries edge — the honest replacement for ranking long/flat P&L.

Method (per feature x horizon, within a family or family x slot subframe):

1. Split by **vintage** (contract expiry year), expanding window: train on all
   earlier vintages, test on the next one, advance. Splitting by vintage — not
   by random rows — is what keeps highly overlapping synthetic contracts from
   leaking across the train/test boundary.
2. On each **test** fold: Spearman IC of the as-of-t feature vs the forward
   return, and a bucketed forward-return profile whose bucket edges are fixed on
   **train** (so the profile is genuinely out-of-sample).
3. Aggregate across folds: mean IC and its t-stat, a monotonicity score, and the
   "reversion spread" (cheap bucket minus expensive bucket) net of a per-family
   cost stub. Report the **effective sample** (distinct vintages), not just rows,
   so a result concentrated in a couple of correlated vintages is visible.

Everything is direction-agnostic points, consistent with ``targets``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# The dataset enrichment and the per-family cost stub are shared with the live
# app, so they live in the installable ``crudewatch`` package; re-exported here
# for the many offline callers (scripts / regime / strategy) that import them
# from :mod:`evaluate`.
from crudewatch.research.dataset import COST_STUB_POINTS, build_dataset
from crudewatch.research.features import FEATURE_NAMES
from crudewatch.research.panel import PANEL_FEATURES
from crudewatch.research.targets import HORIZONS

# Features evaluated by default: the close-only continuous ones plus the
# analogous-panel level features (which need lifecycle columns).
EVAL_FEATURES: list[str] = FEATURE_NAMES + PANEL_FEATURES


def walk_forward_splits(vintages, min_train: int = 3) -> list[tuple[list[int], int]]:
    """Expanding-window splits over sorted distinct vintages.

    Returns ``(train_vintages, test_vintage)`` pairs; the first ``min_train``
    vintages seed the initial training set and are never used as a test fold.
    """
    vs = sorted({int(v) for v in vintages})
    return [(vs[:i], vs[i]) for i in range(min_train, len(vs))]


def _bucket_edges(train_f: np.ndarray, n_buckets: int) -> np.ndarray | None:
    """Quantile bucket edges from TRAIN, or ``None`` if the window is degenerate."""
    edges = np.unique(np.quantile(train_f, np.linspace(0.0, 1.0, n_buckets + 1)))
    return edges if len(edges) == n_buckets + 1 else None


def _assign_buckets(edges: np.ndarray, values: np.ndarray, n_buckets: int) -> np.ndarray:
    """Bucket index in ``0..n_buckets-1`` for each value under fixed TRAIN edges."""
    return np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, n_buckets - 1)


def _bucket_means(edges: np.ndarray, test_f: np.ndarray, test_y: np.ndarray, n_buckets: int):
    """Mean forward outcome per feature bucket on TEST, using fixed TRAIN edges."""
    idx = _assign_buckets(edges, test_f, n_buckets)
    means = np.full(n_buckets, np.nan)
    for b in range(n_buckets):
        vals = test_y[idx == b]
        if len(vals):
            means[b] = float(vals.mean())
    return means


TRADE_COLS = ("contract", "date")


def _accumulate_trades(
    test: pd.DataFrame,
    feature: str,
    edges: np.ndarray,
    sense: float,
    n_buckets: int,
    horizon: int,
    cost: float,
    target: str,
    mfe_col: str,
    mae_col: str,
    pnls: list[float],
    favs: list[float],
    advs: list[float],
    sides: list[float] | None = None,
    dates: list | None = None,
) -> None:
    """Turn the TEST extreme buckets into non-overlapping, cost-adjusted trades.

    The trade side follows the edge measured **on train** (``sense``): a
    reversion feature (``sense < 0``) goes long the cheap/bottom bucket and short
    the expensive/top bucket; a continuation feature does the reverse. Signals in
    the same contract are taken greedily and then locked out for ``horizon`` bars
    so no two trades overlap. Favourable/adverse excursions are oriented by side
    (a short's favourable move is ``-mae``). Everything stays in price points and
    is appended in place to ``pnls`` / ``favs`` / ``advs``.

    Optional confirmation: if ``test`` carries boolean ``_confirm_long`` /
    ``_confirm_short`` columns, a long signal is only taken where ``_confirm_long``
    is true and a short only where ``_confirm_short`` is true (a rejected signal
    consumes no lock-out, so the next qualifying bar can still trade).
    """
    df = test.sort_values(list(TRADE_COLS)).copy()
    if "_pos" not in df.columns:  # caller may precompute true bar positions on the
        df["_pos"] = df.groupby("contract", sort=False).cumcount()  # full (unfiltered) fold
    idx = _assign_buckets(edges, df[feature].to_numpy(), n_buckets)
    is_ext = (idx == 0) | (idx == n_buckets - 1)
    if not is_ext.any():
        return

    bottom_side = 1.0 if sense < 0 else -1.0
    side = np.where(idx == 0, bottom_side, -bottom_side)

    fwd = df[target].to_numpy(dtype=float)
    mfe = df[mfe_col].to_numpy(dtype=float)
    mae = df[mae_col].to_numpy(dtype=float)
    pos = df["_pos"].to_numpy()
    contracts = df["contract"].to_numpy()
    trade_dates = df["date"].to_numpy() if dates is not None else None

    has_confirm = "_confirm_long" in df.columns and "_confirm_short" in df.columns
    confirm_long = df["_confirm_long"].to_numpy() if has_confirm else None
    confirm_short = df["_confirm_short"].to_numpy() if has_confirm else None

    last_exit: dict = {}
    for i in np.flatnonzero(is_ext):
        if np.isnan(fwd[i]) or np.isnan(mfe[i]) or np.isnan(mae[i]):
            continue
        s = side[i]
        if has_confirm and not (confirm_long[i] if s > 0 else confirm_short[i]):
            continue  # level (or other) confirmation disagrees -> no trade, no lock-out
        c = contracts[i]
        if pos[i] < last_exit.get(c, -1):  # still inside a live trade -> skip
            continue
        last_exit[c] = pos[i] + horizon
        pnls.append(float(s * fwd[i] - cost))
        favs.append(float(mfe[i] if s > 0 else -mae[i]))
        advs.append(float(mae[i] if s > 0 else -mfe[i]))
        if sides is not None:
            sides.append(float(s))
        if dates is not None:
            dates.append(trade_dates[i])


def calendar_daily_sharpe(pnls, dates) -> float:
    """Annualised Sharpe of the strategy as a **calendar-time daily P&L series**.

    The per-trade Sharpe with a fixed ``√(252/horizon)`` factor is wrong for a
    pooled book: trades are non-overlapping *within* a contract, but many
    contracts fire on the same day, so per-trade dispersion ignores same-day
    clustering and the true trade intensity. Instead we book each trade's P&L on
    its entry date, sum simultaneous trades, reindex over the business-day span
    (idle days = 0), and annualise the resulting daily series by ``√252``. This
    captures both diversification across concurrent trades and the real capital
    deployment cadence.
    """
    if pnls is None or len(pnls) < 2 or dates is None:
        return np.nan
    s = pd.Series(np.asarray(pnls, dtype=float), index=pd.to_datetime(list(dates)))
    daily = s.groupby(s.index.normalize()).sum()
    if daily.empty or daily.index.min() == daily.index.max():
        return np.nan
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="B")
    daily = daily.reindex(full, fill_value=0.0)
    sd = float(daily.std(ddof=1))
    if not sd or sd <= 0:
        return np.nan
    return float(daily.mean() / sd * np.sqrt(252.0))


def _trade_summary(
    pnls: list[float], favs: list[float], advs: list[float], horizon: int,
    dates: list | None = None,
) -> dict:
    """Aggregate the trade list into headline risk/reward metrics.

    ``sharpe`` is calendar-time (see :func:`calendar_daily_sharpe`) when trade
    ``dates`` are supplied; otherwise it falls back to the per-trade
    ``√(252/horizon)`` approximation for callers that don't track dates.
    """
    summary = {
        "n_trades": len(pnls),
        "win_rate": np.nan,
        "avg_pnl": np.nan,
        "sharpe": np.nan,
        "mfe_mean": np.nan,
        "mae_mean": np.nan,
    }
    if not pnls:
        return summary
    arr = np.asarray(pnls, dtype=float)
    summary["win_rate"] = float((arr > 0).mean())
    summary["avg_pnl"] = float(arr.mean())
    summary["mfe_mean"] = float(np.mean(favs))
    summary["mae_mean"] = float(np.mean(advs))
    if dates is not None and len(dates) == len(arr):
        summary["sharpe"] = calendar_daily_sharpe(arr, dates)
    elif len(arr) > 1:
        sd = float(arr.std(ddof=1))
        if sd > 0:  # fallback: per-trade Sharpe, ~252/horizon trades/yr
            summary["sharpe"] = float(arr.mean() / sd * np.sqrt(252.0 / horizon))
    return summary


def _fold_contribution(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature: str,
    target: str,
    mfe_col: str,
    mae_col: str,
    n_buckets: int,
    horizon: int,
    cost: float,
    can_trade: bool,
    min_rows: int,
) -> dict | None:
    """One walk-forward fold's OOS pieces for a feature, or ``None`` if too small.

    Shared by the plain evaluation and the regime-gated one: the caller passes
    whatever train/test slice it wants (e.g. already filtered to a regime) and
    gets back the fold's IC, bucket means, and trade contributions. Bucket edges
    and the trade side are fixed on ``train`` only, so it stays out-of-sample.
    """
    if len(test) < min_rows or len(train) < min_rows:
        return None

    ic = test[feature].corr(test[target], method="spearman")
    out: dict = {
        "ic": float(ic) if not np.isnan(ic) else None,
        "means": None,
        "n_test": 0,
        "n_ext": 0,
        "n_rev": 0,
        "pnls": [],
        "favs": [],
        "advs": [],
        "dates": [],
    }
    edges = _bucket_edges(train[feature].to_numpy(), n_buckets)
    if edges is None:  # degenerate train window -> no buckets, no trades
        return out

    test_f = test[feature].to_numpy()
    test_y = test[target].to_numpy()
    out["means"] = _bucket_means(edges, test_f, test_y, n_buckets)
    out["n_test"] = len(test)

    # Reversion hits on the extreme buckets: cheap bucket reverts by rising
    # (fwd > 0), expensive bucket by falling (fwd < 0). Pooled -> P(reversion).
    b_idx = _assign_buckets(edges, test_f, n_buckets)
    cheap = (b_idx == 0) & ~np.isnan(test_y)
    dear = (b_idx == n_buckets - 1) & ~np.isnan(test_y)
    out["n_ext"] = int(cheap.sum() + dear.sum())
    out["n_rev"] = int((test_y[cheap] > 0).sum() + (test_y[dear] < 0).sum())
    if can_trade:
        train_ic = train[feature].corr(train[target], method="spearman")
        if not np.isnan(train_ic) and train_ic != 0:
            _accumulate_trades(
                test, feature, edges, float(train_ic), n_buckets, horizon, cost,
                target, mfe_col, mae_col, out["pnls"], out["favs"], out["advs"],
                dates=out["dates"],
            )
    return out


def _confidence(ics: list[float], ic_t: float, n_folds: int, n_trades: int) -> dict:
    """Transparent 0-100 confidence from sample, stability, sign-consistency, trades.

    Each factor is in ``0..1`` and multiplied, so any weak leg drags the score
    down (a stable IC that flips sign across folds, or rests on 2 trades, is not
    trustworthy). Also returns the pieces so the report can show *why*.
    """
    sample_factor = n_folds / (n_folds + 3) if n_folds else 0.0
    stability_factor = min(abs(ic_t) / 3.0, 1.0) if ic_t == ic_t else 0.0  # NaN-safe
    if ics:
        mean_sign = np.sign(np.mean(ics))
        sign_consistency = float(np.mean([np.sign(v) == mean_sign for v in ics]))
    else:
        sign_consistency = 0.0
    # No trades = no economic validation of the edge, so the score must be
    # dragged to zero (a strong IC that never actually trades is not actionable).
    trade_factor = min(n_trades / 100.0, 1.0) if n_trades and n_trades > 0 else 0.0
    confidence = 100.0 * sample_factor * stability_factor * sign_consistency * trade_factor
    return {
        "confidence": float(confidence),
        "sign_consistency": sign_consistency,
        "sample_factor": float(sample_factor),
        "stability_factor": float(stability_factor),
        "trade_factor": float(trade_factor),
    }


def _aggregate_result(
    feature: str,
    horizon: int,
    n_vintages: int,
    ics: list[float],
    bucket_stack: list[np.ndarray],
    trade_pnls: list[float],
    trade_favs: list[float],
    trade_advs: list[float],
    n_obs_test: int,
    n_buckets: int,
    cost: float,
    edges_repr,
    n_ext: int = 0,
    n_rev: int = 0,
    trade_dates: list | None = None,
) -> dict | None:
    """Collapse per-fold pieces into the tidy stats row (shared aggregation)."""
    if not ics:
        return None

    ic_mean = float(np.mean(ics))
    ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else np.nan
    ic_t = ic_mean / (ic_std / np.sqrt(len(ics))) if ic_std and ic_std > 0 else np.nan
    n_folds = len(ics)
    ic_ci_half = 1.96 * ic_std / np.sqrt(n_folds) if ic_std == ic_std and n_folds else np.nan
    ic_ci_low = ic_mean - ic_ci_half if ic_ci_half == ic_ci_half else np.nan
    ic_ci_high = ic_mean + ic_ci_half if ic_ci_half == ic_ci_half else np.nan
    p_reversion = float(n_rev / n_ext) if n_ext else np.nan
    p_continuation = float(1.0 - p_reversion) if p_reversion == p_reversion else np.nan

    bucket_profile = None
    gross_spread = net_spread = monotonicity = np.nan
    if bucket_stack:
        with warnings.catch_warnings():  # a bucket empty across all folds -> NaN, expected
            warnings.simplefilter("ignore", RuntimeWarning)
            bucket_profile = np.nanmean(np.vstack(bucket_stack), axis=0)
        # Reversion spread: cheap (bottom) minus expensive (top) forward move.
        gross_spread = float(bucket_profile[0] - bucket_profile[-1])
        net_spread = float(gross_spread - cost)
        monotonicity = float(
            pd.Series(bucket_profile).corr(pd.Series(np.arange(n_buckets)), method="spearman")
        )

    trades = _trade_summary(trade_pnls, trade_favs, trade_advs, horizon, trade_dates)
    conf = _confidence(ics, ic_t, n_folds, trades["n_trades"])

    return {
        "feature": feature,
        "horizon": horizon,
        "n_vintages": n_vintages,
        "n_folds": n_folds,
        "n_obs_test": n_obs_test,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_t": ic_t,
        "ic_ci_low": ic_ci_low,
        "ic_ci_high": ic_ci_high,
        "gross_spread": gross_spread,
        "net_spread": net_spread,
        "monotonicity": monotonicity,
        "bucket_profile": None if bucket_profile is None else [round(float(v), 4) for v in bucket_profile],
        "bucket_edges": [round(float(v), 4) for v in edges_repr],
        "p_reversion": p_reversion,
        "p_continuation": p_continuation,
        "n_trades": trades["n_trades"],
        "win_rate": trades["win_rate"],
        "avg_pnl": trades["avg_pnl"],
        "sharpe": trades["sharpe"],
        "mfe_mean": trades["mfe_mean"],
        "mae_mean": trades["mae_mean"],
        "confidence": conf["confidence"],
        "sign_consistency": conf["sign_consistency"],
    }


def evaluate_feature(
    frame: pd.DataFrame,
    feature: str,
    horizon: int,
    n_buckets: int = 5,
    min_train: int = 3,
    cost: float = 0.0,
    min_fold_rows: int | None = None,
) -> dict | None:
    """Out-of-sample walk-forward stats for one feature at one horizon.

    Returns ``None`` if there are too few vintages / usable folds to evaluate.
    """
    target = f"fwd_{horizon}"
    mfe_col, mae_col = f"mfe_{horizon}", f"mae_{horizon}"
    if feature not in frame.columns or target not in frame.columns:
        raise KeyError(f"frame must contain {feature!r} and {target!r}")

    min_rows = min_fold_rows if min_fold_rows is not None else n_buckets * 4
    # Trade / excursion stats need per-contract ordering and the MFE/MAE labels;
    # skip them gracefully when the frame is a bare (feature, target) panel.
    can_trade = {*TRADE_COLS, mfe_col, mae_col}.issubset(frame.columns)
    cols = ["vintage", feature, target]
    if can_trade:
        cols += [*TRADE_COLS, mfe_col, mae_col]
    data = frame[cols].dropna(subset=["vintage", feature, target])
    n_vintages = int(data["vintage"].nunique())
    splits = walk_forward_splits(data["vintage"], min_train)

    ics: list[float] = []
    bucket_stack: list[np.ndarray] = []
    trade_pnls: list[float] = []
    trade_favs: list[float] = []
    trade_advs: list[float] = []
    trade_dates: list = []
    n_obs_test = n_ext = n_rev = 0
    for train_vs, test_v in splits:
        train = data[data["vintage"].isin(train_vs)]
        test = data[data["vintage"] == test_v]
        fc = _fold_contribution(
            train, test, feature, target, mfe_col, mae_col,
            n_buckets, horizon, cost, can_trade, min_rows,
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
            trade_pnls += fc["pnls"]
            trade_favs += fc["favs"]
            trade_advs += fc["advs"]
            trade_dates += fc["dates"]

    # Representative bucket cut-offs for display: pooled feature quantiles over
    # the whole evaluated sample (the per-fold trading edges vary slightly, but
    # these describe "what value lands in each bucket" for the reader).
    if not data.empty:
        edges_repr = np.quantile(data[feature].to_numpy(), np.linspace(0.0, 1.0, n_buckets + 1))
    else:
        edges_repr = np.full(n_buckets + 1, np.nan)

    return _aggregate_result(
        feature, horizon, n_vintages, ics, bucket_stack,
        trade_pnls, trade_favs, trade_advs, n_obs_test, n_buckets, cost, edges_repr,
        n_ext=n_ext, n_rev=n_rev, trade_dates=trade_dates,
    )


def evaluate_family(
    frame: pd.DataFrame,
    family: str,
    features: list[str] | None = None,
    horizons: tuple[int, ...] = HORIZONS,
    by_slot: bool = False,
    n_buckets: int = 5,
    min_train: int = 3,
    enrich: bool = True,
) -> pd.DataFrame:
    """Tidy walk-forward results for a family (optionally split by seasonal slot).

    ``enrich=True`` runs :func:`build_dataset` first; pass ``False`` if ``frame``
    already carries lifecycle + feature + forward-return columns.
    """
    data = build_dataset(frame, family) if enrich else frame
    cost = COST_STUB_POINTS.get(family, 0.0)
    if features is None:
        features = [f for f in EVAL_FEATURES if f in data.columns]

    groups: list[tuple[str, pd.DataFrame]] = [("ALL", data)]
    if by_slot:
        groups += [(str(slot), sub) for slot, sub in data.groupby("slot", sort=True)]

    rows: list[dict] = []
    for group_name, sub in groups:
        for feature in features:
            for horizon in horizons:
                stats = evaluate_feature(
                    sub, feature, horizon, n_buckets=n_buckets, min_train=min_train, cost=cost
                )
                if stats is not None:
                    rows.append({"family": family, "group": group_name, **stats})

    columns = [
        "family", "group", "feature", "horizon", "n_vintages", "n_folds", "n_obs_test",
        "ic_mean", "ic_std", "ic_t", "ic_ci_low", "ic_ci_high",
        "gross_spread", "net_spread", "monotonicity", "bucket_profile", "bucket_edges",
        "p_reversion", "p_continuation",
        "n_trades", "win_rate", "avg_pnl", "sharpe", "mfe_mean", "mae_mean",
        "confidence", "sign_consistency",
    ]
    return pd.DataFrame(rows, columns=columns)
