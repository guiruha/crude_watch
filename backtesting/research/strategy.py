"""The selected strategy, simulated trade-by-trade for demonstration.

The research layer concluded that the only robust, cost-surviving edge is
**mean-reversion inside the range regime**, confirmed by the analogous-panel
level, on the calendar-structure families (quarterly / semestral / yearly /
flies). This module replays exactly that strategy and records a **trade ledger**
(date, contract, side, PnL) so the demonstration report can draw a real,
out-of-sample cumulative-PnL curve rather than only aggregate statistics.

It reuses the very same honest machinery as the gated backtest
(:mod:`backtesting.research.regime`): walk-forward by vintage, regime terciles and
bucket edges fixed on train, level confirmation, non-overlapping cost-adjusted
trades. The only addition is that each executed trade is kept with its date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.research.evaluate import (
    TRADE_COLS,
    _assign_buckets,
    _bucket_edges,
    calendar_daily_sharpe,
    walk_forward_splits,
)
from backtesting.research.regime import (
    CONFIRM_FEATURE,
    REGIME_FEATURE,
    REVERSION_CANDIDATES,
    _confirm_columns,
    _pick_feature,
    _quiet_constant_corr,
    _regime_mask,
    _regime_thresholds,
    _with_bar_pos,
)

# Families whose reversion edge is robust (FDR-significant analogous-panel level)
# AND economic (positive calendar-time Sharpe at the headline horizon on the
# executable basis). Calendars is the strongest; yearly/flies flip negative
# beyond D+20 and are dropped. See the research report.
TRADEABLE_FAMILIES: tuple[str, ...] = ("calendars", "quarterly", "semestral")


def _ledger_trades(
    te: pd.DataFrame,
    feature: str,
    edges: np.ndarray,
    sense: float,
    n_buckets: int,
    horizon: int,
    cost: float,
    target: str,
    mfe_col: str | None = None,
    mae_col: str | None = None,
) -> list[dict]:
    """Non-overlapping, cost-adjusted trades from the test extreme buckets, WITH dates.

    Mirrors :func:`evaluate._accumulate_trades` (same side rule, same per-contract
    lock-out, same optional ``_confirm_long`` / ``_confirm_short`` gating) but
    returns a per-trade record instead of only appending PnL.
    """
    df = te.sort_values(list(TRADE_COLS)).copy()
    if "_pos" not in df.columns:
        df["_pos"] = df.groupby("contract", sort=False).cumcount()
    idx = _assign_buckets(edges, df[feature].to_numpy(), n_buckets)
    is_ext = (idx == 0) | (idx == n_buckets - 1)
    trades: list[dict] = []
    if not is_ext.any():
        return trades

    bottom_side = 1.0 if sense < 0 else -1.0
    side = np.where(idx == 0, bottom_side, -bottom_side)
    fwd = df[target].to_numpy(dtype=float)
    pos = df["_pos"].to_numpy()
    contracts = df["contract"].to_numpy()
    dates = df["date"].to_numpy()

    has_confirm = "_confirm_long" in df.columns and "_confirm_short" in df.columns
    confirm_long = df["_confirm_long"].to_numpy() if has_confirm else None
    confirm_short = df["_confirm_short"].to_numpy() if has_confirm else None

    mfe_col = mfe_col or f"mfe_{horizon}"
    mae_col = mae_col or f"mae_{horizon}"
    mfe = df[mfe_col].to_numpy(dtype=float) if mfe_col in df.columns else np.full(len(df), np.nan)
    mae = df[mae_col].to_numpy(dtype=float) if mae_col in df.columns else np.full(len(df), np.nan)

    last_exit: dict = {}
    for i in np.flatnonzero(is_ext):
        # Same NaN gate as evaluate._accumulate_trades: a trade needs a valid
        # forward outcome AND valid excursions, so the ledger matches the research
        # trade set exactly (not just when fwd alone is present).
        if np.isnan(fwd[i]) or np.isnan(mfe[i]) or np.isnan(mae[i]):
            continue
        s = side[i]
        if has_confirm and not (confirm_long[i] if s > 0 else confirm_short[i]):
            continue
        c = contracts[i]
        if pos[i] < last_exit.get(c, -1):
            continue
        last_exit[c] = pos[i] + horizon
        trades.append({
            "date": dates[i],
            "contract": c,
            "side": float(s),
            "pnl": float(s * fwd[i] - cost),
        })
    return trades


def simulate_strategy(
    data: pd.DataFrame,
    family: str,
    horizon: int,
    *,
    reversion: list[str] | None = None,
    regime_feature: str = REGIME_FEATURE,
    confirm_feature: str = CONFIRM_FEATURE,
    n_buckets: int = 5,
    min_train: int = 3,
    cost: float = 0.0,
    confirm: bool = True,
    confirm_q: float = 1 / 3,
    min_fold_rows: int | None = None,
    exec_lag: int = 0,
) -> pd.DataFrame:
    """Replay the range-regime, level-confirmed reversion strategy for one family.

    Returns the out-of-sample trade ledger sorted by date with a running
    ``cum_pnl`` column (empty frame if the strategy never fires).

    The forward target ``fwd_h`` is already on the **executable basis** (entry at
    ``open[t+1]``, exit at ``close[t+h]``; see :mod:`crudewatch.research.targets`),
    so the default ``exec_lag=0`` already reflects realistic next-open fills — no
    same-close look-ahead. ``exec_lag`` remains available as an *extra* stress
    test: it shifts only the REALISED outcome (the fill) a further ``exec_lag``
    bars, while the decision (signal, regime, edges, feature, confirmation) still
    happens at ``t`` and the trade set is unchanged.
    """
    feats = [f for f in (reversion or REVERSION_CANDIDATES) if f in data.columns]
    target = f"fwd_{horizon}"
    mfe_col, mae_col = f"mfe_{horizon}", f"mae_{horizon}"
    cols = list(dict.fromkeys(
        ["vintage", target, mfe_col, mae_col, regime_feature, *TRADE_COLS, *feats, confirm_feature]
    ))
    d = data[[c for c in cols if c in data.columns]].dropna(subset=["vintage", target, regime_feature])

    # Realistic execution: the fill uses the outcome starting `exec_lag` bars
    # later, within the same contract. Feature selection still uses `target`
    # (decision at t), so the trade SET is unchanged and only the fill moves.
    pnl_target, pnl_mfe, pnl_mae = target, mfe_col, mae_col
    if exec_lag:
        g = d.sort_values(list(TRADE_COLS)).groupby("contract", sort=False)
        d = d.sort_values(list(TRADE_COLS)).assign(
            _exec_fwd=g[target].shift(-exec_lag),
            _exec_mfe=g[mfe_col].shift(-exec_lag),
            _exec_mae=g[mae_col].shift(-exec_lag),
        )
        pnl_target, pnl_mfe, pnl_mae = "_exec_fwd", "_exec_mfe", "_exec_mae"

    min_rows = min_fold_rows if min_fold_rows is not None else n_buckets * 4
    splits = walk_forward_splits(d["vintage"], min_train)

    ledger: list[dict] = []
    with _quiet_constant_corr():
        for train_vs, test_v in splits:
            train = d[d["vintage"].isin(train_vs)]
            test = d[d["vintage"] == test_v]
            if train.empty or test.empty:
                continue
            lo, hi = _regime_thresholds(train[regime_feature].to_numpy())
            test = _with_bar_pos(test)
            train_r = train[_regime_mask(train[regime_feature].to_numpy(), lo, hi, "range")]
            test_r = test[_regime_mask(test[regime_feature].to_numpy(), lo, hi, "range")]
            if len(train_r) < min_rows or test_r.empty:
                continue
            feat, feat_ic = _pick_feature(train_r, feats, target, "neg", min_rows)
            if feat is None:
                continue
            tr = train_r[[feat, target]].dropna()
            te = test_r.dropna(subset=[feat])
            if len(tr) < min_rows or te.empty:
                continue
            edges = _bucket_edges(tr[feat].to_numpy(), n_buckets)
            if edges is None:
                continue
            if confirm and confirm_feature in te.columns:
                masks = _confirm_columns(train_r, te, confirm_feature, "range", confirm_q, min_rows)
                if masks is not None:
                    te = te.assign(_confirm_long=masks[0], _confirm_short=masks[1])
            for t in _ledger_trades(te, feat, edges, feat_ic, n_buckets, horizon, cost,
                                    pnl_target, pnl_mfe, pnl_mae):
                t["family"] = family
                t["feature"] = feat
                ledger.append(t)

    led = pd.DataFrame(ledger, columns=["family", "date", "contract", "feature", "side", "pnl"])
    if not led.empty:
        led = led.sort_values("date").reset_index(drop=True)
        led["cum_pnl"] = led["pnl"].cumsum()
    return led


def strategy_stats(ledger: pd.DataFrame, horizon: int) -> dict:
    """Headline risk/reward for a trade ledger (points, cost already deducted)."""
    base = {
        "n_trades": 0, "n_long": 0, "n_short": 0, "win_rate": np.nan,
        "avg_pnl": np.nan, "total_pnl": np.nan, "sharpe": np.nan,
        "max_dd": np.nan, "trades_per_year": np.nan,
    }
    if ledger.empty:
        return base
    pnl = ledger["pnl"].to_numpy(dtype=float)
    n = len(pnl)
    cum = ledger["cum_pnl"].to_numpy(dtype=float)
    peak = np.maximum.accumulate(cum)
    span_days = (pd.to_datetime(ledger["date"].max()) - pd.to_datetime(ledger["date"].min())).days
    years = span_days / 365.25 if span_days else np.nan
    return {
        "n_trades": n,
        "n_long": int((ledger["side"] > 0).sum()),
        "n_short": int((ledger["side"] < 0).sum()),
        "win_rate": float((pnl > 0).mean()),
        "avg_pnl": float(pnl.mean()),
        "total_pnl": float(pnl.sum()),
        # Calendar-time Sharpe: books each trade on its entry date, sums
        # concurrent trades, annualises the daily series by √252 (honest about
        # same-day clustering and real trade cadence, unlike a fixed √(252/h)).
        "sharpe": calendar_daily_sharpe(pnl, ledger["date"].to_numpy()),
        "max_dd": float((cum - peak).min()),
        "trades_per_year": float(n / years) if years and years > 0 else np.nan,
    }
