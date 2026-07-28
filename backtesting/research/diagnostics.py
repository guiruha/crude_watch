"""Robustness diagnostics for the backtest report (Paso 5 + Paso 7).

Two independent, deliberately **descriptive** lenses on the evaluated features —
they never fit weights or optimise, they only expose structure so a human can
judge robustness:

* **Redundancy** (Paso 5): a Spearman correlation matrix of the as-of-``t``
  features, greedy clustering of look-alikes (``|rho| >= threshold``), and an
  **incremental IC** — the IC of a feature's residual after removing the linear
  part explained by its cluster representative. A feature that adds nothing
  beyond its cluster leader shows a near-zero incremental IC.

* **Subgroups** (Paso 7): the headline feature's IC broken down by era
  (pre/post-2020), realized-volatility tercile, contract life-phase (days to
  expiry tercile) and calendar month. An edge that lives in a single slice
  (famously 2020) is not robust; this makes that visible with the sample size.

Everything uses information available at ``t`` only (feature values, realized
vol, dte); the correlation/subgroup IC pool test rows for a robustness read, not
a walk-forward re-run (which would leave subgroups too sparse to interpret).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# WS6 - Redundancy
# --------------------------------------------------------------------------- #
def feature_correlation(
    data: pd.DataFrame, features: list[str], method: str = "spearman"
) -> pd.DataFrame:
    """Feature-vs-feature correlation matrix (as-of-``t`` values, descriptive)."""
    cols = [f for f in features if f in data.columns]
    return data[cols].corr(method=method)


def _connected_components(feats: list[str], adj: pd.DataFrame) -> list[list[str]]:
    """Connected components of the ``|rho| >= threshold`` graph (greedy clusters)."""
    seen: set[str] = set()
    comps: list[list[str]] = []
    for start in feats:
        if start in seen:
            continue
        stack, comp = [start], []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            comp.append(node)
            for other in feats:
                if other not in seen and bool(adj.loc[node, other]):
                    stack.append(other)
        comps.append(sorted(comp))
    return comps


def cluster_features(corr: pd.DataFrame, threshold: float = 0.8) -> list[list[str]]:
    """Group features whose absolute correlation meets ``threshold``."""
    feats = list(corr.columns)
    adj = corr.abs() >= threshold
    return _connected_components(feats, adj)


def incremental_ic(data: pd.DataFrame, feature: str, rep: str, target: str) -> float:
    """Spearman IC of ``feature`` residualised on ``rep`` vs ``target``.

    Near zero => ``feature`` adds no signal beyond its cluster representative.
    """
    if feature == rep:
        return np.nan
    sub = data[[feature, rep, target]].dropna()
    if len(sub) < 30:
        return np.nan
    x = sub[rep].to_numpy(dtype=float)
    y = sub[feature].to_numpy(dtype=float)
    var = np.var(x)
    if var <= 0:
        return np.nan
    beta = np.cov(x, y, ddof=0)[0, 1] / var
    resid = y - (y.mean() - beta * x.mean()) - beta * x
    if np.std(resid) < 1e-9:  # fully explained by rep -> no independent signal
        return 0.0
    ic = pd.Series(resid, index=sub.index).corr(sub[target], method="spearman")
    return float(ic) if not np.isnan(ic) else np.nan


def redundancy_report(
    data: pd.DataFrame,
    family: str,
    features: list[str],
    horizon: int,
    ic_strength: dict[str, float],
    threshold: float = 0.8,
    keep_eps: float = 0.03,
) -> pd.DataFrame:
    """Per-feature redundancy verdict for one family.

    ``ic_strength`` maps feature -> ``|IC t-stat|`` (from the main results) and
    decides each cluster's representative. Verdict: ``representante`` for the
    cluster leader, ``redundante`` when the incremental IC is below ``keep_eps``,
    else ``aporta`` (correlated but still adds independent signal).
    """
    target = f"fwd_{horizon}"
    cols = [f for f in features if f in data.columns]
    corr = feature_correlation(data, cols)
    clusters = cluster_features(corr, threshold)

    rows: list[dict] = []
    for cid, cluster in enumerate(clusters):
        rep = max(cluster, key=lambda f: ic_strength.get(f, 0.0))
        for feat in cluster:
            others = [c for c in cluster if c != feat]
            max_rho = float(corr.loc[feat, others].abs().max()) if others else np.nan
            inc = np.nan if feat == rep else incremental_ic(data, feat, rep, target)
            if feat == rep:
                verdict = "representante"
            elif not np.isnan(inc) and abs(inc) < keep_eps:
                verdict = "redundante"
            else:
                verdict = "aporta"
            rows.append({
                "family": family,
                "feature": feat,
                "cluster": cid,
                "representative": rep,
                "max_abs_corr": max_rho,
                "ic_incremental": inc,
                "verdict": verdict,
            })
    columns = ["family", "feature", "cluster", "representative", "max_abs_corr",
               "ic_incremental", "verdict"]
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------- #
# WS7 - Subgroup robustness
# --------------------------------------------------------------------------- #
def realized_vol(
    data: pd.DataFrame, window: int = 20, price_col: str = "close", contract_col: str = "contract"
) -> pd.Series:
    """Point-in-time realized vol: rolling std of ``Δclose`` per contract (as-of ``t``)."""
    return (
        data.groupby(contract_col, sort=False, group_keys=False)[price_col]
        .apply(lambda s: s.diff().rolling(window).std())
    )


def _tercile_labels(values: pd.Series, names: tuple[str, str, str]) -> pd.Series:
    """Label a numeric series by tercile, robust to ties (falls back to rank)."""
    try:
        return pd.qcut(values, 3, labels=list(names))
    except ValueError:  # too many duplicate edges -> rank then cut
        ranked = values.rank(method="first")
        return pd.qcut(ranked, 3, labels=list(names))


def _subgroup_ic(data: pd.DataFrame, feature: str, target: str, labels: pd.Series) -> list[dict]:
    """Descriptive Spearman IC of ``feature`` vs ``target`` within each label."""
    sub = pd.DataFrame({
        "f": data[feature].to_numpy(),
        "y": data[target].to_numpy(),
        "g": labels.to_numpy(),
    }).dropna()
    out: list[dict] = []
    for grp, gdf in sub.groupby("g", observed=True):
        ic = gdf["f"].corr(gdf["y"], method="spearman") if len(gdf) > 2 else np.nan
        out.append({"group": str(grp), "ic": float(ic) if not np.isnan(ic) else np.nan, "n": int(len(gdf))})
    return out


def conditional_grid(
    data: pd.DataFrame,
    family: str,
    primary: str,
    confirmator: str,
    horizon: int,
    *,
    p_buckets: int = 5,
    c_buckets: int = 3,
) -> pd.DataFrame:
    """Mean forward outcome on a 2-indicator grid (WS-G3).

    Rows are bucketed by ``primary`` (p_buckets, cheap->dear) and columns by
    ``confirmator`` (c_buckets); each cell is the mean forward return at
    ``horizon`` with its sample size. Descriptive (pooled) — an exploratory lens
    on interaction, e.g. "cheap AND confirmed by level" beating "cheap alone".
    Returns tidy rows ``family, primary, confirmator, p_bucket, c_bucket, mean_fwd, n``.
    """
    target = f"fwd_{horizon}"
    cols = ["family", "primary", "confirmator", "p_bucket", "c_bucket", "mean_fwd", "n"]
    if not {primary, confirmator, target}.issubset(data.columns) or primary == confirmator:
        return pd.DataFrame(columns=cols)

    d = data[[primary, confirmator, target]].dropna()
    if len(d) < p_buckets * c_buckets * 10:
        return pd.DataFrame(columns=cols)

    # Rank before qcut so duplicate values don't collapse the bucket edges.
    d = d.assign(
        _p=pd.qcut(d[primary].rank(method="first"), p_buckets, labels=range(1, p_buckets + 1)),
        _c=pd.qcut(d[confirmator].rank(method="first"), c_buckets, labels=range(1, c_buckets + 1)),
    )
    rows: list[dict] = []
    for (p, c), g in d.groupby(["_p", "_c"], observed=True):
        rows.append({
            "family": family,
            "primary": primary,
            "confirmator": confirmator,
            "p_bucket": int(p),
            "c_bucket": int(c),
            "mean_fwd": float(g[target].mean()),
            "n": int(len(g)),
        })
    return pd.DataFrame(rows, columns=cols)


def subgroup_report(
    data: pd.DataFrame,
    family: str,
    feature: str,
    horizon: int,
    vol_window: int = 20,
) -> pd.DataFrame:
    """IC of ``feature`` broken down by era / vol regime / life-phase / month."""
    target = f"fwd_{horizon}"
    if feature not in data.columns or target not in data.columns:
        return pd.DataFrame(columns=["family", "feature", "dimension", "group", "ic", "n"])

    d = data.copy()
    dates = pd.to_datetime(d["date"])
    dims: dict[str, pd.Series] = {}
    dims["era"] = pd.Series(np.where(dates.dt.year >= 2020, "\u22652020", "<2020"), index=d.index)
    dims["vol"] = _tercile_labels(realized_vol(d, vol_window), ("vol baja", "vol media", "vol alta"))
    if "dte" in d.columns:
        dims["fase"] = _tercile_labels(d["dte"], ("pr\u00f3xima", "media", "lejana"))
    dims["mes"] = dates.dt.month.astype("Int64").astype(str)

    rows: list[dict] = []
    for dim, labels in dims.items():
        for rec in _subgroup_ic(d, feature, target, labels):
            rows.append({"family": family, "feature": feature, "dimension": dim, **rec})
    columns = ["family", "feature", "dimension", "group", "ic", "n"]
    return pd.DataFrame(rows, columns=columns)
