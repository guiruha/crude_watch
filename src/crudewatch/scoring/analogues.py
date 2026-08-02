"""Empirical base rates for the current setup: what analogous products did next.

Given an instrument scored as of a date, we look at every historical observation
of the same family that sat in the **same regime and the same level bucket** (the
same indicator configuration) and summarise what happened over the next
``horizon`` bars — average forward move, hit rate, and the outcome signed by the
composite direction. This grounds the composite in "what products with these same
indicators have historically returned", not just the model's number.

Point-in-time: only rows whose forward window had already resolved by the scoring
date enter the cohort, so the base rates never peek past ``as_of``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from crudewatch.research.dataset import COST_STUB_POINTS
from crudewatch.scoring.blocks import (
    FamilyCalibrator,
    fit_calibrator,
    percentile,
    signed_pct,
)
from crudewatch.scoring.score import compute_blocks, compute_opportunity, weights_for, _attach_bar_counts

# Level-composite bucket edges (+ = expensive) and their Spanish labels.
LEVEL_BIN_EDGES: tuple[float, ...] = (-60.0, -20.0, 20.0, 60.0)
LEVEL_BIN_LABELS: tuple[str, ...] = (
    "muy barato",
    "barato",
    "neutral",
    "caro",
    "muy caro",
)

MONTH_COLUMNS: tuple[str, ...] = (
    "month",
    "near_month",
    "wti_month",
    "front_month",
)


def _level_bin(level: float) -> int:
    """Index ``0..4`` of the level-composite bucket for a scalar level score."""
    if level != level:
        return 2
    return int(np.digitize([level], LEVEL_BIN_EDGES)[0])


def _vec_level(data: pd.DataFrame, calibrator: FamilyCalibrator) -> np.ndarray:
    """Vectorised level composite (matches ``blocks.block_level``) for every row."""
    n = len(data)
    rows: list[np.ndarray] = []
    ecdf = calibrator.ecdf.get("level_pct", np.array([], dtype=float))
    if "level_pct" in data.columns and len(ecdf):
        lp = data["level_pct"].to_numpy(dtype=float)
        ranks = np.searchsorted(ecdf, lp, side="right") / len(ecdf)
        rows.append(np.where(np.isnan(lp), np.nan, (2.0 * ranks - 1.0) * 100.0))
    for feat in ("level_z", "z_10", "z_20", "z_50", "keltner_dist_20"):
        if feat in data.columns:
            rows.append(100.0 * np.tanh(data[feat].to_numpy(dtype=float) / 2.0))
    if not rows:
        return np.full(n, np.nan)
    stack = np.vstack(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stack, axis=0)


def _vec_regime(er: np.ndarray, er_lo: float, er_hi: float) -> np.ndarray:
    out = np.full(len(er), "transition", dtype=object)
    out[er <= er_lo] = "range"
    out[er >= er_hi] = "trend"
    return out


def _tenor_bucket(dte: float) -> str:
    if dte != dte:
        return "sin tenor"
    if dte <= 45:
        return "front"
    if dte <= 180:
        return "2-6m"
    if dte <= 365:
        return "6-12m"
    return "+1y"


def _month_series(data: pd.DataFrame) -> pd.Series:
    for col in MONTH_COLUMNS:
        if col in data.columns:
            return data[col]
    return pd.Series(np.nan, index=data.index)


def _far_leg_series(data: pd.DataFrame) -> pd.Series:
    if "far_month" in data.columns:
        return data["far_month"]
    return pd.Series(np.nan, index=data.index)


def _same_or_missing(values: np.ndarray, target) -> np.ndarray:
    if target != target:
        return np.ones(len(values), dtype=bool)
    return values == target


def _non_overlapping_mask(rows: pd.DataFrame, horizon: int) -> np.ndarray:
    if {"date", "_target_date"}.issubset(rows.columns):
        ordered = rows.reset_index(drop=True).reset_index(names="_pos").sort_values(["date", "contract"])
        keep = np.zeros(len(rows), dtype=bool)
        last_exit: pd.Timestamp | None = None
        for _, row in ordered.iterrows():
            entry = pd.Timestamp(row["date"])
            exit_date = pd.Timestamp(row["_target_date"])
            if pd.isna(entry) or pd.isna(exit_date):
                continue
            if last_exit is None or entry > last_exit:
                keep[int(row["_pos"])] = True
                last_exit = exit_date
        return keep

    keep = np.zeros(len(rows), dtype=bool)
    last_by_contract: dict[str, int] = {}
    for i, (contract, bar_idx) in enumerate(rows[["contract", "_bar_idx"]].to_numpy()):
        contract = str(contract)
        bar_idx = int(bar_idx)
        last = last_by_contract.get(contract)
        if last is None or bar_idx - last >= horizon:
            keep[i] = True
            last_by_contract[contract] = bar_idx
    return keep


def _annualized_sample_sharpe(samples: np.ndarray, horizon: int) -> float:
    clean = samples[~np.isnan(samples)]
    if len(clean) < 2:
        return float("nan")
    sd = float(np.std(clean, ddof=1))
    if sd <= 0 or sd != sd:
        return float("nan")
    return float((np.mean(clean) / sd) * np.sqrt(252.0 / max(int(horizon), 1)))


def analogous_outcomes(
    data: pd.DataFrame,
    family: str,
    contract: str,
    horizon: int = 25,
    as_of: pd.Timestamp | None = None,
    calibrator: FamilyCalibrator | None = None,
    action_threshold: float = 0.0,
    cost: float | None = None,
) -> dict:
    """Historical forward outcomes of analogous setups (same regime + level bucket).

    Returns a dict with the matched cohort's effective size, the average/hit
    forward move, and the outcome signed by the composite direction. Empty
    (``n=0``) when no analogous history exists yet.
    """
    ordered = data.sort_values(["contract", "date"]).copy()
    dates = pd.to_datetime(ordered["date"])
    target_dates = ordered.groupby("contract", sort=False)["date"].shift(-horizon)

    if as_of is not None:
        as_of = pd.Timestamp(as_of)
        data = ordered.loc[dates <= as_of]
        target_dates = target_dates.loc[data.index]
        cal = calibrator if calibrator is not None else fit_calibrator(
            data, family, horizon, outcome_asof=as_of
        )
    else:
        data = ordered
        target_dates = target_dates.loc[data.index]
        cal = calibrator if calibrator is not None else fit_calibrator(data, family, horizon)

    enriched = _attach_bar_counts(data)
    sub = enriched.loc[enriched["contract"] == contract]
    empty = {"n": 0, "horizon": horizon}
    if sub.empty:
        return empty
    row = sub.iloc[-1]
    blocks = compute_blocks(row, cal, int(row["_n_contract_bars"]))
    opportunity = compute_opportunity(blocks, row, cal, weights_for(family))
    side = 1.0 if opportunity > action_threshold else -1.0 if opportunity < -action_threshold else 0.0
    cost_points = COST_STUB_POINTS.get(family, 0.0) if cost is None else float(cost)
    target, mfe_col, mae_col = f"fwd_{horizon}", f"mfe_{horizon}", f"mae_{horizon}"
    if target not in enriched.columns:
        return empty

    regime = _vec_regime(enriched["er_20"].to_numpy(dtype=float), cal.er_lo, cal.er_hi)
    level = _vec_level(enriched, cal)
    row_bin = _level_bin(float(blocks.level))
    bins = np.digitize(np.where(np.isnan(level), -999.0, level), LEVEL_BIN_EDGES)
    dte = enriched["dte"].to_numpy(dtype=float) if "dte" in enriched.columns else np.full(len(enriched), np.nan)
    tenor = np.array([_tenor_bucket(x) for x in dte], dtype=object)
    row_tenor = _tenor_bucket(float(row.get("dte", np.nan)))
    months = _month_series(enriched).to_numpy()
    row_month = row.get("month", row.get("near_month", row.get("wti_month", row.get("front_month", np.nan))))
    far_legs = _far_leg_series(enriched).to_numpy()
    row_far_leg = row.get("far_month", np.nan)

    fwd = enriched[target].to_numpy(dtype=float)
    # Only realised outcomes (point-in-time) count toward the base rate.
    if as_of is not None:
        resolved = pd.to_datetime(target_dates.reindex(enriched.index)).to_numpy()
        fwd = np.where(resolved <= np.datetime64(as_of), fwd, np.nan)

    match = (
        (regime == blocks.regime)
        & (bins == row_bin)
        & (tenor == row_tenor)
        & _same_or_missing(months, row_month)
        & _same_or_missing(far_legs, row_far_leg)
        & ~np.isnan(fwd)
        & ~np.isnan(level)
    )
    n_raw = int(match.sum())
    matched_rows = (
        enriched.loc[match, ["contract", "date"]]
        .assign(_bar_idx=enriched.groupby("contract", sort=False).cumcount().loc[match].to_numpy())
        .assign(_target_date=target_dates.reindex(enriched.index).loc[match].to_numpy())
        .sort_values(["contract", "_bar_idx"])
    )
    keep = _non_overlapping_mask(matched_rows, horizon) if n_raw else np.array([], dtype=bool)
    matched_index = matched_rows.index.to_numpy()[keep]
    effective = enriched.index.isin(matched_index)
    n = int(effective.sum())
    if n == 0:
        return {
            **empty,
            "n_raw": n_raw,
            "regime": blocks.regime,
            "level_bin": LEVEL_BIN_LABELS[row_bin],
            "tenor_bucket": row_tenor,
            "month": None if row_month != row_month else int(row_month),
            "far_leg": None if row_far_leg != row_far_leg else int(row_far_leg),
            "side": side,
            "opportunity": float(opportunity),
            "action_threshold": float(action_threshold),
            "cost": float(cost_points),
        }

    fwd_c = fwd[effective]
    matched_effective = enriched.loc[effective]
    dates_effective = pd.to_datetime(matched_effective["date"])
    result = {
        "n": n,
        "n_raw": n_raw,
        "vintage_count": int(matched_effective["vintage"].nunique()) if "vintage" in matched_effective else 0,
        "year_count": int(dates_effective.dt.year.nunique()) if len(dates_effective) else 0,
        "horizon": horizon,
        "regime": blocks.regime,
        "level_bin": LEVEL_BIN_LABELS[row_bin],
        "tenor_bucket": row_tenor,
        "month": None if row_month != row_month else int(row_month),
        "far_leg": None if row_far_leg != row_far_leg else int(row_far_leg),
        "side": side,
        "opportunity": float(opportunity),
        "action_threshold": float(action_threshold),
        "cost": float(cost_points),
        "avg_fwd": float(np.mean(fwd_c)),
        "median_fwd": float(np.median(fwd_c)),
        "up_rate": float(np.mean(fwd_c > 0)),
        "fwd_samples": fwd_c.astype(float).tolist(),
    }
    if side != 0.0 and {mfe_col, mae_col}.issubset(enriched.columns):
        mfe_c = enriched[mfe_col].to_numpy(dtype=float)[effective]
        mae_c = enriched[mae_col].to_numpy(dtype=float)[effective]
        aligned_gross = side * fwd_c
        aligned = aligned_gross - cost_points
        favourable = np.where(side > 0, mfe_c, -mae_c)
        adverse = np.where(side > 0, mae_c, -mfe_c)
        adverse_loss = np.abs(np.minimum(adverse, 0.0))
        result["avg_aligned_gross"] = float(np.nanmean(aligned_gross))
        result["avg_aligned"] = float(np.nanmean(aligned))
        result["median_aligned"] = float(np.nanmedian(aligned))
        result["aligned_win_rate"] = float(np.mean(aligned > 0))
        result["sharpe_aligned"] = _annualized_sample_sharpe(aligned, horizon)
        result["avg_mfe"] = float(np.nanmean(favourable))
        result["avg_mae"] = float(np.nanmean(adverse))
        result["mae_p50"] = float(np.nanpercentile(adverse_loss, 50))
        result["mae_p80"] = float(np.nanpercentile(adverse_loss, 80))
        result["mfe_p50"] = float(np.nanpercentile(favourable, 50))
        result["mfe_p80"] = float(np.nanpercentile(favourable, 80))
        result["aligned_samples"] = aligned.astype(float).tolist()
    return result
