"""Empirical base rates for the current setup: what analogous products did next.

Given an instrument scored as of a date, we look at every historical observation
of the same family that sat in the **same regime and the same level bucket** (the
same indicator configuration) and summarise what happened over the next
``horizon`` bars — average forward move, hit rate, and the outcome *aligned with
the suggested action*. This grounds the Opportunity Score in "what products with
these same indicators have historically returned", not just the model's number.

Point-in-time: only rows whose forward window had already resolved by the scoring
date enter the cohort, so the base rates never peek past ``as_of``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BusinessDay

from crudewatch.scoring.blocks import (
    FamilyCalibrator,
    fit_calibrator,
    percentile,
    signed_pct,
)
from crudewatch.scoring.score import compute_blocks, compute_opportunity, _attach_bar_counts

# Level-composite bucket edges (+ = expensive) and their Spanish labels.
LEVEL_BIN_EDGES: tuple[float, ...] = (-60.0, -20.0, 20.0, 60.0)
LEVEL_BIN_LABELS: tuple[str, ...] = (
    "muy barato",
    "barato",
    "neutral",
    "caro",
    "muy caro",
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


def analogous_outcomes(
    data: pd.DataFrame,
    family: str,
    contract: str,
    horizon: int = 25,
    as_of: pd.Timestamp | None = None,
    calibrator: FamilyCalibrator | None = None,
) -> dict:
    """Historical forward outcomes of analogous setups (same regime + level bucket).

    Returns a dict with the matched cohort's size, the average/hit forward move,
    and the outcome aligned with the suggested action (following ``sign`` of the
    opportunity). Empty (``n=0``) when no analogous history exists yet.
    """
    if as_of is not None:
        as_of = pd.Timestamp(as_of)
        data = data.loc[pd.to_datetime(data["date"]) <= as_of]
        cal = calibrator if calibrator is not None else fit_calibrator(
            data, family, horizon, outcome_asof=as_of
        )
    else:
        cal = calibrator if calibrator is not None else fit_calibrator(data, family, horizon)

    enriched = _attach_bar_counts(data)
    sub = enriched.loc[enriched["contract"] == contract]
    empty = {"n": 0, "horizon": horizon}
    if sub.empty:
        return empty
    row = sub.iloc[-1]
    blocks = compute_blocks(row, cal, int(row["_n_contract_bars"]))
    opportunity = compute_opportunity(blocks, row, cal)
    side = 1.0 if opportunity > 0 else -1.0 if opportunity < 0 else 0.0
    target, mfe_col, mae_col = f"fwd_{horizon}", f"mfe_{horizon}", f"mae_{horizon}"
    if target not in enriched.columns:
        return empty

    regime = _vec_regime(enriched["er_20"].to_numpy(dtype=float), cal.er_lo, cal.er_hi)
    level = _vec_level(enriched, cal)
    row_bin = _level_bin(float(blocks.level))
    bins = np.digitize(np.where(np.isnan(level), -999.0, level), LEVEL_BIN_EDGES)

    fwd = enriched[target].to_numpy(dtype=float)
    # Only realised outcomes (point-in-time) count toward the base rate.
    if as_of is not None:
        realised = (pd.to_datetime(enriched["date"]) + BusinessDay(horizon)).to_numpy()
        fwd = np.where(realised <= np.datetime64(as_of), fwd, np.nan)

    match = (regime == blocks.regime) & (bins == row_bin) & ~np.isnan(fwd) & ~np.isnan(level)
    n = int(match.sum())
    if n == 0:
        return {**empty, "regime": blocks.regime, "level_bin": LEVEL_BIN_LABELS[row_bin], "side": side}

    fwd_c = fwd[match]
    result = {
        "n": n,
        "horizon": horizon,
        "regime": blocks.regime,
        "level_bin": LEVEL_BIN_LABELS[row_bin],
        "side": side,
        "avg_fwd": float(np.mean(fwd_c)),
        "median_fwd": float(np.median(fwd_c)),
        "up_rate": float(np.mean(fwd_c > 0)),
        "fwd_samples": fwd_c.astype(float).tolist(),
    }
    if side != 0.0 and {mfe_col, mae_col}.issubset(enriched.columns):
        mfe_c = enriched[mfe_col].to_numpy(dtype=float)[match]
        mae_c = enriched[mae_col].to_numpy(dtype=float)[match]
        aligned = side * fwd_c
        favourable = np.where(side > 0, mfe_c, -mae_c)
        adverse = np.where(side > 0, mae_c, -mfe_c)
        result["avg_aligned"] = float(np.nanmean(aligned))
        result["aligned_win_rate"] = float(np.mean(aligned > 0))
        result["avg_mfe"] = float(np.nanmean(favourable))
        result["avg_mae"] = float(np.nanmean(adverse))
        result["aligned_samples"] = aligned.astype(float).tolist()
    return result
