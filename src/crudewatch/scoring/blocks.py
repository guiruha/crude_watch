"""Per-block calculators and family-level calibration for the Opportunity Score.

Calibration is pooled over the family's enriched history (``build_dataset``
output). Features and forward outcomes are as-of ``t``; the calibrator never
peeks at future bars when scoring a live row.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BusinessDay

from crudewatch.research.dataset import regime_thresholds as _regime_thresholds

ECDF_FEATURES: tuple[str, ...] = (
    "er_20",
    "slope_20",
    "macd_hist",
    "level_pct",
    "mom_decel_10",
    "ema_align",
    "mom_5",
    "mom_10",
    "mom_20",
    "variance_ratio_5",
    "autocorr_20",
    "macd_div",
    "rsi_div_14",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_METRICS_PATH = _REPO_ROOT / "docs" / "reports" / "backtest" / "research_metrics.csv"


def _resource_root() -> Path:
    env = os.environ.get("CRUDEWATCH_RESOURCE_DIR")
    return Path(env) if env else _REPO_ROOT


def percentile(ecdf_sorted: np.ndarray, value: float) -> float:
    """Fraction of pooled history ``<= value`` in ``[0, 1]``; ``0.5`` if NaN/empty."""
    if ecdf_sorted is None or len(ecdf_sorted) == 0:
        return 0.5
    if value != value:
        return 0.5
    idx = np.searchsorted(ecdf_sorted, value, side="right")
    return float(idx / len(ecdf_sorted))


def signed_pct(p: float) -> float:
    """Map a percentile in ``[0, 1]`` to ``[-100, +100]``."""
    if p != p:
        return 0.0
    return float((2.0 * p - 1.0) * 100.0)


def _safe_tanh(x: float, scale: float = 2.0) -> float:
    if x != x:
        return 0.0
    return float(100.0 * np.tanh(x / scale))


def _clip01(x: float) -> float:
    if x != x:
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


@dataclass
class FamilyCalibrator:
    """Pooled family history used to percentile-normalise live features."""

    family: str
    horizon: int
    er_lo: float
    er_hi: float
    ecdf: dict[str, np.ndarray]
    p_rev_cheap: float
    p_rev_dear: float
    p_cont_up: float
    p_cont_dn: float
    ic_t_level: float


def _sorted_ecdf(values: np.ndarray) -> np.ndarray:
    clean = values[~np.isnan(values)]
    return np.sort(clean) if len(clean) else np.array([], dtype=float)


def _read_ic_t_level(family: str, horizon: int) -> float:
    path = _resource_root() / "docs" / "reports" / "backtest" / "research_metrics.csv"
    if not path.is_file():
        return float("nan")
    metrics = pd.read_csv(path)
    mask = (
        (metrics["family"] == family)
        & (metrics["feature"] == "level_z")
        & (metrics["horizon"] == horizon)
        & (metrics["group"].str.upper() == "ALL")
    )
    rows = metrics.loc[mask, "ic_t"]
    if rows.empty:
        return float("nan")
    return float(abs(rows.iloc[0]))


def _conditional_rate(num: np.ndarray, den: np.ndarray) -> float:
    if len(den) == 0:
        return float("nan")
    return float(np.mean(num))


def fit_calibrator(
    data: pd.DataFrame,
    family: str,
    horizon: int = 25,
    outcome_asof: pd.Timestamp | None = None,
) -> FamilyCalibrator:
    """Fit pooled percentile and conditional-probability tables for one family.

    ``outcome_asof`` makes the calibration point-in-time for a historical scoring
    date: the conditional reversion / continuation probabilities then only use
    rows whose forward window had already been *realised* by that date (a row at
    ``t`` resolves ``horizon`` business days later), so the probabilities never
    peek past ``outcome_asof``. Percentile / regime cut-offs already rely solely
    on as-of-``t`` features, so they are unaffected.
    """
    target = f"fwd_{horizon}"
    er = data["er_20"].to_numpy(dtype=float)
    # ER carries warmup NaNs; np.quantile would propagate them into the tercile
    # cut-offs (making every row "transition"). Drop NaNs first, as the gated
    # backtest does before calling _regime_thresholds.
    er_valid = er[~np.isnan(er)]
    er_lo, er_hi = _regime_thresholds(er_valid) if len(er_valid) else (np.nan, np.nan)

    ecdf: dict[str, np.ndarray] = {}
    for feat in ECDF_FEATURES:
        if feat not in data.columns:
            ecdf[feat] = np.array([], dtype=float)
            continue
        ecdf[feat] = _sorted_ecdf(data[feat].to_numpy(dtype=float))
    if "mom_decel_10" in data.columns:
        ecdf["neg_mom_decel_10"] = _sorted_ecdf(-data["mom_decel_10"].to_numpy(dtype=float))
    else:
        ecdf["neg_mom_decel_10"] = np.array([], dtype=float)
    if "er_drop_20" in data.columns:
        ecdf["neg_er_drop_20"] = _sorted_ecdf(-data["er_drop_20"].to_numpy(dtype=float))
    else:
        ecdf["neg_er_drop_20"] = np.array([], dtype=float)

    level_pct = data["level_pct"].to_numpy(dtype=float) if "level_pct" in data.columns else np.full(len(data), np.nan)
    slope = data["slope_20"].to_numpy(dtype=float) if "slope_20" in data.columns else np.full(len(data), np.nan)
    fwd = data[target].to_numpy(dtype=float) if target in data.columns else np.full(len(data), np.nan)

    # Point-in-time probabilities: drop outcomes that had not yet resolved by the
    # scoring date (a row at t is realised ~horizon business days later).
    if outcome_asof is not None and "date" in data.columns:
        realised = (pd.to_datetime(data["date"]) + BusinessDay(horizon)).to_numpy()
        fwd = np.where(realised <= np.datetime64(pd.Timestamp(outcome_asof)), fwd, np.nan)

    range_mask = er <= er_lo
    trend_mask = er >= er_hi

    cheap_mask = range_mask & (level_pct <= 1.0 / 3.0) & ~np.isnan(level_pct)
    dear_mask = range_mask & (level_pct >= 2.0 / 3.0) & ~np.isnan(level_pct)
    p_rev_cheap = _conditional_rate(fwd[cheap_mask] > 0, fwd[cheap_mask])
    p_rev_dear = _conditional_rate(fwd[dear_mask] < 0, fwd[dear_mask])

    trend_slope = slope[trend_mask]
    if len(trend_slope) >= 3:
        s_lo, s_hi = np.quantile(trend_slope[~np.isnan(trend_slope)], [1.0 / 3.0, 2.0 / 3.0])
        up_mask = trend_mask & (slope >= s_hi)
        dn_mask = trend_mask & (slope <= s_lo)
    else:
        up_mask = np.zeros(len(data), dtype=bool)
        dn_mask = np.zeros(len(data), dtype=bool)

    p_cont_up = _conditional_rate(fwd[up_mask] > 0, fwd[up_mask])
    p_cont_dn = _conditional_rate(fwd[dn_mask] < 0, fwd[dn_mask])

    return FamilyCalibrator(
        family=family,
        horizon=horizon,
        er_lo=float(er_lo),
        er_hi=float(er_hi),
        ecdf=ecdf,
        p_rev_cheap=p_rev_cheap,
        p_rev_dear=p_rev_dear,
        p_cont_up=p_cont_up,
        p_cont_dn=p_cont_dn,
        ic_t_level=_read_ic_t_level(family, horizon),
    )


def regime_label(er_20: float, calibrator: FamilyCalibrator) -> str:
    """Classify efficiency ratio into range / transition / trend."""
    if er_20 != er_20:
        return "transition"
    if er_20 <= calibrator.er_lo:
        return "range"
    if er_20 >= calibrator.er_hi:
        return "trend"
    return "transition"


def block_trendiness(row: pd.Series, calibrator: FamilyCalibrator) -> float:
    """Bloque A: how much *structure / persistence* the market has (regime), from
    the Efficiency Ratio, the variance ratio and return autocorrelation — kept
    distinct from the trend-*quality* metrics of Bloque C (R², persistence)."""
    parts: list[float] = []
    for feat in ("er_20", "variance_ratio_5", "autocorr_20"):
        val = row.get(feat, np.nan)
        if val == val:
            parts.append(percentile(calibrator.ecdf.get(feat, np.array([])), float(val)) * 100.0)
    if not parts:
        return 0.0
    return float(np.mean(parts))


def block_direction(row: pd.Series, calibrator: FamilyCalibrator) -> float:
    parts: list[float] = []
    for feat in ("slope_20", "macd_hist", "ema_align", "mom_5", "mom_10", "mom_20"):
        val = row.get(feat, np.nan)
        if val == val:
            parts.append(signed_pct(percentile(calibrator.ecdf.get(feat, np.array([])), float(val))))
    if not parts:
        return 0.0
    return float(np.mean(parts))


def block_strength(row: pd.Series, direction: float) -> float:
    """Bloque C: trend *quality* (0–100) from its own metrics — the regression R²
    (linearity) and the directional persistence (% of sessions the same way) —
    modulated by the magnitude of the direction. Distinct from Bloque A, which
    only asks whether a regime exists at all."""
    parts: list[float] = []
    for feat in ("r2_20", "dir_persistence_20"):
        val = row.get(feat, np.nan)
        if val == val:
            parts.append(float(np.clip(val, 0.0, 1.0)))
    quality = float(np.mean(parts)) if parts else 0.0
    dir_mag = min(abs(direction) / 100.0, 1.0)
    return float(np.clip(100.0 * quality * (0.4 + 0.6 * dir_mag), 0.0, 100.0))


def block_level(row: pd.Series, calibrator: FamilyCalibrator) -> float:
    parts: list[float] = []
    level_pct = row.get("level_pct", np.nan)
    if level_pct == level_pct:
        parts.append(signed_pct(percentile(calibrator.ecdf.get("level_pct", np.array([])), float(level_pct))))
    for feat in ("level_z", "z_10", "z_20", "z_50", "keltner_dist_20"):
        val = row.get(feat, np.nan)
        if val == val:
            parts.append(_safe_tanh(float(val)))
    if not parts:
        return 0.0
    return float(np.mean(parts))


def _reversion_confirmations(level: float, row: pd.Series, calibrator: FamilyCalibrator) -> list[float]:
    """Signals in [0,1] that the extreme reverts, oriented by side (cheap vs dear).

    Equal-weighted, exhaustion / oscillator / divergence indicators that are not
    shown in any other block (kept distinct from Dirección / Fuerza / Régimen /
    Nivel). NaN inputs are skipped so a missing feature does not drag the mean.
    """
    dear = level > 0  # expensive -> expect a move down; cheap -> up
    out: list[float] = []

    rsi2 = row.get("rsi_2", np.nan)
    if rsi2 == rsi2:
        out.append(_clip01((rsi2 - 80.0) / 20.0 if dear else (20.0 - rsi2) / 20.0))
    rsi14 = row.get("rsi_14", np.nan)
    if rsi14 == rsi14:
        out.append(_clip01((rsi14 - 70.0) / 30.0 if dear else (30.0 - rsi14) / 30.0))
    pctb = row.get("pctb_20_2", np.nan)
    if pctb == pctb:
        out.append(_clip01((pctb - 0.8) / 0.2 if dear else (0.2 - pctb) / 0.2))
    div = row.get("rsi_div_14", np.nan)
    if div == div:
        out.append(_clip01(-div / 1.5 if dear else div / 1.5))
    decel = row.get("mom_decel_10", np.nan)
    if decel == decel:
        # Exhaustion of the prevailing move (family percentile of -deceleration).
        out.append(_clip01(percentile(calibrator.ecdf.get("neg_mom_decel_10", np.array([])), -float(decel))))
    return out


def block_p_reversion(level: float, row: pd.Series, calibrator: FamilyCalibrator) -> float:
    if level < 0:
        p_base = calibrator.p_rev_cheap
    elif level > 0:
        p_base = calibrator.p_rev_dear
    else:
        return 0.5
    if p_base != p_base:
        return 0.5
    confs = _reversion_confirmations(level, row, calibrator)
    confirmation = float(np.mean(confs)) if confs else 0.5
    conf = 0.75 + 0.5 * confirmation             # in [0.75, 1.25]
    w = float(np.clip(min(abs(level) / 60.0, 1.0) * conf, 0.0, 1.0))
    return float(0.5 + (p_base - 0.5) * w)


def block_p_continuation(direction: float, trendiness: float, calibrator: FamilyCalibrator) -> float:
    if direction > 0:
        p_base = calibrator.p_cont_up
    elif direction < 0:
        p_base = calibrator.p_cont_dn
    else:
        return 0.5
    if p_base != p_base:
        return 0.5
    w = trendiness / 100.0
    return float(0.5 + (p_base - 0.5) * w)


def block_confidence(
    row: pd.Series,
    regime: str,
    n_contract_bars: int,
    calibrator: FamilyCalibrator,
) -> float:
    level_pct = row.get("level_pct", np.nan)
    f_level = 1.0 if level_pct == level_pct else 0.4
    f_bars = min(n_contract_bars / 40.0, 1.0)
    f_regime = 1.0 if regime in ("range", "trend") else 0.5
    ic_t = calibrator.ic_t_level
    f_stability = min(ic_t / 3.0, 1.0) if ic_t == ic_t else 0.6
    return float(100.0 * f_level * f_bars * f_regime * f_stability)


def timing_term(row: pd.Series, calibrator: FamilyCalibrator) -> float:
    mom = row.get("mom_decel_10", np.nan)
    if mom != mom:
        return 0.5
    p = percentile(calibrator.ecdf.get("neg_mom_decel_10", np.array([])), -float(mom))
    return _clip01(p)


def vol_term(row: pd.Series) -> float:
    vr = row.get("vol_ratio", np.nan)
    if vr != vr:
        return 0.5
    return _clip01(1.2 - float(vr))
