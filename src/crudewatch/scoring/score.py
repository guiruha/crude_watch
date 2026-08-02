"""Composite stretch/persistence score and public scoring API."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crudewatch.scoring.blocks import (
    FamilyCalibrator,
    _clip01,
    block_direction,
    block_level,
    block_p_continuation,
    block_p_reversion,
    block_strength,
    block_trendiness,
    fit_calibrator,
    percentile,
    regime_label,
    timing_term,
    vol_term,
)

# Equal weighting: every active term in a regime contributes the same. The
# individual raw indicators inside each block are also averaged equally in
# ``blocks.py``. Historical evidence is shown separately from the composite.
BASE_RANGE_TERM_KEYS: tuple[str, ...] = ("rev_term", "lvl_term", "timing_term", "vol_term")
BASE_TREND_TERM_KEYS: tuple[str, ...] = ("dir_term", "qual_term", "cont_term", "ext_low")


def weights_for(family: str) -> dict:
    """Equal weights for every family.

    Weight-search artifacts may exist in the repository for research, but the
    live dashboard intentionally runs one transparent equal-weight configuration.
    """
    return WEIGHTS


FEATURE_SNAPSHOT: tuple[str, ...] = (
    "z_10", "z_20", "z_50", "pctb_20_2", "pctb_10_1_5",
    "keltner_dist_20", "rsi_2", "rsi_14", "rsi_div_14", "macd_div",
    "mom_decel_10", "er_drop_20", "vol_ratio", "slope_20", "macd_hist",
    "ema_align", "mom_5", "mom_10", "mom_20", "er_20", "r2_20",
    "variance_ratio_5", "autocorr_20", "dir_persistence_20",
    "level_z", "level_pct",
)


@dataclass(frozen=True)
class BlockScores:
    regime: str
    trendiness: float
    direction: float
    strength: float
    level: float
    p_reversion: float
    p_continuation: float


@dataclass(frozen=True)
class InstrumentScore:
    family: str
    contract: str
    date: pd.Timestamp
    close: float
    dte: float
    blocks: BlockScores
    opportunity: float
    rationale: list[str]
    risks: list[str]
    features: dict[str, float]
    percentiles: dict[str, float] = field(default_factory=dict)


def _clip_opportunity(x: float) -> float:
    if x != x:
        return 0.0
    return float(np.clip(x, -100.0, 100.0))


def _rev_term(p_reversion: float) -> float:
    p = 0.5 if p_reversion != p_reversion else p_reversion
    return float(np.clip((p - 0.5) / 0.5, 0.0, 1.0))


def _cont_term(p_continuation: float) -> float:
    p = 0.5 if p_continuation != p_continuation else p_continuation
    return float(np.clip((p - 0.5) / 0.5, 0.0, 1.0))


def _lvl_term(level: float) -> float:
    return min(abs(level) / 100.0, 1.0)


def _dir_px_range(level: float) -> float:
    return -1.0 if level > 0 else 1.0


def _dir_px_trend(direction: float) -> float:
    return 1.0 if direction > 0 else -1.0


def _agree_mag(row, cal, feat, dir_px):
    val = row.get(feat, np.nan)
    if val != val:
        return 0.0
    p = percentile(cal.ecdf.get(feat, np.array([])), float(val))
    signed = 2.0 * p - 1.0
    s = 1.0 if signed > 0 else (-1.0 if signed < 0 else 0.0)
    return abs(signed) if s == dir_px else 0.0


def _signed_mag(row, cal, feat):
    """Ungated magnitude of ``feat``'s signed percentile, in ``[0, 1]``.

    Same quantity :func:`_agree_mag` returns, minus the agreement gate. Used
    where a feature carries an *unconditional* relationship to the forward
    outcome — gating such a feature on agreement with the level-implied
    reversion direction discards roughly half its information and drives its
    fitted weight to zero. See the bucket-sweep findings report.
    """
    val = row.get(feat, np.nan)
    if val != val:
        return 0.0
    p = percentile(cal.ecdf.get(feat, np.array([])), float(val))
    return abs(2.0 * p - 1.0)


def _mag_ecdf(row, cal, ecdf_key, src, neg=False, invert=False):
    val = row.get(src, np.nan)
    if val != val:
        return 0.0
    x = -float(val) if neg else float(val)
    p = percentile(cal.ecdf.get(ecdf_key, np.array([])), x)
    return _clip01(1.0 - p if invert else p)


RANGE_TERM_KEYS: tuple[str, ...] = BASE_RANGE_TERM_KEYS
TREND_TERM_KEYS: tuple[str, ...] = BASE_TREND_TERM_KEYS


RANGE_TERMS: dict = {
    "rev_term": lambda b, row, cal, level, direction: _rev_term(b.p_reversion),
    "lvl_term": lambda b, row, cal, level, direction: _lvl_term(level),
    "timing_term": lambda b, row, cal, level, direction: timing_term(row, cal),
    "vol_term": lambda b, row, cal, level, direction: vol_term(row),
}
TREND_TERMS: dict = {
    "dir_term": lambda b, row, cal, level, direction: min(abs(direction) / 100.0, 1.0),
    "qual_term": lambda b, row, cal, level, direction: b.strength / 100.0,
    "cont_term": lambda b, row, cal, level, direction: _cont_term(b.p_continuation),
    "ext_low": lambda b, row, cal, level, direction: 1.0 - _lvl_term(level),
}

RANGE_TERMS.update({
    "macd_div_term": lambda b, row, cal, level, direction: _agree_mag(row, cal, "macd_div", _dir_px_range(level)),
    # Ungated on purpose: the bucket sweep found rsi_div_14 monotone in bucket at
    # all seven horizons in three families with a fully consistent sign, i.e. an
    # unconditional relationship. Gating it on the level-implied reversion
    # direction (as macd_div_term still is) discarded that and fitted to ~0.
    "rsi_div_term": lambda b, row, cal, level, direction: _signed_mag(row, cal, "rsi_div_14"),
    "mom_decel_term": lambda b, row, cal, level, direction: _mag_ecdf(row, cal, "neg_mom_decel_10", neg=True, src="mom_decel_10"),
    "er_drop_term": lambda b, row, cal, level, direction: _mag_ecdf(row, cal, "neg_er_drop_20", neg=True, src="er_drop_20"),
})
TREND_TERMS.update({
    "ema_align_term": lambda b, row, cal, level, direction: _agree_mag(row, cal, "ema_align", _dir_px_trend(direction)),
    "mom10_term": lambda b, row, cal, level, direction: _agree_mag(row, cal, "mom_10", _dir_px_trend(direction)),
    "dirpers_term": lambda b, row, cal, level, direction: _clip01(row.get("dir_persistence_20", np.nan)) if row.get("dir_persistence_20", np.nan) == row.get("dir_persistence_20", np.nan) else 0.0,
})
# Retired 2026-08-01 after the bucket sweep: autocorr_20 ordered forward outcomes
# in only 17.9% of 56 family x horizon triplets and r2_20 in 23.2%, both *below*
# the 33% expected by chance, while carrying non-zero fitted weight. Removed
# rather than left at zero so the weight search cannot re-fit noise into them.
RETIRED_TERM_KEYS: tuple[str, ...] = ("autocorr_term", "r2_term")

RANGE_TERM_KEYS = (*RANGE_TERM_KEYS, "macd_div_term", "rsi_div_term", "mom_decel_term", "er_drop_term")
TREND_TERM_KEYS = (*TREND_TERM_KEYS, "ema_align_term", "mom10_term", "dirpers_term")


def _equal_weights(keys: tuple[str, ...]) -> dict[str, float]:
    w = 1.0 / len(keys)
    return {key: w for key in keys}


WEIGHTS: dict[str, dict[str, float] | float] = {
    "range": _equal_weights(RANGE_TERM_KEYS),
    "trend": _equal_weights(TREND_TERM_KEYS),
    "transition_shrink": 0.4,
}


def _range_opportunity(
    blocks: BlockScores,
    row: pd.Series,
    calibrator: FamilyCalibrator,
    weights=None,
) -> float:
    weights = WEIGHTS if weights is None else weights
    level = 0.0 if blocks.level != blocks.level else blocks.level
    if level == 0.0:
        return 0.0
    w = weights["range"]
    conviction = sum(
        w.get(k, 0.0) * RANGE_TERMS[k](blocks, row, calibrator, level, blocks.direction)
        for k in RANGE_TERM_KEYS
    )
    sign = -1.0 if level < 0 else 1.0
    return _clip_opportunity(-sign * 100.0 * conviction)


def _trend_opportunity(blocks: BlockScores, row: pd.Series, calibrator: FamilyCalibrator, weights=None) -> float:
    weights = WEIGHTS if weights is None else weights
    direction = 0.0 if blocks.direction != blocks.direction else blocks.direction
    if direction == 0.0:
        return 0.0
    level = 0.0 if blocks.level != blocks.level else blocks.level
    w = weights["trend"]
    conviction = sum(
        w.get(k, 0.0) * TREND_TERMS[k](blocks, row, calibrator, level, direction)
        for k in TREND_TERM_KEYS
    )
    sign = 1.0 if direction > 0 else -1.0
    return _clip_opportunity(sign * 100.0 * conviction)


def compute_opportunity(
    blocks: BlockScores,
    row: pd.Series,
    calibrator: FamilyCalibrator,
    weights=None,
) -> float:
    """Regime-specific composite in ``[-100, +100]`` (+ = long)."""
    weights = WEIGHTS if weights is None else weights
    if blocks.regime == "range":
        return _range_opportunity(blocks, row, calibrator, weights)
    if blocks.regime == "trend":
        return _trend_opportunity(blocks, row, calibrator, weights)
    return _clip_opportunity(weights["transition_shrink"] * _range_opportunity(blocks, row, calibrator, weights))


def compute_blocks(
    row: pd.Series,
    calibrator: FamilyCalibrator,
    n_contract_bars: int,
) -> BlockScores:
    """All block scores for one as-of bar."""
    er_20 = float(row.get("er_20", np.nan))
    regime = regime_label(er_20, calibrator)
    trendiness = block_trendiness(row, calibrator)
    direction = block_direction(row, calibrator)
    strength = block_strength(row, direction)
    level = block_level(row, calibrator)
    p_reversion = block_p_reversion(level, row, calibrator)
    p_continuation = block_p_continuation(direction, trendiness, calibrator)
    return BlockScores(
        regime=regime,
        trendiness=trendiness,
        direction=direction,
        strength=strength,
        level=level,
        p_reversion=p_reversion,
        p_continuation=p_continuation,
    )


def build_rationale(blocks: BlockScores, opportunity: float) -> list[str]:
    """Short Spanish bullets explaining the score decomposition."""
    bullets: list[str] = []
    bullets.append(
        f"Régimen {blocks.regime} (trendiness {blocks.trendiness:.0f}/100)."
    )
    bullets.append(
        f"Dirección {blocks.direction:+.0f}, fuerza {blocks.strength:.0f}/100."
    )
    bullets.append(
        f"Nivel {blocks.level:+.0f} (+ = caro); P(reversión)={blocks.p_reversion:.2f}, "
        f"P(continuación)={blocks.p_continuation:.2f}."
    )
    bullets.append(f"Signo interno {opportunity:+.0f}; se usa como color/rank, no como instrucción.")
    return bullets


def build_risks(
    blocks: BlockScores,
    row: pd.Series,
    n_contract_bars: int,
) -> list[str]:
    """Risk flags that apply to this instrument snapshot."""
    risks: list[str] = []
    if n_contract_bars < 15:
        risks.append("Señal joven (poca historia del contrato)")
    level_z = row.get("level_z", np.nan)
    if abs(blocks.level) >= 80 or (level_z == level_z and abs(level_z) >= 2.5):
        risks.append("Mercado muy extendido")
    if blocks.regime == "transition":
        risks.append("Régimen inestable (transición)")
    level_pct = row.get("level_pct", np.nan)
    if level_pct != level_pct:
        risks.append("Muestra histórica baja / sin panel análogo")
    lvl_term = min(abs(blocks.level) / 100.0, 1.0)
    if (
        blocks.regime == "range"
        and not (np.sign(blocks.direction) == np.sign(-blocks.level))
        and abs(blocks.direction) >= 50
        and lvl_term >= 0.5
    ):
        risks.append("Divergencia entre modelos (tendencia vs nivel)")
    vol_ratio = row.get("vol_ratio", np.nan)
    if vol_ratio == vol_ratio and (vol_ratio >= 2 or vol_ratio <= 0.5):
        risks.append("Volatilidad anormal")
    return risks


def _feature_snapshot(row: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in FEATURE_SNAPSHOT:
        val = row.get(name, np.nan)
        if val == val:
            out[name] = float(val)
    return out


def _feature_percentiles(row: pd.Series, calibrator: FamilyCalibrator) -> dict[str, float]:
    """Percentile (0–100) of each snapshot feature within the family history.

    Only features with a fitted ECDF get a percentile; the rest (z-scores, RSI,
    R², …) are absolute by nature and are left out so the UI can fall back to an
    absolute reading for them.
    """
    out: dict[str, float] = {}
    for name in FEATURE_SNAPSHOT:
        ecdf = calibrator.ecdf.get(name)
        val = row.get(name, np.nan)
        if ecdf is not None and len(ecdf) and val == val:
            out[name] = percentile(ecdf, float(val)) * 100.0
    return out


def _attach_bar_counts(data: pd.DataFrame) -> pd.DataFrame:
    ordered = data.sort_values(["contract", "date"]).copy()
    ordered["_n_contract_bars"] = ordered.groupby("contract", sort=False).cumcount() + 1
    return ordered


def score_instrument(
    data: pd.DataFrame,
    family: str,
    contract: str,
    horizon: int = 25,
    calibrator: FamilyCalibrator | None = None,
    as_of: pd.Timestamp | None = None,
) -> InstrumentScore:
    """Score ``contract`` as of ``as_of`` (or its latest bar) within enriched ``data``.

    When ``as_of`` is given the score is strictly point-in-time: only rows up to
    and including that date feed the calibration, and the scored bar is the
    contract's last observation on or before ``as_of``. Features are already
    as-of-``t``, so no future information enters the answer.
    """
    if as_of is not None:
        as_of = pd.Timestamp(as_of)
        cal = calibrator if calibrator is not None else fit_calibrator(
            data, family, horizon, outcome_asof=as_of
        )
        data = data.loc[pd.to_datetime(data["date"]) <= as_of]
    else:
        cal = calibrator if calibrator is not None else fit_calibrator(data, family, horizon)
    enriched = _attach_bar_counts(data)
    sub = enriched.loc[enriched["contract"] == contract]
    if sub.empty:
        raise KeyError(f"contract {contract!r} not found in data on or before {as_of}")
    row = sub.iloc[-1]
    blocks = compute_blocks(row, cal, int(row["_n_contract_bars"]))
    fam_weights = weights_for(family)
    opportunity = compute_opportunity(blocks, row, cal, fam_weights)
    return InstrumentScore(
        family=family,
        contract=str(contract),
        date=pd.Timestamp(row["date"]),
        close=float(row["close"]),
        dte=float(row["dte"]),
        blocks=blocks,
        opportunity=opportunity,
        rationale=build_rationale(blocks, opportunity),
        risks=build_risks(blocks, row, int(row["_n_contract_bars"])),
        features=_feature_snapshot(row),
        percentiles=_feature_percentiles(row, cal),
    )


def score_family(
    data: pd.DataFrame,
    family: str,
    horizon: int = 25,
    active_within_days: int = 45,
    require_unexpired: bool = True,
) -> pd.DataFrame:
    """Rank active contracts by opportunity on their latest bar."""
    cal = fit_calibrator(data, family, horizon)
    enriched = _attach_bar_counts(data)
    max_date = enriched["date"].max()
    cutoff = max_date - pd.Timedelta(days=active_within_days)

    latest = enriched.sort_values(["contract", "date"]).groupby("contract", sort=False).tail(1)
    mask = latest["date"] >= cutoff
    if require_unexpired:
        mask &= latest["dte"] > 0
    active = latest.loc[mask]

    fam_weights = weights_for(family)
    rows: list[dict] = []
    for _, row in active.iterrows():
        blocks = compute_blocks(row, cal, int(row["_n_contract_bars"]))
        opportunity = compute_opportunity(blocks, row, cal, fam_weights)
        rows.append(
            {
                "family": family,
                "contract": row["contract"],
                "date": row["date"],
                "close": row["close"],
                "dte": row["dte"],
                "regime": blocks.regime,
                "trendiness": blocks.trendiness,
                "direction": blocks.direction,
                "strength": blocks.strength,
                "level": blocks.level,
                "p_reversion": blocks.p_reversion,
                "p_continuation": blocks.p_continuation,
                "opportunity": opportunity,
            }
        )

    columns = [
        "family",
        "contract",
        "date",
        "close",
        "dte",
        "regime",
        "trendiness",
        "direction",
        "strength",
        "level",
        "p_reversion",
        "p_continuation",
        "opportunity",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows, columns=columns)
    out["opportunity"] = out["opportunity"].fillna(0.0)
    return out.sort_values("opportunity", ascending=False).reset_index(drop=True)
