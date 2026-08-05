"""Cached composite-rank computations for the Streamlit app."""
from __future__ import annotations

from dataclasses import asdict
import bisect
import warnings

import numpy as np
import pandas as pd
import streamlit as st

from crudewatch.infra import FAMILY_POINT_VALUE_USD
from crudewatch.research import COST_STUB_POINTS, build_dataset

from core.audit import descriptive_bias, pm_description
from core.data import ENRICHED_DIR, load_frame
from core.evidence import MIN_EFFECTIVE_N
from core.validation import validation_for

# The evaluation horizon grid (trading days) offered in the UI.
HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20, 25, 30)
HEADLINE_HORIZON: int = 25


def _float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _life_phase(dte: float) -> str:
    if dte != dte:
        return "sin dato"
    if dte <= 45:
        return "vencimiento cercano"
    if dte <= 180:
        return "vida media"
    if dte <= 365:
        return "diferido"
    return "diferido +1y"


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


def _level_bin_key(level: float) -> int:
    if level != level:
        return 2
    return int(np.digitize([level], (-60.0, -20.0, 20.0, 60.0))[0])


def _vol_regime(vol_ratio: float) -> str:
    if vol_ratio != vol_ratio:
        return "sin dato"
    if vol_ratio >= 1.5:
        return "expansión"
    if vol_ratio <= 0.75:
        return "contracción"
    return "normal"


def _first_present(row: pd.Series, names: tuple[str, ...]):
    for name in names:
        value = row.get(name, np.nan)
        if value == value:
            return value
    return np.nan


def _month_key(value) -> int | None:
    value = _float_or_nan(value)
    return None if value != value else int(value)


__all__ = [
    "HEADLINE_HORIZON",
    "HORIZONS",
    "enriched_frame",
    "calibrator_cached",
    "score_instrument_dict",
    "family_abs_composite_ecdf_cached",
    "active_contract_scores_cached",
    "analogous_outcomes_cached",
    "horizon_outcomes_cached",
    "family_date_bounds",
    "contracts_on_date",
]


@st.cache_data(show_spinner=False, max_entries=16)
def enriched_frame(family: str) -> pd.DataFrame:
    """lifecycle -> features -> level panel -> executable forward outcomes."""
    path = ENRICHED_DIR / f"{family}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    enriched = build_dataset(load_frame(family), family)
    try:
        ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
        enriched.to_parquet(path)
    except OSError:
        pass
    return enriched


@st.cache_resource(show_spinner=False, max_entries=512)
def calibrator_cached(family: str, horizon: int, as_of: str | None = None):
    """Family calibrator fit once per ``(family, horizon, as_of)`` and reused.

    Fitting sorts the whole family history into ECDFs and reads the research
    metrics CSV, so refitting it on every contract / analogue / horizon call is
    the app's main hotspot. Caching it here (and passing it into the scoring
    functions) collapses that to one fit per date+horizon. It reproduces exactly
    what ``score_instrument`` / ``analogous_outcomes`` would fit internally: the
    same ``<= as_of`` slice and ``outcome_asof`` point-in-time cut-off.
    """
    from crudewatch.scoring import fit_calibrator

    frame = enriched_frame(family)
    if as_of:
        stamp = pd.Timestamp(as_of)
        return fit_calibrator(frame, family, horizon, outcome_asof=stamp)
    return fit_calibrator(frame, family, horizon)


@st.cache_data(show_spinner=False, max_entries=2048)
def score_instrument_dict(
    family: str, contract: str, horizon: int, as_of: str | None = None
) -> dict:
    """Score one contract, optionally as of a historical date (ISO ``YYYY-MM-DD``)."""
    from crudewatch.scoring import score_instrument

    stamp = pd.Timestamp(as_of) if as_of else None
    cal = calibrator_cached(family, horizon, as_of)
    return asdict(
        score_instrument(
            enriched_frame(family), family, contract, horizon, calibrator=cal, as_of=stamp
        )
    )


def _percentile_rank(sorted_values: np.ndarray, value: float) -> float:
    if value != value or len(sorted_values) == 0:
        return float("nan")
    return float(bisect.bisect_right(sorted_values, float(value)) / len(sorted_values) * 100.0)


def _col(data: pd.DataFrame, name: str, default: float = np.nan) -> np.ndarray:
    if name in data.columns:
        return data[name].to_numpy(dtype=float)
    return np.full(len(data), default, dtype=float)


def _ecdf_pct(ecdf: np.ndarray, values: np.ndarray) -> np.ndarray:
    if ecdf is None or len(ecdf) == 0:
        return np.where(np.isnan(values), np.nan, 0.5)
    out = np.searchsorted(ecdf, values, side="right") / len(ecdf)
    return np.where(np.isnan(values), np.nan, out)


def _nanmean(parts: list[np.ndarray], n: int, default: float = 0.0) -> np.ndarray:
    if not parts:
        return np.full(n, default, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmean(np.vstack(parts), axis=0)
    return np.where(np.isnan(out), default, out)


def _signed_pct_arr(pct: np.ndarray) -> np.ndarray:
    return (2.0 * pct - 1.0) * 100.0


def _clip01_arr(values: np.ndarray) -> np.ndarray:
    return np.clip(np.where(np.isnan(values), 0.0, values), 0.0, 1.0)


def _timing_term_arr(data: pd.DataFrame, cal) -> np.ndarray:
    mom = _col(data, "mom_decel_10")
    raw = _ecdf_pct(cal.ecdf.get("neg_mom_decel_10", np.array([])), -mom)
    return np.clip(np.where(np.isnan(mom), 0.5, raw), 0.0, 1.0)


def _vol_term_arr(data: pd.DataFrame) -> np.ndarray:
    vr = _col(data, "vol_ratio")
    return np.clip(np.where(np.isnan(vr), 0.5, 1.2 - vr), 0.0, 1.0)


def _agreement_mag(data: pd.DataFrame, cal, feature: str, direction: np.ndarray) -> np.ndarray:
    signed = 2.0 * _ecdf_pct(cal.ecdf.get(feature, np.array([])), _col(data, feature)) - 1.0
    sign = np.sign(signed)
    return np.where(sign == direction, np.abs(signed), 0.0)


def _signed_mag_arr(data: pd.DataFrame, cal, feature: str) -> np.ndarray:
    val = _col(data, feature)
    signed = 2.0 * _ecdf_pct(cal.ecdf.get(feature, np.array([])), val) - 1.0
    return np.where(np.isnan(val), 0.0, np.abs(signed))


def _reversion_confirmation_arr(data: pd.DataFrame, cal, level: np.ndarray) -> np.ndarray:
    dear = level > 0
    parts: list[np.ndarray] = []

    rsi2 = _col(data, "rsi_2")
    parts.append(np.where(np.isnan(rsi2), np.nan, np.where(dear, (rsi2 - 80.0) / 20.0, (20.0 - rsi2) / 20.0)))
    rsi14 = _col(data, "rsi_14")
    parts.append(np.where(np.isnan(rsi14), np.nan, np.where(dear, (rsi14 - 70.0) / 30.0, (30.0 - rsi14) / 30.0)))
    pctb = _col(data, "pctb_20_2")
    parts.append(np.where(np.isnan(pctb), np.nan, np.where(dear, (pctb - 0.8) / 0.2, (0.2 - pctb) / 0.2)))
    pctb_fast = _col(data, "pctb_10_1_5")
    parts.append(np.where(np.isnan(pctb_fast), np.nan, np.where(dear, (pctb_fast - 0.8) / 0.2, (0.2 - pctb_fast) / 0.2)))
    div = _col(data, "rsi_div_14")
    parts.append(np.where(np.isnan(div), np.nan, np.where(dear, -div / 1.5, div / 1.5)))
    decel = _col(data, "mom_decel_10")
    decel_pct = _ecdf_pct(cal.ecdf.get("neg_mom_decel_10", np.array([])), -decel)
    parts.append(np.where(np.isnan(decel), np.nan, decel_pct))

    clipped = [np.clip(part, 0.0, 1.0) for part in parts]
    return _nanmean(clipped, len(data), default=0.5)


def _historical_abs_composite(data: pd.DataFrame, cal, transition_shrink: float, sort: bool = True) -> np.ndarray:
    n = len(data)
    er = _col(data, "er_20")
    regime = np.full(n, "transition", dtype=object)
    regime[er <= cal.er_lo] = "range"
    regime[er >= cal.er_hi] = "trend"

    trendiness = _nanmean(
        [
            _ecdf_pct(cal.ecdf.get("er_20", np.array([])), _col(data, "er_20")) * 100.0,
            _ecdf_pct(cal.ecdf.get("variance_ratio_5", np.array([])), _col(data, "variance_ratio_5")) * 100.0,
            _ecdf_pct(cal.ecdf.get("autocorr_20", np.array([])), _col(data, "autocorr_20")) * 100.0,
        ],
        n,
    )
    direction = _nanmean(
        [
            _signed_pct_arr(_ecdf_pct(cal.ecdf.get(feat, np.array([])), _col(data, feat)))
            for feat in ("slope_20", "macd_hist", "ema_align", "mom_5", "mom_10", "mom_20")
        ],
        n,
    )
    level_parts = []
    level_pct = _col(data, "level_pct")
    level_parts.append(_signed_pct_arr(_ecdf_pct(cal.ecdf.get("level_pct", np.array([])), level_pct)))
    for feat in ("level_z", "z_10", "z_20", "z_50", "keltner_dist_20"):
        level_parts.append(100.0 * np.tanh(_col(data, feat) / 2.0))
    level = _nanmean(level_parts, n)
    quality = _nanmean(
        [
            np.clip(_col(data, "r2_20"), 0.0, 1.0),
            np.clip(_col(data, "dir_persistence_20"), 0.0, 1.0),
        ],
        n,
    )
    strength = np.clip(100.0 * quality * (0.4 + 0.6 * np.minimum(np.abs(direction) / 100.0, 1.0)), 0.0, 100.0)

    p_rev_base = np.where(level < 0, cal.p_rev_cheap, np.where(level > 0, cal.p_rev_dear, 0.5))
    p_rev_base = np.where(np.isnan(p_rev_base), 0.5, p_rev_base)
    confirmation = _reversion_confirmation_arr(data, cal, level)
    conf = 0.75 + 0.5 * confirmation
    p_reversion = 0.5 + (p_rev_base - 0.5) * np.clip(np.minimum(np.abs(level) / 60.0, 1.0) * conf, 0.0, 1.0)
    p_cont_base = np.where(direction > 0, cal.p_cont_up, np.where(direction < 0, cal.p_cont_dn, 0.5))
    p_cont_base = np.where(np.isnan(p_cont_base), 0.5, p_cont_base)
    p_continuation = 0.5 + (p_cont_base - 0.5) * (trendiness / 100.0)

    lvl_term = np.minimum(np.abs(level) / 100.0, 1.0)
    rev_term = np.clip((p_reversion - 0.5) / 0.5, 0.0, 1.0)
    timing_term = _timing_term_arr(data, cal)
    vol_term = _vol_term_arr(data)
    range_dir = np.where(level > 0, -1.0, 1.0)
    range_terms = np.vstack(
        [
            rev_term,
            lvl_term,
            timing_term,
            vol_term,
            _agreement_mag(data, cal, "macd_div", range_dir),
            _signed_mag_arr(data, cal, "rsi_div_14"),
            _clip01_arr(np.where(np.isnan(_col(data, "mom_decel_10")), 0.0, _ecdf_pct(cal.ecdf.get("neg_mom_decel_10", np.array([])), -_col(data, "mom_decel_10")))),
            _clip01_arr(np.where(np.isnan(_col(data, "er_drop_20")), 0.0, _ecdf_pct(cal.ecdf.get("neg_er_drop_20", np.array([])), -_col(data, "er_drop_20")))),
        ]
    )
    range_conviction = np.nanmean(range_terms, axis=0)
    range_conviction = np.where(level == 0.0, 0.0, range_conviction)

    cont_term = np.clip((p_continuation - 0.5) / 0.5, 0.0, 1.0)
    trend_dir = np.where(direction > 0, 1.0, -1.0)
    trend_terms = np.vstack(
        [
            np.minimum(np.abs(direction) / 100.0, 1.0),
            strength / 100.0,
            cont_term,
            1.0 - lvl_term,
            _agreement_mag(data, cal, "ema_align", trend_dir),
            _agreement_mag(data, cal, "mom_10", trend_dir),
            _clip01_arr(_col(data, "dir_persistence_20")),
        ]
    )
    trend_conviction = np.nanmean(trend_terms, axis=0)
    trend_conviction = np.where(direction == 0.0, 0.0, trend_conviction)
    magnitude = np.where(
        regime == "trend",
        trend_conviction * 100.0,
        np.where(regime == "transition", transition_shrink * range_conviction * 100.0, range_conviction * 100.0),
    )
    clean = magnitude[~np.isnan(magnitude)]
    return np.sort(clean) if sort else magnitude


def _batch_analogous_outcomes(
    family: str,
    horizon: int,
    as_of: str,
    frame: pd.DataFrame,
    enriched: pd.DataFrame,
    scored_rows: list[dict],
    cal,
) -> dict[str, dict]:
    """Compute analogue cohorts for all active contracts in one family pass."""
    from crudewatch.scoring.analogues import (
        LEVEL_BIN_LABELS,
        _annualized_sample_sharpe,
        _far_leg_series,
        _month_series,
        _non_overlapping_mask,
        _tenor_bucket,
        _vec_level,
        _vec_regime,
    )

    target = f"fwd_{int(horizon)}"
    mfe_col = f"mfe_{int(horizon)}"
    mae_col = f"mae_{int(horizon)}"
    if target not in enriched.columns or not scored_rows:
        return {}

    stamp = pd.Timestamp(as_of)
    full_ordered = frame.sort_values(["contract", "date"])
    full_target_dates = full_ordered.groupby("contract", sort=False)["date"].shift(-int(horizon))
    target_dates = pd.to_datetime(full_target_dates.reindex(enriched.index)).to_numpy()
    resolved = target_dates <= np.datetime64(stamp)

    er = enriched["er_20"].to_numpy(dtype=float) if "er_20" in enriched.columns else np.full(len(enriched), np.nan)
    regime = _vec_regime(er, cal.er_lo, cal.er_hi)
    level = _vec_level(enriched, cal)
    bins = np.digitize(np.where(np.isnan(level), -999.0, level), (-60.0, -20.0, 20.0, 60.0))
    dte = enriched["dte"].to_numpy(dtype=float) if "dte" in enriched.columns else np.full(len(enriched), np.nan)
    tenor = np.array([_tenor_bucket(x) for x in dte], dtype=object)
    months = _month_series(enriched).to_numpy()
    far_legs = _far_leg_series(enriched).to_numpy()
    fwd = enriched[target].to_numpy(dtype=float)
    valid_outcome = resolved & ~np.isnan(fwd) & ~np.isnan(level)

    unique_keys = {item["cohort_key"] for item in scored_rows}
    samples_by_key: dict[tuple, dict] = {}
    bar_idx = enriched["_n_contract_bars"].to_numpy(dtype=int) - 1
    for key in unique_keys:
        key_regime, key_bin, key_tenor, key_month, key_far_leg = key
        match = (regime == key_regime) & (bins == key_bin) & (tenor == key_tenor) & valid_outcome
        if key_month is not None:
            match &= months == key_month
        if key_far_leg is not None:
            match &= far_legs == key_far_leg

        n_raw = int(match.sum())
        matched_rows = (
            enriched.loc[match, ["contract", "date"]]
            .assign(_bar_idx=bar_idx[match])
            .assign(_target_date=target_dates[match])
            .sort_values(["contract", "_bar_idx"])
        )
        keep = _non_overlapping_mask(matched_rows, int(horizon)) if n_raw else np.array([], dtype=bool)
        idx = matched_rows.index.to_numpy()[keep]
        matched_effective = enriched.loc[idx]
        fwd_c = enriched.loc[idx, target].to_numpy(dtype=float)
        dates_effective = pd.to_datetime(matched_effective["date"]) if len(matched_effective) else pd.Series(dtype="datetime64[ns]")
        samples_by_key[key] = {
            "n": int(len(idx)),
            "n_raw": n_raw,
            "idx": idx,
            "fwd": fwd_c,
            "vintage_count": int(matched_effective["vintage"].nunique()) if "vintage" in matched_effective else 0,
            "year_count": int(dates_effective.dt.year.nunique()) if len(dates_effective) else 0,
        }

    cost_points = float(COST_STUB_POINTS.get(family, 0.0))
    out: dict[str, dict] = {}
    for item in scored_rows:
        key = item["cohort_key"]
        samples = samples_by_key.get(key, {})
        fwd_c = samples.get("fwd", np.array([], dtype=float))
        n = int(samples.get("n", 0))
        side = 1.0 if item["opportunity"] > 0 else -1.0 if item["opportunity"] < 0 else 0.0
        result = {
            "n": n,
            "n_raw": int(samples.get("n_raw", 0)),
            "vintage_count": int(samples.get("vintage_count", 0)),
            "year_count": int(samples.get("year_count", 0)),
            "horizon": int(horizon),
            "regime": key[0],
            "level_bin": LEVEL_BIN_LABELS[int(key[1])],
            "tenor_bucket": key[2],
            "month": key[3],
            "far_leg": key[4],
            "side": side,
            "opportunity": float(item["opportunity"]),
            "action_threshold": 0.0,
            "cost": cost_points,
        }
        if n:
            result.update(
                {
                    "avg_fwd": float(np.nanmean(fwd_c)),
                    "median_fwd": float(np.nanmedian(fwd_c)),
                    "up_rate": float(np.mean(fwd_c > 0)),
                }
            )
        if n and side != 0.0 and {mfe_col, mae_col}.issubset(enriched.columns):
            idx = samples["idx"]
            mfe_c = enriched.loc[idx, mfe_col].to_numpy(dtype=float)
            mae_c = enriched.loc[idx, mae_col].to_numpy(dtype=float)
            aligned_gross = side * fwd_c
            aligned = aligned_gross - cost_points
            favourable = np.where(side > 0, mfe_c, -mae_c)
            adverse = np.where(side > 0, mae_c, -mfe_c)
            adverse_loss = np.abs(np.minimum(adverse, 0.0))
            result.update(
                {
                    "avg_aligned_gross": float(np.nanmean(aligned_gross)),
                    "avg_aligned": float(np.nanmean(aligned)),
                    "median_aligned": float(np.nanmedian(aligned)),
                    "aligned_win_rate": float(np.mean(aligned > 0)),
                    "sharpe_aligned": _annualized_sample_sharpe(aligned, int(horizon)),
                    "avg_mfe": float(np.nanmean(favourable)),
                    "avg_mae": float(np.nanmean(adverse)),
                    "mae_p50": float(np.nanpercentile(adverse_loss, 50)),
                    "mae_p80": float(np.nanpercentile(adverse_loss, 80)),
                    "mfe_p50": float(np.nanpercentile(favourable, 50)),
                    "mfe_p80": float(np.nanpercentile(favourable, 80)),
                }
            )
        out[str(item["contract"])] = result
    return out


@st.cache_data(show_spinner=False, max_entries=256)
def family_abs_composite_ecdf_cached(family: str, horizon: int, as_of: str) -> np.ndarray:
    """Historical ECDF of abs(composite) for one family up to ``as_of``."""
    from crudewatch.scoring.score import WEIGHTS

    stamp = pd.Timestamp(as_of)
    frame = enriched_frame(family)
    window = frame.loc[pd.to_datetime(frame["date"]) <= stamp]
    if window.empty:
        return np.array([], dtype=float)
    cal = calibrator_cached(family, horizon, as_of)
    return _historical_abs_composite(window, cal, float(WEIGHTS["transition_shrink"]))


@st.cache_data(show_spinner="Rankeando contratos activos…", max_entries=512)
def active_contract_scores_cached(family: str, horizon: int, as_of: str) -> pd.DataFrame:
    """Score every contract live on ``as_of`` for the overview ranking."""
    from crudewatch.scoring.score import (
        _attach_bar_counts,
        build_rationale,
        build_risks,
        compute_blocks,
        compute_opportunity,
        weights_for,
    )

    contracts = contracts_on_date(family, as_of)
    frame = enriched_frame(family)
    stamp = pd.Timestamp(as_of)
    window = frame.loc[pd.to_datetime(frame["date"]) <= stamp]
    if window.empty or not contracts:
        return pd.DataFrame()
    cal = calibrator_cached(family, horizon, as_of)
    weighted = weights_for(family)
    enriched = _attach_bar_counts(window)
    latest = (
        enriched.loc[enriched["contract"].isin(contracts)]
        .sort_values(["contract", "date"])
        .groupby("contract", sort=False)
        .tail(1)
    )
    latest_by_contract = {str(r["contract"]): r for _, r in latest.iterrows()}
    point_value = FAMILY_POINT_VALUE_USD.get(family, 1000.0)
    cost_points = float(COST_STUB_POINTS.get(family, 0.0))
    cost_usd = cost_points * point_value
    abs_ecdf = family_abs_composite_ecdf_cached(family, horizon, as_of)
    validation = validation_for(family, horizon)
    scored_rows: list[dict] = []
    rows: list[dict] = []
    for contract in contracts:
        try:
            row = latest_by_contract[str(contract)]
            blocks_obj = compute_blocks(row, cal, int(row["_n_contract_bars"]))
            opportunity = compute_opportunity(blocks_obj, row, cal, weighted)
            blocks = asdict(blocks_obj)
            score = {
                "family": family,
                "contract": str(contract),
                "date": pd.Timestamp(row["date"]),
                "close": float(row["close"]),
                "dte": float(row["dte"]),
                "blocks": blocks,
                "opportunity": float(opportunity),
                "rationale": build_rationale(blocks_obj, opportunity),
                "risks": build_risks(blocks_obj, row, int(row["_n_contract_bars"])),
            }
        except Exception as exc:  # noqa: BLE001 - keep one bad contract from breaking the view
            rows.append(
                {
                    "contract": contract,
                    "error": str(exc),
                    "opportunity": float("nan"),
                    "abs_opportunity": float("nan"),
                }
            )
            continue

        opportunity = float(score["opportunity"])
        season_month = _first_present(row, ("month", "near_month", "wti_month", "front_month"))
        scored_rows.append(
            {
                "contract": str(contract),
                "row": row,
                "score": score,
                "blocks": blocks,
                "opportunity": opportunity,
                "cohort_key": (
                    blocks["regime"],
                    _level_bin_key(float(blocks["level"])),
                    _tenor_bucket(float(score["dte"])),
                    _month_key(season_month),
                    _month_key(row.get("far_month", np.nan)),
                ),
            }
        )

    cohort_by_contract = _batch_analogous_outcomes(
        family,
        horizon,
        as_of,
        frame,
        enriched,
        scored_rows,
        cal,
    )
    for item in scored_rows:
        contract = item["contract"]
        row = item["row"]
        score = item["score"]
        blocks = item["blocks"]
        opportunity = item["opportunity"]
        volume = _float_or_nan(row.get("volume"))
        season_month = _first_present(row, ("month", "near_month", "wti_month", "front_month"))
        far_month = _first_present(row, ("far_month", "brent_month"))
        vol_ratio_value = _float_or_nan(row.get("vol_ratio"))
        cohort = cohort_by_contract.get(str(contract), {"n": 0, "horizon": horizon})
        cohort_n = int(cohort.get("n", 0))
        cohort_sharpe = (
            float(cohort.get("sharpe_aligned", float("nan")))
            if cohort_n >= MIN_EFFECTIVE_N
            else float("nan")
        )
        preview_rank_row = {
            "liquidity": "sin dato" if volume != volume else "baja" if volume < 50 else "media" if volume < 500 else "alta",
            "vol_regime": _vol_regime(vol_ratio_value),
        }
        rows.append(
            {
                "contract": contract,
                "date": score["date"],
                "close": float(score["close"]),
                "dte": float(score["dte"]),
                "life_phase": _life_phase(float(score["dte"])),
                "slot": str(row.get("slot", "")),
                "vintage": _float_or_nan(row.get("vintage")),
                "season_month": _float_or_nan(season_month),
                "far_month": _float_or_nan(far_month),
                "volume": volume,
                "point_value_usd": point_value,
                "cost_points": cost_points,
                "cost_usd": cost_usd,
                "liquidity": preview_rank_row["liquidity"],
                "vol_regime": _vol_regime(vol_ratio_value),
                "pm_read": pm_description(score, cohort, preview_rank_row),
                "descriptive_bias": descriptive_bias(score),
                "opportunity": opportunity,
                "abs_opportunity": abs(opportunity),
                "signed_composite": opportunity,
                "sign": "+" if opportunity > 0 else "-" if opportunity < 0 else "0",
                "regime": blocks["regime"],
                "trendiness": float(blocks["trendiness"]),
                "direction": float(blocks["direction"]),
                "strength": float(blocks["strength"]),
                "level": float(blocks["level"]),
                "p_reversion": float(blocks["p_reversion"]),
                "p_continuation": float(blocks["p_continuation"]),
                "validation_state": validation["state"],
                "validation_n": validation["n_trades"],
                "validation_hit": validation["win_rate"],
                "cohort_n": cohort_n,
                "cohort_sharpe": cohort_sharpe,
                "cohort_median": (
                    float(cohort.get("median_aligned", float("nan")))
                    if cohort_n >= MIN_EFFECTIVE_N
                    else float("nan")
                ),
                "cohort_mae_p80": (
                    float(cohort.get("mae_p80", float("nan")))
                    if cohort_n >= MIN_EFFECTIVE_N
                    else float("nan")
                ),
                "risk_count": len(score["risks"]),
                "risks": " · ".join(score["risks"][:6]),
                "error": "",
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame.from_records(rows)
    out["stretch_rank"] = [
        _percentile_rank(abs_ecdf, value) for value in out.get("abs_opportunity", pd.Series(dtype=float))
    ]
    return out.sort_values(
        ["stretch_rank", "volume", "dte"],
        ascending=[False, False, True],
        na_position="last",
    )


@st.cache_data(show_spinner=False, max_entries=4096)
def analogous_outcomes_cached(
    family: str, contract: str, horizon: int, as_of: str | None = None
) -> dict:
    """Historical outcomes of analogous setups (same regime + level bucket)."""
    from crudewatch.scoring import analogous_outcomes

    stamp = pd.Timestamp(as_of) if as_of else None
    cal = calibrator_cached(family, horizon, as_of)
    return analogous_outcomes(
        enriched_frame(family), family, contract, horizon, as_of=stamp, calibrator=cal
    )


@st.cache_data(show_spinner="Calculando horizontes…", max_entries=1024)
def horizon_outcomes_cached(family: str, contract: str, as_of: str | None = None) -> pd.DataFrame:
    """Analogous forward outcomes across all HORIZONS for one contract.

    One row per horizon (same regime + level bucket cohort): sample size, average
    forward move, hit rate, and, when the composite is directional, the
    sign-adjusted net move plus excursion percentiles. Point-in-time when
    ``as_of`` is given.
    """
    from crudewatch.scoring import analogous_outcomes

    stamp = pd.Timestamp(as_of) if as_of else None
    frame = enriched_frame(family)
    records = []
    for h in HORIZONS:
        cal = calibrator_cached(family, h, as_of)
        coh = analogous_outcomes(frame, family, contract, h, as_of=stamp, calibrator=cal)
        records.append(
            {
                "horizon": h,
                "n": int(coh.get("n", 0)),
                "avg_fwd": coh.get("avg_fwd", float("nan")),
                "up_rate": coh.get("up_rate", float("nan")),
                "avg_aligned": coh.get("avg_aligned", float("nan")),
                "aligned_win_rate": coh.get("aligned_win_rate", float("nan")),
                "sharpe_aligned": coh.get("sharpe_aligned", float("nan")),
                "median_aligned": coh.get("median_aligned", float("nan")),
                "mae_p50": coh.get("mae_p50", float("nan")),
                "mae_p80": coh.get("mae_p80", float("nan")),
                "mfe_p80": coh.get("mfe_p80", float("nan")),
                "avg_mfe": coh.get("avg_mfe", float("nan")),
                "avg_mae": coh.get("avg_mae", float("nan")),
            }
        )
    return pd.DataFrame.from_records(records)


@st.cache_data(show_spinner=False, max_entries=16)
def family_date_bounds(family: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Earliest / latest observation date available for a family."""
    dates = enriched_frame(family)["date"]
    return pd.Timestamp(dates.min()), pd.Timestamp(dates.max())


@st.cache_data(show_spinner=False, max_entries=2048)
def contracts_on_date(family: str, as_of: str) -> list[str]:
    """Contracts trading on ``as_of`` (a bar bracketing that date), most-recent first.

    A contract qualifies when it has history on or before ``as_of`` and had not
    yet expired (its own series still runs at or beyond that date), so the
    instrument view only offers names that were actually live that day.
    """
    frame = enriched_frame(family)
    stamp = pd.Timestamp(as_of)
    dates = pd.to_datetime(frame["date"])
    grp = frame.assign(_d=dates).groupby("contract")["_d"]
    spans = pd.DataFrame({"min": grp.min(), "max": grp.max()})
    live = spans[(spans["min"] <= stamp) & (spans["max"] >= stamp)]
    if live.empty:
        return []
    current = (
        frame.loc[frame["contract"].isin(live.index) & (dates <= stamp)]
        .sort_values(["contract", "date"])
        .groupby("contract", sort=False)
        .tail(1)
    )
    if "dte" in current.columns:
        current = current.sort_values(["dte", "contract"], ascending=[True, True])
        return current["contract"].astype(str).tolist()
    return sorted(live.index.tolist())
