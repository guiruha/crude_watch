"""Cached opportunity-score computations for the Streamlit app."""
from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import streamlit as st

from crudewatch.research import build_dataset

from core.data import load_frames

# The evaluation horizon grid (trading days) offered in the UI.
HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20, 25, 30)
HEADLINE_HORIZON: int = 25

__all__ = [
    "HEADLINE_HORIZON",
    "HORIZONS",
    "enriched_frame",
    "calibrator_cached",
    "score_family_cached",
    "score_instrument_dict",
    "analogous_outcomes_cached",
    "horizon_outcomes_cached",
    "backtest_contract_cached",
    "family_date_bounds",
    "contracts_on_date",
]


@st.cache_data(show_spinner=False)
def enriched_frame(family: str) -> pd.DataFrame:
    """lifecycle -> features -> level panel -> executable forward outcomes."""
    frames = load_frames()
    return build_dataset(frames[family], family)


@st.cache_resource(show_spinner=False)
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
        frame = frame.loc[pd.to_datetime(frame["date"]) <= stamp]
        return fit_calibrator(frame, family, horizon, outcome_asof=stamp)
    return fit_calibrator(frame, family, horizon)


@st.cache_data(show_spinner="Calculando oportunidades…")
def score_family_cached(family: str, horizon: int) -> pd.DataFrame:
    from crudewatch.scoring import score_family

    return score_family(enriched_frame(family), family, horizon)


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner="Calculando horizontes…")
def horizon_outcomes_cached(family: str, contract: str, as_of: str | None = None) -> pd.DataFrame:
    """Analogous forward outcomes across all HORIZONS for one contract.

    One row per horizon (same regime + level bucket cohort): sample size, average
    forward move, hit rate, and — when the setup has a directional side — the
    action-aligned average and the favourable / adverse excursions (MFE / MAE).
    Point-in-time when ``as_of`` is given.
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
                "avg_mfe": coh.get("avg_mfe", float("nan")),
                "avg_mae": coh.get("avg_mae", float("nan")),
            }
        )
    return pd.DataFrame.from_records(records)


@st.cache_data(show_spinner="Simulando backtest del contrato…")
def backtest_contract_cached(family: str, contract: str, horizon: int):
    """Follow-the-score backtest for one contract (strict point-in-time).

    Heavy on first call (per-bar calibrator refit) but cached thereafter. Returns
    a ``crudewatch.scoring.BacktestResult``.
    """
    from crudewatch.scoring import backtest_contract

    return backtest_contract(enriched_frame(family), family, contract, horizon)


@st.cache_data(show_spinner=False)
def family_date_bounds(family: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Earliest / latest observation date available for a family."""
    dates = enriched_frame(family)["date"]
    return pd.Timestamp(dates.min()), pd.Timestamp(dates.max())


@st.cache_data(show_spinner=False)
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
    return sorted(live.index.tolist())
