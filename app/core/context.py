"""Shared cached view-models for dashboard screens."""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from core.scoring import (
    active_contract_scores_cached,
    analogous_outcomes_cached,
    horizon_outcomes_cached,
    score_instrument_dict,
)
from core.runtime import low_memory_mode
from core.validation import validation_for


@st.cache_data(show_spinner=False, max_entries=1024)
def selected_context_cached(
    family: str,
    contract: str,
    horizon: int,
    as_of: str,
    include_horizons: bool = False,
) -> dict:
    """Score/cohort bundle shared by Radar and Score.

    Keeping this as one cached unit prevents each screen from independently
    recomputing the same score and analogue cohort for the selected instrument.
    """
    t0 = time.perf_counter()
    score = score_instrument_dict(family, contract, horizon, as_of)
    t_score = time.perf_counter()
    if low_memory_mode():
        rank_data = {}
        stretch_rank = float("nan")
    else:
        active = active_contract_scores_cached(family, horizon, as_of)
        rank_row = active.loc[active["contract"].astype(str) == str(contract)] if not active.empty else pd.DataFrame()
        rank_data = rank_row.iloc[0].to_dict() if not rank_row.empty else {}
        stretch_rank = float(rank_row["stretch_rank"].iloc[0]) if not rank_row.empty else float("nan")
    cohort = analogous_outcomes_cached(family, contract, horizon, as_of)
    t_cohort = time.perf_counter()
    horizons = (
        horizon_outcomes_cached(family, contract, as_of)
        if include_horizons and not low_memory_mode()
        else pd.DataFrame()
    )
    t_done = time.perf_counter()
    return {
        "score": score,
        "stretch_rank": stretch_rank,
        "rank_row": rank_data,
        "cohort": cohort,
        "validation": validation_for(family, horizon),
        "horizons": horizons,
        "timings": {
            "score_ms": (t_score - t0) * 1000.0,
            "cohort_ms": (t_cohort - t_score) * 1000.0,
            "horizons_ms": (t_done - t_cohort) * 1000.0,
            "total_ms": (t_done - t0) * 1000.0,
        },
    }
