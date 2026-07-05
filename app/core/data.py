"""Cached dataset loading for the Streamlit app.

First run builds every dataframe from the raw Excel (~30s) and caches the result
to ``data/processed/*.parquet``; later runs read the parquet, which is fast. The
in-process ``st.cache_data`` layer means reruns within a session never rebuild.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from crudewatch.analytics import (
    BY_KEY,
    MAX_DAYS_TO_EXPIRY,
    Lifecycle,
    PriceMatrix,
    SeasonalBias,
    alignment_backtest,
    available_years,
    buffer_cone,
    build_price_matrix,
    build_series,
    current_percentile,
    forward_tendency,
    lifecycle_phase,
    monthly_heatmap,
    seasonal_bias,
    seasonal_stack,
)
from crudewatch.data_preparation import build_all
from crudewatch.infra import load_raw, save_frames

# Resource dir holds bundled, read-only inputs (repo root in dev; the
# PyInstaller extract dir in the frozen exe). Cache dir is writable and persists
# the parquet cache between launches (repo root in dev; next to the exe frozen).
_RESOURCE_ROOT = Path(os.environ.get("CRUDEWATCH_RESOURCE_DIR", Path(__file__).resolve().parents[2]))
_CACHE_ROOT = Path(os.environ.get("CRUDEWATCH_CACHE_DIR", _RESOURCE_ROOT))

REPO_ROOT = _RESOURCE_ROOT
RAW_PATH = _RESOURCE_ROOT / "data" / "raw_files.xlsx"
# Prefer a parquet cache baked alongside the resources; otherwise fall back to
# the writable cache dir (built on first launch and reused thereafter).
_BAKED_PROCESSED = _RESOURCE_ROOT / "data" / "processed"
PROCESSED_DIR = _CACHE_ROOT / "data" / "processed"

# Insertion order defines the structure picker order in the UI.
FRAME_NAMES = [
    "outrights", "calendars", "cracks", "brent_wti",
    "quarterly", "semestral", "yearly", "flies",
]


@st.cache_data(show_spinner="Building CrudeWatch dataset…")
def load_frames() -> dict[str, pd.DataFrame]:
    """Return every published dataframe, using the parquet cache when present.

    Read order: a parquet cache baked alongside the app, then the writable
    cache, then (last resort) rebuild from the raw workbook and cache it.
    """
    for source in (_BAKED_PROCESSED, PROCESSED_DIR):
        parquet = {name: source / f"{name}.parquet" for name in FRAME_NAMES}
        if all(p.exists() for p in parquet.values()):
            return {name: pd.read_parquet(p) for name, p in parquet.items()}

    frames = build_all(load_raw(RAW_PATH))
    try:
        save_frames(frames, PROCESSED_DIR, "parquet")
    except OSError:
        pass  # read-only location; keep serving the in-memory frames
    return frames


@st.cache_data(show_spinner=False)
def load_price_matrix() -> PriceMatrix:
    """Wide outright close/volume matrix backing every fixed-date structure."""
    return build_price_matrix(load_frames()["outrights"])


@st.cache_data(show_spinner=False)
def spread_years(structure_key: str) -> list[int]:
    """Vintage base years available for a structure."""
    return available_years(load_price_matrix(), BY_KEY[structure_key])


@st.cache_data(show_spinner=False)
def spread_series(structure_key: str, base_year: int) -> pd.DataFrame:
    """The fixed-date series for one structure vintage (cached per selection)."""
    return build_series(load_price_matrix(), BY_KEY[structure_key], base_year)


@st.cache_data(show_spinner=False)
def spread_seasonal(structure_key: str) -> pd.DataFrame:
    """Day-of-year stack of every vintage of a structure (cached per structure)."""
    return seasonal_stack(load_price_matrix(), BY_KEY[structure_key])


@st.cache_data(show_spinner=False)
def spread_current_vintage(structure_key: str) -> int | None:
    """The vintage that is trading *now*: the most recent data, nearest to expiry.

    Among vintages sharing the latest observation date, pick the front one
    (smallest days-to-expiry) so screens open on the live structure rather than a
    barely-started long-dated year.
    """
    stack = spread_seasonal(structure_key)
    if stack.empty:
        return None
    last = stack.groupby("vintage")["date"].max()
    live = last[last == last.max()].index
    front = stack[stack["vintage"].isin(live)].groupby("vintage")["days_to_expiry"].last()
    return int(front.idxmin())


@st.cache_data(show_spinner=False)
def spread_cone(structure_key: str, base_year: int, buffer_days: int,
                lo: int = 0, hi: int = MAX_DAYS_TO_EXPIRY) -> pd.DataFrame:
    """±buffer-day seasonal cone over the [lo, hi] dte window, excluding the active vintage."""
    return buffer_cone(spread_seasonal(structure_key), buffer_days, exclude_vintage=base_year,
                       max_dte=hi, min_dte=lo)


@st.cache_data(show_spinner=False)
def spread_percentile(structure_key: str, base_year: int, buffer_days: int,
                      lo: int = 0, hi: int = MAX_DAYS_TO_EXPIRY) -> float | None:
    """Current seasonal percentile of the active vintage within the [lo, hi] window."""
    return current_percentile(spread_seasonal(structure_key), base_year, buffer_days, max_dte=hi, min_dte=lo)


@st.cache_data(show_spinner=False)
def spread_seasonal_bias(structure_key: str, base_year: int, buffer_days: int, horizon: int = 20,
                         lo: int = 0, hi: int = MAX_DAYS_TO_EXPIRY) -> SeasonalBias:
    """Seasonal directional bias + consistency for the active vintage within the window."""
    return seasonal_bias(spread_seasonal(structure_key), base_year, buffer_days, horizon,
                         max_dte=hi, min_dte=lo)


@st.cache_data(show_spinner=False)
def spread_forward_tendency(structure_key: str, base_year: int, buffer_days: int,
                            lo: int = 0, hi: int = MAX_DAYS_TO_EXPIRY) -> dict[int, SeasonalBias]:
    """Seasonal bias at the 10 / 20 / 30-day forward horizons within the window."""
    return forward_tendency(spread_seasonal(structure_key), base_year, buffer_days, max_dte=hi, min_dte=lo)


@st.cache_data(show_spinner=False)
def spread_lifecycle(structure_key: str, base_year: int) -> Lifecycle:
    """Lifecycle phase of the active vintage (Quiet … Expiry-risk) over the full window."""
    return lifecycle_phase(spread_seasonal(structure_key), base_year)


@st.cache_data(show_spinner="Backtesting seasonal alignment…")
def spread_alignment_backtest(structure_key: str, horizon: int = 20, buffer_days: int = 5,
                              lo: int = 0, hi: int = MAX_DAYS_TO_EXPIRY) -> pd.DataFrame:
    """Tailwind/Headwind/Mixed vs forward continuation across all vintages (§8 test)."""
    return alignment_backtest(spread_seasonal(structure_key), horizon, buffer_days, max_dte=hi, min_dte=lo)


@st.cache_data(show_spinner=False)
def spread_monthly_heatmap(structure_key: str) -> pd.DataFrame:
    """Monthly move matrix (month x vintage) for the structure."""
    return monthly_heatmap(spread_seasonal(structure_key))
