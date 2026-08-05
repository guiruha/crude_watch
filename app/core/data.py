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
ENRICHED_DIR = _CACHE_ROOT / "data" / "enriched"

# Insertion order defines the structure picker order in the UI.
FRAME_NAMES = [
    "outrights", "calendars", "cracks", "brent_wti",
    "quarterly", "semestral", "yearly", "flies",
]


@st.cache_data(show_spinner="Building CrudeWatch dataset…", max_entries=1)
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


@st.cache_data(show_spinner="Loading CrudeWatch family…", max_entries=2)
def load_frame(family: str) -> pd.DataFrame:
    """Load one published family without materialising the whole dataset."""
    family = str(family)
    if family not in FRAME_NAMES:
        raise KeyError(f"Unknown family: {family}")
    for source in (_BAKED_PROCESSED, PROCESSED_DIR):
        parquet = source / f"{family}.parquet"
        if parquet.exists():
            return pd.read_parquet(parquet)

    frames = build_all(load_raw(RAW_PATH))
    try:
        save_frames(frames, PROCESSED_DIR, "parquet")
    except OSError:
        pass
    return frames[family]


def enriched_cache_available(family: str) -> bool:
    """Whether a prebuilt enriched family cache exists for runtime scoring."""
    return (ENRICHED_DIR / f"{family}.parquet").exists()
