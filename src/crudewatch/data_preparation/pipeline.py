"""Orchestration: build every published dataframe from the raw feed."""
from __future__ import annotations

import pandas as pd

from crudewatch.infra.constants import CONTRACT_KEYS, SPREAD_STRUCTURES
from crudewatch.data_preparation.outrights import build_outrights
from crudewatch.data_preparation.calendars import build_calendars
from crudewatch.data_preparation.cracks import build_cracks
from crudewatch.data_preparation.brent import build_brent_wti
from crudewatch.data_preparation.spreads import build_calendar_spread, build_flies


def build_all(df: pd.DataFrame, trim_synthetic: bool = True) -> dict[str, pd.DataFrame]:
    """Build every published dataframe from the raw feed.

    ``trim_synthetic`` trims synthetic spreads/flies to each contract's last 12
    months (matching the dataset convention); set False to keep full history.
    """
    frames: dict[str, pd.DataFrame] = {
        "outrights": build_outrights(df, trim=True),
        "calendars": build_calendars(df),
        "cracks": build_cracks(df),
        "brent_wti": build_brent_wti(df),
    }

    # Untrimmed legs so deferred spreads/flies still have overlapping dates.
    legs = build_outrights(df, trim=False)
    for name, gap in SPREAD_STRUCTURES.items():
        frames[name] = build_calendar_spread(legs, gap, name, trim=trim_synthetic)
    frames["flies"] = build_flies(legs, "fly", trim=trim_synthetic)
    return frames


def summarize(frames: dict[str, pd.DataFrame]) -> None:
    """Print a one-line-per-frame summary of row and contract counts."""
    print("\nSummary")
    print("-" * 64)
    for name, frame in frames.items():
        keys = CONTRACT_KEYS if set(CONTRACT_KEYS).issubset(frame.columns) else ["contract"]
        contracts = frame.groupby(keys).ngroups
        print(f"{name:10s} rows: {len(frame):>8} | contracts: {contracts:>6}")
