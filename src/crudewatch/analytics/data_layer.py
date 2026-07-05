"""Fixed-date series construction from the outright price history.

Every structure/vintage series is a linear combination of outright legs, so the
whole data layer reduces to:

1. pivot the outright closes (and volumes) into a wide date x contract matrix, and
2. for a given :class:`~crudewatch.analytics.structures.Structure` and base year,
   dot the relevant leg columns with their coefficients, dropping any day on
   which a leg does not trade.

This uniformly covers monthly spreads, quarterly spreads and butterflies
(``A - 2B + C``). The functions are pure (no Streamlit), so the app can wrap
them in ``st.cache_data``.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crudewatch.analytics.structures import Leg, Structure
from crudewatch.infra.constants import MONTH_CODES


def _contract(month_code: str, year: int) -> str:
    """Outright contract label for a leg, e.g. ("Z", 2024) -> "CLZ2024"."""
    return f"CL{month_code}{year}"


@dataclass(frozen=True)
class PriceMatrix:
    """Wide outright matrices: index = date, columns = contract label."""
    close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def contracts(self) -> set[str]:
        return set(self.close.columns)


def build_price_matrix(outrights: pd.DataFrame) -> PriceMatrix:
    """Pivot the tidy outrights frame into wide close & volume matrices."""
    close = outrights.pivot_table(index="date", columns="contract", values="close")
    volume = outrights.pivot_table(index="date", columns="contract", values="volume")
    close = close.sort_index()
    volume = volume.reindex(index=close.index, columns=close.columns)
    return PriceMatrix(close=close, volume=volume)


def _leg_contracts(structure: Structure, base_year: int) -> list[tuple[Leg, str]]:
    return [(leg, _contract(leg.month_code, base_year + leg.year_offset)) for leg in structure.legs]


def available_years(matrix: PriceMatrix, structure: Structure) -> list[int]:
    """Base years for which *every* leg of the structure has a listed contract."""
    contracts = matrix.contracts
    years: list[int] = []
    # Outright years present in the matrix bound the search.
    all_years = {int(c[3:]) for c in contracts if c.startswith("CL") and c[3:].isdigit()}
    for year in sorted(all_years):
        wanted = [c for _, c in _leg_contracts(structure, year)]
        if all(c in contracts for c in wanted):
            years.append(year)
    return years


def build_series(matrix: PriceMatrix, structure: Structure, base_year: int) -> pd.DataFrame:
    """The fixed-date OHLCV-lite series for one structure vintage.

    Returns columns ``date``, ``close`` and (when every leg has volume that day)
    ``volume`` = the min leg volume, the standard bottleneck proxy for a
    multi-leg product. Days on which any leg is missing are dropped so the linear
    combination is always fully populated.
    """
    legs = _leg_contracts(structure, base_year)
    close_cols = matrix.close[[c for _, c in legs]].dropna(how="any")
    if close_cols.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])

    combo = sum(leg.coefficient * close_cols[c] for leg, c in legs)

    vol_cols = matrix.volume.reindex(index=close_cols.index)[[c for _, c in legs]]
    volume = vol_cols.min(axis=1) if vol_cols.notna().all(axis=None) else pd.Series(index=close_cols.index, dtype=float)

    out = pd.DataFrame({"date": close_cols.index, "close": combo.to_numpy(), "volume": volume.to_numpy()})
    return out.reset_index(drop=True)


SEASONAL_COLUMNS = ["vintage", "date", "day_of_year", "days_to_expiry", "close"]


def near_leg_expiry(structure: Structure, base_year: int) -> pd.Timestamp:
    """Approximate expiry of the structure's near (earliest) leg.

    WTI crude terminates trading ~3 business days before the 25th of the month
    *preceding* the delivery month; the 20th of the preceding month is a close
    enough anchor for a seasonal (±days) axis and, crucially, is a fixed date
    independent of how far the dataset extends — so live/future vintages get a
    correct days-to-expiry instead of collapsing to the data's end.
    """
    near = structure.legs[0]  # legs are ordered earliest-first
    month = MONTH_CODES[near.month_code]
    year = base_year + near.year_offset
    prior_month = 12 if month == 1 else month - 1
    prior_year = year - 1 if month == 1 else year
    return pd.Timestamp(prior_year, prior_month, 20)


def seasonal_stack(matrix: PriceMatrix, structure: Structure) -> pd.DataFrame:
    """All vintages of a structure aligned for cross-vintage seasonal comparison.

    Long frame with ``vintage`` (base year), ``date``, ``day_of_year``,
    ``days_to_expiry`` and ``close``. Because a fixed-date spread trades for
    ~2 calendar years and crosses the year-end, the raw day-of-year wraps; the
    canonical seasonal axis is therefore ``days_to_expiry`` — days before the
    vintage's last observation (its expiry proxy), which is monotonic and aligns
    every vintage at a common reference (0 = expiry). This is the backbone of
    Chapter 2.
    """
    parts: list[pd.DataFrame] = []
    for year in available_years(matrix, structure):
        s = build_series(matrix, structure, year)
        if s.empty:
            continue
        expiry = near_leg_expiry(structure, year)
        s = s.assign(
            vintage=year,
            day_of_year=s["date"].dt.dayofyear,
            days_to_expiry=(expiry - s["date"]).dt.days,
        )
        parts.append(s[SEASONAL_COLUMNS])
    if not parts:
        return pd.DataFrame(columns=SEASONAL_COLUMNS)
    return pd.concat(parts, ignore_index=True)
