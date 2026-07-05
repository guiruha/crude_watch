"""Auxiliary functions shared across the data-cleaning and preparation builders.

Two concerns live here:

* **Cleaning** - deduping, dropping malformed bars, and trimming each contract
  to its most-liquid final 12 months.
* **Expiry resolution** - turning the decade-ambiguous single-digit year carried
  by a futures symbol into a full 4-digit expiry year, resolved once per
  physical contract.
"""
from __future__ import annotations

import pandas as pd

from crudewatch.infra.constants import CONTRACT_KEYS


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def last_12_months(frame: pd.DataFrame, keys: list[str] = CONTRACT_KEYS) -> pd.DataFrame:
    """Keep the final 12 months of each contract's trading history.

    For every contract we broadcast its last trade date back to each row (via
    groupby-transform, so the mask stays row-aligned) and keep rows within a
    calendar-accurate 12-month window ending on that date. Keying on
    CONTRACT_KEYS (not symbol alone) means each decade-reused contract keeps its
    own last year, instead of only the most recent reuse.
    """
    last = frame.groupby(keys)["date"].transform("max")
    return frame[frame["date"] >= last - pd.DateOffset(months=12)]


def dedup_clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Cleaning steps common to every symbol family, WITHOUT the 12-month trim:
    one bar per symbol/day, drop malformed OHLC bars, drop constant columns.
    """
    out = frame.drop_duplicates(subset=["date", "symbol"]).copy()  # one bar per symbol/day
    out = out[out["high"] >= out["low"]]                           # drop malformed OHLC bars
    out = out.drop(columns=["rtype", "publisher_id"])              # constant columns, no signal
    return out


def basic_clean(frame: pd.DataFrame) -> pd.DataFrame:
    """``dedup_clean`` plus the last-12-months trim used by the published frames."""
    return last_12_months(dedup_clean(frame))


# ---------------------------------------------------------------------------
# Expiry-year resolution
# ---------------------------------------------------------------------------
# Near-dated futures symbols only carry the LAST digit of the expiry year (the
# "9" in "CLZ9"), which is decade-ambiguous and gets reused every 10 years. A
# single trade date can't disambiguate reliably near decade boundaries, but each
# instrument_id IS a single physical contract, so we resolve the year once per
# instrument from the range of dates it traded. Deep-deferred contracts sidestep
# the ambiguity by quoting an explicit two-digit year (e.g. "CLZ29" = 2029),
# which we detect by a token value >= 10 and read directly as 20xx.

def resolve_expiry_year(month: int, year_digit: int, ref_date: pd.Timestamp) -> int:
    """Full 4-digit expiry year for a contract, from its LAST observed trade date.

    A two-digit token (value >= 10) is an explicit year and read as ``2000 + n``.
    For a single digit we pick the smallest year ending in ``year_digit`` whose
    contract month has not already passed as of ``ref_date`` (a contract trades
    right up to its expiry, so its last trade date sits just before the contract
    month; the 1st of the contract month is a safe upper bound because WTI
    expires the month *before* the contract month).
    """
    if year_digit >= 10:
        return 2000 + year_digit
    base = ref_date.year - (ref_date.year % 10) + year_digit
    for y in (base - 10, base, base + 10, base + 20):
        if pd.Timestamp(year=y, month=month, day=1) >= ref_date:
            return y
    return base + 20


def resolve_far_year(far_month: int, far_digit: int, near_year: int, near_month: int) -> int:
    """Expiry year of a calendar spread's far leg.

    A two-digit token is an explicit year (``2000 + n``); otherwise it's the
    first year ending in ``far_digit`` that falls strictly after the (already
    resolved) near leg."""
    if far_digit >= 10:
        return 2000 + far_digit
    base = near_year - (near_year % 10) + far_digit
    for y in (base - 10, base, base + 10, base + 20):
        if (y, far_month) > (near_year, near_month):
            return y
    return base + 20


def add_expiry_year(frame, month_col="month", digit_col="year_digit", out_col="expiry_year"):
    """Resolve the expiry year once per (instrument_id, symbol) and broadcast it.

    month/digit are constant within a contract, and the reference date is that
    contract's last trade date.
    """
    g = frame.groupby(CONTRACT_KEYS)
    ref = g["date"].max()
    meta = g[[month_col, digit_col]].first()
    years = {
        key: resolve_expiry_year(int(meta.at[key, month_col]), int(meta.at[key, digit_col]), ref[key])
        for key in ref.index
    }
    frame[out_col] = frame.set_index(CONTRACT_KEYS).index.map(years).to_numpy()
    return frame
