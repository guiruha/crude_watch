"""Contract lifecycle: real WTI expiry dates, days-to-expiry, slot and vintage.

The scoring model has to compare a contract against *analogous* contracts at the
**same point of their life** (Bloque D: level / cheap-expensive by vintage and
seasonality). That requires a proper life-phase axis, and the honest way to get
it is the actual expiration date — not a proxy deduced from the last observed
trade date (a thin, decade-recycled contract can stop printing early).

WTI (CL) termination-of-trading rule (NYMEX rulebook 200.07):

    Trading ceases on the THIRD business day prior to the 25th calendar day of
    the month preceding the delivery month. If that 25th is a non-business day,
    trading ceases on the third business day prior to the last business day
    preceding the 25th.

Both branches collapse to: take the 25th of the prior month, roll it *back* to
the previous business day if it is not one, then step back three business days.

"Business day" here means an Exchange business day, so we use a NYMEX energy
holiday calendar (New Year, MLK, Presidents, Good Friday, Memorial, Juneteenth
from 2022, Independence, Labor, Thanksgiving, Christmas) — deliberately WITHOUT
Columbus Day / Veterans Day, which CME energy trades through. No extra
dependency: the calendar is built from ``pandas.tseries.holiday``.

The synthetic and inter-commodity families are governed by their *near / front*
CL leg — the position dies when the nearest leg expires — so the CL rule applies
exactly to outrights, calendars, quarterly/semestral/yearly spreads and flies.
For ``brent_wti`` we use the WTI leg as the governing expiry (the Brent leg has
its own rule; near-leg alignment is the right approximation for life-phase).
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay


class NYMEXEnergyCalendar(AbstractHolidayCalendar):
    """Full-closure holidays for NYMEX energy (CL) markets.

    Fixed-date holidays use ``nearest_workday`` observance (the exchange shifts
    them to the nearest weekday), matching CME practice. Columbus Day and
    Veterans Day are intentionally excluded — energy futures trade on them.
    """

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


_CBD = CustomBusinessDay(calendar=NYMEXEnergyCalendar())


@lru_cache(maxsize=None)
def wti_last_trading_day(delivery_year: int, delivery_month: int) -> pd.Timestamp:
    """Last trading day of the CL contract for a given delivery month/year.

    Implements NYMEX rule 200.07 (see module docstring). Cached because it is
    called once per distinct (year, month) across many contract rows.
    """
    ref_year, ref_month = (delivery_year, delivery_month - 1) if delivery_month > 1 else (delivery_year - 1, 12)
    twenty_fifth = pd.Timestamp(year=ref_year, month=ref_month, day=25)
    anchor = _CBD.rollback(twenty_fifth)  # the 25th if a business day, else the previous one
    return (anchor - 3 * _CBD).normalize()


# Per-family: which (near/front) leg governs expiry, and how to build the
# seasonal ``slot`` key used to group analogous contracts across vintages.
FAMILY_LIFECYCLE: dict[str, dict[str, object]] = {
    "outrights": {"month": "month", "year": "expiry_year", "slot": ("month_code",)},
    "cracks": {"month": "month", "year": "expiry_year", "slot": ("product", "month_code")},
    "calendars": {"month": "near_month", "year": "near_year", "slot": ("near_month_code", "far_month_code")},
    "brent_wti": {"month": "wti_month", "year": "wti_year", "slot": ("wti_month_code", "brent_month_code")},
    "quarterly": {"month": "near_month", "year": "near_year", "slot": ("near_month_code",)},
    "semestral": {"month": "near_month", "year": "near_year", "slot": ("near_month_code",)},
    "yearly": {"month": "near_month", "year": "near_year", "slot": ("near_month_code",)},
    "flies": {"month": "month", "year": "front_year", "slot": ("month_code",)},
}


def add_lifecycle(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    """Return a copy of ``frame`` with ``expiry_date``, ``dte``, ``vintage`` and ``slot``.

    - ``expiry_date`` — real CL last-trading-day of the governing (near/front) leg.
    - ``dte`` — integer days-to-expiry (``expiry_date - date``); negative after expiry.
    - ``vintage`` — the governing leg's expiry year (the "which contract" axis).
    - ``slot`` — seasonal key shared by analogous contracts across vintages
      (e.g. ``"Z"`` for December flies, ``"Z-H"`` for a Dec-Mar calendar).
    """
    try:
        cfg = FAMILY_LIFECYCLE[family]
    except KeyError as exc:
        raise ValueError(f"unknown family {family!r}; expected one of {list(FAMILY_LIFECYCLE)}") from exc

    month_col, year_col = cfg["month"], cfg["year"]
    slot_cols: tuple[str, ...] = cfg["slot"]
    missing = [c for c in (month_col, year_col, *slot_cols) if c not in frame.columns]
    if missing:
        raise KeyError(f"{family} frame is missing lifecycle columns: {missing}")

    out = frame.copy()
    years = out[year_col].astype(int)
    months = out[month_col].astype(int)

    unique_pairs = {(int(y), int(m)) for y, m in zip(years, months)}
    expiry_by_pair = {pair: wti_last_trading_day(*pair) for pair in unique_pairs}

    out["expiry_date"] = pd.to_datetime([expiry_by_pair[(y, m)] for y, m in zip(years, months)])
    out["dte"] = (out["expiry_date"] - pd.to_datetime(out["date"])).dt.days
    out["vintage"] = years.to_numpy()

    slot = out[slot_cols[0]].astype(str)
    for col in slot_cols[1:]:
        slot = slot + "-" + out[col].astype(str)
    out["slot"] = slot.to_numpy()
    return out
