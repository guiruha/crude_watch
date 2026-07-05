"""Synthetic spreads & flies, built from outright leg closes.

Each outright leg contributes its full-history close. A structure is priced
only on dates where EVERY leg traded (inner join across legs), avoiding
stale-leg artifacts.

Sign convention: front minus back for spreads, so a positive value means
backwardation (nearer contract richer than deferred). A fly is
``P(front) - 2*P(mid) + P(back)``; positive means the middle is cheap relative
to the wings.
"""
from __future__ import annotations

import pandas as pd

from crudewatch.infra.constants import CODE_BY_MONTH, FLY_MONTHS
from crudewatch.data_preparation.helpers import last_12_months


def _leg_close(outright_legs: pd.DataFrame) -> pd.DataFrame:
    """Minimal (date, contract, close) view used to look up each leg's price."""
    return outright_legs[["date", "contract", "close"]]


def _shift_contract(month: pd.Series, expiry_year: pd.Series, gap_months: int) -> pd.Series:
    """Label of the contract ``gap_months`` after (month, expiry_year)."""
    idx = expiry_year * 12 + (month - 1) + gap_months
    far_year = idx // 12
    far_month = idx % 12 + 1
    far_code = far_month.map(CODE_BY_MONTH)
    return "CL" + far_code + far_year.astype(int).astype(str)


def build_calendar_spread(
    outright_legs: pd.DataFrame, gap_months: int, structure: str, trim: bool = True
) -> pd.DataFrame:
    """Two-leg calendar spread, ``close = P(near) - P(far)`` where the far leg is
    ``gap_months`` after the near leg (3=quarterly, 6=semestral, 12=yearly)."""
    near = outright_legs[["date", "contract", "close", "month", "month_code", "expiry_year"]].rename(
        columns={"contract": "near_contract", "close": "near_close",
                 "month_code": "near_month_code", "month": "near_month", "expiry_year": "near_year"}
    )
    near["far_contract"] = _shift_contract(near["near_month"], near["near_year"], gap_months)

    far = _leg_close(outright_legs).rename(columns={"contract": "far_contract", "close": "far_close"})

    merged = near.merge(far, on=["date", "far_contract"], how="inner")
    merged["close"] = merged["near_close"] - merged["far_close"]
    merged["contract"] = merged["near_contract"] + "-" + merged["far_contract"]
    merged["structure"] = structure
    merged["gap_months"] = gap_months
    merged["n_legs"] = 2

    if trim:
        merged = last_12_months(merged, keys=["contract"])
    return merged.sort_values(["date", "near_year", "near_month"]).reset_index(drop=True)


def build_flies(
    outright_legs: pd.DataFrame,
    structure: str = "fly",
    trim: bool = True,
    months: tuple[str, ...] = FLY_MONTHS,
) -> pd.DataFrame:
    """Same-month, consecutive-year butterfly:
    ``close = P(year) - 2*P(year+1) + P(year+2)`` (e.g. Dec-Dec-Dec).

    Only front months in ``months`` are built (default: the liquid ones), since
    a fly's 2-year-deferred back leg barely trades for other months.
    """
    front = outright_legs[["date", "contract", "close", "month", "month_code", "expiry_year"]].rename(
        columns={"contract": "front_contract", "close": "front_close", "expiry_year": "front_year"}
    )
    front = front[front["month_code"].isin(months)]
    front["mid_contract"] = "CL" + front["month_code"] + (front["front_year"] + 1).astype(str)
    front["back_contract"] = "CL" + front["month_code"] + (front["front_year"] + 2).astype(str)

    mid = _leg_close(outright_legs).rename(columns={"contract": "mid_contract", "close": "mid_close"})
    back = _leg_close(outright_legs).rename(columns={"contract": "back_contract", "close": "back_close"})

    merged = (
        front.merge(mid, on=["date", "mid_contract"], how="inner")
             .merge(back, on=["date", "back_contract"], how="inner")
    )
    merged["close"] = merged["front_close"] - 2 * merged["mid_close"] + merged["back_close"]
    merged["contract"] = (
        merged["front_contract"] + "-" + merged["mid_contract"] + "-" + merged["back_contract"]
    )
    merged["structure"] = structure
    merged["n_legs"] = 3

    if trim:
        merged = last_12_months(merged, keys=["contract"])
    return merged.sort_values(["date", "front_year", "month"]).reset_index(drop=True)
