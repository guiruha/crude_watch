"""CALENDAR SPREADS (WTI): two WTI legs traded as one listed instrument,
e.g. "CLF0-CLG0". These come straight from the feed (not synthesized)."""
from __future__ import annotations

import pandas as pd

from crudewatch.infra.constants import CAL_RE, CONTRACT_KEYS, MONTH_CODES
from crudewatch.data_preparation.helpers import add_expiry_year, basic_clean, resolve_far_year


def build_calendars(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["symbol"].str.extract(CAL_RE)
    calendars = basic_clean(df[parts["m1"].notna()])

    p = calendars["symbol"].str.extract(CAL_RE)
    calendars["near_month_code"] = p["m1"]
    calendars["far_month_code"] = p["m2"]
    calendars["near_month"] = p["m1"].map(MONTH_CODES)
    calendars["far_month"] = p["m2"].map(MONTH_CODES)
    calendars["near_year_digit"] = p["y1"].astype(int)
    calendars["far_year_digit"] = p["y2"].astype(int)

    # Near leg: resolve per instrument from its last trade date.
    add_expiry_year(calendars, month_col="near_month", digit_col="near_year_digit", out_col="near_year")

    # Far leg: first year ending in far_digit that falls strictly after the near
    # leg (constant per contract, so resolve once and broadcast).
    far_meta = calendars.groupby(CONTRACT_KEYS)[
        ["far_month", "far_year_digit", "near_year", "near_month"]
    ].first()
    far_years = {
        key: resolve_far_year(int(r.far_month), int(r.far_year_digit), int(r.near_year), int(r.near_month))
        for key, r in far_meta.iterrows()
    }
    calendars["far_year"] = calendars.set_index(CONTRACT_KEYS).index.map(far_years).to_numpy()

    calendars["contract"] = (
        "CL" + calendars["near_month_code"] + calendars["near_year"].astype(str)
        + "-CL" + calendars["far_month_code"] + calendars["far_year"].astype(str)
    )

    return calendars.sort_values(["date", "near_year", "near_month"]).reset_index(drop=True)
