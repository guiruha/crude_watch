"""BRENT-WTI SPREAD: Brent (BZ) vs WTI (CL) inter-commodity spread,
e.g. "CLZ2-BZZ2" (same-month arb) or "CLF3-BZG3" (cross-month).

The exchange lists this with WTI as the first leg, so its quoted price is
``WTI - Brent`` (typically negative, since WTI trades at a discount). We flip it
to the conventional **Brent - WTI** premium (typically positive) by negating the
OHLC bar (high/low swap to keep the bar well-formed).
"""
from __future__ import annotations

import pandas as pd

from crudewatch.infra.constants import BRENT_WTI_RE, MONTH_CODES
from crudewatch.data_preparation.helpers import add_expiry_year, basic_clean


def build_brent_wti(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["symbol"].str.extract(BRENT_WTI_RE)
    brent_wti = basic_clean(df[parts["wti_m"].notna()])

    p = brent_wti["symbol"].str.extract(BRENT_WTI_RE)
    brent_wti["wti_month_code"] = p["wti_m"]
    brent_wti["brent_month_code"] = p["bz_m"]
    brent_wti["wti_month"] = p["wti_m"].map(MONTH_CODES)
    brent_wti["brent_month"] = p["bz_m"].map(MONTH_CODES)
    brent_wti["wti_year_digit"] = p["wti_y"].astype(int)
    brent_wti["brent_year_digit"] = p["bz_y"].astype(int)

    # Resolve each leg's expiry year once per instrument from its last trade date.
    add_expiry_year(brent_wti, month_col="wti_month", digit_col="wti_year_digit", out_col="wti_year")
    add_expiry_year(brent_wti, month_col="brent_month", digit_col="brent_year_digit", out_col="brent_year")

    # Flip WTI - Brent (listed) to the conventional Brent - WTI premium.
    o, h, low, c = brent_wti["open"], brent_wti["high"], brent_wti["low"], brent_wti["close"]
    brent_wti["open"] = -o
    brent_wti["high"] = -low  # swap so high >= low still holds after negation
    brent_wti["low"] = -h
    brent_wti["close"] = -c

    brent_wti["contract"] = (
        "CL" + brent_wti["wti_month_code"] + brent_wti["wti_year"].astype(str)
        + "-BZ" + brent_wti["brent_month_code"] + brent_wti["brent_year"].astype(str)
    )

    return brent_wti.sort_values(["date", "wti_year", "wti_month"]).reset_index(drop=True)
