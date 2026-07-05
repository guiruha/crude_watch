"""OUTRIGHTS: single WTI contracts, e.g. "CLM1" = WTI June 2011."""
from __future__ import annotations

import pandas as pd

from crudewatch.infra.constants import MONTH_CODES, OUTRIGHT_RE
from crudewatch.data_preparation.helpers import add_expiry_year, basic_clean, dedup_clean


def build_outrights(df: pd.DataFrame, trim: bool = True) -> pd.DataFrame:
    """Single WTI legs.

    ``trim=True`` keeps each contract's last 12 months (the published
    ``outrights``); ``trim=False`` keeps full history, used as the price source
    for synthetic spreads so deferred legs still overlap in time.
    """
    parts = df["symbol"].str.extract(OUTRIGHT_RE)
    clean = basic_clean if trim else dedup_clean
    outrights = clean(df[parts["month_code"].notna()])

    p = outrights["symbol"].str.extract(OUTRIGHT_RE)
    outrights["month_code"] = p["month_code"]
    outrights["month"] = outrights["month_code"].map(MONTH_CODES)
    outrights["year_digit"] = p["year_digit"].astype(int)

    add_expiry_year(outrights)

    # Unique per-contract label using the full expiry year, e.g. "CLZ2019".
    outrights["contract"] = "CL" + outrights["month_code"] + outrights["expiry_year"].astype(str)

    return outrights.sort_values(["date", "expiry_year", "month"]).reset_index(drop=True)
