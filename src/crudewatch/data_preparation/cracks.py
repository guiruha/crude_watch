"""CRACK SPREADS: refined product vs. crude in CME "CL:C1" notation,
e.g. "CL:C1 HO-CL Z7" (heating oil) or "CL:C1 RB-CL Z7" (gasoline)."""
from __future__ import annotations

import pandas as pd

from crudewatch.infra.constants import CRACK_RE, MONTH_CODES
from crudewatch.data_preparation.helpers import add_expiry_year, basic_clean


def build_cracks(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["symbol"].str.extract(CRACK_RE)
    cracks = basic_clean(df[parts["product"].notna()])

    p = cracks["symbol"].str.extract(CRACK_RE)
    cracks["product"] = p["product"]  # HO = heating-oil crack, RB = gasoline (RBOB) crack
    cracks["month_code"] = p["month_code"]
    cracks["month"] = p["month_code"].map(MONTH_CODES)
    cracks["year_digit"] = p["year_digit"].astype(int)

    add_expiry_year(cracks)

    cracks["contract"] = (
        "CL:C1 " + cracks["product"] + "-CL " + cracks["month_code"] + cracks["expiry_year"].astype(str)
    )

    return cracks.sort_values(["date", "product", "expiry_year", "month"]).reset_index(drop=True)
