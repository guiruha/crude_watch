"""Unit tests for synthetic-structure OHLC reconstruction (executable basis)."""
from __future__ import annotations

import pandas as pd

from crudewatch.data_preparation.spreads import build_calendar_spread, build_flies


def _legs(rows: list[dict]) -> pd.DataFrame:
    """Minimal outright-leg frame with the columns the spread builders need."""
    return pd.DataFrame(rows)


def test_calendar_spread_open_is_leg_open_difference():
    # Quarterly (gap=3): near CLF2024 (Jan), far CLJ2024 (Apr).
    d = pd.Timestamp("2023-06-01")
    legs = _legs([
        {"date": d, "contract": "CLF2024", "close": 100.0, "open": 99.0,
         "month": 1, "month_code": "F", "expiry_year": 2024},
        {"date": d, "contract": "CLJ2024", "close": 95.0, "open": 94.0,
         "month": 4, "month_code": "J", "expiry_year": 2024},
    ])
    spr = build_calendar_spread(legs, gap_months=3, structure="quarterly", trim=False)
    assert len(spr) == 1
    row = spr.iloc[0]
    assert row["close"] == 100.0 - 95.0        # near_close - far_close
    assert row["open"] == 99.0 - 94.0          # near_open - far_open (executable)
    assert row["contract"] == "CLF2024-CLJ2024"


def test_fly_open_is_wing_minus_two_body():
    # Dec fly across 3 consecutive years: front CLZ2020, mid CLZ2021, back CLZ2022.
    d = pd.Timestamp("2020-03-02")
    legs = _legs([
        {"date": d, "contract": "CLZ2020", "close": 50.0, "open": 49.0,
         "month": 12, "month_code": "Z", "expiry_year": 2020},
        {"date": d, "contract": "CLZ2021", "close": 48.0, "open": 47.0,
         "month": 12, "month_code": "Z", "expiry_year": 2021},
        {"date": d, "contract": "CLZ2022", "close": 47.0, "open": 46.5,
         "month": 12, "month_code": "Z", "expiry_year": 2022},
    ])
    fly = build_flies(legs, trim=False)
    assert len(fly) == 1
    row = fly.iloc[0]
    assert row["close"] == 50.0 - 2 * 48.0 + 47.0     # front - 2*mid + back
    assert row["open"] == 49.0 - 2 * 47.0 + 46.5
    assert row["contract"] == "CLZ2020-CLZ2021-CLZ2022"
