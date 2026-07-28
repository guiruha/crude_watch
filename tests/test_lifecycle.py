"""Tests for the WTI expiry calculation and the lifecycle enrichment."""
from __future__ import annotations

import pandas as pd

from crudewatch.research.lifecycle import add_lifecycle, wti_last_trading_day


def test_wti_last_trading_day_known_dates():
    # Verified against CME/NYMEX published last-trading-days.
    # 25th of prior month IS a business day -> step back 3 business days.
    assert wti_last_trading_day(2024, 7) == pd.Timestamp("2024-06-20")  # CLN2024
    # 25th falls on a weekend -> roll back, then 3 business days.
    assert wti_last_trading_day(2024, 6) == pd.Timestamp("2024-05-21")  # CLM2024
    assert wti_last_trading_day(2020, 5) == pd.Timestamp("2020-04-21")  # CLK2020 (negative-price day)
    # Holiday interaction near the 25th (Thanksgiving / Christmas).
    assert wti_last_trading_day(2023, 12) == pd.Timestamp("2023-11-20")  # CLZ2023
    assert wti_last_trading_day(2024, 1) == pd.Timestamp("2023-12-19")  # CLF2024


def test_january_contract_uses_prior_december():
    # Delivery month January -> reference month is December of the previous year.
    assert wti_last_trading_day(2025, 1).year == 2024


def test_add_lifecycle_outrights():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-05-02", "2023-11-20"]),
            "close": [80.0, 76.0],
            "month": [12, 12],
            "month_code": ["Z", "Z"],
            "expiry_year": [2023, 2023],
            "contract": ["CLZ2023", "CLZ2023"],
        }
    )
    out = add_lifecycle(df, "outrights")

    assert (out["expiry_date"] == pd.Timestamp("2023-11-20")).all()
    assert out.loc[out["date"] == pd.Timestamp("2023-11-20"), "dte"].iloc[0] == 0
    assert out.loc[out["date"] == pd.Timestamp("2023-05-02"), "dte"].iloc[0] == 202
    assert out["vintage"].tolist() == [2023, 2023]
    assert out["slot"].tolist() == ["Z", "Z"]


def test_add_lifecycle_fly_slot_and_leg():
    # A December fly is governed by its front (Dec) leg; slot is the front month.
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-06"]),
            "close": [0.35],
            "month": [12],
            "month_code": ["Z"],
            "front_year": [2027],
            "contract": ["CLZ2027-CLZ2028-CLZ2029"],
        }
    )
    out = add_lifecycle(df, "flies")

    assert out["vintage"].iloc[0] == 2027
    assert out["slot"].iloc[0] == "Z"
    assert out["expiry_date"].iloc[0] == wti_last_trading_day(2027, 12)


def test_unknown_family_raises():
    try:
        add_lifecycle(pd.DataFrame({"date": []}), "not_a_family")
    except ValueError as exc:
        assert "unknown family" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown family")
