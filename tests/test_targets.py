"""Tests for forward-outcome (label) construction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from crudewatch.research.targets import HORIZONS, add_forward_returns


def test_default_horizon_grid():
    # The definitive research/strategy grid; scripts rely on this staying in sync.
    assert HORIZONS == (1, 3, 5, 10, 20, 25, 30)


def _frame(closes: list[float], contract: str = "A") -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "contract": [contract] * n,
            "close": closes,
        }
    )


def _frame_ohlc(opens: list[float], closes: list[float], contract: str = "A") -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "contract": [contract] * n,
            "open": opens,
            "close": closes,
        }
    )


def test_default_horizons_emit_all_short_and_long_columns():
    # Need enough rows for max(HORIZONS)=30 plus one tail NaN row.
    closes = list(np.linspace(10.0, 40.0, 40))
    out = add_forward_returns(_frame(closes))
    for h in HORIZONS:
        for prefix in ("fwd", "mfe", "mae", "fwd_vol", "mfe_vol", "mae_vol"):
            assert f"{prefix}_{h}" in out.columns
    # Spot-check short-horizon columns are populated (not all NaN).
    assert not out["fwd_1"].iloc[:-1].isna().all()
    assert not out["mfe_3"].iloc[:-3].isna().all()
    assert not out["mae_5"].iloc[:-5].isna().all()
    assert not out["fwd_10"].iloc[:-10].isna().all()


def test_short_horizon_h1_monotonic_legacy_entry():
    # No ``open`` column: entry = close[t]; +1/bar ramp -> fwd_1 = close[t+1]-close[t].
    closes = [10.0 + i for i in range(8)]
    out = add_forward_returns(_frame(closes), horizons=(1,))
    assert out["fwd_1"].tolist()[:-1] == [1.0] * 7
    # h=1 window is a single bar: mfe_1 and mae_1 match fwd_1.
    assert out["mfe_1"].tolist()[:-1] == [1.0] * 7
    assert out["mae_1"].tolist()[:-1] == [1.0] * 7


def test_short_horizon_h1_monotonic_executable_basis():
    # With ``open``: entry = open[t+1]; monotonic path with fixed close-open gap.
    opens = [10.0 + i for i in range(8)]
    closes = [10.5 + i for i in range(8)]
    out = add_forward_returns(_frame_ohlc(opens, closes), horizons=(1,))
    # fwd_1[t] = close[t+1] - open[t+1] = 0.5 on every row with a next open.
    assert out["fwd_1"].tolist()[:-1] == [0.5] * 7
    assert out["mfe_1"].tolist()[:-1] == [0.5] * 7
    assert out["mae_1"].tolist()[:-1] == [0.5] * 7


def test_executable_basis_entry_is_next_open():
    # With an ``open`` column the entry is open[t+1], not close[t].
    opens = [9.0, 7.0, 11.0, 10.0, 14.0]
    closes = [10.0, 8.0, 12.0, 9.0, 15.0]
    out = add_forward_returns(_frame_ohlc(opens, closes), horizons=(1, 3))

    # fwd_1[t] = close[t+1] - open[t+1]
    assert out["fwd_1"].tolist()[:-1] == [1.0, 1.0, -1.0, 1.0]
    assert np.isnan(out["fwd_1"].iloc[-1])  # no next open on the tail

    # fwd_3[t0] = close[3] - open[1] = 9 - 7 = 2; fwd_3[t1] = close[4]-open[2] = 15-11 = 4
    assert out["fwd_3"].iloc[0] == 2.0
    assert out["fwd_3"].iloc[1] == 4.0


def test_executable_basis_excursions_from_entry():
    opens = [9.0, 7.0, 11.0, 10.0, 14.0]
    closes = [10.0, 8.0, 12.0, 9.0, 15.0]
    out = add_forward_returns(_frame_ohlc(opens, closes), horizons=(2, 3))
    # Entry = open[1] = 7. Window close[1..2] = [8, 12] -> mfe = 5, mae = 1.
    assert out["mfe_2"].iloc[0] == 5.0
    assert out["mae_2"].iloc[0] == 1.0
    # bars-to over horizon 3: rel = [8,12,9]-7 = [1,5,2] -> max at bar 2, min at bar 1.
    assert out["bars_to_mfe"].iloc[0] == 2
    assert out["bars_to_mae"].iloc[0] == 1


def test_forward_returns_point_values():
    df = _frame([10.0, 8.0, 12.0, 9.0, 15.0])
    out = add_forward_returns(df, horizons=(1, 2, 3))

    assert out["fwd_1"].tolist()[:-1] == [-2.0, 4.0, -3.0, 6.0]
    assert np.isnan(out["fwd_1"].iloc[-1])  # tail has no next bar

    # fwd_3 needs t+3; only the first two rows have it.
    assert out["fwd_3"].iloc[0] == 9.0 - 10.0  # close[3]-close[0]
    assert out["fwd_3"].iloc[1] == 15.0 - 8.0  # close[4]-close[1]
    assert np.isnan(out["fwd_3"].iloc[2])


def test_mfe_mae_excursions():
    df = _frame([10.0, 8.0, 12.0, 9.0, 15.0])
    out = add_forward_returns(df, horizons=(1, 2, 3))

    # Over [t+1, t+2] from t0: values 8, 12 -> mfe = +2, mae = -2.
    assert out["mfe_2"].iloc[0] == 2.0
    assert out["mae_2"].iloc[0] == -2.0
    # Over [t+1, t+3] from t0: values 8, 12, 9 -> mfe = +2, mae = -2.
    assert out["mfe_3"].iloc[0] == 2.0
    assert out["mae_3"].iloc[0] == -2.0


def test_bars_to_extreme():
    df = _frame([10.0, 8.0, 12.0, 9.0, 15.0])
    out = add_forward_returns(df, horizons=(1, 2, 3))
    # Longest horizon = 3. From t0 over next 3 bars (8,12,9): max at bar 2, min at bar 1.
    assert out["bars_to_mfe"].iloc[0] == 2
    assert out["bars_to_mae"].iloc[0] == 1
    # Rows without a full 3-bar window are NaN.
    assert np.isnan(out["bars_to_mfe"].iloc[-1])


def test_vol_normalised_forward_return():
    # Long ramp so the ATR(10) is well defined by the tail rows.
    closes = list(np.linspace(10.0, 40.0, 40))
    df = _frame(closes)
    out = add_forward_returns(df, horizons=(5,), price_col="close")
    row = out.iloc[20]
    sigma = out["close"].diff().abs().ewm(alpha=1 / 10, adjust=False, min_periods=10).mean().iloc[20]
    expected = row["fwd_5"] / (sigma * np.sqrt(5))
    assert np.isclose(row["fwd_vol_5"], expected)
    # Vol-normalised excursions divide by sigma (no sqrt(h)).
    assert np.isclose(row["mfe_vol_5"], row["mfe_5"] / sigma)
    assert np.isclose(row["mae_vol_5"], row["mae_5"] / sigma)


def test_no_leak_across_contracts():
    a = _frame([10.0, 11.0, 12.0], contract="A")
    b = _frame([100.0, 90.0, 80.0], contract="B")
    out = add_forward_returns(pd.concat([a, b], ignore_index=True), horizons=(1,))

    a_out = out[out["contract"] == "A"].sort_values("date")
    b_out = out[out["contract"] == "B"].sort_values("date")
    # Last bar of each contract must be NaN (no bleed from the other contract).
    assert np.isnan(a_out["fwd_1"].iloc[-1])
    assert np.isnan(b_out["fwd_1"].iloc[-1])
    assert a_out["fwd_1"].tolist()[:-1] == [1.0, 1.0]
    assert b_out["fwd_1"].tolist()[:-1] == [-10.0, -10.0]
