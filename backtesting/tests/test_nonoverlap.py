"""Tests for the non-overlapping Opportunity Score backtest.

The property that makes this backtest worth having is that **no two trades share
a day**. Overlapping holds are what inflated every statistic in the bucket
sweep; here each trade is a genuinely independent observation, so the reported
Sharpe can be annualised honestly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.research.nonoverlap import (
    nonoverlapping_trades,
    summarize,
)


def _panel(opp, n_contracts=1, start="2015-01-01", price=None):
    """One contract per column of ``opp``; ramping prices unless given."""
    n = len(opp[0])
    dates = pd.bdate_range(start, periods=n)
    parts = []
    for c in range(n_contracts):
        px = np.arange(n, dtype=float) if price is None else np.asarray(price, dtype=float)
        parts.append(pd.DataFrame({
            "date": dates, "contract": f"C{c}",
            "opp": np.asarray(opp[c], dtype=float),
            "open": px, "close": px,
        }))
    return pd.concat(parts, ignore_index=True)


def test_no_two_trades_share_a_day():
    """The load-bearing property: holding periods are pairwise disjoint."""
    rng = np.random.default_rng(0)
    opp = [rng.uniform(-100, 100, 200)]          # fires constantly
    t = nonoverlapping_trades(_panel(opp), hold_bars=5, enter_at=50.0)

    assert len(t) > 5, "expected several trades from a noisy signal"
    days = []
    for r in t.itertuples():
        days.append(pd.date_range(r.entry_date, r.exit_date, freq="D"))
    allday = np.concatenate([d.to_numpy() for d in days])
    assert len(allday) == len(np.unique(allday)), "two trades overlap in time"
    # and entries are strictly ordered after the previous exit
    assert (t["entry_date"].to_numpy()[1:] > t["exit_date"].to_numpy()[:-1]).all()


def test_hold_length_is_fixed():
    opp = [[0.0] * 3 + [80.0] + [0.0] * 30]
    t = nonoverlapping_trades(_panel(opp), hold_bars=7, enter_at=50.0)
    assert len(t) == 1
    assert t["bars"].iloc[0] == 7


def test_entry_fills_at_next_open_not_the_signal_close():
    """Signal at close[t]; the fill must be open[t+1] — no same-bar look-ahead."""
    opp = [[0.0, 90.0] + [0.0] * 20]
    px = np.arange(22, dtype=float) * 10.0      # open == close == 0,10,20,...
    t = nonoverlapping_trades(_panel(opp, price=px), hold_bars=3, enter_at=50.0)
    assert len(t) == 1
    # signal on bar 1 -> entry on bar 2 (price 20), exit on bar 5 (price 50)
    assert t["entry_price"].iloc[0] == pytest.approx(20.0)
    assert t["exit_price"].iloc[0] == pytest.approx(50.0)


def test_long_and_short_pnl_signs():
    up = np.arange(30, dtype=float)
    down = np.arange(30, dtype=float)[::-1].copy()
    long_t = nonoverlapping_trades(_panel([[0.0, 90.0] + [0.0] * 28], price=up),
                                   hold_bars=5, enter_at=50.0)
    short_t = nonoverlapping_trades(_panel([[0.0, -90.0] + [0.0] * 28], price=up),
                                    hold_bars=5, enter_at=50.0)
    assert long_t["side"].iloc[0] == 1 and long_t["gross_pnl"].iloc[0] > 0
    assert short_t["side"].iloc[0] == -1 and short_t["gross_pnl"].iloc[0] < 0
    # a short into a falling market makes money
    short_win = nonoverlapping_trades(_panel([[0.0, -90.0] + [0.0] * 28], price=down),
                                      hold_bars=5, enter_at=50.0)
    assert short_win["gross_pnl"].iloc[0] > 0


def test_cost_is_charged_once_per_trade():
    opp = [[0.0, 90.0] + [0.0] * 20]
    free = nonoverlapping_trades(_panel(opp), hold_bars=5, enter_at=50.0, cost=0.0)
    paid = nonoverlapping_trades(_panel(opp), hold_bars=5, enter_at=50.0, cost=0.25)
    assert paid["cost"].iloc[0] == pytest.approx(0.25)
    assert free["net_pnl"].iloc[0] - paid["net_pnl"].iloc[0] == pytest.approx(0.25)


def test_threshold_suppresses_weak_signals():
    opp = [[10.0] * 40]                       # never reaches 50
    t = nonoverlapping_trades(_panel(opp), hold_bars=5, enter_at=50.0)
    assert t.empty
    assert list(t.columns)                     # still a typed, usable frame


def test_strongest_contract_wins_on_a_shared_date():
    """With several eligible contracts on one date, take the highest conviction."""
    p = pd.concat([
        _panel([[0.0, 60.0] + [0.0] * 20]).assign(contract="WEAK"),
        _panel([[0.0, 95.0] + [0.0] * 20]).assign(contract="STRONG"),
    ], ignore_index=True)
    t = nonoverlapping_trades(p, hold_bars=5, enter_at=50.0)
    assert len(t) == 1
    assert t["contract"].iloc[0] == "STRONG"


def test_summarize_annualises_on_independent_trades():
    t = pd.DataFrame({
        "net_pnl": [1.0, -0.5, 1.5, -0.5, 1.0],
        "entry_date": pd.to_datetime(
            ["2015-01-05", "2015-04-05", "2015-07-05", "2015-10-05", "2016-01-05"]),
        "exit_date": pd.to_datetime(
            ["2015-02-05", "2015-05-05", "2015-08-05", "2015-11-05", "2016-02-05"]),
        "side": [1, -1, 1, -1, 1],
    })
    s = summarize(t)
    assert s["trades"] == 5
    assert s["total_pnl"] == pytest.approx(2.5)
    assert s["mean_pnl"] == pytest.approx(0.5)
    assert s["hit_rate"] == pytest.approx(0.6)
    # Sharpe scales by sqrt(trades per year), not by sqrt(252)
    expected = (np.mean(t["net_pnl"]) / np.std(t["net_pnl"], ddof=1)) * np.sqrt(s["trades_per_year"])
    assert s["sharpe"] == pytest.approx(expected)


def test_summarize_handles_empty():
    s = summarize(pd.DataFrame(columns=["net_pnl", "entry_date", "exit_date", "side"]))
    assert s["trades"] == 0
    assert np.isnan(s["sharpe"])
