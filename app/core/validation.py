"""Backtest-summary lookups for dashboard context."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
_STATS_PATH = _ROOT / "docs" / "reports" / "backtest" / "strategy_stats.csv"


def _empty() -> dict:
    return {
        "state": "Sin validación",
        "n_trades": 0,
        "win_rate": float("nan"),
        "sharpe": float("nan"),
    }


@st.cache_data(show_spinner=False, max_entries=1)
def validation_table() -> pd.DataFrame:
    if not _STATS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(_STATS_PATH)


def validation_for(family: str, horizon: int) -> dict:
    table = validation_table()
    if table.empty:
        return _empty()
    sub = table[(table["family"] == family) & (table["horizon"] == int(horizon))]
    if sub.empty:
        return _empty()
    row = sub.iloc[0]
    return {
        "state": "OOS",
        "n_trades": int(row.get("n_trades", 0)),
        "win_rate": float(row.get("win_rate", float("nan"))),
        "sharpe": float(row.get("sharpe", float("nan"))),
    }
