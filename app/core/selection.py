"""Global top-bar selection shared across every screen.

A single row of pickers (family · date · contract · horizon) rendered above the
active screen. The chosen values live in ``st.session_state`` under fixed keys,
so they persist when the user switches sidebar tabs: each screen just reads the
returned :class:`Selection` instead of drawing its own pickers.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from crudewatch.infra import FAMILY_LABELS

from core.data import load_frame
from core.runtime import low_memory_mode
from core.scoring import (
    HEADLINE_HORIZON,
    HORIZONS,
    contracts_on_date,
    family_date_bounds,
)
from core.validation import validation_for

_FAMILIES: tuple[str, ...] = tuple(FAMILY_LABELS)


def _family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, family.title())


def _horizon_label(family: str):
    def label(horizon: int) -> str:
        validation = validation_for(family, int(horizon))
        suffix = " · OOS" if validation["state"] == "OOS" else ""
        return f"D+{int(horizon)}{suffix}"

    return label


def _family_date_bounds_light(family: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    frame = load_frame(family)
    if frame is None or frame.empty:
        today = pd.Timestamp.today().normalize()
        return today, today
    dates = pd.to_datetime(frame["date"])
    return pd.Timestamp(dates.min()), pd.Timestamp(dates.max())


def _contracts_on_date_light(family: str, as_of: str) -> list[str]:
    frame = load_frame(family)
    if frame is None or frame.empty or "contract" not in frame:
        return []
    stamp = pd.Timestamp(as_of)
    dates = pd.to_datetime(frame["date"])
    grp = frame.assign(_d=dates).groupby("contract")["_d"]
    spans = pd.DataFrame({"min": grp.min(), "max": grp.max()})
    live = spans[(spans["min"] <= stamp) & (spans["max"] >= stamp)]
    if live.empty:
        return []
    current = (
        frame.loc[frame["contract"].isin(live.index) & (dates <= stamp)]
        .sort_values(["contract", "date"])
        .groupby("contract", sort=False)
        .tail(1)
    )
    if "dte" in current.columns:
        current = current.sort_values(["dte", "contract"], ascending=[True, True])
        return current["contract"].astype(str).tolist()
    return sorted(live.index.astype(str).tolist())


@dataclass(frozen=True)
class Selection:
    """The globally selected instrument context, shared by all screens."""

    family: str
    as_of: pd.Timestamp
    as_of_iso: str
    contract: str | None
    horizon: int


def render_selection_bar() -> Selection:
    """Draw the persistent top selector and return the current selection.

    Widget state is keyed on stable names (``sel_family`` / ``sel_date`` /
    ``sel_contract`` / ``sel_horizon``) so it survives navigation. Date and
    contract are clamped to what the chosen family actually offers before their
    widgets read session state, which avoids out-of-range / stale-option errors
    when the family changes.
    """
    with st.container(border=False, key="cw_topbar"):
        st.markdown(
            '<div class="cw-topbar-head"><span></span>Instrumento</div>',
            unsafe_allow_html=True,
        )
        c_family, c_date, c_contract, c_horizon = st.columns(
            [1.4, 1, 2.2, 1], vertical_alignment="bottom"
        )

        with c_family:
            family = st.selectbox(
                "Familia", _FAMILIES, format_func=_family_label, key="sel_family"
            )

        if low_memory_mode():
            min_date, max_date = _family_date_bounds_light(family)
        else:
            min_date, max_date = family_date_bounds(family)
        lo, hi = min_date.date(), max_date.date()
        if "sel_date" not in st.session_state or not (lo <= st.session_state["sel_date"] <= hi):
            st.session_state["sel_date"] = hi
        with c_date:
            as_of = st.date_input("Fecha", min_value=lo, max_value=hi, key="sel_date")
        as_of_iso = pd.Timestamp(as_of).strftime("%Y-%m-%d")

        contracts = (
            _contracts_on_date_light(family, as_of_iso)
            if low_memory_mode()
            else contracts_on_date(family, as_of_iso)
        )
        with c_contract:
            if contracts:
                if st.session_state.get("sel_contract") not in contracts:
                    st.session_state["sel_contract"] = contracts[0]
                contract = st.selectbox(
                    f"Contrato ({len(contracts)} activos esa fecha)",
                    contracts,
                    key="sel_contract",
                )
            else:
                st.selectbox("Contrato", ["—"], disabled=True)
                st.caption("Ningún contrato cotizaba en esa fecha.")
                contract = None

        with c_horizon:
            if "sel_horizon" not in st.session_state:
                st.session_state["sel_horizon"] = HEADLINE_HORIZON
            horizon = st.selectbox(
                "Horizonte (D+)",
                HORIZONS,
                key="sel_horizon",
                format_func=_horizon_label(family),
                help="Días de trading a los que se miden retornos, probabilidades y análogos.",
            )

    return Selection(
        family=family,
        as_of=pd.Timestamp(as_of),
        as_of_iso=as_of_iso,
        contract=contract,
        horizon=int(horizon),
    )
