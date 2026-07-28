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

from core.scoring import (
    HEADLINE_HORIZON,
    HORIZONS,
    contracts_on_date,
    family_date_bounds,
)

_FAMILIES: tuple[str, ...] = tuple(FAMILY_LABELS)


def _family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, family.title())


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

        min_date, max_date = family_date_bounds(family)
        lo, hi = min_date.date(), max_date.date()
        if "sel_date" not in st.session_state or not (lo <= st.session_state["sel_date"] <= hi):
            st.session_state["sel_date"] = hi
        with c_date:
            as_of = st.date_input("Fecha", min_value=lo, max_value=hi, key="sel_date")
        as_of_iso = pd.Timestamp(as_of).strftime("%Y-%m-%d")

        contracts = contracts_on_date(family, as_of_iso)
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
                help="Días de trading a los que se miden retornos, probabilidades y análogos.",
            )

    return Selection(
        family=family,
        as_of=pd.Timestamp(as_of),
        as_of_iso=as_of_iso,
        contract=contract,
        horizon=int(horizon),
    )
