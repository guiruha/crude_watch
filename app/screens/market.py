"""Cross-family radar board."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from crudewatch.infra import FAMILY_LABELS

from core.scoring import active_contract_scores_cached
from core.selection import Selection
from screens.opportunity import _REGIME_ES
from theme.palette import title_block


class MarketScreen:
    """One board for the best live structures across every instrument family."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def display(self, selection: Selection) -> None:
        title_block(
            "Mercado",
            "Radar cross-family: outrights, calendars, flies, cracks y spreads en una sola tabla.",
        )
        rows = []
        for family in FAMILY_LABELS:
            if family not in self.frames:
                continue
            panel = active_contract_scores_cached(family, selection.horizon, selection.as_of_iso)
            if panel.empty:
                continue
            rows.append(panel.head(25).assign(family=family, family_label=FAMILY_LABELS.get(family, family)))
        if not rows:
            st.info("No hay contratos activos para la fecha seleccionada.")
            return

        board = pd.concat(rows, ignore_index=True)
        quick = st.toggle("Top por familia", value=True)
        if quick:
            board = (
                board.sort_values(["family", "stretch_rank", "volume"], ascending=[True, False, False])
                .groupby("family", sort=False)
                .head(5)
            )
        board = board.sort_values(["stretch_rank", "volume", "dte"], ascending=[False, False, True])
        view = board.rename(
            columns={
                "family_label": "Familia",
                "contract": "Contrato",
                "sign": "Signo",
                "stretch_rank": "Extremo vs familia",
                "signed_composite": "Valor interno",
                "regime": "Régimen",
                "level": "Nivel",
                "volume": "Volume",
                "liquidity": "Liquidez",
                "point_value_usd": "$ / pt",
                "cost_points": "Coste pts",
                "cost_usd": "Coste USD",
                "validation_state": "OOS familia",
                "validation_hit": "Hit %",
                "cohort_n": "Cohorte n",
                "cohort_sharpe": "Sharpe cohorte anual.",
                "pm_read": "Lectura PM",
                "descriptive_bias": "Sesgo descriptivo",
                "slot": "Slot",
                "life_phase": "Vida",
                "vol_regime": "Vol",
                "risk_count": "Flags",
                "close": "Close",
                "dte": "DTE",
            }
        )
        view["Régimen"] = view["Régimen"].map(lambda x: _REGIME_ES.get(x, x))
        cols = [
            "Familia",
            "Contrato",
            "Extremo vs familia",
            "Lectura PM",
            "Sesgo descriptivo",
            "Régimen",
            "Nivel",
            "Cohorte n",
            "Sharpe cohorte anual.",
            "Volume",
            "Liquidez",
            "Vol",
            "Coste USD",
            "Flags",
            "DTE",
        ]
        if "Hit %" in view:
            view["Hit %"] = view["Hit %"] * 100.0
        st.caption(f"{len(view)} estructuras vivas · ordenadas por extremo vs familia y volumen.")
        st.dataframe(
            view[cols],
            width=1900,
            hide_index=True,
            column_config={
                "Extremo vs familia": st.column_config.NumberColumn(format="%.0f"),
                "Valor interno": st.column_config.NumberColumn(format="%+.0f"),
                "Nivel": st.column_config.NumberColumn(format="%+.0f"),
                "Cohorte n": st.column_config.NumberColumn(format="%.0f"),
                "Sharpe cohorte anual.": st.column_config.NumberColumn(format="%+.2f"),
                "Volume": st.column_config.NumberColumn(format="%.0f"),
                "$ / pt": st.column_config.NumberColumn(format="$%.0f"),
                "Coste pts": st.column_config.NumberColumn(format="%.3f"),
                "Coste USD": st.column_config.NumberColumn(format="$%.0f"),
                "Hit %": st.column_config.NumberColumn(format="%.0f%%"),
                "Close": st.column_config.NumberColumn(format="%.2f"),
                "DTE": st.column_config.NumberColumn(format="%.0f"),
            },
        )
