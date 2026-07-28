"""Contract Exploration screen: chart the globally selected contract's history.

Family and contract come from the shared top selector (``Selection``); this
screen shows the full price history of that contract (the date/horizon in the
selector are ignored here — it is a browse view, not a point-in-time score).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from crudewatch.plots import price_volume_figure

from core.selection import Selection
from theme.palette import CHART_ACCENT, title_block

# family key -> (chart label, y-axis title, fill area to zero)
_CHART_STYLE: dict[str, tuple[str, str, bool]] = {
    "outrights": ("Outright", "Close ($/bbl)", True),
    "calendars": ("Calendar", "Spread ($/bbl)", False),
    "quarterly": ("Quarterly", "Spread ($/bbl)", False),
    "semestral": ("Semestral", "Spread ($/bbl)", False),
    "yearly": ("Yearly", "Spread ($/bbl)", False),
    "flies": ("Fly", "Fly ($/bbl)", False),
    "cracks": ("Crack", "Crack ($/bbl)", False),
    "brent_wti": ("Brent\u2013WTI", "Brent \u2212 WTI ($/bbl)", False),
}


class ContractExplorationScreen:
    """Browse the full price history of the globally selected contract."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def display(self, selection: Selection) -> None:
        title_block(
            "Contract Exploration",
            "Historia de precios completa del contrato seleccionado en el menú superior.",
        )

        if selection.contract is None:
            st.info("Selecciona una familia y una fecha con contratos activos en el menú superior.")
            return

        family = selection.family
        contract = selection.contract
        label, y_title, fill = _CHART_STYLE.get(family, (family.title(), "Precio", False))

        frame = self.frames.get(family)
        if frame is None or "contract" not in frame:
            st.warning("No hay datos para esta familia.")
            return
        series = frame[frame["contract"] == contract].sort_values("date")
        if series.empty:
            st.warning("Sin serie de precios para este contrato.")
            return

        self._render_stats(series)
        self._render_chart(series, contract, label, y_title, fill)
        self._render_table(series)

    def _render_stats(self, series: pd.DataFrame) -> None:
        close = series["close"]
        cols = st.columns(6)
        cells = [
            ("Last", f"{close.iloc[-1]:.2f}"),
            ("Min", f"{close.min():.2f}"),
            ("Max", f"{close.max():.2f}"),
            ("Mean", f"{close.mean():.2f}"),
            ("Observations", f"{len(series):,}"),
            ("Span", f"{series['date'].min():%b %Y} \u2013 {series['date'].max():%b %Y}"),
        ]
        for col, (label, value) in zip(cols, cells):
            col.metric(label, value)

    def _render_chart(
        self, series: pd.DataFrame, contract: str, label: str, y_title: str, fill: bool
    ) -> None:
        fig = price_volume_figure(
            series,
            title=f"{label} \u2014 {contract}",
            y_title=y_title,
            fill_to_zero=fill,
            color=CHART_ACCENT,
        )
        st.plotly_chart(fig, width="stretch")

    def _render_table(self, series: pd.DataFrame) -> None:
        with st.expander("Underlying data"):
            cols = [c for c in ("date", "contract", "close", "volume") if c in series.columns]
            st.dataframe(
                series[cols].sort_values("date", ascending=False),
                width="stretch",
                hide_index=True,
            )
