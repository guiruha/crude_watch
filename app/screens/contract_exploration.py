"""Contract Exploration screen: pick a structure and a contract, then chart it."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from crudewatch.plots import price_volume_figure

from theme.palette import CHART_ACCENT, title_block


@dataclass(frozen=True)
class Structure:
    """One selectable instrument family."""
    key: str                    # dict key in the frames mapping
    label: str                  # UI label
    y_title: str                # chart y-axis label
    fill: bool                  # area fill to zero (only sensible for outrights)
    category: str               # grouping shown in the top-level type picker
    product: str | None = None  # optional row filter on the frame's ``product`` column


# Ordered instrument families, grouped by category for the two-tier menu.
STRUCTURES: list[Structure] = [
    Structure("outrights", "Outright", "Close ($/bbl)", True, "Outright"),
    Structure("calendars", "Calendar", "Spread ($/bbl)", False, "Spread"),
    Structure("quarterly", "Quarterly", "Spread ($/bbl)", False, "Spread"),
    Structure("semestral", "Semestral", "Spread ($/bbl)", False, "Spread"),
    Structure("yearly", "Yearly", "Spread ($/bbl)", False, "Spread"),
    Structure("flies", "Fly", "Fly ($/bbl)", False, "Fly"),
    Structure("cracks", "HO crack", "Crack ($/bbl)", False, "Inter-commodity", product="HO"),
    Structure("cracks", "RB crack", "Crack ($/bbl)", False, "Inter-commodity", product="RB"),
    Structure("brent_wti", "Brent\u2013WTI", "Brent \u2212 WTI ($/bbl)", False, "Inter-commodity"),
]

# Category -> its structures, preserving the order above.
CATEGORIES: dict[str, list[Structure]] = {}
for _s in STRUCTURES:
    CATEGORIES.setdefault(_s.category, []).append(_s)


class ContractExplorationScreen:
    """Browse a single contract's price history across every instrument family."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def display(self) -> None:
        title_block(
            "Contract Exploration",
            "Select an instrument family, then a contract, to explore its price history.",
        )

        with st.container(border=True):
            structure = self._pick_structure()
            frame = self.frames[structure.key]
            if structure.product is not None:
                frame = frame[frame["product"] == structure.product]
            contract = self._pick_contract(frame)

        if contract is None:
            return

        series = frame[frame["contract"] == contract].sort_values("date")
        self._render_stats(series, structure)
        self._render_chart(series, contract, structure)
        self._render_table(series)

    # -- menu ----------------------------------------------------------------

    def _pick_structure(self) -> Structure:
        """Two-tier picker: instrument type, then the structure within it."""
        categories = list(CATEGORIES)
        category = st.segmented_control(
            "Instrument type",
            options=categories,
            default=categories[0],
            key="cw_category",
        ) or categories[0]

        members = CATEGORIES[category]
        if len(members) == 1:
            return members[0]

        labels = [s.label for s in members]
        label = st.segmented_control(
            "Structure",
            options=labels,
            default=labels[0],
            key=f"cw_structure_{category}",
        ) or labels[0]
        return next(s for s in members if s.label == label)

    def _pick_contract(self, frame: pd.DataFrame) -> str | None:
        contracts = sorted(frame["contract"].unique())
        if not contracts:
            st.warning("No contracts available for this structure.")
            return None

        left, right = st.columns([3, 1], vertical_alignment="bottom")
        with right:
            query = st.text_input(
                "Filter contracts",
                placeholder="e.g. Z2022",
                help="Type part of a contract code to narrow the list.",
            )
        options = [c for c in contracts if query.upper() in c.upper()] if query else contracts
        if not options:
            with left:
                st.info(f"No contracts match '{query}'.")
            return None
        with left:
            return st.selectbox(f"Contract ({len(options)} of {len(contracts)})", options)

    def _render_stats(self, series: pd.DataFrame, structure: Structure) -> None:
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

    def _render_chart(self, series: pd.DataFrame, contract: str, structure: Structure) -> None:
        fig = price_volume_figure(
            series,
            title=f"{structure.label} \u2014 {contract}",
            y_title=structure.y_title,
            fill_to_zero=structure.fill,
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
