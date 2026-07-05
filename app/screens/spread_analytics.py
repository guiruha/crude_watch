"""Spread Analytics screens — one per analytical view.

Technical Analysis, Seasonality and Bollinger are each their own left-menu
entry. They share the structure selector (Level -> Structure -> Vintage) and a
common cached data layer via session state, so switching views keeps the active
spread. The composite Score is rendered on the main page above every view.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from crudewatch.analytics import Level, Structure, structures_for

from chapters import (
    BollingerChapter,
    Chapter,
    ChapterContext,
    ScoreChapter,
    SeasonalityChapter,
    TechnicalChapter,
)
from core.data import spread_current_vintage, spread_seasonal, spread_series, spread_years
from theme.palette import SUBTEXT, title_block

# The default structure every screen opens on: the Mar-Jun (m3-m6) quarterly spread.
_DEFAULT_LEVEL = Level.QUARTERLY
_DEFAULT_LABEL = "Mar-Jun"


class _SpreadScreen:
    """Base screen: shared structure selector + one analytical chapter."""

    #: The analytical chapter this screen renders. Set by subclasses.
    chapter: Chapter

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames  # series come from the cached matrix, not here

    def display(self) -> None:
        title_block(self.chapter.name, self.chapter.subtitle)

        with st.container(border=True):
            selection = self._select_structure()
        if selection is None:
            return
        structure, base_year = selection

        series = spread_series(structure.key, base_year)
        seasonal = spread_seasonal(structure.key)
        ctx = ChapterContext(structure, base_year, series, seasonal)

        self.chapter.render(ctx)

    # -- shared structure selector ------------------------------------------

    def _select_structure(self) -> tuple[Structure, int] | None:
        levels = [lvl.value for lvl in Level]
        # Wide Level column so its four options stay on one line; Structure and
        # Vintage sit to the right, with a trailing spacer keeping them compact.
        left, mid, right, _spacer = st.columns([2.3, 1.5, 1, 0.5])

        with left:
            level_label = st.segmented_control(
                "Level", options=levels, default=_DEFAULT_LEVEL.value, key="sa_level",
            ) or _DEFAULT_LEVEL.value
        level = Level(level_label)
        members = structures_for(level)

        with mid:
            labels = [s.label for s in members]
            label = st.selectbox("Structure", labels, key=f"sa_struct_{level.value}")
        structure = next(s for s in members if s.label == label)

        years = spread_years(structure.key)
        if not years:
            with right:
                st.warning("No vintages available.")
            return None
        current = spread_current_vintage(structure.key)
        default_idx = years.index(current) if current in years else len(years) - 1
        with right:
            base_year = st.selectbox("Vintage", years, index=default_idx, key="sa_year")

        legs = " ".join(
            f"{'+' if leg.coefficient > 0 else ''}{leg.coefficient:g}·CL{leg.month_code}"
            f"{base_year + leg.year_offset}"
            for leg in structure.legs
        )
        st.markdown(
            f'<div style="color:{SUBTEXT};font-size:12px;margin-top:4px">Legs: {legs}</div>',
            unsafe_allow_html=True,
        )
        return structure, base_year


class CompositeScoreScreen(_SpreadScreen):
    chapter = ScoreChapter()


class TechnicalScreen(_SpreadScreen):
    chapter = TechnicalChapter()


class SeasonalityScreen(_SpreadScreen):
    chapter = SeasonalityChapter()


class BollingerScreen(_SpreadScreen):
    chapter = BollingerChapter()
