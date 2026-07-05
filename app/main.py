"""CrudeWatch — entry point.

Run with:  uv run streamlit run app/main.py

Screens are registered in ``SCREENS`` below; add a new entry to grow the app.
Each screen is a class taking the frames mapping and exposing ``display()``.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.data import load_frames
from screens.contract_exploration import ContractExplorationScreen
from screens.spread_analytics import (
    BollingerScreen,
    CompositeScoreScreen,
    SeasonalityScreen,
    TechnicalScreen,
)
from theme.palette import (
    inject_css,
    nav_label,
    sidebar_brand,
    sidebar_card,
    sidebar_footer,
)

st.set_page_config(
    page_title="CrudeWatch",
    page_icon=":material/water_drop:",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# name -> screen class. New screens slot in here.
SCREENS = {
    "Composite Score": CompositeScoreScreen,
    "Contract Exploration": ContractExplorationScreen,
    "Technical Analysis": TechnicalScreen,
    "Seasonality": SeasonalityScreen,
    "Bollinger": BollingerScreen,
}


def _dataset_summary(frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Compact stats for the sidebar status card."""
    outrights = frames.get("outrights")
    contracts = sum(f["contract"].nunique() for f in frames.values() if "contract" in f)
    span = "—"
    if outrights is not None and not outrights.empty:
        dates = outrights["date"]
        span = f"{dates.min():%b %Y} \u2013 {dates.max():%b %Y}"
    return {
        "Families": str(len(frames)),
        "Contracts": f"{contracts:,}",
        "Coverage": span,
    }


def main() -> None:
    frames = load_frames()

    with st.sidebar:
        sidebar_brand()
        st.divider()

        nav_label("Navigation")
        choice = st.radio("Screen", list(SCREENS), label_visibility="collapsed")

        st.divider()
        nav_label("Dataset")
        sidebar_card(_dataset_summary(frames))

        sidebar_footer("CrudeWatch \u00b7 v0.1 \u00b7 Data: CME / ICE")
        st.markdown(
            '<div class="cw-side-copy">\u00a9 guiruha</div>',
            unsafe_allow_html=True,
        )

    SCREENS[choice](frames).display()


if __name__ == "__main__":
    main()
