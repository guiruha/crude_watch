"""CrudeWatch — entry point.

Run with:  uv run streamlit run app/main.py

Screens are registered in ``SCREENS`` below; add a new entry to grow the app.
Each screen is a class taking the frames mapping and exposing ``display()``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the ``crudewatch`` package importable when it isn't pip-installed (e.g. on
# Streamlit Community Cloud, which only installs requirements.txt).
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.auth import require_login, sidebar_account
from core.data import load_frame, load_frames
from core.preload import start_preload
from core.runtime import low_memory_mode
from core.selection import render_selection_bar
from core.self_assessment import self_assessment_status, start_self_assessment
from screens.component import ComponentScreen
from screens.contract_exploration import ContractExplorationScreen
from screens.curve import CurveScreen
from screens.pm import PMScreen
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
    "PM": PMScreen,
    "Curva": CurveScreen,
    "Exploración": ContractExplorationScreen,
    "Régimen": lambda frames: ComponentScreen(frames, "regime"),
    "Dirección": lambda frames: ComponentScreen(frames, "direction"),
    "Fuerza": lambda frames: ComponentScreen(frames, "strength"),
    "Nivel": lambda frames: ComponentScreen(frames, "level"),
    "Probabilidades": lambda frames: ComponentScreen(frames, "probabilities"),
    "Evidencia": lambda frames: ComponentScreen(frames, "evidence"),
}

_INFO_SCREENS = {
    "Régimen",
    "Dirección",
    "Fuerza",
    "Nivel",
    "Probabilidades",
    "Evidencia",
}


def _nav_format(name: str) -> str:
    return f":material/info: {name}" if name in _INFO_SCREENS else name


def _dataset_summary(frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Compact stats for the sidebar status card."""
    outrights = frames.get("outrights")
    contracts = sum(f["contract"].nunique() for f in frames.values() if "contract" in f)
    span = "—"
    if outrights is not None and not outrights.empty:
        dates = outrights["date"]
        span = f"{dates.min():%b %Y} \u2013 {dates.max():%b %Y}"
    return {
        "Familias": str(len(frames)),
        "Contratos": f"{contracts:,}",
        "Cobertura": span,
    }


def _assessment_summary(status: dict) -> dict[str, str]:
    state = status.get("state")
    if state == "done":
        agreement = status.get("agreement", float("nan"))
        lo = status.get("ci_low", float("nan"))
        hi = status.get("ci_high", float("nan"))
        if agreement != agreement:
            rate = "sin muestra"
            ci = "—"
        else:
            rate = f"{agreement * 100:.0f}%"
            ci = f"{lo * 100:.0f}-{hi * 100:.0f}%"
        return {
            "Medida": "top-1 por fecha",
            "Acuerdo top-1": rate,
            "IC 95%": ci,
            "Muestra": f"{int(status.get('usable', 0))}/{int(status.get('inspected', 0))}",
            "Warm-up": f"+{int(status.get('warmup_years', 3))} años",
        }
    if state == "error":
        return {"Estado": "error", "Detalle": str(status.get("error", ""))[:28]}
    return {
        "Estado": "calculando",
        "Medida": "top-1 por fecha",
        "Muestra": "~200 fechas",
        "Warm-up": f"+{int(status.get('warmup_years', 3))} años",
    }


if hasattr(st, "fragment"):

    @st.fragment(run_every="10s")
    def _self_assessment_card(family: str, horizon: int) -> None:
        if low_memory_mode():
            sidebar_card({"Estado": "desactivada", "Modo": "memoria baja"})
            return
        start_self_assessment(family, horizon)
        sidebar_card(_assessment_summary(self_assessment_status(family, horizon)))

else:

    def _self_assessment_card(family: str, horizon: int) -> None:
        if low_memory_mode():
            sidebar_card({"Estado": "desactivada", "Modo": "memoria baja"})
            return
        start_self_assessment(family, horizon)
        sidebar_card(_assessment_summary(self_assessment_status(family, horizon)))


def main() -> None:
    require_login()
    if low_memory_mode():
        selection = render_selection_bar()
        frames = {selection.family: load_frame(selection.family)}
    else:
        frames = load_frames()
        selection = render_selection_bar()
        start_preload(selection, list(frames))

    with st.sidebar:
        sidebar_brand()
        st.divider()

        nav_label("Navegación")
        screen_names = ["Exploración"] if low_memory_mode() else list(SCREENS)
        default_screen = "Exploración" if low_memory_mode() else "PM"
        choice = st.radio(
            "Pantalla",
            screen_names,
            index=screen_names.index(default_screen),
            format_func=_nav_format,
            label_visibility="collapsed",
        )

        st.divider()
        nav_label("Datos")
        sidebar_card(_dataset_summary(frames))

        st.divider()
        nav_label("Autoevaluación")
        _self_assessment_card(selection.family, selection.horizon)

        sidebar_account()
        sidebar_footer("CrudeWatch \u00b7 v0.1 \u00b7 Data: CME / ICE")
        st.markdown(
            '<div class="cw-side-copy">\u00a9 guiruha</div>',
            unsafe_allow_html=True,
        )

    SCREENS[choice](frames).display(selection)


if __name__ == "__main__":
    main()
