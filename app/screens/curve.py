"""Forward-curve and structure board."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data import load_frames
from core.scoring import active_contract_scores_cached
from core.selection import Selection
from screens.opportunity import _base_layout
from theme.palette import ACCENT, BEAR, BORDER, BULL, SUBTEXT, SURFACE, TEXT, title_block


def _latest_live(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(frame["date"])
    spans = frame.assign(_d=dates).groupby("contract")["_d"].agg(["min", "max"])
    live = spans[(spans["min"] <= as_of) & (spans["max"] >= as_of)]
    if live.empty:
        return pd.DataFrame()
    return (
        frame.loc[frame["contract"].isin(live.index) & (dates <= as_of)]
        .sort_values(["contract", "date"])
        .groupby("contract", sort=False)
        .tail(1)
        .copy()
    )


@st.cache_data(show_spinner=False, max_entries=512)
def _latest_live_cached(family: str, as_of_iso: str) -> pd.DataFrame:
    frame = load_frames().get(family)
    if frame is None or frame.empty:
        return pd.DataFrame()
    return _latest_live(frame, pd.Timestamp(as_of_iso))


def _month_label(code, year) -> str:
    if pd.isna(code) or pd.isna(year):
        return "-"
    return f"{code}{int(year) % 100:02d}"


def _fmt(value: float, spec: str = ".2f", dash: str = "-") -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return dash
    if value != value:
        return dash
    return format(value, spec)


def _read_box(title: str, rows: list[tuple[str, str]], tone: str = ACCENT) -> None:
    body = "".join(
        f"<div style='display:flex;gap:10px;margin-top:5px;'>"
        f"<div style='min-width:132px;color:{SUBTEXT};font-size:12px;font-weight:800;text-transform:uppercase;'>{k}</div>"
        f"<div style='color:{TEXT};font-size:14px;font-weight:600;'>{v}</div>"
        f"</div>"
        for k, v in rows
    )
    st.markdown(
        f"""
        <div style="border:1px solid {BORDER};background:{SURFACE};border-left:4px solid {tone};
                    border-radius:8px;padding:12px 14px;margin:8px 0 12px 0;">
            <div style="color:{TEXT};font-size:17px;font-weight:800;">{title}</div>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _volume_read(row: pd.Series) -> str:
    vol = row.get("volume", float("nan"))
    if vol != vol:
        return "sin volumen publicado"
    if vol < 50:
        return f"liquidez baja ({vol:.0f})"
    if vol < 500:
        return f"liquidez media ({vol:.0f})"
    return f"liquidez alta ({vol:.0f})"


class CurveScreen:
    """Curve/structure view for the selected as-of date."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def display(self, selection: Selection) -> None:
        title_block(
            "Curva",
            "Strip, calendario y liquidez en una sola pantalla para leer estructura antes que señal.",
        )
        as_of = selection.as_of
        self._outright_strip(as_of)
        self._calendar_matrix(selection)
        self._fly_grid(selection)

    def _outright_strip(self, as_of: pd.Timestamp) -> None:
        st.markdown("### WTI strip")
        frame = self.frames.get("outrights")
        if frame is None or frame.empty:
            st.info("Sin outrights para construir la curva.")
            return
        current = _latest_live_cached("outrights", pd.Timestamp(as_of).strftime("%Y-%m-%d"))
        if current.empty:
            st.info("Sin contratos outright vivos en la fecha seleccionada.")
            return
        current["label"] = [
            _month_label(c, y) for c, y in zip(current["month_code"], current["expiry_year"])
        ]
        current = current.sort_values(["expiry_year", "month"]).head(36)
        self._strip_read(current)
        vol = current["volume"] if "volume" in current else pd.Series([0] * len(current))
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=current["label"],
                y=current["close"],
                mode="lines+markers",
                marker=dict(
                    size=(vol.fillna(0).clip(lower=0) ** 0.35 + 5),
                    color=vol.fillna(0),
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Volume"),
                ),
                line=dict(color=ACCENT, width=2),
                name="CL",
                customdata=current[["contract", "volume"]].to_numpy(),
                hovertemplate="%{customdata[0]}<br>Close %{y:.2f}<br>Vol %{customdata[1]:.0f}<extra></extra>",
            )
        )
        fig.update_yaxes(title_text="$ / bbl")
        fig.update_xaxes(title_text="Contrato")
        st.plotly_chart(_base_layout(fig, 360), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        front = current.iloc[0]
        back = current.iloc[min(len(current) - 1, 11)]
        c1.metric("Front", f"{front['contract']} · {front['close']:.2f}")
        c2.metric("12m spread", f"{front['close'] - back['close']:+.2f}")
        c3.metric("Front volume", f"{float(front.get('volume', 0)):.0f}")

    def _strip_read(self, current: pd.DataFrame) -> None:
        front = current.iloc[0]
        back_12 = current.iloc[min(len(current) - 1, 11)]
        back_last = current.iloc[-1]
        spread_12 = float(front["close"] - back_12["close"])
        full_slope = float(front["close"] - back_last["close"])
        shape = "backwardation" if spread_12 > 0 else "contango" if spread_12 < 0 else "plana"
        tone = BULL if spread_12 > 0 else BEAR if spread_12 < 0 else SUBTEXT
        top_vol = current.sort_values("volume", ascending=False).iloc[0] if "volume" in current else front
        _read_box(
            "Cómo leer el WTI strip",
            [
                ("Qué es", "cada punto es un contrato mensual; la línea enseña la forma de la curva."),
                ("Lectura", f"curva en {shape}: front {front['contract']} contra {back_12['contract']} = {spread_12:+.2f} $/bbl."),
                ("PM", "front caro vs deferred = tightness cercana; front barato vs deferred = exceso/almacenaje."),
                ("Liquidez", f"mayor volumen en {top_vol['contract']} ({float(top_vol.get('volume', 0)):.0f}); tratar deferred finos con cuidado."),
                ("Rango total", f"{front['contract']} - {back_last['contract']} = {full_slope:+.2f} $/bbl."),
            ],
            tone,
        )

    def _calendar_matrix(self, selection: Selection) -> None:
        st.markdown("### Calendar matrix")
        frame = self.frames.get("calendars")
        if frame is None or frame.empty:
            st.info("Sin calendars para construir matriz.")
            return
        current = _latest_live_cached("calendars", selection.as_of_iso)
        if current.empty:
            st.info("Sin calendars vivos en la fecha seleccionada.")
            return

        scores = active_contract_scores_cached("calendars", selection.horizon, selection.as_of_iso)
        rank_map = dict(zip(scores.get("contract", []), scores.get("stretch_rank", [])))
        current["stretch_rank"] = current["contract"].map(rank_map)
        current["near_label"] = [
            _month_label(c, y) for c, y in zip(current["near_month_code"], current["near_year"])
        ]
        current["far_label"] = [
            _month_label(c, y) for c, y in zip(current["far_month_code"], current["far_year"])
        ]
        current = current.sort_values(["near_year", "near_month", "far_year", "far_month"]).head(250)
        self._calendar_read(current)

        pivot = current.pivot_table(
            index="near_label",
            columns="far_label",
            values="close",
            aggfunc="last",
            sort=False,
        )
        hover = current.pivot_table(
            index="near_label",
            columns="far_label",
            values="stretch_rank",
            aggfunc="last",
            sort=False,
        ).reindex_like(pivot)
        tab_spread, tab_score = st.tabs(["Spread", "Extremo"])
        with tab_spread:
            fig = go.Figure(
                go.Heatmap(
                    z=pivot.to_numpy(),
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale=[[0, BEAR], [0.5, "#202620"], [1, BULL]],
                    zmid=0,
                    colorbar=dict(title="Spread"),
                    customdata=hover.to_numpy(),
                    hovertemplate="Near %{y}<br>Far %{x}<br>Spread %{z:.2f}<br>Extremo vs familia %{customdata:.0f}<extra></extra>",
                )
            )
            fig.update_xaxes(title_text="Far leg")
            fig.update_yaxes(title_text="Near leg", autorange="reversed")
            st.plotly_chart(_base_layout(fig, 520), use_container_width=True)
        with tab_score:
            fig = go.Figure(
                go.Heatmap(
                    z=hover.to_numpy(),
                    x=hover.columns,
                    y=hover.index,
                    colorscale=[[0, BEAR], [0.5, "#202620"], [1, BULL]],
                    zmin=0,
                    zmax=100,
                    colorbar=dict(title="Extremo"),
                    customdata=pivot.to_numpy(),
                    hovertemplate="Near %{y}<br>Far %{x}<br>Extremo vs familia %{z:.0f}<br>Spread %{customdata:.2f}<extra></extra>",
                )
            )
            fig.update_xaxes(title_text="Far leg")
            fig.update_yaxes(title_text="Near leg", autorange="reversed")
            st.plotly_chart(_base_layout(fig, 520), use_container_width=True)

        cols = [c for c in ["contract", "close", "volume", "stretch_rank", "near_label", "far_label"] if c in current.columns]
        table = current[cols].rename(
            columns={
                "contract": "Contrato",
                "close": "Spread",
                "volume": "Volume",
                "stretch_rank": "Extremo vs familia",
                "near_label": "Near",
                "far_label": "Far",
            }
        )
        st.dataframe(table, width=1600, hide_index=True)

    def _calendar_read(self, current: pd.DataFrame) -> None:
        if current.empty:
            return
        ranked = current.dropna(subset=["stretch_rank"]).copy()
        extreme = ranked.sort_values("stretch_rank", ascending=False).iloc[0] if not ranked.empty else current.iloc[0]
        spread = float(extreme.get("close", float("nan")))
        sign_read = (
            "near leg por encima del far leg"
            if spread > 0
            else "near leg por debajo del far leg"
            if spread < 0
            else "near y far alineados"
        )
        near_idx = float(extreme.get("near_year", 0)) * 12 + float(extreme.get("near_month", 0))
        far_idx = float(extreme.get("far_year", 0)) * 12 + float(extreme.get("far_month", 0))
        gap = far_idx - near_idx
        zone = "front/near" if gap <= 3 else "media" if gap <= 12 else "larga"
        _read_box(
            "Cómo leer el Calendar matrix",
            [
                ("Qué es", "cada celda es near - far; filas = vencimiento cercano, columnas = vencimiento lejano."),
                ("Mayor extremo", f"{extreme['contract']} · spread {_fmt(spread, '+.2f')} · extremo {_fmt(extreme.get('stretch_rank'), '.0f')}."),
                ("Lectura", f"{sign_read}; estructura {zone} ({_fmt(gap, '.0f')} meses entre legs)."),
                ("Operabilidad", _volume_read(extreme)),
                ("PM", "elige la celda extrema, revisa volumen y vuelve a PM para evidencia/cohorte antes de priorizar."),
            ],
            BULL if spread > 0 else BEAR if spread < 0 else SUBTEXT,
        )

    def _fly_grid(self, selection: Selection) -> None:
        st.markdown("### Fly grid")
        frame = self.frames.get("flies")
        if frame is None or frame.empty:
            st.info("Sin flies para construir grid.")
            return
        current = _latest_live_cached("flies", selection.as_of_iso)
        if current.empty:
            st.info("Sin flies vivos en la fecha seleccionada.")
            return

        scores = active_contract_scores_cached("flies", selection.horizon, selection.as_of_iso)
        rank_map = dict(zip(scores.get("contract", []), scores.get("stretch_rank", [])))
        current["stretch_rank"] = current["contract"].map(rank_map)
        current["month_label"] = current["month_code"].astype(str)
        current = current.sort_values(["front_year", "month"]).head(180)
        self._fly_read(current)

        spread = current.pivot_table(
            index="month_label",
            columns="front_year",
            values="close",
            aggfunc="last",
            sort=False,
        )
        opp = current.pivot_table(
            index="month_label",
            columns="front_year",
            values="stretch_rank",
            aggfunc="last",
            sort=False,
        ).reindex_like(spread)

        tab_spread, tab_score = st.tabs(["Fly", "Extremo"])
        with tab_spread:
            fig = go.Figure(
                go.Heatmap(
                    z=spread.to_numpy(),
                    x=spread.columns,
                    y=spread.index,
                    colorscale=[[0, BEAR], [0.5, "#202620"], [1, BULL]],
                    zmid=0,
                    colorbar=dict(title="Fly"),
                    customdata=opp.to_numpy(),
                    hovertemplate="Mes %{y}<br>Front %{x}<br>Fly %{z:.2f}<br>Extremo vs familia %{customdata:.0f}<extra></extra>",
                )
            )
            fig.update_xaxes(title_text="Front year")
            fig.update_yaxes(title_text="Month")
            st.plotly_chart(_base_layout(fig, 420), use_container_width=True)
        with tab_score:
            fig = go.Figure(
                go.Heatmap(
                    z=opp.to_numpy(),
                    x=opp.columns,
                    y=opp.index,
                    colorscale=[[0, BEAR], [0.5, "#202620"], [1, BULL]],
                    zmin=0,
                    zmax=100,
                    colorbar=dict(title="Extremo"),
                    customdata=spread.to_numpy(),
                    hovertemplate="Mes %{y}<br>Front %{x}<br>Extremo vs familia %{z:.0f}<br>Fly %{customdata:.2f}<extra></extra>",
                )
            )
            fig.update_xaxes(title_text="Front year")
            fig.update_yaxes(title_text="Month")
            st.plotly_chart(_base_layout(fig, 420), use_container_width=True)

        st.caption("Los flies son estructuras derivadas de legs; no se muestra volumen directo si la fuente no lo publica.")
        table = current[
            ["contract", "close", "stretch_rank", "month_code", "front_year", "front_contract", "mid_contract", "back_contract"]
        ].rename(
            columns={
                "contract": "Contrato",
                "close": "Fly",
                "stretch_rank": "Extremo vs familia",
                "month_code": "Mes",
                "front_year": "Front year",
                "front_contract": "Front",
                "mid_contract": "Mid",
                "back_contract": "Back",
            }
        )
        st.dataframe(table, width=1800, hide_index=True)

    def _fly_read(self, current: pd.DataFrame) -> None:
        if current.empty:
            return
        ranked = current.dropna(subset=["stretch_rank"]).copy()
        extreme = ranked.sort_values("stretch_rank", ascending=False).iloc[0] if not ranked.empty else current.iloc[0]
        fly = float(extreme.get("close", float("nan")))
        _read_box(
            "Cómo leer el Fly grid",
            [
                ("Qué es", "cada celda mide la curvatura: front - 2*mid + back."),
                ("Mayor extremo", f"{extreme['contract']} · fly {_fmt(fly, '+.2f')} · extremo {_fmt(extreme.get('stretch_rank'), '.0f')}."),
                ("Lectura", "un fly extremo dice que el centro de esa estructura está deformado frente a sus alas."),
                ("PM", "útil para ver dónde la curva tiene una joroba o un valle; confirmar en Evidencia/PM antes de priorizar."),
            ],
            BULL if fly > 0 else BEAR if fly < 0 else SUBTEXT,
        )
