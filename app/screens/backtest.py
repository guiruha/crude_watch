"""Backtest screen — follow-the-score results for the selected contract.

Simulates a stateful hysteresis rule over the contract's life (enter at ±50,
close at ±20; flips allowed), strictly point-in-time, net of the per-family cost
stub, and shows: summary stats, equity vs buy-and-hold, the price line with
entry/exit markers, and the trade ledger.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.scoring import backtest_contract_cached
from core.selection import Selection
from theme.palette import (
    ACCENT,
    BEAR,
    BORDER,
    BULL,
    SUBTEXT,
    SURFACE,
    TEXT,
    title_block,
)


def _base_layout(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(color=TEXT, size=13, family="Inter, -apple-system, sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=SURFACE, bordercolor=BORDER,
            font=dict(color=TEXT, size=12, family="Inter, sans-serif"),
        ),
    )
    grid = "#161C19"
    fig.update_xaxes(gridcolor=grid, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor=grid, zeroline=True, zerolinecolor=BORDER, showline=False)
    return fig


def _fmt(value, spec: str = ".2f", dash: str = "—") -> str:
    if value is None:
        return dash
    if isinstance(value, float) and (np.isnan(value) or not np.isfinite(value)):
        return dash
    return format(value, spec)


class BacktestScreen:
    """Per-contract 'follow the Opportunity Score' backtest."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def display(self, selection: Selection) -> None:
        title_block(
            "Backtest",
            "Qué habría pasado siguiendo el Opportunity Score en este contrato: "
            "entra a ±50, cierra a ±20 (con giros), point-in-time y neto de costes.",
        )

        if selection.contract is None:
            st.info("Selecciona una familia y una fecha con contratos activos en el menú superior.")
            return

        family, contract, horizon = selection.family, selection.contract, selection.horizon
        try:
            res = backtest_contract_cached(family, contract, horizon)
        except Exception as exc:  # noqa: BLE001 — surface any scoring failure to the UI
            st.error(f"No se pudo simular el backtest: {exc}")
            return

        if res.score_df.empty or len(res.score_df) < 2:
            st.info("El contrato no tiene suficiente historia para simular.")
            return

        st.caption(
            f"Regla: entrar a **±{res.enter_at:.0f}**, cerrar a **±{res.exit_at:.0f}** · "
            f"horizonte de calibración **{res.horizon}** · coste ida/vuelta **{res.cost:.3f}** pts · "
            f"unidades = **puntos de precio**."
        )

        self._stat_cards(res.stats)
        st.markdown("#### Equity vs comprar-y-mantener")
        self._equity_chart(res)
        st.markdown("#### Precio con entradas y salidas")
        self._price_markers(res)
        st.markdown("#### Operaciones")
        self._trades_table(res.trades)

    # -- sections ------------------------------------------------------------

    def _stat_cards(self, s: dict) -> None:
        total, bench, excess = s.get("total_pnl"), s.get("bench_total"), s.get("excess_total")
        row1 = st.columns(4)
        row1[0].metric("PnL estrategia (pts)", _fmt(total))
        row1[1].metric("Comprar-y-mantener (pts)", _fmt(bench))
        row1[2].metric("Exceso vs B&H (pts)", _fmt(excess))
        row1[3].metric("Nº operaciones", str(int(s.get("n_trades", 0))))

        wr = s.get("win_rate")
        tim = s.get("time_in_market")
        row2 = st.columns(4)
        row2[0].metric("% acierto", _fmt(wr * 100 if wr == wr else wr, ".0f") + ("%" if wr == wr else ""))
        row2[1].metric("Sharpe (anual.)", _fmt(s.get("sharpe")))
        row2[2].metric("Máx drawdown (pts)", _fmt(s.get("max_drawdown")))
        row2[3].metric("% tiempo en mercado", _fmt(tim * 100 if tim == tim else tim, ".0f") + ("%" if tim == tim else ""))

    def _equity_chart(self, res) -> None:
        eq, bench = res.equity, res.benchmark
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=bench.index, y=bench.to_numpy(), mode="lines", name="Comprar-y-mantener",
                line=dict(color=SUBTEXT, width=1.5, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=eq.index, y=eq.to_numpy(), mode="lines", name="Seguir el score",
                line=dict(color=ACCENT, width=2.6),
            )
        )
        fig.update_yaxes(title_text="PnL acumulado (pts)")
        st.plotly_chart(_base_layout(fig, 320), width="stretch")

    def _price_markers(self, res) -> None:
        sdf = res.score_df
        trades = res.trades
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(sdf["date"]), y=sdf["close"], mode="lines", name="Precio",
                line=dict(color=SUBTEXT, width=1.6), hoverinfo="skip",
            )
        )
        if not trades.empty:
            longs = trades[trades["side"] == "long"]
            shorts = trades[trades["side"] == "short"]
            if not longs.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(longs["entry_date"]), y=longs["entry_price"], mode="markers",
                        name="Entrada larga", marker=dict(symbol="triangle-up", size=11, color=BULL),
                    )
                )
            if not shorts.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(shorts["entry_date"]), y=shorts["entry_price"], mode="markers",
                        name="Entrada corta", marker=dict(symbol="triangle-down", size=11, color=BEAR),
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(trades["exit_date"]), y=trades["exit_price"], mode="markers",
                    name="Salida",
                    marker=dict(symbol="circle-open", size=10, color=TEXT, line=dict(width=1.5)),
                )
            )
        fig.update_yaxes(title_text="precio")
        st.plotly_chart(_base_layout(fig, 340), width="stretch")

    def _trades_table(self, trades: pd.DataFrame) -> None:
        if trades.empty:
            st.info("La regla no habría abierto ninguna operación en este contrato.")
            return
        def _num(series: pd.Series, spec: str = ".2f") -> list[str]:
            return [("—" if v != v else format(float(v), spec)) for v in series]

        view = pd.DataFrame(
            {
                "Entrada": pd.to_datetime(trades["entry_date"]).dt.strftime("%d %b %Y"),
                "Salida": pd.to_datetime(trades["exit_date"]).dt.strftime("%d %b %Y"),
                "Lado": trades["side"].map({"long": "Largo", "short": "Corto"}),
                "Días": trades["bars"].astype(int),
                "P. entrada": _num(trades["entry_price"]),
                "P. salida": _num(trades["exit_price"]),
                "PnL neto (pts)": _num(trades["net_pnl"], ".3f"),
                "MFE": _num(trades["mfe"]),
                "MAE": _num(trades["mae"]),
                "Abierta": trades["open"].map({True: "sí", False: ""}),
            }
        )
        st.dataframe(view, width="stretch", hide_index=True)
