"""Chapter 3 — Bollinger bands & volatility diagnostics (Phase 4).

Detects the curve regime (contango vs. backwardation) and draws the ±1/2/3σ
bands around the SMA20, with %B, bandwidth and ATR-percentile diagnostics.

Note: the strategy layer (mean-reversion / breakout plans, dynamic TP/SL and the
simulated position tracker) is parked pending feedback before it goes back in.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crudewatch.analytics import (
    Regime,
    bollinger_layers,
    detect_regime,
    risk_metrics,
)
from crudewatch.plots import BLACK, GRID, TEXT

from chapters.base import Chapter, ChapterContext
from theme.palette import ACCENT, AMBER, FLAT

_COOL = "#3B82F6"   # contango
_WARM = "#F59E0B"   # backwardation
_MIN_OBS = 20


class BollingerChapter(Chapter):
    name = "Bollinger"
    subtitle = "Regime-aware Bollinger bands and volatility diagnostics."
    phase = "Phase 4"

    def render(self, ctx: ChapterContext) -> None:
        close = ctx.series.set_index("date")["close"].astype(float)
        if len(close) < _MIN_OBS:
            st.info("Not enough observations in this vintage to compute Bollinger bands.")
            return

        regime = detect_regime(close)
        rm = risk_metrics(close)

        self._risk_metrics(rm)
        st.plotly_chart(self._bands_fig(ctx, close, regime), width="stretch")

    # -- diagnostics ---------------------------------------------------------

    def _risk_metrics(self, rm) -> None:
        c1, c2, c3 = st.columns(3)
        c1.metric("%B", "—" if rm.pctb is None else f"{rm.pctb:.2f}",
                  help="Position within the ±2σ band (0 = lower, 1 = upper).")
        c2.metric("Bandwidth", "—" if rm.bandwidth is None else f"{rm.bandwidth:.3f}",
                  help="Band width / mean — contraction flags a squeeze.")
        c3.metric("ATR percentile", "—" if rm.atr_percentile is None else f"{rm.atr_percentile:.0f}th",
                  help="Current ATR vs. its own history.")

    # -- figure --------------------------------------------------------------

    def _bands_fig(self, ctx: ChapterContext, close: pd.Series, regime: Regime) -> go.Figure:
        bb = bollinger_layers(close)
        fig = go.Figure()
        # ±2σ envelope shading.
        fig.add_trace(go.Scatter(x=bb.index, y=bb["u2"], mode="lines", line=dict(color="rgba(0,0,0,0)"),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=bb.index, y=bb["l2"], mode="lines", line=dict(color="rgba(0,0,0,0)"),
                                 fill="tonexty", fillcolor="rgba(139,150,145,0.08)",
                                 name="±2σ", hoverinfo="skip"))
        for k, dash, width in ((1, "dot", 0.8), (2, "dash", 1.1), (3, "dot", 0.8)):
            for side in ("u", "l"):
                fig.add_trace(go.Scatter(
                    x=bb.index, y=bb[f"{side}{k}"], mode="lines",
                    name=f"±{k}σ" if side == "u" else None, showlegend=(side == "u"),
                    line=dict(color=FLAT, width=width, dash=dash), opacity=0.7, hoverinfo="skip",
                ))
        fig.add_trace(go.Scatter(x=bb.index, y=bb["mid"], mode="lines", name="SMA20",
                                 line=dict(color=AMBER, width=1.4)))
        fig.add_trace(go.Scatter(x=close.index, y=close, mode="lines", name="Spread",
                                 line=dict(color=ACCENT, width=2),
                                 hovertemplate="%{x|%b %d, %Y}<br>%{y:.3f}<extra></extra>"))
        accent = _COOL if regime == Regime.CONTANGO else _WARM
        fig.update_layout(
            title=dict(text=f"{ctx.title} — Bollinger ({regime.value})", x=0.5, xanchor="center",
                       font=dict(color=accent, size=18)),
            template="plotly_dark", paper_bgcolor=BLACK, plot_bgcolor=BLACK,
            font=dict(color=TEXT, family="Arial"), hovermode="x unified",
            margin=dict(l=60, r=40, t=60, b=40), height=470,
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.02, yanchor="bottom"),
        )
        fig.update_xaxes(gridcolor=GRID, showline=True, linecolor=accent)
        fig.update_yaxes(title_text="Spread ($/bbl)", gridcolor=GRID, showline=True, linecolor=accent, zeroline=False)
        return fig

