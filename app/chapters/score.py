"""Composite scoring chapter (Phase 5).

Synthesises every directional indicator vote into one regime-weighted score in
[-100, +100] — the spread-specific answer to TradingView's "Technical Summary" —
shown on a Plotly gauge with the full vote breakdown, alongside the QUANT
mean-reversion diagnostics (Hurst, half-life, ADF, variance ratio) that gauge how
much to trust the mean-reversion premise.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crudewatch.analytics import composite_score, detect_regime
from crudewatch.analytics.indicators import Bias, all_signals, regime_diagnostics
from crudewatch.plots import BLACK, TEXT

from chapters.base import Chapter, ChapterContext
from theme.palette import ACCENT, BEAR, BULL, FLAT, SUBTEXT, SURFACE

_MIN_OBS = 30
_TOKEN_COLOR = {
    "strong_up": "#059669", "up": BULL, "neutral": FLAT, "down": "#F97316", "strong_down": BEAR,
}
_BIAS_LABEL = {Bias.BULLISH: "Bullish", Bias.BEARISH: "Bearish", Bias.NEUTRAL: "Neutral"}


class ScoreChapter(Chapter):
    name = "Composite Score"
    subtitle = "Regime-weighted synthesis of every indicator into one actionable read."
    phase = "Phase 5"

    def render(self, ctx: ChapterContext) -> None:
        close = ctx.series.set_index("date")["close"].astype(float)
        if len(close) < _MIN_OBS:
            st.info("Not enough observations in this vintage to compute a composite score.")
            return

        regime = detect_regime(close)
        signals = all_signals(close)
        score = composite_score(signals, regime)

        left, right = st.columns([1.1, 1])
        with left:
            st.plotly_chart(self._gauge(score), width="stretch", key="score_gauge")
            st.markdown(
                f"<div style='text-align:center;color:{_TOKEN_COLOR[score.color_token]};"
                f"font-size:16px;font-weight:800;margin-top:-10px'>{score.reading}</div>"
                f"<div style='text-align:center;color:{SUBTEXT};font-size:12px'>"
                f"Regime weighting: {regime.value}</div>",
                unsafe_allow_html=True,
            )
        with right:
            self._diagnostics(close)

        self._vote_table(score)

    # -- gauge ---------------------------------------------------------------

    def _gauge(self, score) -> go.Figure:
        bar = _TOKEN_COLOR[score.color_token]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score.value,
            number=dict(font=dict(color=TEXT, size=40)),
            gauge=dict(
                axis=dict(range=[-100, 100], tickcolor=SUBTEXT, tickfont=dict(color=SUBTEXT)),
                bar=dict(color=bar, thickness=0.28),
                bgcolor=SURFACE,
                borderwidth=0,
                steps=[
                    dict(range=[-100, -60], color="rgba(229,72,77,0.35)"),
                    dict(range=[-60, -20], color="rgba(249,115,22,0.22)"),
                    dict(range=[-20, 20], color="rgba(139,150,145,0.18)"),
                    dict(range=[20, 60], color="rgba(16,185,129,0.20)"),
                    dict(range=[60, 100], color="rgba(5,150,105,0.35)"),
                ],
                threshold=dict(line=dict(color=TEXT, width=2), thickness=0.75, value=score.value),
            ),
        ))
        fig.update_layout(
            paper_bgcolor=BLACK, font=dict(color=TEXT, family="Arial"),
            margin=dict(l=20, r=20, t=30, b=0), height=300,
            title=dict(text="Composite score", x=0.5, xanchor="center", font=dict(color=ACCENT, size=16)),
        )
        return fig

    # -- diagnostics ---------------------------------------------------------

    def _diagnostics(self, close: pd.Series) -> None:
        d = regime_diagnostics(close)
        st.markdown(f"<div style='color:{SUBTEXT};font-size:11px;font-weight:700;"
                    f"letter-spacing:.6px;text-transform:uppercase;margin-bottom:6px'>"
                    f"Quantitative diagnostics</div>", unsafe_allow_html=True)
        a, b = st.columns(2)
        a.metric("Hurst", "—" if d.hurst is None else f"{d.hurst:.2f}",
                 help="Hurst exponent from the aggregated variance of lagged differences.")
        b.metric("Half-life", "—" if d.half_life is None else f"{d.half_life:.0f}d",
                 help="Ornstein-Uhlenbeck half-life in days: -ln2 / b from Δy regressed on lagged y.")
        c, e = st.columns(2)
        c.metric("ADF p-value", "—" if d.adf_pvalue is None else f"{d.adf_pvalue:.2f}",
                 help="Augmented Dickey-Fuller unit-root test p-value.")
        e.metric("Variance ratio", "—" if d.variance_ratio is None else f"{d.variance_ratio:.2f}",
                 help="Lo-MacKinlay variance ratio: Var(k-step) / (k · Var(1-step)).")

    # -- votes ---------------------------------------------------------------

    def _vote_table(self, score) -> None:
        with st.expander("Vote breakdown", expanded=False):
            rows = [
                {
                    "Indicator": c.name,
                    "Category": c.category.title(),
                    "Bias": _BIAS_LABEL[c.bias],
                    "Weight": round(c.weight, 2),
                    "Contribution": round(c.contribution, 2),
                }
                for c in sorted(score.contributions, key=lambda c: c.contribution, reverse=True)
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption("Score = 100 × Σ(weight × vote) ⁄ Σ(weight). Weights are regime-dependent and auditable.")
