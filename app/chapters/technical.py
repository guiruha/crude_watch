"""Chapter 1 — Technical analysis (Phase 2).

A traffic-light signal panel (Trend / Momentum / Volatility / Statistical) over
the active spread, then one collapsible section per theme. Each expander holds
that family's indicator charts and a detail table, so the indicators are grouped
by theme rather than toggled from a flat list. All indicators are close-based (a
fixed-date spread has no intraday high/low).
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crudewatch.analytics import MATURITY_BUCKETS, bucket_bounds
from crudewatch.analytics.indicators import (
    Bias,
    FamilyVerdict,
    adaptive_rsi_bands,
    bollinger,
    dmi_adx,
    ma_ribbon,
    macd,
    signal_panel,
    supertrend,
    zscore,
)
from crudewatch.plots import BLACK, GRID, TEXT

from chapters.base import Chapter, ChapterContext
from theme.palette import ACCENT, AMBER, BEAR, BULL, FLAT, SUBTEXT, signal_chip

_TONE = {Bias.BULLISH: "bull", Bias.BEARISH: "bear", Bias.NEUTRAL: "flat"}
_DOT = {Bias.BULLISH: "\U0001F7E2", Bias.BEARISH: "\U0001F534", Bias.NEUTRAL: "\u26AA"}
_BIAS_LABEL = {Bias.BULLISH: "Bullish", Bias.BEARISH: "Bearish", Bias.NEUTRAL: "Neutral"}

# What each theme contributes to the chart stack.
_THEME_BLURB = {
    "Trend": "Direction & structure — MA ribbon, Supertrend trailing stop and the ADX/DMI regime filter.",
    "Momentum": "Speed of the move — RSI with adaptive bands and the MACD histogram.",
    "Volatility": "Dispersion & stretch — Bollinger bands around the active spread.",
    "Statistical": "Mean-reversion — the 60-day Z-Score against ±2σ triggers.",
}

# Per-indicator explanation, split into three sections: what it is, how it's
# calculated, and how to read it. Rendered as a sectioned card in the popover.
_INDICATOR_HELP: dict[str, dict[str, str]] = {
    "MA ribbon": {
        "what": (
            "A triple moving-average \u201cribbon\u201d that reads the trend at three speeds at once. It is the "
            "primary trend anchor for the spread \u2014 the fast line reacts to the latest sessions, the slow "
            "line is the long-run backbone, and the mid line sits in between."
        ),
        "calc": (
            "Fast = EMA(20), mid = EMA(50), slow = SMA(200). The two EMAs (exponential means) weight recent "
            "sessions more heavily so they turn quickly; the SMA is a plain 200-session average that changes "
            "slowly. All three run on the close of the spread."
        ),
        "read": (
            "Cleanly stacked up (EMA20 > EMA50 > SMA200) is bullish; cleanly stacked down is bearish; when the "
            "lines interleave there is no clean trend, so it reads neutral. Wider gaps between the lines mean a "
            "stronger, more extended move. If a vintage is too short for 200 sessions it falls back to the "
            "fast-vs-mid (EMA20 vs EMA50) cross."
        ),
    },
    "ADX/DMI": {
        "what": (
            "A regime filter that answers two separate questions: is the market trending at all, and if so, in "
            "which direction? ADX measures trend strength (never direction); +DI and \u2212DI measure direction."
        ),
        "calc": (
            "Computed from close-to-close moves (a fixed-date spread has no intraday range, so ATR here is a "
            "close-to-close average). +DI and \u2212DI are the smoothed up-moves and down-moves normalised by ATR; "
            "ADX is the smoothed, scaled distance between them: |+DI \u2212 \u2212DI| / (+DI + \u2212DI) \u00d7 100."
        ),
        "read": (
            "ADX below 20 is a choppy range \u2014 treated as neutral, and trend signals are distrusted. ADX at or "
            "above 25 is trending, with direction set by whichever DI is on top (+DI above \u2192 up, \u2212DI above \u2192 "
            "down). A rising ADX means the trend is strengthening."
        ),
    },
    "Supertrend": {
        "what": (
            "A trailing stop-and-reverse line that hugs price and flips only when the trend genuinely turns. It "
            "is designed to keep you with a move while filtering out day-to-day noise."
        ),
        "calc": (
            "Basic bands = close \u00b1 3 \u00d7 ATR(10), with ATR taken close-to-close. In an up-trend the line locks "
            "to the lower band and only ratchets upward; in a down-trend it locks to the upper band and only "
            "ratchets down. It switches sides when the close crosses through it."
        ),
        "read": (
            "Line below price (up-trend) is bullish; line above price (down-trend) is bearish. A flip is an "
            "early trend-change flag, and the distance from price to the line acts as a rough trailing-stop level."
        ),
    },
    "MACD": {
        "what": (
            "A momentum and trend-follow oscillator that tracks the gap between a fast and a slow average \u2014 in "
            "other words, how quickly momentum is building or fading."
        ),
        "calc": (
            "MACD line = EMA(12) \u2212 EMA(26); signal line = EMA(9) of the MACD line; histogram = MACD \u2212 signal. "
            "The histogram is the part most watched because it leads the crossovers."
        ),
        "read": (
            "Histogram above zero and rising = strengthening bullish momentum; below zero and falling = bearish. "
            "A MACD-crosses-signal event is the classic trigger. We also flag price/MACD divergence \u2014 price "
            "making a new high or low the MACD does not confirm \u2014 as an early exhaustion warning."
        ),
    },
    "RSI (adaptive)": {
        "what": (
            "A bounded 0\u2013100 momentum oscillator, but with self-adjusting overbought/oversold levels so it "
            "adapts to each spread\u2019s own volatility instead of the fixed 70/30 lines."
        ),
        "calc": (
            "RSI(14) = 100 \u2212 100 / (1 + average gain / average loss) over the last 14 sessions. Rather than "
            "static 70/30 thresholds, the bands are RSI\u2019s own rolling mean \u00b1 k \u00d7 its rolling standard deviation."
        ),
        "read": (
            "RSI above its adaptive upper band is overbought (bearish lean); below the lower band is oversold "
            "(bullish lean); inside the bands is neutral. The adaptive bands give fewer false extremes in a quiet "
            "spread and catch turns earlier in a volatile one."
        ),
    },
    "Bollinger %B": {
        "what": (
            "Normalises where the spread sits inside its Bollinger band into a single 0\u20131 number, so \u201cstretch\u201d "
            "is directly comparable across different periods."
        ),
        "calc": (
            "Bands = SMA(20) \u00b1 2 \u00d7 rolling std(20). %B = (close \u2212 lower) / (upper \u2212 lower), so 0 sits on the "
            "lower band, 0.5 at the mean, and 1 on the upper band."
        ),
        "read": (
            "%B > 1 means the spread closed above the upper band (stretched rich \u2192 bearish fade lean); %B < 0 "
            "means it closed below the lower band (stretched cheap \u2192 bullish fade lean). This is a mean-reversion "
            "read, so treat it as context alongside the trend, not a standalone trigger."
        ),
    },
    "Z-Score": {
        "what": (
            "Measures how far the spread has strayed from its recent mean, in standard deviations \u2014 a pure "
            "statistical stretch gauge used for mean-reversion context."
        ),
        "calc": (
            "z = (close \u2212 mean\u2086\u2080) / std\u2086\u2080 over a rolling 60-session window. z = 0 sits exactly on the 60-day "
            "mean; \u00b11 is one standard deviation away from it."
        ),
        "read": (
            "z \u2265 +2 is roughly 2\u03c3 rich (potential fade-short context); z \u2264 \u22122 is ~2\u03c3 cheap (potential "
            "fade-long context); anything between \u00b12 is a normal range. As with %B, this is context \u2014 a strong "
            "trend can stay stretched for a long time."
        ),
    },
}

_HELP_SECTIONS = (("what", "What it is"), ("calc", "How it\u2019s calculated"), ("read", "How to read it"))


def _dark(fig: go.Figure, title: str, y_title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(color=ACCENT, size=18)),
        template="plotly_dark", paper_bgcolor=BLACK, plot_bgcolor=BLACK,
        font=dict(color=TEXT, family="Arial"), hovermode="x unified",
        margin=dict(l=60, r=30, t=60, b=40), height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.02, yanchor="bottom"),
    )
    fig.update_xaxes(gridcolor=GRID, showline=True, linecolor=ACCENT)
    fig.update_yaxes(title_text=y_title, gridcolor=GRID, showline=True, linecolor=ACCENT, zeroline=False)
    return fig


class TechnicalChapter(Chapter):
    name = "Technical Analysis"
    subtitle = "Trend, momentum, volatility and statistical layers over the active spread."
    phase = "Phase 2"

    def render(self, ctx: ChapterContext) -> None:
        close = ctx.series.set_index("date")["close"].astype(float)
        if len(close) < 5:
            st.info("Not enough observations in this vintage to compute indicators.")
            return

        bucket = self._select_maturity(ctx)
        close = self._window_close(ctx, close, bucket)
        if len(close) < 5:
            st.info(
                f"The **{bucket}** window holds fewer than 5 observations for this vintage — "
                f"pick a wider maturity bucket to compute indicators."
            )
            return
        if bucket != "All":
            st.caption(
                f"Indicators are computed on the slice of this vintage where it sat in the **{bucket}** "
                f"window ({len(close)} sessions). Long-lookback indicators (e.g. SMA200) may stay neutral "
                f"on short slices."
            )

        verdicts = signal_panel(close)
        self._signal_panel(verdicts)
        st.caption("Expand a theme to inspect its indicators.")
        for verdict in verdicts:
            self._family_section(ctx, close, verdict)

    # -- maturity split (12–10 / 10–6 / 6–1 / 1–0 months to expiry) ----------

    def _select_maturity(self, ctx: ChapterContext) -> str:
        key = f"tech_bucket_{ctx.structure.key}"
        return st.segmented_control(
            "Maturity (months to expiry)", list(MATURITY_BUCKETS), default="All",
            key=key, help="Restrict the indicators to the sessions where the contract sat in this window.",
        ) or "All"

    def _window_close(self, ctx: ChapterContext, close: pd.Series, bucket: str) -> pd.Series:
        if bucket == "All":
            return close
        stack = ctx.seasonal
        if stack.empty:
            return close
        lo, hi = bucket_bounds(bucket)
        active = stack[stack["vintage"] == ctx.base_year]
        if active.empty:
            return close
        dte = active.set_index("date")["days_to_expiry"]
        keep = dte[(dte >= lo) & (dte <= hi)].index
        return close[close.index.isin(keep)]

    # -- header --------------------------------------------------------------

    def _signal_panel(self, verdicts: list[FamilyVerdict]) -> None:
        cols = st.columns(len(verdicts))
        for col, v in zip(cols, verdicts):
            with col:
                signal_chip(v.family, v.label, v.conviction, _TONE[v.bias])
        st.write("")

    # -- per-theme sections --------------------------------------------------

    def _family_section(self, ctx: ChapterContext, close: pd.Series, v: FamilyVerdict) -> None:
        conviction = f" · {v.conviction:.0%} agree" if v.bias != Bias.NEUTRAL else ""
        header = f"{_DOT[v.bias]}  {v.family} — {v.label}{conviction}"
        with st.expander(header, expanded=(v.family == "Trend")):
            if v.family in _THEME_BLURB:
                st.markdown(
                    f'<div style="color:{SUBTEXT};font-size:12px;margin:-4px 0 6px 0">'
                    f'{_THEME_BLURB[v.family]}</div>',
                    unsafe_allow_html=True,
                )
            self._help_row(v)
            for i, fig in enumerate(self._family_figs(ctx, close, v.family)):
                st.plotly_chart(fig, width="stretch", key=f"tech_{v.family}_{i}")
            self._family_table(v)

    def _family_figs(self, ctx: ChapterContext, close: pd.Series, family: str) -> list[go.Figure]:
        if family == "Trend":
            return [self._price_fig(ctx, close, ("MA ribbon", "Supertrend")), self._adx_fig(close)]
        if family == "Momentum":
            return [self._rsi_fig(close), self._macd_fig(close)]
        if family == "Volatility":
            return [self._price_fig(ctx, close, ("Bollinger",))]
        if family == "Statistical":
            return [self._zscore_fig(close)]
        return []

    def _help_row(self, v: FamilyVerdict) -> None:
        """A row of info popovers; click a chip to see how that indicator is computed."""
        st.markdown(
            f'<div style="color:{SUBTEXT};font-size:11px;margin:0 0 4px 2px">'
            f'Click a chip for how each indicator is calculated.</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(v.signals))
        for col, s in zip(cols, v.signals):
            with col, st.popover(f"{s.name}  \u24D8", use_container_width=True):
                st.markdown(self._help_html(s.name), unsafe_allow_html=True)

    @staticmethod
    def _help_html(name: str) -> str:
        help_ = _INDICATOR_HELP.get(name)
        header = (
            f'<div style="color:{ACCENT};font-size:18px;font-weight:800;letter-spacing:.2px;'
            f'margin:2px 0 12px 0">{name}</div>'
        )
        if not help_:
            return header
        sections = "".join(
            f'<div style="margin:0 0 14px 0">'
            f'<div style="color:{AMBER};font-size:12px;font-weight:800;letter-spacing:.8px;'
            f'text-transform:uppercase;margin:0 0 4px 0">{label}</div>'
            f'<div style="color:{TEXT};font-size:15px;line-height:1.7">{help_[key]}</div>'
            f'</div>'
            for key, label in _HELP_SECTIONS
        )
        return (
            f'<div style="min-width:340px;max-width:460px;padding:2px 2px 4px 2px">{header}{sections}</div>'
        )

    def _family_table(self, v: FamilyVerdict) -> None:
        rows = [
            {
                "Indicator": s.name,
                "Bias": _BIAS_LABEL[s.bias],
                "Value": None if s.value is None else round(s.value, 3),
                "Note": s.note,
            }
            for s in v.signals
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # -- figures -------------------------------------------------------------

    def _price_fig(self, ctx: ChapterContext, close: pd.Series, overlays: tuple[str, ...]) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=close.index, y=close, mode="lines", name="Spread",
            line=dict(color=ACCENT, width=2),
            hovertemplate="%{x|%b %d, %Y}<br>%{y:.3f}<extra></extra>",
        ))
        if "MA ribbon" in overlays:
            ribbon = ma_ribbon(close)
            for col_name, color in zip(ribbon.columns, (BULL, AMBER, FLAT)):
                fig.add_trace(go.Scatter(
                    x=ribbon.index, y=ribbon[col_name], mode="lines", name=col_name.upper(),
                    line=dict(color=color, width=1.3), opacity=0.9,
                ))
        if "Bollinger" in overlays:
            bb = bollinger(close)
            fig.add_trace(go.Scatter(x=bb.index, y=bb["upper"], mode="lines", name="BB upper",
                                     line=dict(color=FLAT, width=1, dash="dot")))
            fig.add_trace(go.Scatter(x=bb.index, y=bb["lower"], mode="lines", name="BB lower",
                                     line=dict(color=FLAT, width=1, dash="dot"),
                                     fill="tonexty", fillcolor="rgba(139,150,145,0.08)"))
        if "Supertrend" in overlays:
            stt = supertrend(close)
            fig.add_trace(go.Scatter(x=stt.index, y=stt["supertrend"], mode="lines", name="Supertrend",
                                     line=dict(color=AMBER, width=1.4, dash="dash")))
        return _dark(fig, ctx.title, "Spread ($/bbl)", height=460)

    def _adx_fig(self, close: pd.Series) -> go.Figure:
        d = dmi_adx(close)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.index, y=d["adx"], mode="lines", name="ADX", line=dict(color=ACCENT, width=1.8)))
        fig.add_trace(go.Scatter(x=d.index, y=d["plus_di"], mode="lines", name="+DI", line=dict(color=BULL, width=1)))
        fig.add_trace(go.Scatter(x=d.index, y=d["minus_di"], mode="lines", name="-DI", line=dict(color=BEAR, width=1)))
        fig.add_hline(y=25, line=dict(color=FLAT, width=1, dash="dot"), annotation_text="trend")
        return _dark(fig, "ADX / DMI — regime filter", "level", height=260)

    def _rsi_fig(self, close: pd.Series) -> go.Figure:
        bands = adaptive_rsi_bands(close)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bands.index, y=bands["rsi"], mode="lines", name="RSI",
                                 line=dict(color=ACCENT, width=1.6)))
        fig.add_trace(go.Scatter(x=bands.index, y=bands["upper"], mode="lines", name="Adaptive upper",
                                 line=dict(color=BEAR, width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=bands.index, y=bands["lower"], mode="lines", name="Adaptive lower",
                                 line=dict(color=BULL, width=1, dash="dot")))
        fig = _dark(fig, "RSI (adaptive bands)", "RSI", height=260)
        fig.update_yaxes(range=[0, 100])
        return fig

    def _macd_fig(self, close: pd.Series) -> go.Figure:
        m = macd(close)
        colors = [BULL if h >= 0 else BEAR for h in m["hist"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=m.index, y=m["hist"], name="Histogram", marker_color=colors, marker_line_width=0))
        fig.add_trace(go.Scatter(x=m.index, y=m["macd"], mode="lines", name="MACD", line=dict(color=ACCENT, width=1.5)))
        fig.add_trace(go.Scatter(x=m.index, y=m["signal"], mode="lines", name="Signal", line=dict(color=AMBER, width=1.3)))
        return _dark(fig, "MACD (12-26-9)", "MACD", height=260)

    def _zscore_fig(self, close: pd.Series) -> go.Figure:
        z = zscore(close, 60)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=z.index, y=z, mode="lines", name="Z-Score (60d)", line=dict(color=ACCENT, width=1.6)))
        for level, color in ((2, BEAR), (-2, BULL), (0, FLAT)):
            fig.add_hline(y=level, line=dict(color=color, width=1, dash="dot"))
        return _dark(fig, "Z-Score (60d) — mean-reversion", "\u03c3", height=260)
