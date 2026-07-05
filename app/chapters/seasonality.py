"""Chapter 2 — Seasonality as a context layer (v1 spec).

Seasonality here is *context*, never a mean-reversion trigger. The chapter shows
a clean seasonal overlay (current year loud, history quiet, median dotted, ±1σ
band shaded), the monthly heatmap, the fundamental context and a validation
backtest of whether alignment actually improves forward continuation.
"""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crudewatch.analytics import (
    BUFFER_DAYS,
    MATURITY_BUCKETS,
    MAX_DAYS_TO_EXPIRY,
    SEASONAL_DRIVERS,
    bucket_bounds,
    cap_to_year,
)
from crudewatch.infra.constants import MONTH_CODES
from crudewatch.plots import BLACK, GRID, TEXT

from chapters.base import Chapter, ChapterContext
from core.data import (
    spread_alignment_backtest,
    spread_cone,
    spread_monthly_heatmap,
    spread_percentile,
)
from theme.palette import ACCENT, AMBER, BEAR, BULL, FLAT, SUBTEXT, SURFACE

# Forward window used by the validation backtest (the 10/20/30 breakdown stays
# visible in the forward-tendency row for shorter/longer views).
HEADLINE_HORIZON = 20


class SeasonalityChapter(Chapter):
    name = "Seasonality"
    subtitle = "Context layer — does the season give the current trend a tailwind, headwind or neither?"
    phase = "Phase 3"

    def render(self, ctx: ChapterContext) -> None:
        stack = ctx.seasonal
        if stack.empty or stack["vintage"].nunique() < 2:
            st.info("Not enough vintages to build a seasonal profile for this structure.")
            return

        self._select_maturity(ctx)

        st.caption(
            f"Window fixed to the final **{MAX_DAYS_TO_EXPIRY} days** before expiry; smoothing buffer "
            f"fixed at **±{BUFFER_DAYS} days**. Seasonality is **context**, not a buy/sell trigger."
        )

        if self._lo > self._hi:
            st.info(
                f"The **{self._bucket}** window sits entirely beyond the {MAX_DAYS_TO_EXPIRY}-day seasonal "
                f"cap, so there is no seasonal history to show here. Pick a nearer maturity bucket."
            )
            return

        if self._bucket != "All":
            st.caption(
                f"The overlay and the validation backtest are restricted to the **{self._bucket}** window "
                f"(**{self._lo}–{self._hi} days** to expiry). The monthly heatmap remains full-life for reference."
            )

        st.plotly_chart(self._overlay_fig(ctx), width="stretch")
        self._percentile_caption(ctx)

        self._fundamentals(ctx)
        self._heatmap_description()
        st.plotly_chart(self._heatmap_fig(ctx), width="stretch")
        self._validation(ctx)

    # -- maturity split (12–10 / 10–6 / 6–1 / 1–0 months to expiry) ----------

    def _select_maturity(self, ctx: ChapterContext) -> None:
        key = f"seas_bucket_{ctx.structure.key}"
        self._bucket = st.segmented_control(
            "Maturity (months to expiry)", list(MATURITY_BUCKETS), default="All",
            key=key, help="Restrict the whole seasonal read to one point in the contract's life.",
        ) or "All"
        lo, hi = bucket_bounds(self._bucket)
        # Seasonality spans the final year, so intersect the bucket with that cap
        # (the 12–10m bucket now sits fully inside the window).
        self._lo, self._hi = lo, min(hi, MAX_DAYS_TO_EXPIRY)

    # -- figures -------------------------------------------------------------

    def _overlay_fig(self, ctx: ChapterContext) -> go.Figure:
        stack = cap_to_year(ctx.seasonal, self._hi, min_dte=self._lo)
        cone = spread_cone(ctx.structure.key, ctx.base_year, BUFFER_DAYS, self._lo, self._hi)

        fig = go.Figure()
        # ±1σ band (drawn first, behind everything).
        if not cone.empty:
            fig.add_trace(go.Scatter(
                x=cone.index, y=cone["upper"], mode="lines",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=cone.index, y=cone["lower"], mode="lines", name="±1σ band",
                line=dict(color="rgba(0,0,0,0)"), fill="tonexty",
                fillcolor="rgba(245,166,35,0.12)", hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=cone.index, y=cone["median"], mode="lines", name="Historical median",
                line=dict(color=AMBER, width=1.6, dash="dot"),
                hovertemplate="%{x} days to expiry<br>median %{y:.2f}<extra></extra>",
            ))
        # Prior vintages: thin, quiet grey. Active vintage: thick, loud emerald.
        for vintage, grp in stack.groupby("vintage"):
            grp = grp.sort_values("days_to_expiry")
            active = vintage == ctx.base_year
            fig.add_trace(go.Scatter(
                x=grp["days_to_expiry"], y=grp["close"], mode="lines",
                name=f"{vintage} (current)" if active else str(vintage),
                line=dict(color=ACCENT if active else "rgba(150,160,155,0.22)",
                          width=3.2 if active else 1),
                showlegend=active,
                hovertemplate=f"{vintage}<br>%{{x}} days to expiry<br>%{{y:.2f}}<extra></extra>",
            ))
        fig = self._dark(fig, f"{ctx.structure.label} — seasonal overlay",
                         "Days to expiry", "Spread ($/bbl)")
        fig.update_xaxes(autorange="reversed")  # expiry (0) on the right
        return fig

    def _heatmap_description(self) -> None:
        st.markdown(
            f'<div style="color:{SUBTEXT};font-size:11px;font-weight:700;letter-spacing:.6px;'
            f'text-transform:uppercase;margin:6px 0 2px 0">Monthly move by vintage</div>'
            f'<div style="color:{SUBTEXT};font-size:13px;line-height:1.6;margin:0 0 4px 0">'
            f'A <b style="color:{TEXT}">month × vintage</b> grid of how far the spread moved during each '
            f'calendar month (Δ = last minus first value that month, shown on the colour bar). '
            f'<b style="color:{BULL}">Green</b> = the spread rose that month, <b style="color:{BEAR}">red</b> '
            f'= it fell, and deeper colour means a larger move. Read <b style="color:{TEXT}">across a row</b> '
            f'to see whether a given month is consistently up or down across years (recurring seasonality); '
            f'read <b style="color:{TEXT}">down a column</b> to trace one vintage\u2019s month-by-month path. '
            f'This view is full-life and not restricted by the maturity bucket.</div>',
            unsafe_allow_html=True,
        )

    def _heatmap_fig(self, ctx: ChapterContext) -> go.Figure:
        hm = spread_monthly_heatmap(ctx.structure.key)
        fig = go.Figure()
        if not hm.empty:
            months = [calendar.month_abbr[m] for m in hm.index]
            zmax = float(np.nanmax(np.abs(hm.to_numpy()))) or 1.0
            fig.add_trace(go.Heatmap(
                z=hm.to_numpy(), x=[str(v) for v in hm.columns], y=months,
                colorscale=[[0.0, BEAR], [0.5, SURFACE], [1.0, BULL]],
                zmid=0, zmin=-zmax, zmax=zmax,
                colorbar=dict(title="Δ / mo"),
                hovertemplate="%{x} %{y}<br>Δ %{z:.2f}<extra></extra>",
            ))
        fig = self._dark(fig, f"{ctx.structure.label} — monthly move by vintage", "Vintage", "Month", height=360)
        fig.update_yaxes(autorange="reversed")
        return fig

    # -- context -------------------------------------------------------------

    def _percentile_caption(self, ctx: ChapterContext) -> None:
        pct = spread_percentile(ctx.structure.key, ctx.base_year, BUFFER_DAYS, self._lo, self._hi)
        if pct is None:
            return
        st.caption(
            f"Context only: the current level sits around the **{pct:.0f}th percentile** vs history at this "
            f"point in the life — informational, not a reason to fade the trend."
        )

    def _fundamentals(self, ctx: ChapterContext) -> None:
        near_code = ctx.structure.legs[0].month_code
        near_month = MONTH_CODES[near_code]
        driver = SEASONAL_DRIVERS.get(near_month, "")
        st.markdown(
            f"""
            <div style="background:{SURFACE};border:1px solid {FLAT}22;border-left:4px solid {ACCENT};
                        border-radius:10px;padding:10px 14px;margin:2px 0 6px 0">
                <span style="color:{SUBTEXT};font-size:11px;font-weight:700;letter-spacing:.5px;
                             text-transform:uppercase">Fundamental context · {calendar.month_name[near_month]}</span>
                <div style="color:{TEXT};font-size:13px;margin-top:3px">{driver}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -- validation backtest (§8) --------------------------------------------

    def _validation(self, ctx: ChapterContext) -> None:
        horizon = HEADLINE_HORIZON
        with st.expander("Validation — does alignment improve continuation?", expanded=False):
            bt = spread_alignment_backtest(ctx.structure.key, horizon, BUFFER_DAYS, self._lo, self._hi)
            if bt.empty:
                st.info("Not enough history to backtest alignment for this structure.")
                return
            show = bt.rename(columns={
                "alignment": "Alignment", "n": "Samples",
                "avg_continuation": f"Avg {horizon}d continuation",
                "hit_rate": "Hit rate", "median_return": "Median return",
            }).copy()
            show["Hit rate"] = show["Hit rate"].map(lambda v: "—" if np.isnan(v) else f"{v:.0%}")
            for c in (f"Avg {horizon}d continuation", "Median return"):
                show[c] = show[c].map(lambda v: "—" if np.isnan(v) else f"{v:+.3f}")
            st.dataframe(show, width="stretch", hide_index=True)
            st.caption(
                "Forward continuation is measured **in the trend's direction** (positive = the trend extended). "
                "Key question: does *Tailwind* beat *Headwind*? If yes, alignment earns weight; if not, it stays "
                "visual context only."
            )
            self._validation_verdict(bt, horizon)

    def _validation_verdict(self, bt: pd.DataFrame, horizon: int) -> None:
        rows = {r["alignment"]: r for _, r in bt.iterrows()}
        tw, hw = rows.get("Tailwind"), rows.get("Headwind")
        if tw is None or hw is None or np.isnan(tw["avg_continuation"]) or np.isnan(hw["avg_continuation"]):
            return
        edge = tw["avg_continuation"] - hw["avg_continuation"]
        if edge > 0:
            st.success(
                f"Tailwind continuation exceeds headwind by {edge:+.3f} over {horizon}d "
                f"({tw['hit_rate']:.0%} vs {hw['hit_rate']:.0%} hit rate) — alignment adds information here.",
                icon="\u2705",
            )
        else:
            st.info(
                f"Tailwind does not beat headwind here ({edge:+.3f}) — treat seasonality as visual context "
                f"only for this structure, not a weighted input.",
                icon="\u2139\uFE0F",
            )

    # -- shared --------------------------------------------------------------

    @staticmethod
    def _dark(fig: go.Figure, title: str, x_title: str, y_title: str, height: int = 460) -> go.Figure:
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center", font=dict(color=ACCENT, size=18)),
            template="plotly_dark", paper_bgcolor=BLACK, plot_bgcolor=BLACK,
            font=dict(color=TEXT, family="Arial"), hovermode="x unified",
            margin=dict(l=60, r=30, t=60, b=45), height=height,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        fig.update_xaxes(title_text=x_title, gridcolor=GRID, showline=True, linecolor=ACCENT)
        fig.update_yaxes(title_text=y_title, gridcolor=GRID, showline=True, linecolor=ACCENT, zeroline=False)
        return fig
