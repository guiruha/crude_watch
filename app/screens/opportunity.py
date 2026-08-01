"""Opportunity Score screen.

A single view: at the top, how the Opportunity Score is built; below, one tab per
instrument family. Each family tab has a date picker and a picker of the
contracts trading on that date, then shows the score (as-of, point-in-time) and
every element that composes it, each with a plain-language explanation of what it
means, plus the empirical outcome of analogous historical setups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crudewatch.infra import FAMILY_LABELS
from crudewatch.scoring.reliability import HEADLINE, MEASURED_ON, reliability_for

from core.scoring import (
    analogous_outcomes_cached,
    enriched_frame,
    horizon_outcomes_cached,
    score_instrument_dict,
)
from core.selection import Selection
from theme.palette import (
    ACCENT,
    BEAR,
    BORDER,
    BULL,
    FLAT,
    SUBTEXT,
    SURFACE,
    TEXT,
    caveat_note,
    reliability_chip,
    title_block,
)

_REGIME_ES = {"range": "Rango", "transition": "Transición", "trend": "Direccional"}

def _label(family: str) -> str:
    return FAMILY_LABELS.get(family, family.title())


def _fmt(value: float, spec: str = ".2f", dash: str = "—") -> str:
    return dash if value is None or (isinstance(value, float) and np.isnan(value)) else format(value, spec)


def _base_layout(fig: go.Figure, height: int = 380) -> go.Figure:
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
    # Softer gridlines than the plotly_dark default (best-practice: keep grids
    # subtle, carry info via labels), and a hairline zero baseline.
    grid = "#161C19"
    fig.update_xaxes(gridcolor=grid, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor=grid, zeroline=True, zerolinecolor=BORDER, showline=False)
    return fig


def _action_tone(opportunity: float) -> str:
    if opportunity >= 20:
        return BULL
    if opportunity <= -20:
        return BEAR
    return SUBTEXT


class OpportunityScreen:
    """Opportunity-Score view for the globally selected instrument."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def display(self, selection: Selection) -> None:
        title_block(
            "Opportunity Score",
            "Una puntuación compuesta (−100…+100) por instrumento: reversión en rango, "
            "continuación en tendencia, calibrada por familia y en base ejecutable open[t+1].",
        )
        caveat_note(
            f"<p>{HEADLINE}</p>"
            "<p>Los pesos ajustados se rechazaron en <b>3 de 4</b> familias evaluadas: "
            "batían a los pesos iguales dentro de muestra, pero <b>perdían fuera de "
            "muestra</b>. Esas familias usan pesos iguales. En <code>flies</code> el "
            "ajuste llegaba a <b>−0.174</b> fuera de muestra frente a <b>0.348</b> con "
            "pesos iguales.</p>"
            f"<p>Cuatro familias (<code>outrights</code>, <code>calendars</code>, "
            "<code>cracks</code>, <code>brent_wti</code>) <b>no están evaluadas</b>: "
            f"no hay medida de fiabilidad para ellas. Última medición: {MEASURED_ON}.</p>"
        )
        self._how_it_works()

        if selection.contract is None:
            st.info("Selecciona una familia y una fecha con contratos activos en el menú superior.")
            return
        self._render_instrument(selection)

    # -- explanation ---------------------------------------------------------

    def _how_it_works(self) -> None:
        with st.expander("¿Cómo se calcula el Opportunity Score?", expanded=False):
            st.markdown(
                """
El **Opportunity Score** va de **−100** (oportunidad corta muy fuerte) a **+100**
(oportunidad larga muy fuerte). No es un indicador más: resume en un número la
decisión, combinando los bloques calculados sobre el último dato disponible y
**calibrados con la historia de cada familia** (no un modelo único para todo).

| Bloque | Qué mide |
|--------|----------|
| **Régimen / trendiness** | ¿Existe régimen? Rango, tendencia o transición; trendiness = ER + variance ratio + autocorrelación |
| **Dirección** | Sesgo (−100…+100): pendiente, MACD, EMA 20/50/100 y momentum 5/10/20 |
| **Fuerza** | Calidad/limpieza de la tendencia (0…100): R² (linealidad) + persistencia direccional |
| **Nivel** | Caro vs barato: z-scores 10/20/50, Keltner y panel análogo |
| **P(reversión)** | Probabilidad histórica de que el extremo revierta |
| **P(continuación)** | Probabilidad histórica de que el movimiento continúe |
| **Fiabilidad** | *Informativa* (no entra en el score): track record de setups análogos — cuánto acertó seguir la acción |

**Cómo se combinan.** Por defecto, **pesos iguales**. Sólo se usan pesos ajustados
en una familia donde han superado a los pesos iguales *fuera de muestra*
(validación walk-forward); en el resto se descartaron por sobreajuste.
- **Rango** → `0.25·reversión + 0.25·nivel + 0.25·timing + 0.25·volatilidad`, y se opera *en contra* del extremo (caro → corto, barato → largo).
- **Tendencia** → `0.25·dirección + 0.25·fuerza + 0.25·continuación + 0.25·extensión`, siguiendo la dirección.
- **Transición** → se encoge la señal (mayor incertidumbre).

**Base ejecutable:** todo se mide con entrada en `open[t+1]` (`fwd = close[t+h] − open[t+1]`);
la señal se forma con datos as-of `t`. Al elegir una fecha, el cálculo es
**point-in-time**: solo usa información disponible hasta esa fecha.

*Es una herramienta de apoyo a la decisión, no una recomendación de inversión.*
                """
            )

    # -- instrument view -----------------------------------------------------

    def _render_instrument(self, selection: Selection) -> None:
        family = selection.family
        contract = selection.contract
        horizon = selection.horizon
        as_of_iso = selection.as_of_iso

        try:
            score = score_instrument_dict(family, contract, horizon, as_of_iso)
        except Exception as exc:
            st.error(f"No se pudo calcular el instrumento: {exc}")
            return

        scored_date = pd.Timestamp(score["date"])
        if scored_date.normalize() != selection.as_of.normalize():
            st.caption(f"Última barra disponible ≤ fecha elegida: {scored_date:%d %b %Y}.")

        blocks = score["blocks"]
        opportunity = float(score["opportunity"])
        action = score["action"]
        tone = _action_tone(opportunity)

        st.markdown(
            f"<div style='display:inline-block;padding:10px 18px;border-radius:10px;"
            f"border:1px solid {BORDER};background:{SURFACE};border-left:4px solid {tone};"
            f"font-size:20px;font-weight:700;color:{tone};margin:8px 0 16px 0;'>"
            f"Opportunity Score {opportunity:+.0f} — {action}</div>",
            unsafe_allow_html=True,
        )
        # Right under the number, so the score is never read without its
        # measured reliability alongside it.
        reliability_chip(_label(family), reliability_for(family))

        left, right = st.columns([1, 1])
        with left:
            self._gauge(opportunity)
        with right:
            self._block_bars(blocks)

        st.markdown("#### Productos análogos (mismos indicadores)")
        self._analogues(family, contract, horizon, as_of_iso, opportunity)

        st.markdown("#### Retorno esperado por horizonte (D+1 … D+30)")
        self._horizon_curve(family, contract, as_of_iso)

        with st.expander("Explicación"):
            for bullet in score.get("rationale", []):
                st.markdown(f"- {bullet}")

        risks = score.get("risks", [])
        if risks:
            st.markdown("##### Riesgos")
            chips = " ".join(
                f"<span style='display:inline-block;margin:4px 6px 4px 0;padding:4px 10px;"
                f"border-radius:8px;border:1px solid {BORDER};background:{SURFACE};"
                f"color:{SUBTEXT};font-size:13px;'>{r}</span>"
                for r in risks
            )
            st.markdown(chips, unsafe_allow_html=True)

        st.markdown("##### Precio reciente (hasta la fecha)")
        self._price_line(family, contract, scored_date)

    def _block_bars(self, blocks: dict) -> None:
        labels = [
            "Trendiness (régimen)",
            "Dirección",
            "Fuerza",
            "Nivel",
            "P(reversión) %",
            "P(continuación) %",
        ]
        values = [
            blocks["trendiness"],
            blocks["direction"],
            blocks["strength"],
            blocks["level"],
            blocks["p_reversion"] * 100,
            blocks["p_continuation"] * 100,
        ]
        colors = [
            ACCENT if blocks["regime"] == "trend" else SUBTEXT if blocks["regime"] == "transition" else FLAT,
            BULL if blocks["direction"] >= 0 else BEAR,
            ACCENT,
            BULL if blocks["level"] >= 0 else BEAR,
            ACCENT,
            ACCENT,
        ]
        fig = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color=colors,
                text=[_fmt(v, ".1f") for v in values],
                textposition="outside",
            )
        )
        fig.update_xaxes(title_text="Puntuación / %")
        st.plotly_chart(_base_layout(fig, 320), width="stretch")
        st.caption(f"Régimen: {_REGIME_ES.get(blocks['regime'], blocks['regime'])}")

    # -- analogues -----------------------------------------------------------

    def _analogues(
        self, family: str, contract: str, horizon: int, as_of_iso: str, opportunity: float
    ) -> None:
        try:
            coh = analogous_outcomes_cached(family, contract, horizon, as_of_iso)
        except Exception as exc:
            st.caption(f"No se pudo calcular análogos: {exc}")
            return

        n = int(coh.get("n", 0))
        if n == 0:
            st.caption(
                "Sin historia análoga suficiente para este setup en la fecha elegida "
                f"({_REGIME_ES.get(coh.get('regime',''), coh.get('regime',''))}, "
                f"nivel {coh.get('level_bin','—')})."
            )
            return

        regime_es = _REGIME_ES.get(coh["regime"], coh["regime"])
        st.caption(
            f"Media histórica de contratos de {_label(family)} en el **mismo régimen "
            f"({regime_es})** y **mismo nivel ({coh['level_bin']})** — resultado a D+{horizon}, "
            f"base ejecutable. n = {n} observaciones (solo resultados ya realizados a esa fecha)."
        )

        side = float(coh.get("side", 0.0))
        avg_fwd = float(coh.get("avg_fwd", float("nan")))
        c = st.columns(4)
        c[0].metric("Casos análogos", f"{n}")
        c[1].metric("Movimiento medio (pts)", _fmt(avg_fwd, "+.3f"))
        c[2].metric("% subidas", _fmt(coh.get("up_rate", float("nan")) * 100, ".0f") + "%")
        if side != 0.0 and "avg_aligned" in coh:
            aligned = float(coh["avg_aligned"])
            c[3].metric(
                "PnL medio siguiendo la acción (pts)",
                _fmt(aligned, "+.3f"),
                delta=f"{coh.get('aligned_win_rate', float('nan')) * 100:.0f}% aciertos",
            )
        else:
            c[3].metric("PnL medio siguiendo la acción", "—")

        if side != 0.0 and "avg_mfe" in coh:
            c2 = st.columns(2)
            c2[0].metric("Excursión favorable media (MFE)", _fmt(coh["avg_mfe"], "+.3f"))
            c2[1].metric("Excursión adversa media (MAE)", _fmt(coh["avg_mae"], "+.3f"))

    def _horizon_curve(self, family: str, contract: str, as_of_iso: str) -> None:
        try:
            hz = horizon_outcomes_cached(family, contract, as_of_iso)
        except Exception as exc:
            st.caption(f"No se pudieron calcular los horizontes: {exc}")
            return
        if hz.empty or (hz["n"] == 0).all():
            st.caption("Sin historia análoga suficiente para construir la curva de horizontes.")
            return

        st.caption(
            "Resultado medio de setups análogos (mismo régimen y nivel) a cada horizonte, "
            "en base ejecutable. Movimiento y excursiones favorable (MFE) / adversa (MAE) en puntos."
        )

        table = hz.assign(
            **{
                "D+": hz["horizon"].astype(int),
                "n": hz["n"].astype(int),
            }
        )[["D+", "n", "avg_fwd", "up_rate", "avg_mfe", "avg_mae"]].rename(
            columns={
                "avg_fwd": "Mov. medio (pts)",
                "up_rate": "% subidas",
                "avg_mfe": "MFE media (pts)",
                "avg_mae": "MAE media (pts)",
            }
        )
        st.dataframe(
            table.style.format(
                {
                    "Mov. medio (pts)": "{:+.3f}",
                    "% subidas": lambda v: "—" if v != v else f"{v * 100:.0f}%",
                    "MFE media (pts)": "{:+.3f}",
                    "MAE media (pts)": "{:+.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        valid = hz[hz["n"] > 0]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=valid["horizon"], y=valid["avg_mfe"], name="MFE media",
                mode="lines", line=dict(color=BULL, width=1, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=valid["horizon"], y=valid["avg_mae"], name="MAE media",
                mode="lines", line=dict(color=BEAR, width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(139,150,145,0.10)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=valid["horizon"], y=valid["avg_fwd"], name="Movimiento medio",
                mode="lines+markers", line=dict(color=ACCENT, width=2),
            )
        )
        fig.update_xaxes(title_text="Horizonte (D+, días de trading)")
        fig.update_yaxes(title_text="Puntos")
        st.plotly_chart(_base_layout(fig, 320), width="stretch")

    # -- charts --------------------------------------------------------------

    def _gauge(self, opportunity: float) -> None:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=opportunity,
                number={"font": {"color": TEXT, "size": 36}},
                gauge={
                    "axis": {"range": [-100, 100], "tickwidth": 1, "tickcolor": BORDER},
                    "bar": {"color": _action_tone(opportunity), "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [-100, -20], "color": "rgba(229,72,77,0.25)"},
                        {"range": [-20, 20], "color": "rgba(139,150,145,0.15)"},
                        {"range": [20, 100], "color": "rgba(16,185,129,0.25)"},
                    ],
                    "threshold": {
                        "line": {"color": ACCENT, "width": 2},
                        "thickness": 0.8,
                        "value": opportunity,
                    },
                },
                title={"text": "Oportunidad", "font": {"color": SUBTEXT, "size": 14}},
            )
        )
        st.plotly_chart(_base_layout(fig, 320), width="stretch")

    def _price_line(self, family: str, contract: str, as_of: pd.Timestamp | None = None) -> None:
        data = enriched_frame(family)
        sub = data[data["contract"] == contract].sort_values("date")
        if as_of is not None:
            sub = sub[pd.to_datetime(sub["date"]) <= pd.Timestamp(as_of)]
        if sub.empty or "close" not in sub.columns:
            st.caption("Sin serie de precios.")
            return
        tail = sub.tail(60)
        fig = go.Figure(
            go.Scatter(
                x=tail["date"],
                y=tail["close"],
                mode="lines",
                line=dict(color=ACCENT, width=2),
                name=contract,
            )
        )
        fig.update_yaxes(title_text="Cierre")
        st.plotly_chart(_base_layout(fig, 260), width="stretch")
