"""Composite-extreme breakdown screen.

Shows the selected instrument's composite extreme/persistence rank, the block
decomposition, and empirical outcomes of analogous historical setups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crudewatch.infra import FAMILY_LABELS

from core.audit import audit_rows, descriptive_bias, pm_description
from core.context import selected_context_cached
from core.evidence import MIN_EFFECTIVE_N, evidence_read
from core.scoring import (
    enriched_frame,
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


def _signed_tone(value: float) -> str:
    if value > 0:
        return BULL
    if value < 0:
        return BEAR
    return SUBTEXT


class ExtensionScreen:
    """Composite extreme-rank view for the globally selected instrument."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def display(self, selection: Selection) -> None:
        title_block(
            "Extremo",
            "Qué tan extremo está el instrumento frente a su familia, con bloques y evidencia auditable.",
        )
        caveat_note(
            "<p>Este panel no da setups automáticos ni instrucciones de posición. "
            "El número principal es un <b>rank</b> para comparar estructuras dentro de su familia.</p>"
        )
        self._how_it_works()

        if selection.contract is None:
            st.info("Selecciona una familia y una fecha con contratos activos en el menú superior.")
            return
        self._render_instrument(selection)

    # -- explanation ---------------------------------------------------------

    def _how_it_works(self) -> None:
        with st.expander("¿Cómo se calcula la extensión?", expanded=False):
            st.markdown(
                """
**Extremo vs familia** va de **0 a 100** y compara el contrato contra la
historia de su propia familia disponible hasta la fecha elegida. Es un percentil:
alto significa "más extremo que la historia familiar", no "mejor trade".

El **signo interno** se conserva como dato técnico para ver el signo del
modelo interno, pero no es una frontera operativa ni una magnitud de convicción.

| Bloque | Qué mide |
|--------|----------|
| **Régimen / trendiness** | ¿Existe régimen? Rango, tendencia o transición; trendiness = ER + variance ratio + autocorrelación |
| **Dirección** | Inclinación del movimiento (−100…+100): pendiente, MACD, EMA 20/50/100 y momentum 5/10/20 |
| **Fuerza** | Calidad/limpieza de la tendencia (0…100): R² (linealidad) + persistencia direccional |
| **Nivel** | Caro vs barato: z-scores 10/20/50, Keltner y panel análogo |
| **P(reversión)** | Probabilidad histórica de que el extremo revierta |
| **P(continuación)** | Probabilidad histórica de que el movimiento continúe |
| **Evidencia histórica** | Cohorte auditable de setups análogos: n, movimiento medio, acierto y excursiones |

**Cómo se combinan (pesos iguales, sin overrides por familia):**
- **Rango** → reversión, nivel, timing, volatilidad, divergencia MACD, divergencia RSI, desaceleración de momentum y caída de ER pesan igual.
- **Tendencia** → dirección, fuerza, continuación, baja extensión, alineación EMA, momentum 10d y persistencia direccional pesan igual.
- **Transición** → se encoge el composite.

**Base ejecutable:** los retornos históricos se miden desde `open[t+1]`
(`fwd = close[t+h] − open[t+1]`);
la señal se forma con datos as-of `t`. Al elegir una fecha, el cálculo es
**point-in-time**: solo usa información disponible hasta esa fecha.

*Es una herramienta de lectura de mercado; no da instrucciones de posición.*
                """
            )

    # -- instrument view -----------------------------------------------------

    def _render_instrument(self, selection: Selection) -> None:
        family = selection.family
        contract = selection.contract
        horizon = selection.horizon
        as_of_iso = selection.as_of_iso

        try:
            context = selected_context_cached(
                family,
                contract,
                horizon,
                as_of_iso,
                include_horizons=True,
            )
        except Exception as exc:
            st.error(f"No se pudo calcular el instrumento: {exc}")
            return
        score = context["score"]

        scored_date = pd.Timestamp(score["date"])
        if scored_date.normalize() != selection.as_of.normalize():
            st.caption(f"Última barra disponible ≤ fecha elegida: {scored_date:%d %b %Y}.")

        blocks = score["blocks"]
        signed_composite = float(score["opportunity"])
        stretch_rank = float(context.get("stretch_rank", float("nan")))
        tone = _signed_tone(signed_composite)

        st.markdown(
            f"<div style='display:inline-block;padding:10px 18px;border-radius:10px;"
            f"border:1px solid {BORDER};background:{SURFACE};border-left:4px solid {tone};"
            f"font-size:20px;font-weight:700;color:{tone};margin:8px 0 16px 0;'>"
            f"Extremo vs familia {_fmt(stretch_rank, '.0f')} · signo interno {signed_composite:+.0f}</div>",
            unsafe_allow_html=True,
        )
        self._trade_context(context)
        left, right = st.columns([1, 1])
        with left:
            self._gauge(stretch_rank)
        with right:
            self._block_bars(blocks)

        self._audit_panel(family, horizon, score, context["cohort"], context.get("rank_row", {}))

        st.markdown("#### Productos análogos (mismos indicadores)")
        self._analogues(family, horizon, context["cohort"])

        st.markdown("#### Retorno esperado por horizonte (D+1 … D+30)")
        self._horizon_curve(context["horizons"])

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

    def _audit_panel(self, family: str, horizon: int, score: dict, cohort: dict, rank_row: dict) -> None:
        with st.expander("Auditoría de la lectura", expanded=False):
            rows = audit_rows(family, horizon, score, cohort, rank_row)
            st.dataframe(
                pd.DataFrame({"Métrica": list(rows), "Lectura": list(rows.values())}),
                width=1200,
                hide_index=True,
            )

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
            BEAR if blocks["level"] >= 0 else BULL,
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
        st.plotly_chart(_base_layout(fig, 320), use_container_width=True)
        st.caption(f"Régimen: {_REGIME_ES.get(blocks['regime'], blocks['regime'])}")

    def _trade_context(self, context: dict) -> None:
        score = context["score"]
        cohort = context["cohort"]
        c = st.columns(5)
        c[0].metric("Extremo vs familia", _fmt(context.get("stretch_rank", float("nan")), ".0f"))
        c[1].metric("Signo interno", _fmt(float(score["opportunity"]), "+.0f"))
        read = evidence_read(score, cohort)
        c[2].metric("Evidencia", read["label"])
        if int(cohort.get("n", 0)) >= MIN_EFFECTIVE_N:
            c[3].metric("Tasa positiva", _fmt(cohort.get("aligned_win_rate", float("nan")) * 100, ".0f") + "%")
            c[4].metric("Sharpe cohorte anual.", _fmt(cohort.get("sharpe_aligned", float("nan")), "+.2f"))
        else:
            c[3].metric("Casos efectivos", f"{int(cohort.get('n', 0))}")
            c[4].metric("Sharpe cohorte anual.", "oculto")
        st.caption(read["detail"])
        rank_row = context.get("rank_row", {})
        st.markdown(f"**Lectura PM:** {pm_description(score, cohort, rank_row)}")
        st.caption(f"Sesgo descriptivo: {descriptive_bias(score)}")
        risks = score.get("risks", [])
        if risks:
            st.caption("Alertas: " + " · ".join(risks[:4]))

    # -- analogues -----------------------------------------------------------

    def _analogues(self, family: str, horizon: int, coh: dict) -> None:
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
            f"base ejecutable. n efectivo = {n}; raw = {int(coh.get('n_raw', n))}. "
            f"Tenor {coh.get('tenor_bucket', '—')} · mes {coh.get('month', '—')}."
        )

        side = float(coh.get("side", 0.0))
        avg_fwd = float(coh.get("avg_fwd", float("nan")))
        c = st.columns(4)
        c[0].metric("Casos análogos", f"{n}")
        if n < MIN_EFFECTIVE_N:
            c[1].metric("Raw", f"{int(coh.get('n_raw', n))}")
            c[2].metric("Mediana", "oculta")
            samples = coh.get("aligned_samples") or coh.get("fwd_samples") or []
            if samples:
                with st.expander("Muestras raw", expanded=False):
                    st.dataframe(
                        pd.DataFrame({"resultado": pd.Series(samples, dtype=float)}).head(100),
                        width=520,
                        hide_index=True,
                        column_config={"resultado": st.column_config.NumberColumn(format="%+.3f")},
                    )
            return
        c[1].metric("Movimiento medio (pts)", _fmt(avg_fwd, "+.3f"))
        c[2].metric("% subidas", _fmt(coh.get("up_rate", float("nan")) * 100, ".0f") + "%")
        if side != 0.0 and "avg_aligned" in coh:
            aligned = float(coh["avg_aligned"])
            c[3].metric(
                "Mediana con signo composite (pts)",
                _fmt(coh.get("median_aligned", aligned), "+.3f"),
                delta=f"{coh.get('aligned_win_rate', float('nan')) * 100:.0f}% aciertos",
            )
        else:
            c[3].metric("Mediana con signo composite", "—")

        if side != 0.0 and "mae_p80" in coh:
            c2 = st.columns(3)
            c2[0].metric("MFE p80", _fmt(coh.get("mfe_p80", float("nan")), ".3f"))
            c2[1].metric("MAE p80", _fmt(coh.get("mae_p80", float("nan")), ".3f"))
            c2[2].metric("Sharpe cohorte anual.", _fmt(coh.get("sharpe_aligned", float("nan")), "+.2f"))

    def _horizon_curve(self, hz: pd.DataFrame) -> None:
        if hz.empty or (hz["n"] == 0).all():
            st.caption("Sin historia análoga suficiente para construir la curva de horizontes.")
            return
        hz = hz.copy()
        weak = hz["n"] < MIN_EFFECTIVE_N
        hz.loc[weak, ["median_aligned", "mae_p50", "mae_p80", "mfe_p80", "sharpe_aligned"]] = float("nan")

        st.caption(
            "Resultado de setups análogos a cada horizonte, en base ejecutable. "
            "Se muestra mediana neta y distribución de excursión adversa."
        )

        table = hz.assign(
            **{
                "D+": hz["horizon"].astype(int),
                "n": hz["n"].astype(int),
            }
        )[["D+", "n", "avg_fwd", "up_rate", "median_aligned", "sharpe_aligned", "mae_p50", "mae_p80"]].rename(
            columns={
                "avg_fwd": "Mov. medio (pts)",
                "up_rate": "% subidas",
                "median_aligned": "Mediana neta (pts)",
                "sharpe_aligned": "Sharpe cohorte anual.",
                "mae_p50": "MAE p50 (pts)",
                "mae_p80": "MAE p80 (pts)",
            }
        )
        st.dataframe(
            table.style.format(
                {
                    "Mov. medio (pts)": "{:+.3f}",
                    "% subidas": lambda v: "—" if v != v else f"{v * 100:.0f}%",
                    "Mediana neta (pts)": "{:+.3f}",
                    "Sharpe cohorte anual.": "{:+.2f}",
                    "MAE p50 (pts)": "{:.3f}",
                    "MAE p80 (pts)": "{:.3f}",
                }
            ),
            width=1200,
            hide_index=True,
        )

        valid = hz[hz["n"] >= MIN_EFFECTIVE_N]
        if valid.empty:
            st.caption("Sin horizontes con n efectivo suficiente para graficar percentiles.")
            return
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=valid["horizon"], y=valid["mfe_p80"], name="MFE p80",
                mode="lines", line=dict(color=BULL, width=1, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=valid["horizon"], y=-valid["mae_p80"], name="MAE p80",
                mode="lines", line=dict(color=BEAR, width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(139,150,145,0.10)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=valid["horizon"], y=valid["median_aligned"], name="Mediana con signo",
                mode="lines+markers", line=dict(color=ACCENT, width=2),
            )
        )
        fig.update_xaxes(title_text="Horizonte (D+, días de trading)")
        fig.update_yaxes(title_text="Puntos")
        st.plotly_chart(_base_layout(fig, 320), use_container_width=True)

    # -- charts --------------------------------------------------------------

    def _gauge(self, stretch_rank: float) -> None:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=0.0 if stretch_rank != stretch_rank else stretch_rank,
                number={"font": {"color": TEXT, "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": BORDER},
                    "bar": {"color": ACCENT, "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 70], "color": "rgba(139,150,145,0.12)"},
                        {"range": [70, 90], "color": "rgba(34,211,238,0.16)"},
                        {"range": [90, 100], "color": "rgba(16,185,129,0.22)"},
                    ],
                    "threshold": {
                        "line": {"color": ACCENT, "width": 2},
                        "thickness": 0.8,
                        "value": 0.0 if stretch_rank != stretch_rank else stretch_rank,
                    },
                },
                title={"text": "Extremo vs familia", "font": {"color": SUBTEXT, "size": 14}},
            )
        )
        st.plotly_chart(_base_layout(fig, 320), use_container_width=True)

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
        st.plotly_chart(_base_layout(fig, 260), use_container_width=True)
