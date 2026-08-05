"""PM summary screen: conclusions, drivers and audit for the selected read."""
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from crudewatch.infra import FAMILY_LABELS, FAMILY_POINT_VALUE_USD
from crudewatch.plots import price_volume_figure

from core.audit import audit_rows, descriptive_bias, pm_description
from core.context import selected_context_cached
from core.evidence import MIN_EFFECTIVE_N, evidence_read
from core.indicator_buckets import (
    bucket_signal_windows_cached,
    indicator_bucket_combinations_cached,
    indicator_bucket_outcomes_cached,
)
from core.runtime import low_memory_mode
from core.scoring import HORIZONS
from core.selection import Selection
from screens.opportunity import _REGIME_ES, _base_layout
from theme.palette import ACCENT, BEAR, BORDER, BULL, CHART_ACCENT, SUBTEXT, SURFACE, TEXT, title_block

_CHART_STYLE: dict[str, tuple[str, str, bool]] = {
    "outrights": ("Outright", "Close ($/bbl)", True),
    "calendars": ("Calendar", "Spread ($/bbl)", False),
    "quarterly": ("Quarterly", "Spread ($/bbl)", False),
    "semestral": ("Semestral", "Spread ($/bbl)", False),
    "yearly": ("Yearly", "Spread ($/bbl)", False),
    "flies": ("Fly", "Fly ($/bbl)", False),
    "cracks": ("Crack", "Crack ($/bbl)", False),
    "brent_wti": ("Brent-WTI", "Brent-WTI ($/bbl)", False),
}


def _num(value, spec: str = ".2f", dash: str = "-") -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return dash
    if value != value:
        return dash
    return format(value, spec)


def _family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, family.title())


class PMScreen:
    """Daily PM-oriented synthesis for one selected instrument."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def display(self, selection: Selection) -> None:
        title_block(
            "PM",
            "Conclusión discrecional documentada: qué importa, por qué importa y qué mirar antes de actuar.",
        )
        if selection.contract is None:
            st.info("Selecciona una familia y un contrato en la barra superior.")
            return

        try:
            context = selected_context_cached(
                selection.family,
                selection.contract,
                selection.horizon,
                selection.as_of_iso,
                include_horizons=True,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo construir la lectura PM: {exc}")
            return

        score = context["score"]
        cohort = context["cohort"]
        rank_row = context.get("rank_row", {})
        read = evidence_read(score, cohort)

        self._headline(selection, context, read)
        self._executive_summary(selection, context, read)
        self._metrics(selection, context, read)
        self._conclusions(selection, score, cohort, rank_row, read)
        self._signal_windows(selection)
        self._directional_expectation(selection)
        self._indicator_buckets(selection)
        self._bucket_combinations(selection)
        self._horizon_table(context["horizons"])
        self._price_history(selection)
        self._audit(selection, score, cohort, rank_row)

    def _headline(self, selection: Selection, context: dict, read: dict) -> None:
        score = context["score"]
        cohort = context["cohort"]
        rank_row = context.get("rank_row", {})
        st.markdown(
            f"### {_family_label(selection.family)} · {selection.contract} · D+{selection.horizon}"
        )
        st.markdown(f"**Lectura PM:** {pm_description(score, cohort, rank_row)}")
        st.caption(
            f"Sesgo descriptivo: {descriptive_bias(score)} · Evidencia: {read['label']} · "
            f"fecha usada {pd.Timestamp(score['date']):%d %b %Y}"
        )

    def _executive_summary(self, selection: Selection, context: dict, read: dict) -> None:
        score = context["score"]
        cohort = context["cohort"]
        rank_row = context.get("rank_row", {})
        blocks = score["blocks"]
        n = int(cohort.get("n", 0))
        active = n >= MIN_EFFECTIVE_N and float(cohort.get("side", 0.0)) != 0.0
        pm_read = pm_description(score, cohort, rank_row)
        bias = descriptive_bias(score)
        engine = {
            "range": "reversión",
            "trend": "continuación",
            "transition": "transición",
        }.get(blocks.get("regime"), "-")
        tone = (
            BEAR if "en contra" in pm_read.lower() or "baja prioridad" in pm_read.lower()
            else BULL if "vigilar" in pm_read.lower() or read.get("tone") == "positive"
            else SUBTEXT
        )
        bucket = self._top_bucket(selection)
        bucket_text = (
            f"{bucket['indicator_label']} · {bucket['bucket']} · Sharpe {_num(bucket['sharpe_bucket'], '+.2f')}"
            if bucket is not None
            else "sin bucket dominante"
        )
        direction = self._bucket_direction_read(selection)
        cells = [
            ("Lectura PM", pm_read, tone),
            ("Sesgo", bias, ACCENT),
            ("Motor", f"{engine} · {_REGIME_ES.get(blocks['regime'], blocks['regime'])}", ACCENT),
            ("Evidencia", read["label"], BULL if read.get("tone") == "positive" else BEAR if read.get("tone") == "negative" else SUBTEXT),
            ("Direccionalidad", direction["label"], direction["tone"]),
            ("Bucket dominante", bucket_text, ACCENT if bucket is not None else SUBTEXT),
        ]
        html = "<div style='display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:12px 0 14px 0;'>"
        for label, value, color in cells:
            html += (
                f"<div style='border:1px solid {BORDER};background:{SURFACE};border-left:4px solid {color};"
                "border-radius:8px;padding:12px 14px;min-height:86px;'>"
                f"<div style='color:{SUBTEXT};font-size:12px;font-weight:800;text-transform:uppercase;'>{label}</div>"
                f"<div style='color:{TEXT};font-size:18px;font-weight:800;margin-top:6px;line-height:1.25;'>{value}</div>"
                "</div>"
            )
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
        notes = [
            f"Extremo vs familia {_num(context.get('stretch_rank'), '.0f')} con nivel {blocks['level']:+.0f}.",
            f"Cohorte: régimen {cohort.get('regime', '-')}, nivel {cohort.get('level_bin', '-')}, tenor {cohort.get('tenor_bucket', '-')}.",
            (
                f"Distribución neta: mediana {_num(cohort.get('median_aligned'), '+.3f')} pts, "
                f"Sharpe anual. {_num(cohort.get('sharpe_aligned'), '+.2f')}, MAE p80 {_num(cohort.get('mae_p80'), '.3f')} pts."
                if active
                else "Distribución agregada oculta porque falta muestra suficiente o signo activo."
            ),
            direction["detail"],
            f"Operabilidad: liquidez {rank_row.get('liquidity', '-')}, vol {rank_row.get('vol_regime', '-')}, coste ${_num(rank_row.get('cost_usd'), '.0f')}.",
        ]
        st.markdown("#### Lo relevante ahora")
        st.markdown("\n".join(f"- {note}" for note in notes))

    def _top_bucket(self, selection: Selection) -> dict | None:
        horizons = (int(selection.horizon),) if low_memory_mode() else HORIZONS
        buckets = indicator_bucket_outcomes_cached(
            selection.family,
            selection.contract,
            selection.as_of_iso,
            horizons=horizons,
        )
        if buckets.empty:
            return None
        selected = buckets[
            (buckets["horizon"] == int(selection.horizon))
            & (buckets["n"] >= MIN_EFFECTIVE_N)
            & (buckets["sharpe_bucket"].notna())
        ].copy()
        if selected.empty:
            return None
        selected["_abs_sharpe"] = selected["sharpe_bucket"].abs()
        return selected.sort_values(["_abs_sharpe", "n"], ascending=[False, False]).iloc[0].to_dict()

    def _combined_bucket_combos(self, selection: Selection) -> pd.DataFrame:
        if low_memory_mode():
            horizons = (int(selection.horizon),)
            max_evolution_pairs = None
        else:
            horizons = (int(selection.horizon),) + tuple(
                horizon for horizon in HORIZONS if int(horizon) != int(selection.horizon)
            )
            max_evolution_pairs = 16
        combos = indicator_bucket_combinations_cached(
            selection.family,
            selection.contract,
            selection.as_of_iso,
            horizons=horizons,
            focus_horizon=int(selection.horizon),
            max_evolution_pairs=max_evolution_pairs,
            combo_size=4,
        )
        if not combos.empty:
            combos["combo_type"] = combos["combo_size"].map({4: "4 bloques"}).fillna("Combo")
        return combos

    def _bucket_direction_read(self, selection: Selection) -> dict:
        horizons = (int(selection.horizon),) if low_memory_mode() else HORIZONS
        buckets = indicator_bucket_outcomes_cached(
            selection.family,
            selection.contract,
            selection.as_of_iso,
            horizons=horizons,
        )
        combos = self._combined_bucket_combos(selection)
        rows: list[dict] = []
        if not buckets.empty:
            b = buckets[
                (buckets["horizon"] == int(selection.horizon))
                & (buckets["n"] >= MIN_EFFECTIVE_N)
                & (buckets["sharpe_bucket"] > 0)
                & (buckets["median_fwd"].notna())
            ].copy()
            for _, row in b.iterrows():
                rows.append(
                    {
                        "label": str(row["indicator_label"]),
                        "side": str(row["historical_side"]),
                        "sharpe": float(row["sharpe_bucket"]),
                        "sharpe_low": float(row["sharpe_bucket_cons"]),
                        "n": int(row["n"]),
                    }
                )
        if not combos.empty:
            c = combos[
                (combos["horizon"] == int(selection.horizon))
                & (combos["n"] >= MIN_EFFECTIVE_N)
                & (combos["sharpe_combo"] > 0)
                & (combos["median_fwd"].notna())
            ].copy()
            for _, row in c.iterrows():
                rows.append(
                    {
                        "label": str(row["combo_label"]),
                        "side": str(row["historical_side"]),
                        "sharpe": float(row["sharpe_combo"]),
                        "sharpe_low": float(row["sharpe_combo_cons"]),
                        "n": int(row["n"]),
                    }
                )
        if not rows:
            return {
                "label": "sin lectura por buckets",
                "detail": "Buckets: no hay muestra suficiente con Sharpe positivo para sostener una dirección histórica.",
                "tone": SUBTEXT,
            }
        panel = pd.DataFrame(rows).sort_values(["sharpe", "n"], ascending=[False, False]).head(10)
        up = int((panel["side"] == "subidas").sum())
        down = int((panel["side"] == "bajadas").sum())
        total = len(panel)
        best = panel.iloc[0]
        avg_sharpe = float(panel["sharpe"].mean())
        median_n = int(panel["n"].median())
        if total >= 3 and up / total >= 0.65:
            label = "histórico inclina a subidas"
            tone = BULL
        elif total >= 3 and down / total >= 0.65:
            label = "histórico inclina a bajadas"
            tone = BEAR
        else:
            label = "lectura histórica mixta"
            tone = SUBTEXT
        sample_note = "muestra justa" if median_n < 30 else "muestra amplia"
        if label != "lectura histórica mixta":
            label = f"{label} · {sample_note}"
        return {
            "label": label,
            "detail": (
                f"Buckets: top {total} con Sharpe positivo; {up} subidas / {down} bajadas, "
                f"Sharpe no-overlap medio {_num(avg_sharpe, '+.2f')}, n mediano {median_n} ({sample_note}). "
                f"Mejor lectura: {best['label']} ({best['side']}, Sharpe {_num(best['sharpe'], '+.2f')}, "
                f"rango bajo {_num(best['sharpe_low'], '+.2f')})."
            ),
            "tone": tone,
        }

    def _directional_expectation(self, selection: Selection) -> None:
        panel = self._directional_signal_panel(selection)
        if panel.empty:
            return
        summary = []
        for side, label in (("long", "Largo"), ("short", "Corto")):
            side_rows = panel[panel["favored_side"] == side].copy()
            if side_rows.empty:
                summary.append(
                    {
                        "Lado": label,
                        "Señales": 0,
                        "Media neta señales": float("nan"),
                        "Sharpe medio": float("nan"),
                        "Mejor Sharpe": float("nan"),
                        "n mediano": 0,
                    }
                )
                continue
            side_rows = side_rows.sort_values(["side_sharpe", "n"], ascending=[False, False]).head(12)
            weights = side_rows["n"].clip(lower=1)
            summary.append(
                {
                    "Lado": label,
                    "Señales": len(side_rows),
                    "Media neta señales": float((side_rows["side_avg"] * weights).sum() / weights.sum()),
                    "Sharpe medio": float(side_rows["side_sharpe"].mean()),
                    "Mejor Sharpe": float(side_rows["side_sharpe"].max()),
                    "n mediano": int(side_rows["n"].median()),
                }
            )
        out = pd.DataFrame(summary)
        long_row = out[out["Lado"] == "Largo"].iloc[0]
        short_row = out[out["Lado"] == "Corto"].iloc[0]
        long_signals = int(long_row["Señales"])
        short_signals = int(short_row["Señales"])
        long_sharpe = float(long_row["Sharpe medio"])
        short_sharpe = float(short_row["Sharpe medio"])
        long_avg = float(long_row["Media neta señales"])
        short_avg = float(short_row["Media neta señales"])
        sharpe_gap = long_sharpe - short_sharpe
        avg_gap = long_avg - short_avg
        out["Ventaja vs otro lado"] = [
            avg_gap if row["Lado"] == "Largo" else -avg_gap for _, row in out.iterrows()
        ]
        both_sides_live = long_signals > 0 and short_signals > 0
        close_call = both_sides_live and abs(sharpe_gap) < 0.25 and abs(avg_gap) < 0.35
        if close_call:
            verdict = "Lectura dividida: largo y corto salen parecidos"
            tone = SUBTEXT
            verdict_detail = (
                f"Largo y corto tienen buckets favorables con diferencias pequeñas "
                f"(gap media {avg_gap:+.3f}, gap Sharpe {sharpe_gap:+.2f}). "
                "No leer esto como dirección única; revisar las señales concretas."
            )
        elif long_signals and (not short_signals or sharpe_gap > 0):
            verdict = "Largo más favorable por buckets"
            tone = BULL
            verdict_detail = (
                f"Largo supera a corto por gap media {avg_gap:+.3f} y gap Sharpe {sharpe_gap:+.2f}."
            )
        elif short_signals:
            verdict = "Corto más favorable por buckets"
            tone = BEAR
            verdict_detail = (
                f"Corto supera a largo por gap media {-avg_gap:+.3f} y gap Sharpe {-sharpe_gap:+.2f}."
            )
        else:
            verdict = "sin lado favorable claro"
            tone = SUBTEXT
            verdict_detail = "No hay suficientes señales con media neta y Sharpe positivos para separar lado."

        st.markdown("#### Largo vs Corto por buckets")
        st.caption(
            "Agrega indicadores, pares y tríos cuyo bucket actual favorece cada lado. "
            "Media neta = media histórica neta si se hubiera mantenido ese lado a D+; "
            "no suma señales como si fueran trades independientes."
        )
        st.markdown(
            f"""
            <div style="border:1px solid {BORDER};background:{SURFACE};border-left:4px solid {tone};
                        border-radius:8px;padding:12px 14px;margin-bottom:10px;">
                <div style="color:{SUBTEXT};font-size:12px;font-weight:800;text-transform:uppercase;">Lectura direccional</div>
                <div style="color:{TEXT};font-size:20px;font-weight:800;margin-top:5px;">{verdict}</div>
                <div style="color:{SUBTEXT};font-size:13px;margin-top:5px;">{verdict_detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            out,
            width=1000,
            hide_index=True,
            column_config={
                "Media neta señales": st.column_config.NumberColumn(format="%+.3f"),
                "Sharpe medio": st.column_config.NumberColumn(format="%+.2f"),
                "Mejor Sharpe": st.column_config.NumberColumn(format="%+.2f"),
                "Ventaja vs otro lado": st.column_config.NumberColumn(format="%+.3f"),
            },
        )
        detail = panel.sort_values(["side_sharpe", "n"], ascending=[False, False]).head(18)[
            ["Tipo", "Señal", "favored_label", "side_avg", "side_edge", "side_sharpe", "n", "n_raw"]
        ].rename(
            columns={
                "favored_label": "Lado favorecido",
                "side_avg": "Media neta",
                "side_edge": "Edge vs contrario",
                "side_sharpe": "Sharpe",
                "n": "n efectivo",
                "n_raw": "raw",
            }
        )
        with st.expander("Señales que alimentan Largo vs Corto", expanded=False):
            st.dataframe(
                detail,
                width=1400,
                hide_index=True,
                column_config={
                    "Media neta": st.column_config.NumberColumn(format="%+.3f"),
                    "Edge vs contrario": st.column_config.NumberColumn(format="%+.3f"),
                    "Sharpe": st.column_config.NumberColumn(format="%+.2f"),
                },
            )

    def _directional_signal_panel(self, selection: Selection) -> pd.DataFrame:
        horizons = (int(selection.horizon),) if low_memory_mode() else HORIZONS
        buckets = indicator_bucket_outcomes_cached(
            selection.family,
            selection.contract,
            selection.as_of_iso,
            horizons=horizons,
        )
        combos = self._combined_bucket_combos(selection)
        rows: list[dict] = []
        if not buckets.empty:
            current = buckets[buckets["horizon"] == int(selection.horizon)].copy()
            for _, row in current.iterrows():
                rows.append(
                    {
                        "Tipo": "Indicador",
                        "Señal": str(row["indicator_label"]),
                        "n": int(row["n"]),
                        "n_raw": int(row["n_raw"]),
                        "avg_long": float(row.get("avg_long", float("nan"))),
                        "avg_short": float(row.get("avg_short", float("nan"))),
                        "sharpe_long": float(row.get("sharpe_long", float("nan"))),
                        "sharpe_short": float(row.get("sharpe_short", float("nan"))),
                    }
                )
        if not combos.empty:
            current = combos[combos["horizon"] == int(selection.horizon)].copy()
            for _, row in current.iterrows():
                rows.append(
                    {
                        "Tipo": str(row.get("combo_type", "Combo")),
                        "Señal": str(row["combo_label"]),
                        "n": int(row["n"]),
                        "n_raw": int(row["n_raw"]),
                        "avg_long": float(row.get("avg_long", float("nan"))),
                        "avg_short": float(row.get("avg_short", float("nan"))),
                        "sharpe_long": float(row.get("sharpe_long", float("nan"))),
                        "sharpe_short": float(row.get("sharpe_short", float("nan"))),
                    }
                )
        panel = pd.DataFrame(rows)
        if panel.empty:
            return panel
        panel = panel[panel["n"] >= MIN_EFFECTIVE_N].copy()
        if panel.empty:
            return panel
        long_ok = (panel["avg_long"] > 0) & (panel["sharpe_long"] > 0)
        short_ok = (panel["avg_short"] > 0) & (panel["sharpe_short"] > 0)
        panel["favored_side"] = ""
        panel.loc[long_ok & ~short_ok, "favored_side"] = "long"
        panel.loc[short_ok & ~long_ok, "favored_side"] = "short"
        both = long_ok & short_ok
        panel.loc[both & (panel["sharpe_long"] >= panel["sharpe_short"]), "favored_side"] = "long"
        panel.loc[both & (panel["sharpe_short"] > panel["sharpe_long"]), "favored_side"] = "short"
        panel = panel[panel["favored_side"] != ""].copy()
        if panel.empty:
            return panel
        panel["favored_label"] = panel["favored_side"].map({"long": "Largo", "short": "Corto"})
        panel["side_avg"] = np.where(panel["favored_side"] == "long", panel["avg_long"], panel["avg_short"])
        panel["side_edge"] = np.where(
            panel["favored_side"] == "long",
            panel["avg_long"] - panel["avg_short"],
            panel["avg_short"] - panel["avg_long"],
        )
        panel["side_sharpe"] = np.where(panel["favored_side"] == "long", panel["sharpe_long"], panel["sharpe_short"])
        return panel

    def _metrics(self, selection: Selection, context: dict, read: dict) -> None:
        score = context["score"]
        cohort = context["cohort"]
        rank_row = context.get("rank_row", {})
        point_value = FAMILY_POINT_VALUE_USD.get(selection.family, 1000.0)
        n = int(cohort.get("n", 0))
        active = n >= MIN_EFFECTIVE_N and float(cohort.get("side", 0.0)) != 0.0
        median = cohort.get("median_aligned", float("nan"))
        mae_p80 = cohort.get("mae_p80", float("nan"))
        c = st.columns(7)
        c[0].metric("Extremo", _num(context.get("stretch_rank"), ".0f"))
        c[1].metric("Régimen", _REGIME_ES.get(score["blocks"]["regime"], score["blocks"]["regime"]))
        c[2].metric("Nivel", _num(score["blocks"]["level"], "+.0f"))
        c[3].metric("Cohorte n", f"{n}")
        c[4].metric("Sharpe anual.", _num(cohort.get("sharpe_aligned"), "+.2f") if active else "-")
        c[5].metric("Mediana USD", _num(median * point_value, "+.0f") if active else "-")
        c[6].metric("MAE p80 USD", _num(mae_p80 * point_value, ".0f") if active else "-")
        st.caption(
            f"Liquidez {rank_row.get('liquidity', '-')} · vol {rank_row.get('vol_regime', '-')} · "
            f"coste asumido {_num(rank_row.get('cost_points'), '.3f')} pts / ${_num(rank_row.get('cost_usd'), '.0f')}"
        )

    def _conclusions(
        self,
        selection: Selection,
        score: dict,
        cohort: dict,
        rank_row: dict,
        read: dict,
    ) -> None:
        blocks = score["blocks"]
        n = int(cohort.get("n", 0))
        active = n >= MIN_EFFECTIVE_N and float(cohort.get("side", 0.0)) != 0.0
        rows = [
            {
                "Punto": "Prioridad PM",
                "Valor": pm_description(score, cohort, rank_row),
                "Lectura": "Resume evidencia, muestra, liquidez y volatilidad.",
                "Qué revisar": "Curva, liquidez y coherencia con el libro antes de convertirlo en idea.",
            },
            {
                "Punto": "Extremo",
                "Valor": _num(rank_row.get("stretch_rank"), ".0f"),
                "Lectura": "Percentil histórico de extremo dentro de la familia.",
                "Qué revisar": "Un valor alto prioriza atención; no mide por sí solo rentabilidad.",
            },
            {
                "Punto": "Evidencia",
                "Valor": f"{read['label']} · {read['detail']}",
                "Lectura": "Compara el signo interno con cohortes históricos análogos.",
                "Qué revisar": "Si va en contra, tratar el extremo como advertencia, no como oportunidad limpia.",
            },
            {
                "Punto": "Cobertura",
                "Valor": f"n {n}; raw {int(cohort.get('n_raw', n))}; vintages {int(cohort.get('vintage_count', 0))}",
                "Lectura": "Muestra efectiva tras quitar solape.",
                "Qué revisar": "Por debajo de 15 se muestran muestras raw y se ocultan métricas agregadas.",
            },
            {
                "Punto": "Régimen",
                "Valor": _REGIME_ES.get(blocks["regime"], blocks["regime"]),
                "Lectura": f"Trendiness {blocks['trendiness']:.0f}; dirección {blocks['direction']:+.0f}; fuerza {blocks['strength']:.0f}.",
                "Qué revisar": "Define si pesa más reversión, continuación o transición.",
            },
            {
                "Punto": "Nivel",
                "Valor": _num(blocks["level"], "+.0f"),
                "Lectura": "Caro/barato contra z-scores y panel de slot/vintage/fase de vida.",
                "Qué revisar": f"Slot {rank_row.get('slot', '-')}; vida {rank_row.get('life_phase', '-')}; vintage {_num(rank_row.get('vintage'), '.0f')}.",
            },
            {
                "Punto": "Riesgo histórico",
                "Valor": (
                    f"Sharpe anual. {_num(cohort.get('sharpe_aligned'), '+.2f')}; "
                    f"MAE p80 {_num(cohort.get('mae_p80'), '.3f')} pts"
                    if active
                    else "oculto por muestra/signo"
                ),
                "Lectura": "Distribución neta del cohorte, coste incluido.",
                "Qué revisar": "Comparar MAE con tolerancia de riesgo y profundidad disponible.",
            },
            {
                "Punto": "Coste y operabilidad",
                "Valor": f"{rank_row.get('liquidity', '-')} · ${_num(rank_row.get('cost_usd'), '.0f')}",
                "Lectura": "Coste asumido y volumen disponible en el dato.",
                "Qué revisar": "Bid/offer real sigue pendiente; no sobreponderar estructuras ilíquidas.",
            },
        ]
        st.markdown("#### Conclusiones documentadas")
        st.dataframe(pd.DataFrame(rows), width=1600, hide_index=True)

    def _indicator_buckets(self, selection: Selection) -> None:
        horizons = (int(selection.horizon),) if low_memory_mode() else HORIZONS
        buckets = indicator_bucket_outcomes_cached(
            selection.family,
            selection.contract,
            selection.as_of_iso,
            horizons=horizons,
        )
        if buckets.empty:
            return
        st.markdown("#### Buckets por indicador")
        st.caption(
            "Para cada indicador: bucket actual frente a la historia familiar y resultado histórico "
            "si se observa ese mismo bucket a D+X. Sharpe anual. neto según el sesgo histórico del bucket; "
            "se oculta si n efectivo < 15. La mediana marca el centro; el Sharpe usa media neta y dispersión."
        )
        selected = buckets[buckets["horizon"] == int(selection.horizon)].copy()
        selected = selected.sort_values(
            ["sharpe_bucket", "n"], ascending=[False, False], na_position="last"
        )
        self._bucket_cards(selected)
        self._bucket_chart(selected)
        table = selected[
            [
                "indicator_label",
                "value",
                "bucket",
                "n",
                "median_fwd",
                "avg_aligned",
                "historical_side",
                "sharpe_bucket",
                "sharpe_bucket_cons",
                "mae_p80",
            ]
        ].rename(
            columns={
                "indicator_label": "Indicador",
                "value": "Valor actual",
                "bucket": "Bucket actual",
                "n": "n efectivo",
                "median_fwd": "Mediana D+",
                "avg_aligned": "Media neta",
                "historical_side": "Sesgo hist.",
                "sharpe_bucket": "Sharpe bucket anual.",
                "sharpe_bucket_cons": "Rango bajo",
                "mae_p80": "MAE p80",
            }
        )
        st.dataframe(
            table.head(18),
            width=1400,
            hide_index=True,
            column_config={
                "Valor actual": st.column_config.NumberColumn(format="%+.3f"),
                "Mediana D+": st.column_config.NumberColumn(format="%+.3f"),
                "Media neta": st.column_config.NumberColumn(format="%+.3f"),
                "Sharpe bucket anual.": st.column_config.NumberColumn(format="%+.2f"),
                "Rango bajo": st.column_config.NumberColumn(format="%+.2f"),
                "MAE p80": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        with st.expander("Buckets por todos los horizontes", expanded=False):
            full = buckets.sort_values(
                ["indicator_label", "horizon"], ascending=[True, True]
            )[
                [
                    "indicator_label",
                    "bucket",
                    "horizon",
                    "n",
                    "median_fwd",
                    "avg_aligned",
                    "historical_side",
                    "sharpe_bucket",
                    "sharpe_bucket_cons",
                    "mae_p80",
                ]
            ].rename(
                columns={
                    "indicator_label": "Indicador",
                    "bucket": "Bucket",
                    "horizon": "D+",
                    "n": "n",
                    "median_fwd": "Mediana",
                    "avg_aligned": "Media neta",
                    "historical_side": "Sesgo hist.",
                    "sharpe_bucket": "Sharpe anual.",
                    "sharpe_bucket_cons": "Rango bajo",
                    "mae_p80": "MAE p80",
                }
            )
            st.dataframe(
                full,
                width=1400,
                hide_index=True,
                column_config={
                    "Mediana": st.column_config.NumberColumn(format="%+.3f"),
                    "Media neta": st.column_config.NumberColumn(format="%+.3f"),
                    "Sharpe anual.": st.column_config.NumberColumn(format="%+.2f"),
                    "Rango bajo": st.column_config.NumberColumn(format="%+.2f"),
                    "MAE p80": st.column_config.NumberColumn(format="%.3f"),
                },
            )

    def _bucket_cards(self, selected: pd.DataFrame) -> None:
        usable = selected[selected["n"] >= MIN_EFFECTIVE_N].copy()
        if usable.empty:
            st.info("No hay buckets con n efectivo suficiente para esta fecha/horizonte.")
            return
        usable = usable[usable["sharpe_bucket"].notna()].copy()
        usable["_abs_sharpe"] = usable["sharpe_bucket"].abs()
        highlights = usable.sort_values(["_abs_sharpe", "n"], ascending=[False, False]).head(4)
        cols = st.columns(len(highlights))
        for col, (_, row) in zip(cols, highlights.iterrows()):
            sharpe = float(row.get("sharpe_bucket", float("nan")))
            tone = BULL if sharpe > 0 else BEAR if sharpe < 0 else SUBTEXT
            col.markdown(
                f"""
                <div style="border:1px solid {BORDER};background:{SURFACE};border-left:4px solid {tone};
                            border-radius:8px;padding:12px 13px;min-height:138px;">
                    <div style="color:{SUBTEXT};font-size:12px;font-weight:700;text-transform:uppercase;">
                        {row['indicator_label']}
                    </div>
                    <div style="color:{TEXT};font-size:17px;font-weight:800;margin-top:4px;">
                        {row['bucket']} · {row['historical_side']}
                    </div>
                    <div style="color:{tone};font-size:28px;font-weight:800;margin-top:6px;">
                        {_num(sharpe, '+.2f')}
                    </div>
                    <div style="color:{SUBTEXT};font-size:12px;margin-top:4px;">
                        media {_num(row.get('avg_aligned'), '+.3f')} · rango bajo {_num(row.get('sharpe_bucket_cons'), '+.2f')} · n {int(row['n'])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def _bucket_chart(self, selected: pd.DataFrame) -> None:
        usable = selected[(selected["n"] >= MIN_EFFECTIVE_N) & selected["sharpe_bucket"].notna()].copy()
        if usable.empty:
            return
        usable = usable.sort_values("sharpe_bucket", ascending=True).tail(14)
        colors = [BULL if v > 0 else BEAR if v < 0 else SUBTEXT for v in usable["sharpe_bucket"]]
        fig = go.Figure(
            go.Bar(
                x=usable["sharpe_bucket"],
                y=usable["indicator_label"],
                orientation="h",
                marker_color=colors,
                customdata=usable[["bucket", "historical_side", "n", "median_fwd", "avg_aligned", "sharpe_bucket_cons"]].to_numpy(),
                hovertemplate=(
                    "<b>%{y}</b><br>Bucket: %{customdata[0]}<br>"
                    "Sesgo hist.: %{customdata[1]}<br>n: %{customdata[2]}<br>"
                    "Mediana: %{customdata[3]:+.3f}<br>Media neta: %{customdata[4]:+.3f}<br>"
                    "Sharpe: %{x:+.2f}<br>Rango bajo: %{customdata[5]:+.2f}<extra></extra>"
                ),
            )
        )
        fig.update_xaxes(title_text="Sharpe no-overlap", zeroline=True, zerolinecolor=BORDER)
        fig.update_yaxes(title_text="")
        st.plotly_chart(_base_layout(fig, 360), use_container_width=True)

    def _bucket_combinations(self, selection: Selection) -> None:
        combos = self._combined_bucket_combos(selection)
        if combos.empty:
            return
        st.markdown("#### Combinaciones de buckets")
        st.caption(
            "Cruza una señal de Régimen, una de Dirección, una de Fuerza y una de Nivel, "
            "tomando hasta tres candidatas por bloque. Responde: cuando estos estados "
            "aparecieron juntos, qué pasó a D+X. Se ocultan Sharpe/mediana si n efectivo < 15; "
            "Mediana y Sharpe pueden discrepar si la cola adversa domina la media neta."
        )
        selected = combos[combos["horizon"] == int(selection.horizon)].copy()
        selected = selected.sort_values(
            ["sharpe_combo", "n"], ascending=[False, False], na_position="last"
        )
        self._combo_cards(selected)
        self._combo_chart(selected)
        table = selected[
            [
                "combo_type",
                "combo_label",
                "combo_bucket",
                "n",
                "median_fwd",
                "avg_aligned",
                "historical_side",
                "sharpe_combo",
                "sharpe_combo_cons",
                "mae_p80",
            ]
        ].rename(
            columns={
                "combo_type": "Tipo",
                "combo_label": "Combinación",
                "combo_bucket": "Buckets actuales",
                "n": "n efectivo",
                "median_fwd": "Mediana D+",
                "avg_aligned": "Media neta",
                "historical_side": "Sesgo hist.",
                "sharpe_combo": "Sharpe combo anual.",
                "sharpe_combo_cons": "Rango bajo",
                "mae_p80": "MAE p80",
            }
        )
        st.dataframe(
            table.head(20),
            width=1500,
            hide_index=True,
            column_config={
                "Mediana D+": st.column_config.NumberColumn(format="%+.3f"),
                "Media neta": st.column_config.NumberColumn(format="%+.3f"),
                "Sharpe combo anual.": st.column_config.NumberColumn(format="%+.2f"),
                "Rango bajo": st.column_config.NumberColumn(format="%+.2f"),
                "MAE p80": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        with st.expander("Evolución de combinaciones top por horizonte", expanded=False):
            valid = combos[
                (combos["n"] >= MIN_EFFECTIVE_N) & combos["sharpe_combo"].notna()
            ].copy()
            if not valid.empty:
                valid = valid[valid["sharpe_combo"].notna()].copy()
                valid["_abs_sharpe"] = valid["sharpe_combo"].abs()
                top_labels = (
                    valid.sort_values(["_abs_sharpe", "n"], ascending=[False, False])
                    ["combo_label"]
                    .drop_duplicates()
                    .head(16)
                    .tolist()
                )
                heat = (
                    valid[valid["combo_label"].isin(top_labels)]
                    .pivot_table(
                        index="combo_label",
                        columns="horizon",
                        values="sharpe_combo",
                        aggfunc="last",
                    )
                    .reindex(top_labels)
                )
                fig = go.Figure(
                    go.Heatmap(
                        z=heat.to_numpy(),
                        x=[f"D+{int(h)}" for h in heat.columns],
                        y=heat.index,
                        colorscale=[[0, BEAR], [0.5, "#202620"], [1, BULL]],
                        zmid=0,
                        colorbar=dict(title="Sharpe"),
                        hovertemplate="%{y}<br>%{x}<br>Sharpe no-overlap %{z:+.2f}<extra></extra>",
                    )
                )
                fig.update_xaxes(title_text="Horizonte")
                fig.update_yaxes(title_text="")
                st.plotly_chart(_base_layout(fig, 440), use_container_width=True)
            full = combos.sort_values(
                ["combo_label", "horizon"], ascending=[True, True]
            )[
                [
                    "combo_type",
                    "combo_label",
                    "combo_bucket",
                    "horizon",
                    "n",
                   "median_fwd",
                    "avg_aligned",
                    "historical_side",
                    "sharpe_combo",
                    "sharpe_combo_cons",
                    "mae_p80",
                ]
            ].rename(
                columns={
                    "combo_type": "Tipo",
                    "combo_label": "Combinación",
                    "combo_bucket": "Buckets",
                    "horizon": "D+",
                    "n": "n",
                    "median_fwd": "Mediana",
                    "avg_aligned": "Media neta",
                    "historical_side": "Sesgo hist.",
                    "sharpe_combo": "Sharpe anual.",
                    "sharpe_combo_cons": "Rango bajo",
                    "mae_p80": "MAE p80",
                }
            )
            st.dataframe(
                full,
                width=1500,
                hide_index=True,
                column_config={
                    "Mediana": st.column_config.NumberColumn(format="%+.3f"),
                    "Media neta": st.column_config.NumberColumn(format="%+.3f"),
                    "Sharpe anual.": st.column_config.NumberColumn(format="%+.2f"),
                    "Rango bajo": st.column_config.NumberColumn(format="%+.2f"),
                    "MAE p80": st.column_config.NumberColumn(format="%.3f"),
                },
            )

    def _combo_cards(self, selected: pd.DataFrame) -> None:
        usable = selected[(selected["n"] >= MIN_EFFECTIVE_N) & selected["sharpe_combo"].notna()].copy()
        if usable.empty:
            st.info("No hay combinaciones de buckets con n efectivo suficiente para esta fecha/horizonte.")
            return
        usable["_abs_sharpe"] = usable["sharpe_combo"].abs()
        highlights = usable.sort_values(["_abs_sharpe", "n"], ascending=[False, False]).head(4)
        cols = st.columns(len(highlights))
        for col, (_, row) in zip(cols, highlights.iterrows()):
            sharpe = float(row.get("sharpe_combo", float("nan")))
            tone = BULL if sharpe > 0 else BEAR if sharpe < 0 else SUBTEXT
            col.markdown(
                f"""
                <div style="border:1px solid {BORDER};background:{SURFACE};border-left:4px solid {tone};
                            border-radius:8px;padding:12px 13px;min-height:154px;">
                    <div style="color:{SUBTEXT};font-size:12px;font-weight:700;text-transform:uppercase;">
                        {row.get('combo_type', 'Combinación')}
                    </div>
                    <div style="color:{TEXT};font-size:15px;font-weight:800;margin-top:4px;line-height:1.25;">
                        {row['combo_label']}
                    </div>
                    <div style="color:{SUBTEXT};font-size:12px;margin-top:4px;">
                        {row['combo_bucket']} · {row['historical_side']}
                    </div>
                    <div style="color:{tone};font-size:28px;font-weight:800;margin-top:6px;">
                        {_num(sharpe, '+.2f')}
                    </div>
                    <div style="color:{SUBTEXT};font-size:12px;margin-top:4px;">
                        media {_num(row.get('avg_aligned'), '+.3f')} · rango bajo {_num(row.get('sharpe_combo_cons'), '+.2f')} · n {int(row['n'])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def _combo_chart(self, selected: pd.DataFrame) -> None:
        usable = selected[(selected["n"] >= MIN_EFFECTIVE_N) & selected["sharpe_combo"].notna()].copy()
        if usable.empty:
            return
        usable["_abs_sharpe"] = usable["sharpe_combo"].abs()
        usable = usable.sort_values("_abs_sharpe", ascending=False).head(14)
        usable = usable.sort_values("sharpe_combo", ascending=True)
        colors = [BULL if v > 0 else BEAR if v < 0 else SUBTEXT for v in usable["sharpe_combo"]]
        fig = go.Figure(
            go.Bar(
                x=usable["sharpe_combo"],
                y=usable["combo_label"],
                orientation="h",
                marker_color=colors,
                customdata=usable[["combo_bucket", "historical_side", "n", "median_fwd", "avg_aligned", "sharpe_combo_cons"]].to_numpy(),
                hovertemplate=(
                    "<b>%{y}</b><br>Buckets: %{customdata[0]}<br>"
                    "Sesgo hist.: %{customdata[1]}<br>n: %{customdata[2]}<br>"
                    "Mediana: %{customdata[3]:+.3f}<br>Media neta: %{customdata[4]:+.3f}<br>"
                    "Sharpe: %{x:+.2f}<br>Rango bajo: %{customdata[5]:+.2f}<extra></extra>"
                ),
            )
        )
        fig.update_xaxes(title_text="Sharpe no-overlap", zeroline=True, zerolinecolor=BORDER)
        fig.update_yaxes(title_text="")
        st.plotly_chart(_base_layout(fig, 420), use_container_width=True)

    def _signal_windows(self, selection: Selection) -> None:
        candidates = self._signal_window_candidates(selection)
        if not candidates:
            return
        with st.expander("Análogos históricos de entrada por señal", expanded=True):
            st.caption(
                "Vista principal para auditar la señal: muestra todas las veces históricas en que el "
                "bucket actual del indicador o combinación volvió a aparecer, con entrada, salida D+ "
                "y P&L de la dirección que marca la propia señal."
            )
            available_types = [
                kind for kind in ("Indicador", "Combinaciones") if any(c["kind"] == kind for c in candidates)
            ]
            type_choice = st.segmented_control(
                "Tipo de señal",
                available_types,
                default=available_types[0],
                key=f"signal_type_{selection.family}_{selection.contract}_{selection.horizon}",
            )
            filtered = [item for item in candidates if item["kind"] == type_choice]
            labels = [item["label"] for item in filtered]
            chosen = st.selectbox(
                "Indicador / combinación",
                labels,
                key=f"signal_windows_{selection.family}_{selection.contract}_{selection.horizon}_{type_choice}",
            )
            item = filtered[labels.index(chosen)]
            windows = bucket_signal_windows_cached(
                selection.family,
                selection.contract,
                selection.as_of_iso,
                int(selection.horizon),
                tuple(item["indicators"]),
                "historical",
            )
            if windows.empty:
                st.info("No hay ventanas efectivas para esta señal.")
                return
            wins = int((windows["net_points"] > 0).sum())
            losses = int((windows["net_points"] < 0).sum())
            direction = str(windows["side"].iloc[0])
            if direction == "Plano":
                st.info("Esta señal no tiene dirección histórica clara para entrar.")
                return
            hit_rate = wins / len(windows) if len(windows) else float("nan")
            cols = st.columns(7)
            cols[0].metric("Entradas", f"{len(windows)}", delta=f"raw {int(windows['n_raw'].iloc[0])}")
            cols[1].metric("Dirección señal", direction)
            cols[2].metric("Tasa acierto", _num(hit_rate * 100, ".0f") + "%")
            cols[3].metric("Positivas", f"{wins}")
            cols[4].metric("Negativas", f"{losses}")
            cols[5].metric("Media neta", _num(windows["net_points"].mean(), "+.3f"))
            cols[6].metric("Mediana neta", _num(windows["net_points"].median(), "+.3f"))

            ordered_windows = windows.sort_values("entry_date", ascending=False).reset_index(drop=True)
            analogue_labels = [
                self._signal_analogue_label(row, i)
                for i, (_, row) in enumerate(ordered_windows.iterrows(), start=1)
            ]
            selected_label = st.selectbox(
                "Análogo histórico",
                analogue_labels,
                key=f"signal_analogue_{selection.family}_{selection.contract}_{selection.horizon}_{type_choice}_{chosen}",
            )
            selected_window = ordered_windows.iloc[analogue_labels.index(selected_label)]
            self._render_signal_analogue_series(selection, selected_window)

            st.markdown(f"##### Todas las series efectivas ({len(ordered_windows)})")
            self._render_all_signal_analogue_series(selection, ordered_windows)

            plot = windows.sort_values("entry_date").copy()
            plot["cum_net"] = plot["net_points"].cumsum()
            colors = [BULL if v > 0 else BEAR if v < 0 else SUBTEXT for v in plot["net_points"]]
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=plot["entry_date"],
                    y=plot["net_points"],
                    name="Neto por ventana",
                    marker_color=colors,
                    customdata=plot[["contract", "exit_date", "fwd_points", "side"]].to_numpy(),
                    hovertemplate=(
                        "%{customdata[0]}<br>Entrada %{x|%d %b %Y}<br>"
                        "Salida %{customdata[1]|%d %b %Y}<br>Lado %{customdata[3]}<br>"
                        "Mov. D+ %{customdata[2]:+.3f}<br>Neto %{y:+.3f}<extra></extra>"
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=plot["entry_date"],
                    y=plot["cum_net"],
                    name="Neto acumulado",
                    mode="lines",
                    line=dict(color=ACCENT, width=2),
                    yaxis="y2",
                )
            )
            fig.update_layout(
                yaxis=dict(title="Neto por ventana"),
                yaxis2=dict(title="Neto acumulado", overlaying="y", side="right", showgrid=False),
            )
            st.plotly_chart(_base_layout(fig, 380), use_container_width=True)

            table = windows.sort_values("entry_date", ascending=False).rename(
                columns={
                    "contract": "Contrato",
                    "entry_date": "Entrada",
                    "exit_date": "Salida",
                    "side": "Lado",
                    "historical_side": "Sesgo hist.",
                    "signal_close": "Close señal",
                    "entry_price": "Entrada ejec.",
                    "exit_price": "Salida close",
                    "volume": "Volume",
                    "fwd_points": "Mov. D+",
                    "net_points": "Neto",
                    "result": "Resultado",
                }
            )
            columns = [
                "Contrato",
                "Entrada",
                "Salida",
                "Lado",
                "Sesgo hist.",
                "Close señal",
                "Entrada ejec.",
                "Salida close",
                "Mov. D+",
                "Neto",
                "Resultado",
            ]
            if "Volume" in table.columns:
                columns.insert(6, "Volume")
            st.dataframe(
                table[columns],
                width=1500,
                hide_index=True,
                column_config={
                    "Entrada": st.column_config.DateColumn(format="DD MMM YYYY"),
                    "Salida": st.column_config.DateColumn(format="DD MMM YYYY"),
                    "Close señal": st.column_config.NumberColumn(format="%.3f"),
                    "Entrada ejec.": st.column_config.NumberColumn(format="%.3f"),
                    "Salida close": st.column_config.NumberColumn(format="%.3f"),
                    "Mov. D+": st.column_config.NumberColumn(format="%+.3f"),
                    "Neto": st.column_config.NumberColumn(format="%+.3f"),
                },
            )

    def _signal_analogue_label(self, row: pd.Series, number: int) -> str:
        entry = pd.Timestamp(row["entry_date"]).strftime("%d %b %Y")
        exit_date = pd.Timestamp(row["exit_date"]).strftime("%d %b %Y")
        return (
            f"{number:02d}. {row['contract']} · {entry} -> {exit_date} · "
            f"{row['side']} · precio {_num(row['fwd_points'], '+.3f')} · P&L {_num(row['net_points'], '+.3f')}"
        )

    def _render_signal_analogue_series(self, selection: Selection, window: pd.Series) -> None:
        frame = self.frames.get(selection.family)
        if frame is None or "contract" not in frame:
            return
        contract = str(window["contract"])
        series = frame[frame["contract"].astype(str) == contract].sort_values("date").copy()
        if series.empty:
            return

        series["date"] = pd.to_datetime(series["date"])
        signal_date = pd.Timestamp(window["entry_date"])
        exit_date = pd.Timestamp(window["exit_date"])
        entry_candidates = series.loc[series["date"] > signal_date, "date"]
        exec_date = pd.Timestamp(entry_candidates.iloc[0]) if not entry_candidates.empty else signal_date

        left = signal_date - pd.Timedelta(days=max(int(selection.horizon) * 3, 45))
        right = exit_date + pd.Timedelta(days=max(int(selection.horizon), 20))
        plot = series[(series["date"] >= left) & (series["date"] <= right)].copy()
        if plot.empty:
            plot = series.copy()
        held = plot[(plot["date"] >= signal_date) & (plot["date"] <= exit_date)]

        side = str(window["side"])
        net = float(window.get("net_points", np.nan))
        result_color = BULL if net > 0 else BEAR if net < 0 else SUBTEXT
        result_label = "Ganador" if net > 0 else "Perdedor" if net < 0 else "Plano"
        label, y_title, _ = _CHART_STYLE.get(selection.family, (selection.family.title(), "Precio", False))

        cols = st.columns(6)
        cols[0].metric("Contrato análogo", contract)
        cols[1].metric("Dirección", side)
        cols[2].metric("Señal", f"{signal_date:%d %b %Y}")
        cols[3].metric("Entrada ejec.", _num(window.get("entry_price"), ".3f"))
        cols[4].metric("Salida D+", _num(window.get("exit_price"), ".3f"))
        cols[5].metric(result_label, _num(window.get("net_points"), "+.3f"))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=plot["date"],
                y=plot["close"],
                mode="lines",
                name=f"{label} histórico",
                line=dict(color=SUBTEXT, width=1),
                hovertemplate="%{x|%d %b %Y}<br>Close %{y:.3f}<extra></extra>",
            )
        )
        if not held.empty:
            fig.add_trace(
                go.Scatter(
                    x=held["date"],
                    y=held["close"],
                    mode="lines",
                    name="Periodo mantenido",
                    line=dict(color=result_color, width=3),
                    hovertemplate="%{x|%d %b %Y}<br>Close %{y:.3f}<extra></extra>",
                )
            )
        marker_points = pd.DataFrame(
            [
                {
                    "date": signal_date,
                    "price": float(window.get("signal_close", window.get("close", np.nan))),
                    "name": "Señal",
                },
                {"date": exec_date, "price": float(window.get("entry_price", np.nan)), "name": "Entrada"},
                {"date": exit_date, "price": float(window.get("exit_price", np.nan)), "name": "Salida"},
            ]
        ).dropna(subset=["price"])
        fig.add_trace(
            go.Scatter(
                x=marker_points["date"],
                y=marker_points["price"],
                mode="markers+text",
                name="Puntos",
                text=marker_points["name"],
                textposition=["top center", "bottom center", "top center"][: len(marker_points)],
                marker=dict(color=[ACCENT, TEXT, result_color][: len(marker_points)], size=11),
                hovertemplate="%{text}<br>%{x|%d %b %Y}<br>%{y:.3f}<extra></extra>",
            )
        )
        fig.add_vrect(
            x0=signal_date,
            x1=exit_date,
            fillcolor=result_color,
            opacity=0.10,
            line_width=0,
        )
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text=y_title)
        fig.update_layout(
            title=(
                f"{contract}: {side} desde {signal_date:%d %b %Y} hasta {exit_date:%d %b %Y} · "
                f"movimiento precio {_num(window.get('fwd_points'), '+.3f')} · P&L señal {_num(window.get('net_points'), '+.3f')}"
            )
        )
        st.plotly_chart(_base_layout(fig, 430), use_container_width=True)

    def _render_all_signal_analogue_series(self, selection: Selection, windows: pd.DataFrame) -> None:
        frame = self.frames.get(selection.family)
        if frame is None or "contract" not in frame or windows.empty:
            return
        label, y_title, _ = _CHART_STYLE.get(selection.family, (selection.family.title(), "Precio", False))
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        by_contract = {
            str(contract): group.sort_values("date").copy()
            for contract, group in frame.groupby(frame["contract"].astype(str), sort=False)
        }

        columns = st.columns(2)
        for number, (_, window) in enumerate(windows.iterrows(), start=1):
            contract = str(window["contract"])
            series = by_contract.get(contract)
            if series is None or series.empty:
                continue

            signal_date = pd.Timestamp(window["entry_date"])
            exit_date = pd.Timestamp(window["exit_date"])
            entry_candidates = series.loc[series["date"] > signal_date, "date"]
            exec_date = pd.Timestamp(entry_candidates.iloc[0]) if not entry_candidates.empty else signal_date
            left = signal_date - pd.Timedelta(days=max(int(selection.horizon) * 2, 30))
            right = exit_date + pd.Timedelta(days=max(int(selection.horizon), 14))
            plot = series[(series["date"] >= left) & (series["date"] <= right)].copy()
            if plot.empty:
                plot = series.copy()
            held = plot[(plot["date"] >= signal_date) & (plot["date"] <= exit_date)]

            side = str(window["side"])
            net = float(window.get("net_points", np.nan))
            result_color = BULL if net > 0 else BEAR if net < 0 else SUBTEXT
            marker_points = pd.DataFrame(
                [
                    {
                        "date": signal_date,
                        "price": float(window.get("signal_close", window.get("close", np.nan))),
                        "name": "Señal",
                    },
                    {"date": exec_date, "price": float(window.get("entry_price", np.nan)), "name": "Entrada"},
                    {"date": exit_date, "price": float(window.get("exit_price", np.nan)), "name": "Salida"},
                ]
            ).dropna(subset=["price"])

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=plot["date"],
                    y=plot["close"],
                    mode="lines",
                    name=label,
                    line=dict(color=SUBTEXT, width=1),
                    hovertemplate="%{x|%d %b %Y}<br>Close %{y:.3f}<extra></extra>",
                )
            )
            if not held.empty:
                fig.add_trace(
                    go.Scatter(
                        x=held["date"],
                        y=held["close"],
                        mode="lines",
                        name="Mantenido",
                        line=dict(color=result_color, width=3),
                        hovertemplate="%{x|%d %b %Y}<br>Close %{y:.3f}<extra></extra>",
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=marker_points["date"],
                    y=marker_points["price"],
                    mode="markers+text",
                    text=marker_points["name"],
                    textposition=["top center", "bottom center", "top center"][: len(marker_points)],
                    marker=dict(color=[ACCENT, TEXT, result_color][: len(marker_points)], size=8),
                    hovertemplate="%{text}<br>%{x|%d %b %Y}<br>%{y:.3f}<extra></extra>",
                    showlegend=False,
                )
            )
            fig.add_vrect(
                x0=signal_date,
                x1=exit_date,
                fillcolor=result_color,
                opacity=0.10,
                line_width=0,
            )
            fig.update_layout(
                title=(
                    f"{number:02d}. {contract} · {side} · "
                    f"{signal_date:%d %b %Y} -> {exit_date:%d %b %Y} · "
                    f"P&L {_num(window.get('net_points'), '+.3f')}"
                ),
                showlegend=False,
                margin=dict(l=20, r=14, t=46, b=20),
            )
            fig.update_xaxes(title_text="")
            fig.update_yaxes(title_text=y_title)
            columns[(number - 1) % 2].plotly_chart(_base_layout(fig, 240), use_container_width=True)

    def _signal_window_candidates(self, selection: Selection) -> list[dict]:
        candidates: list[dict] = []
        horizons = (int(selection.horizon),) if low_memory_mode() else HORIZONS
        buckets = indicator_bucket_outcomes_cached(
            selection.family,
            selection.contract,
            selection.as_of_iso,
            horizons=horizons,
        )
        if not buckets.empty:
            current = buckets[
                (buckets["horizon"] == int(selection.horizon))
                & (buckets["n"] >= MIN_EFFECTIVE_N)
                & (buckets["sharpe_bucket"].notna())
                & (buckets["sharpe_bucket"] > 0)
                & (buckets["avg_aligned"] > 0)
                & (buckets["historical_side"].isin(["subidas", "bajadas"]))
            ].copy()
            current["_rank"] = current["sharpe_bucket"].abs()
            for _, row in current.sort_values(["_rank", "n"], ascending=[False, False]).iterrows():
                candidates.append(
                    {
                        "kind": "Indicador",
                        "label": f"{row['indicator_label']} · bucket {row['bucket']} · {row['historical_side']} · Sharpe {_num(row['sharpe_bucket'], '+.2f')} · hit {_num(row['hit_rate'] * 100, '.0f')}% · n {int(row['n'])}",
                        "indicators": (str(row["indicator"]),),
                    }
                )
        combos = self._combined_bucket_combos(selection)
        if not combos.empty:
            current = combos[
                (combos["horizon"] == int(selection.horizon))
                & (combos["n"] >= MIN_EFFECTIVE_N)
                & (combos["sharpe_combo"].notna())
                & (combos["sharpe_combo"] > 0)
                & (combos["avg_aligned"] > 0)
                & (combos["historical_side"].isin(["subidas", "bajadas"]))
            ].copy()
            current["_rank"] = current["sharpe_combo"].abs()
            for _, row in current.sort_values(["_rank", "n"], ascending=[False, False]).iterrows():
                combo_type = str(row.get("combo_type", "Combo"))
                candidates.append(
                    {
                        "kind": "Combinaciones",
                        "label": f"{combo_type} · {row['combo_label']} · buckets {row['combo_bucket']} · {row['historical_side']} · Sharpe {_num(row['sharpe_combo'], '+.2f')} · hit {_num(row['hit_rate'] * 100, '.0f')}% · n {int(row['n'])}",
                        "indicators": tuple(str(row["combo_key"]).split("|")),
                    }
                )
        return candidates

    def _horizon_table(self, horizons: pd.DataFrame) -> None:
        if horizons.empty:
            return
        view = horizons.copy()
        weak = view["n"] < MIN_EFFECTIVE_N
        view.loc[weak, ["median_aligned", "mae_p80", "sharpe_aligned"]] = float("nan")
        table = view.assign(**{"D+": view["horizon"].astype(int)})[
            ["D+", "n", "median_aligned", "mae_p80", "sharpe_aligned"]
        ].rename(
            columns={
                "median_aligned": "Mediana neta",
                "mae_p80": "MAE p80",
                "sharpe_aligned": "Sharpe anual.",
            }
        )
        with st.expander("Horizontes", expanded=False):
            valid = view[view["n"] >= MIN_EFFECTIVE_N]
            if not valid.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=valid["horizon"],
                        y=valid["median_aligned"],
                        name="Mediana neta",
                        mode="lines+markers",
                        line=dict(color=ACCENT, width=2),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=valid["horizon"],
                        y=valid["mfe_p80"],
                        name="MFE p80",
                        mode="lines",
                        line=dict(color=BULL, width=1, dash="dot"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=valid["horizon"],
                        y=-valid["mae_p80"],
                        name="MAE p80",
                        mode="lines",
                        line=dict(color=BEAR, width=1, dash="dot"),
                    )
                )
                fig.update_xaxes(title_text="Horizonte D+")
                fig.update_yaxes(title_text="Puntos")
                st.plotly_chart(_base_layout(fig, 280), use_container_width=True)
            st.dataframe(
                table,
                width=900,
                hide_index=True,
                column_config={
                    "Mediana neta": st.column_config.NumberColumn(format="%+.3f"),
                    "MAE p80": st.column_config.NumberColumn(format="%.3f"),
                    "Sharpe anual.": st.column_config.NumberColumn(format="%+.2f"),
                },
            )

    def _price_history(self, selection: Selection) -> None:
        frame = self.frames.get(selection.family)
        if frame is None or selection.contract is None or "contract" not in frame:
            return
        series = frame[frame["contract"].astype(str) == str(selection.contract)].sort_values("date")
        if series.empty:
            return
        with st.expander("Serie completa", expanded=False):
            close = series["close"]
            cols = st.columns(6)
            cells = [
                ("Último", f"{close.iloc[-1]:.2f}"),
                ("Mín", f"{close.min():.2f}"),
                ("Máx", f"{close.max():.2f}"),
                ("Media", f"{close.mean():.2f}"),
                ("Obs.", f"{len(series):,}"),
                ("Periodo", f"{series['date'].min():%b %Y} - {series['date'].max():%b %Y}"),
            ]
            for col, (label, value) in zip(cols, cells):
                col.metric(label, value)
            label, y_title, fill = _CHART_STYLE.get(selection.family, (selection.family.title(), "Precio", False))
            fig = price_volume_figure(
                series,
                title=f"{label} - {selection.contract}",
                y_title=y_title,
                fill_to_zero=fill,
                color=CHART_ACCENT,
            )
            st.plotly_chart(fig, use_container_width=True)
            raw_cols = [c for c in ("date", "contract", "close", "volume") if c in series.columns]
            st.dataframe(series[raw_cols].sort_values("date", ascending=False), width=1200, hide_index=True)

    def _audit(self, selection: Selection, score: dict, cohort: dict, rank_row: dict) -> None:
        with st.expander("Auditoría completa", expanded=False):
            rows = audit_rows(selection.family, selection.horizon, score, cohort, rank_row)
            st.dataframe(
                pd.DataFrame({"Métrica": list(rows), "Lectura": list(rows.values())}),
                width=1200,
                hide_index=True,
            )
