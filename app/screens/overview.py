"""Overview screen: selected-instrument snapshot plus active-contract ranking."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crudewatch.infra import FAMILY_LABELS, FAMILY_POINT_VALUE_USD

from core.audit import audit_rows
from core.context import selected_context_cached
from core.evidence import MIN_EFFECTIVE_N, evidence_read
from core.scoring import (
    active_contract_scores_cached,
    enriched_frame,
)
from core.selection import Selection
from screens.opportunity import _REGIME_ES, _base_layout, _fmt
from theme.palette import ACCENT, BEAR, BORDER, BULL, SUBTEXT, SURFACE, TEXT, title_block


def _family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, family.title())


def _tone(signed_composite: float) -> str:
    if signed_composite > 0:
        return BULL
    if signed_composite < 0:
        return BEAR
    return SUBTEXT


def _rank_band(rank: float) -> str:
    if rank != rank:
        return "-"
    if rank >= 90:
        return "Alta"
    if rank >= 70:
        return "Media"
    return "Baja"


def _evidence_tone(read: dict) -> str:
    if read["tone"] == "positive":
        return BULL
    if read["tone"] == "negative":
        return BEAR
    return SUBTEXT


def _fmt_num(value: float, spec: str = ".2f") -> str:
    if value is None or value != value:
        return "-"
    return format(float(value), spec)


def _compact_date(value) -> str:
    if value is None or value != value:
        return "-"
    return f"{pd.Timestamp(value):%d %b %Y}"


class OverviewScreen:
    """Operational first screen for the selected instrument and its live peers."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def display(self, selection: Selection) -> None:
        title_block(
            "Radar",
            "Radar de extremos: ranking de estructuras vivas y desglose del contrato seleccionado.",
        )

        if selection.contract is None:
            st.info("Selecciona una familia y una fecha con contratos activos en el menú superior.")
            return

        context = self._selected_context(selection)
        if context is None:
            return
        score = context["score"]

        self._headline(selection, score, context)
        self._freshness(selection, score)
        self._selected_body(selection, score, context)
        self._ranking(selection)

    def _selected_context(self, selection: Selection) -> dict | None:
        try:
            context = selected_context_cached(
                selection.family,
                selection.contract,
                selection.horizon,
                selection.as_of_iso,
                include_horizons=False,
            )
        except Exception as exc:  # noqa: BLE001 - Streamlit should show scoring failures clearly
            st.error(f"No se pudo calcular el overview: {exc}")
            return None

        score = context["score"]
        scored_date = pd.Timestamp(score["date"])
        if scored_date.normalize() != selection.as_of.normalize():
            st.caption(f"Última barra disponible <= fecha elegida: {scored_date:%d %b %Y}.")
        return context

    def _freshness(self, selection: Selection, score: dict) -> None:
        scored_date = pd.Timestamp(score["date"])
        contracts = active_contract_scores_cached(
            selection.family,
            selection.horizon,
            selection.as_of_iso,
        )
        exact = scored_date.normalize() == selection.as_of.normalize()
        status = "barra exacta" if exact else f"última barra <= fecha elegida: {scored_date:%d %b %Y}"
        c1, c2, c3 = st.columns(3)
        c1.metric("Fecha elegida", f"{selection.as_of:%d %b %Y}")
        c2.metric("Dato usado", status)
        c3.metric("Contratos activos", f"{len(contracts)}")

    def _headline(self, selection: Selection, score: dict, context: dict) -> None:
        blocks = score["blocks"]
        cohort = context["cohort"]
        read = evidence_read(score, cohort)
        signed_composite = float(score["opportunity"])
        stretch_rank = float(context.get("stretch_rank", float("nan")))
        tone = _tone(signed_composite)
        ev_tone = _evidence_tone(read)
        regime = _REGIME_ES.get(blocks["regime"], blocks["regime"])

        st.markdown(
            f"""
            <div style="border:1px solid {BORDER};background:{SURFACE};border-left:5px solid {tone};
                        border-radius:8px;padding:16px 18px;margin:4px 0 14px 0;">
                <div style="display:flex;justify-content:space-between;gap:18px;align-items:flex-start;flex-wrap:wrap;">
                    <div>
                        <div style="color:{SUBTEXT};font-size:13px;font-weight:700;text-transform:uppercase;">
                            {_family_label(selection.family)} · {selection.contract} · D+{selection.horizon}
                        </div>
                        <div style="color:{tone};font-size:34px;font-weight:800;line-height:1.15;margin-top:3px;">
                            Extremo vs familia {_fmt_num(stretch_rank, ".0f")}
                        </div>
                        <div style="color:{SUBTEXT};font-size:13px;margin-top:4px;">
                            Signo interno {signed_composite:+.0f}; color y ordenación, no veredicto de trade.
                        </div>
                    </div>
                    <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;">
                        <span style="border:1px solid {BORDER};border-radius:8px;padding:7px 10px;color:{TEXT};">
                            Régimen <b>{regime}</b>
                        </span>
                        <span style="border:1px solid {ev_tone};border-radius:8px;padding:7px 10px;color:{TEXT};">
                            Evidencia <b style="color:{ev_tone};">{read["label"]}</b>
                        </span>
                        <span style="border:1px solid {BORDER};border-radius:8px;padding:7px 10px;color:{TEXT};">
                            Extremo <b>{_rank_band(stretch_rank)}</b>
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _selected_body(self, selection: Selection, score: dict, context: dict) -> None:
        blocks = score["blocks"]
        cohort = context["cohort"]
        validation = context["validation"]
        stretch_rank = float(context.get("stretch_rank", float("nan")))

        self._decision_metrics(blocks, score, stretch_rank, validation, cohort)

        left, mid, right = st.columns([1.0, 1.0, 1.15], vertical_alignment="top")
        with left:
            self._setup_panel(selection, score, cohort, context.get("rank_row", {}))
        with mid:
            self._evidence_panel(selection.family, score, cohort)
        with right:
            self._price_mini(selection.family, selection.contract, pd.Timestamp(score["date"]))

        self._audit_panel(selection, score, cohort, context.get("rank_row", {}))

        st.markdown("#### Lectura por bloques")
        self._block_snapshot(blocks)

    def _audit_panel(self, selection: Selection, score: dict, cohort: dict, rank_row: dict) -> None:
        with st.expander("Auditoría de la lectura", expanded=False):
            rows = audit_rows(selection.family, selection.horizon, score, cohort, rank_row)
            st.dataframe(
                pd.DataFrame({"Métrica": list(rows), "Lectura": list(rows.values())}),
                width=1200,
                hide_index=True,
            )

    def _decision_metrics(
        self,
        blocks: dict,
        score: dict,
        stretch_rank: float,
        validation: dict,
        cohort: dict,
    ) -> None:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Extremo vs familia", _fmt_num(stretch_rank, ".0f"))
        c2.metric("Signo interno", f"{float(score['opportunity']):+.0f}")
        cohort_sharpe = (
            float(cohort.get("sharpe_aligned", float("nan")))
            if int(cohort.get("n", 0)) >= MIN_EFFECTIVE_N
            else float("nan")
        )
        c3.metric("Sharpe cohorte anual.", _fmt_num(cohort_sharpe, "+.2f"))
        c4.metric("Nivel", f"{blocks['level']:+.0f}")
        c5.metric("P(reversión)", f"{blocks['p_reversion'] * 100:.0f}%")
        c6.metric("P(continuación)", f"{blocks['p_continuation'] * 100:.0f}%")

    def _raw_samples(self, cohort: dict) -> None:
        samples = cohort.get("aligned_samples") or cohort.get("fwd_samples") or []
        if not samples:
            return
        with st.expander("Muestras raw", expanded=False):
            st.dataframe(
                pd.DataFrame({"resultado": pd.Series(samples, dtype=float)}).head(80),
                width=420,
                hide_index=True,
                column_config={"resultado": st.column_config.NumberColumn(format="%+.3f")},
            )

    def _evidence_panel(self, family: str, score: dict, cohort: dict) -> None:
        st.markdown("##### Evidencia histórica")
        n = int(cohort.get("n", 0))
        if n == 0:
            st.caption("Sin cohorte análogo suficiente para este setup.")
            return
        read = evidence_read(score, cohort)
        st.markdown(f"**{read['label']}** · {read['detail']}")
        if n < MIN_EFFECTIVE_N:
            self._raw_samples(cohort)
            return
        side = float(cohort.get("side", 0.0))
        median_aligned = cohort.get("median_aligned", float("nan"))
        avg_fwd = cohort.get("avg_fwd", float("nan"))
        hit = cohort.get("aligned_win_rate", float("nan"))
        mae_p80 = cohort.get("mae_p80", float("nan"))
        point_value = FAMILY_POINT_VALUE_USD.get(family, 1000.0)
        value = median_aligned if side != 0 else avg_fwd
        c1, c2 = st.columns(2)
        c1.metric("Casos efectivos", f"{n}", delta=f"raw {int(cohort.get('n_raw', n))}")
        c2.metric("Acierto", "-" if hit != hit else f"{hit * 100:.0f}%")
        c7, c8 = st.columns(2)
        c7.metric("Sharpe cohorte anual.", _fmt(cohort.get("sharpe_aligned", float("nan")), "+.2f"))
        c8.metric("Coste", _fmt(cohort.get("cost", float("nan")), ".3f") + " pts")
        c3, c4 = st.columns(2)
        c3.metric("Mediana con signo", _fmt(value, "+.3f"))
        c4.metric("MAE p80", _fmt(mae_p80, ".3f"))
        c5, c6 = st.columns(2)
        c5.metric("Mediana USD", _fmt(value * point_value, "+.0f"))
        c6.metric("MAE p80 USD", _fmt(mae_p80 * point_value, ".0f"))
        st.caption(
            f"Mismo régimen, bucket de nivel, tenor {cohort.get('tenor_bucket', '-')}, "
            f"mes {cohort.get('month', '-')}, far leg {cohort.get('far_leg', '-')}; "
            f"muestra de-overlap a D+{cohort.get('horizon', '-') }."
        )

    def _setup_panel(self, selection: Selection, score: dict, cohort: dict, rank_row: dict) -> None:
        st.markdown("##### Setup seleccionado")
        c1, c2 = st.columns(2)
        c1.metric("Signo interno", f"{float(score['opportunity']):+.0f}")
        c2.metric("Close ref.", _fmt_num(score.get("close", float("nan")), ".2f"))
        if rank_row:
            st.caption(
                f"Slot {rank_row.get('slot', '-')} · vintage {_fmt_num(rank_row.get('vintage', float('nan')), '.0f')} · "
                f"fase {rank_row.get('life_phase', '-')} · vol {rank_row.get('vol_regime', '-')}"
            )
            st.markdown(f"**Lectura PM:** {rank_row.get('pm_read', '-')}")
            st.caption(f"Sesgo descriptivo: {rank_row.get('descriptive_bias', '-')}")
        read = evidence_read(score, cohort)
        st.caption(f"{read['label']} · {read['detail']}")
        if int(cohort.get("n", 0)) < MIN_EFFECTIVE_N:
            for warning in score.get("risks", [])[:5]:
                st.caption(f"- {warning}")
            return
        c3, c4 = st.columns(2)
        c3.metric("Mediana con signo", _fmt_num(cohort.get("median_aligned", float("nan")), "+.3f"))
        c4.metric("MAE p80", _fmt_num(cohort.get("mae_p80", float("nan")), ".3f"))
        point_value = FAMILY_POINT_VALUE_USD.get(selection.family, 1000.0)
        c5, c6 = st.columns(2)
        c5.metric("Mediana USD", _fmt_num(cohort.get("median_aligned", float("nan")) * point_value, "+.0f"))
        c6.metric("MAE p80 USD", _fmt_num(cohort.get("mae_p80", float("nan")) * point_value, ".0f"))
        st.caption(f"Horizonte D+{selection.horizon}; sin niveles automáticos inventados.")
        for warning in score.get("risks", [])[:5]:
            st.caption(f"- {warning}")

    def _block_snapshot(self, blocks: dict) -> None:
        labels = ["Régimen", "Dirección", "Fuerza", "Nivel", "P(rev)", "P(cont)"]
        values = [
            blocks["trendiness"],
            blocks["direction"],
            blocks["strength"],
            blocks["level"],
            blocks["p_reversion"] * 100,
            blocks["p_continuation"] * 100,
        ]
        colors = [
            ACCENT,
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
                text=[_fmt(v, ".0f") for v in values],
                textposition="outside",
            )
        )
        fig.update_xaxes(range=[-100, 100], title_text="Bloques")
        st.plotly_chart(_base_layout(fig, 285), use_container_width=True)

    def _price_mini(self, family: str, contract: str, as_of: pd.Timestamp) -> None:
        data = enriched_frame(family)
        sub = data[data["contract"] == contract].sort_values("date")
        sub = sub[pd.to_datetime(sub["date"]) <= as_of].tail(90)
        if sub.empty:
            st.caption("Sin serie reciente.")
            return
        fig = go.Figure(
            go.Scatter(
                x=sub["date"],
                y=sub["close"],
                mode="lines",
                line=dict(color=ACCENT, width=2),
                name=contract,
            )
        )
        fig.add_vline(x=as_of, line_color=SUBTEXT, line_dash="dot", line_width=1)
        fig.add_annotation(
            x=as_of,
            y=1,
            yref="paper",
            yanchor="bottom",
            showarrow=False,
            text="fecha usada",
            font=dict(color=SUBTEXT, size=11),
        )
        fig.update_yaxes(title_text="Cierre")
        st.plotly_chart(_base_layout(fig, 285), use_container_width=True)

    def _ranking(self, selection: Selection) -> None:
        st.markdown("### Contratos activos")
        ranking = active_contract_scores_cached(
            selection.family,
            selection.horizon,
            selection.as_of_iso,
        )
        if ranking.empty:
            st.info("No hay contratos activos para rankear en esta fecha.")
            return

        quick, min_score, min_volume, validation_filter, regime_filter, side_filter = self._filters(ranking)
        view = self._apply_filters(
            ranking,
            quick,
            min_score,
            min_volume,
            validation_filter,
            regime_filter,
            side_filter,
        )
        st.caption(
            f"{len(view)} de {len(ranking)} contratos activos · ordenados por extremo vs familia y volumen."
        )
        if view.empty:
            st.info("Ningún contrato pasa los filtros actuales.")
            return

        display = view.head(40).copy().reset_index(drop=True)
        display["Fecha"] = display["date"].map(_compact_date)
        display["regime"] = display["regime"].map(lambda x: _REGIME_ES.get(x, x))
        display = display.rename(
            columns={
                "contract": "Contrato",
                "sign": "Interno | Signo",
                "stretch_rank": "Extremo | Vs familia",
                "signed_composite": "Interno | Valor",
                "regime": "Bloques | Régimen",
                "level": "Bloques | Nivel",
                "p_reversion": "Bloques | P(rev)",
                "p_continuation": "Bloques | P(cont)",
                "validation_state": "OOS familia | Estado",
                "validation_n": "OOS familia | Trades",
                "validation_hit": "OOS familia | Hit %",
                "cohort_n": "Cohorte | n",
                "cohort_sharpe": "Cohorte | Sharpe anual.",
                "pm_read": "PM | Lectura",
                "descriptive_bias": "PM | Sesgo",
                "slot": "Cohorte | Slot",
                "vintage": "Cohorte | Vintage",
                "life_phase": "Cohorte | Vida",
                "risk_count": "Riesgo | Flags",
                "risks": "Riesgo | Detalle",
                "close": "Dato | Close",
                "volume": "Dato | Volume",
                "liquidity": "Dato | Liquidez",
                "vol_regime": "Dato | Vol",
                "season_month": "Dato | Mes",
                "far_month": "Dato | Far",
                "point_value_usd": "Dato | $/pt",
                "cost_points": "Coste asumido | pts",
                "cost_usd": "Coste asumido | USD",
                "dte": "Dato | DTE",
            }
        )
        fast_cols = [
            "Contrato",
            "Fecha",
            "Extremo | Vs familia",
            "PM | Lectura",
            "PM | Sesgo",
            "Bloques | Régimen",
            "Bloques | Nivel",
            "Cohorte | n",
            "Cohorte | Sharpe anual.",
            "Dato | Volume",
            "Dato | Liquidez",
            "Dato | Vol",
            "Coste asumido | USD",
            "Riesgo | Flags",
            "Dato | DTE",
        ]
        full_cols = [
            *fast_cols,
            "Interno | Signo",
            "Interno | Valor",
            "Cohorte | Vida",
            "Cohorte | Slot",
            "OOS familia | Estado",
            "OOS familia | Trades",
            "OOS familia | Hit %",
            "Dato | $/pt",
            "Coste asumido | pts",
            "Dato | Close",
            "Bloques | P(rev)",
            "Bloques | P(cont)",
            "Cohorte | Vintage",
            "Dato | Mes",
            "Dato | Far",
            "Riesgo | Detalle",
        ]
        if "OOS familia | Hit %" in display:
            display["OOS familia | Hit %"] = display["OOS familia | Hit %"] * 100.0
        cols = fast_cols if quick else full_cols
        event = st.dataframe(
            display[cols],
            width=1900,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="overview_active_contracts",
            column_config={
                "Extremo | Vs familia": st.column_config.NumberColumn(format="%.0f"),
                "Interno | Valor": st.column_config.NumberColumn(format="%+.0f"),
                "Bloques | Nivel": st.column_config.NumberColumn(format="%+.0f"),
                "Bloques | P(rev)": st.column_config.NumberColumn(format="%.0%"),
                "Bloques | P(cont)": st.column_config.NumberColumn(format="%.0%"),
                "OOS familia | Hit %": st.column_config.NumberColumn(format="%.0f%%"),
                "Cohorte | n": st.column_config.NumberColumn(format="%.0f"),
                "Cohorte | Sharpe anual.": st.column_config.NumberColumn(format="%+.2f"),
                "Cohorte | Vintage": st.column_config.NumberColumn(format="%.0f"),
                "Dato | Mes": st.column_config.NumberColumn(format="%.0f"),
                "Dato | Far": st.column_config.NumberColumn(format="%.0f"),
                "Dato | Close": st.column_config.NumberColumn(format="%.2f"),
                "Dato | Volume": st.column_config.NumberColumn(format="%.0f"),
                "Dato | $/pt": st.column_config.NumberColumn(format="$%.0f"),
                "Coste asumido | pts": st.column_config.NumberColumn(format="%.3f"),
                "Coste asumido | USD": st.column_config.NumberColumn(format="$%.0f"),
                "Dato | DTE": st.column_config.NumberColumn(format="%.0f"),
            },
        )
        if quick:
            with st.expander("Detalle completo de ranking", expanded=False):
                st.dataframe(
                    display[full_cols],
                    width=2600,
                    hide_index=True,
                    column_config={
                        "Extremo | Vs familia": st.column_config.NumberColumn(format="%.0f"),
                        "Interno | Valor": st.column_config.NumberColumn(format="%+.0f"),
                        "Bloques | Nivel": st.column_config.NumberColumn(format="%+.0f"),
                        "Bloques | P(rev)": st.column_config.NumberColumn(format="%.0%"),
                        "Bloques | P(cont)": st.column_config.NumberColumn(format="%.0%"),
                        "OOS familia | Hit %": st.column_config.NumberColumn(format="%.0f%%"),
                        "Cohorte | n": st.column_config.NumberColumn(format="%.0f"),
                        "Cohorte | Sharpe anual.": st.column_config.NumberColumn(format="%+.2f"),
                        "Cohorte | Vintage": st.column_config.NumberColumn(format="%.0f"),
                        "Dato | Mes": st.column_config.NumberColumn(format="%.0f"),
                        "Dato | Far": st.column_config.NumberColumn(format="%.0f"),
                        "Dato | Close": st.column_config.NumberColumn(format="%.2f"),
                        "Dato | Volume": st.column_config.NumberColumn(format="%.0f"),
                        "Dato | $/pt": st.column_config.NumberColumn(format="$%.0f"),
                        "Coste asumido | pts": st.column_config.NumberColumn(format="%.3f"),
                        "Coste asumido | USD": st.column_config.NumberColumn(format="$%.0f"),
                        "Dato | DTE": st.column_config.NumberColumn(format="%.0f"),
                    },
                )
        selection = getattr(event, "selection", None)
        rows = selection.get("rows", []) if isinstance(selection, dict) else getattr(selection, "rows", [])
        if rows:
            selected = str(display.iloc[int(rows[0])]["Contrato"])
            if selected != st.session_state.get("sel_contract"):
                st.session_state["sel_contract"] = selected
                st.rerun()

    def _filters(self, ranking: pd.DataFrame) -> tuple[bool, int, int, str, str, str]:
        c0, c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1, 1], vertical_alignment="bottom")
        with c0:
            quick = st.toggle("Lectura rápida", value=True)
        with c1:
            default_score = 70 if quick else 0
            min_score = st.slider("Extremo mín.", 0, 100, default_score, 5)
        with c2:
            has_volume = "volume" in ranking and ranking["volume"].notna().any()
            max_volume = int(ranking["volume"].max()) if has_volume else 0
            min_volume = st.slider("Vol. mín.", 0, max(1000, max_volume), 0, 50, disabled=not has_volume)
        with c3:
            validation_filter = st.selectbox("Validación OOS", ["Todos", "OOS", "Sin validación"])
        with c4:
            regimes = ["Todos", *_REGIME_ES.values()]
            regime_filter = st.selectbox("Régimen", regimes)
        with c5:
            side_filter = st.selectbox("Signo", ["Todos", "+", "-", "0"])
        return quick, min_score, min_volume, validation_filter, regime_filter, side_filter

    def _apply_filters(
        self,
        ranking: pd.DataFrame,
        quick: bool,
        min_score: int,
        min_volume: int,
        validation_filter: str,
        regime_filter: str,
        side_filter: str,
    ) -> pd.DataFrame:
        view = ranking.copy()
        view = view[view["stretch_rank"].notna()]
        view = view[view["stretch_rank"] >= min_score]
        if min_volume > 0 and "volume" in view:
            view = view[view["volume"].fillna(0) >= min_volume]
        if validation_filter != "Todos" and "validation_state" in view:
            view = view[view["validation_state"] == validation_filter]
        if quick:
            view = view[view["sign"] != "0"]
        if regime_filter != "Todos":
            inverse = {v: k for k, v in _REGIME_ES.items()}
            view = view[view["regime"] == inverse.get(regime_filter, regime_filter)]
        if side_filter != "Todos":
            view = view[view["sign"] == side_filter]
        return view.sort_values(
            ["stretch_rank", "volume", "dte"],
            ascending=[False, False, True],
            na_position="last",
        )
