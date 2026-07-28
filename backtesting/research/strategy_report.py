"""Single-page demonstration report for the selected strategy.

Deliberately narrow: it states the research conclusion and then shows **only**
the out-of-sample results of applying the one strategy that survived scrutiny —
range-regime, level-confirmed mean reversion on the calendar-structure families.
The centrepiece is a real cumulative-PnL curve built from the trade ledger.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from crudewatch.infra.constants import FAMILY_LABELS
from backtesting.research.report import (
    ACCENT, BACKGROUND, BORDER, BRAND_MARK_SVG, CHART_GRID, SUBTEXT, SURFACE,
    SURFACE_2, TEXT, BEAR, _PLOTLYJS, _div, _flabel, _kpi, _num,
)

_LINE_COLORS = [
    "#38BDF8", "#F59E0B", "#A78BFA", "#F472B6", "#34D399", "#FB7185",
    "#FBBF24", "#60A5FA", "#C084FC",
]


def _equity_fig(ledgers: dict[str, pd.DataFrame], horizon: int, recommended: set[str]) -> str:
    """Cumulative PnL (points) over time: combined (recommended) + one line per family.

    The combined line pools only the recommended families (the proposed portfolio);
    non-recommended families are drawn dashed/dimmed for context.
    """
    fig = go.Figure()
    rec_frames = [
        led for fam, led in ledgers.items()
        if fam in recommended and led is not None and not led.empty
    ]
    if rec_frames:
        combined = pd.concat(rec_frames, ignore_index=True).sort_values("date")
        combined["cum_pnl"] = combined["pnl"].cumsum()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(combined["date"]), y=combined["cum_pnl"],
            mode="lines", name="Combinada (recomendadas)", line=dict(color=ACCENT, width=2.6),
        ))
    for i, (fam, led) in enumerate(ledgers.items()):
        if led is None or led.empty:
            continue
        is_rec = fam in recommended
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(led["date"]), y=led["cum_pnl"], mode="lines",
            name=FAMILY_LABELS.get(fam, fam) + ("" if is_rec else " (no rec.)"),
            line=dict(
                color=_LINE_COLORS[i % len(_LINE_COLORS)],
                width=1.4 if is_rec else 1.0,
                dash="solid" if is_rec else "dot",
            ),
            opacity=0.85 if is_rec else 0.45,
        ))
    fig.update_layout(
        title=dict(text=f"PnL acumulado fuera de muestra (D+{horizon}, coste descontado)",
                   x=0.5, xanchor="center", font=dict(color=ACCENT, size=17)),
        template="plotly_dark", paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND,
        font=dict(color=TEXT, family="Arial"), height=460,
        margin=dict(l=60, r=30, t=56, b=60), showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=SUBTEXT, size=11)),
        xaxis=dict(title="Fecha", gridcolor=CHART_GRID, showline=True, linecolor=ACCENT),
        yaxis=dict(title="PnL acumulado (puntos)", gridcolor=CHART_GRID, showline=True,
                   linecolor=ACCENT, zeroline=True, zerolinecolor=CHART_GRID),
    )
    return _div(fig)


def _family_order(stats: pd.DataFrame) -> list[str]:
    """Recommended families first (in FAMILY_LABELS order), then the rest."""
    def rank(fam: str) -> int:
        rec = stats.loc[stats["family"] == fam, "recommended"]
        return 0 if (not rec.empty and bool(rec.iloc[0])) else 1
    fams = [f for f in FAMILY_LABELS if (stats["family"] == f).any()]
    return sorted(fams, key=lambda f: (rank(f), list(FAMILY_LABELS).index(f)))


def _stats_table(stats: pd.DataFrame, headline: int) -> str:
    """Family x horizon results of applying the strategy (all families)."""
    has_rec = "recommended" in stats.columns
    rows = []
    for fam in _family_order(stats):
        sub = stats[stats["family"] == fam]
        if sub.empty:
            continue
        rec = bool(sub["recommended"].iloc[0]) if has_rec else True
        rec_cell = f'<span style="color:{ACCENT}">\u2713</span>' if rec else f'<span style="color:{SUBTEXT}">\u2014</span>'
        for _, r in sub.sort_values("horizon").iterrows():
            hl = ' class="hl"' if int(r["horizon"]) == headline else ""
            sh_col = ACCENT if (not pd.isna(r["sharpe"]) and r["sharpe"] > 0) else BEAR
            wr = "" if pd.isna(r["win_rate"]) else f"{r['win_rate'] * 100:.0f}%"
            name_style = "" if rec else f" style='color:{SUBTEXT}'"
            rows.append(
                f"<tr{hl}>"
                f"<td{name_style}>{FAMILY_LABELS.get(fam, fam)}</td>"
                f"<td>{rec_cell}</td>"
                f"<td>D+{int(r['horizon'])}</td>"
                f"<td>{int(r['n_trades'])}</td>"
                f"<td>{int(r['n_long'])}/{int(r['n_short'])}</td>"
                f"<td>{wr}</td>"
                f"<td>{_num(r['avg_pnl'], 3)}</td>"
                f"<td>{_num(r['total_pnl'], 1)}</td>"
                f"<td style='color:{sh_col}'><b>{_num(r['sharpe'])}</b></td>"
                f"<td>{_num(r['max_dd'], 1)}</td>"
                f"<td>{_num(r['trades_per_year'], 0)}</td>"
                "</tr>"
            )
    head = (
        "<tr><th>Familia</th><th title='Familia recomendada por el backtest (ventaja robusta y econ\u00f3mica).'>Rec.</th>"
        "<th title='Sesiones mantenidas por operaci\u00f3n.'>Horizonte</th>"
        "<th title='N\u00famero de operaciones fuera de muestra.'>Trades</th>"
        "<th title='Operaciones largas / cortas.'>L/S</th>"
        "<th title='% de operaciones con PnL positivo.'>Win</th>"
        "<th title='PnL medio por operaci\u00f3n, en puntos, coste descontado.'>PnL medio</th>"
        "<th title='PnL total acumulado en puntos.'>PnL total</th>"
        "<th title='Sharpe en tiempo-calendario: PnL diario (sumando trades simult\u00e1neos, d\u00edas ociosos=0) anualizado por \u221a252.'>Sharpe</th>"
        "<th title='M\u00e1xima ca\u00edda del PnL acumulado (puntos).'>Max DD</th>"
        "<th title='Operaciones por a\u00f1o.'>Trades/a\u00f1o</th></tr>"
    )
    body = "".join(rows)
    return (
        '<div class="tablewrap"><table><thead>' + head + '</thead><tbody>' + body + '</tbody></table></div>'
    )


def _excluded_table(excluded: pd.DataFrame | None, headline: int = 25) -> str:
    if excluded is None or excluded.empty:
        return ""
    rows = []
    for _, r in excluded.iterrows():
        rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                FAMILY_LABELS.get(r["family"], r["family"]),
                _num(r.get("ic_mean"), 3), _num(r.get("sharpe")), r.get("reason", ""),
            )
        )
    head = (f"<tr><th>Familia</th><th>IC (D+{headline})</th><th>Sharpe estrat.</th><th>Motivo de exclusi\u00f3n</th></tr>")
    return (
        '<h2 class="cat" id="excluidas">Familias excluidas</h2>'
        '<p class="cat-lead">Por qu\u00e9 esta primera estrategia no opera el resto de familias.</p>'
        '<div class="card"><h3>No fiables / no econ\u00f3micas (todav\u00eda)</h3>'
        '<div class="tablewrap"><table><thead>' + head + '</thead><tbody>' + "".join(rows) +
        '</tbody></table></div></div>'
    )


def _kpis(stats: pd.DataFrame, headline: int) -> str:
    sub = stats[stats["horizon"] == headline]
    has_rec = "recommended" in sub.columns
    rec = sub[sub["recommended"]] if has_rec else sub
    n_fam = int(sub["family"].nunique()) if not sub.empty else 0
    n_rec = int(rec["family"].nunique()) if not rec.empty else 0
    n_trades = int(rec["n_trades"].sum()) if not rec.empty else 0
    total = rec["total_pnl"].sum() if not rec.empty else 0.0
    pos = int((sub["sharpe"] > 0).sum()) if not sub.empty else 0
    best = sub.reindex(sub["sharpe"].sort_values(ascending=False).index).iloc[0] if not sub.empty else None
    kpis = [
        _kpi(f"{n_rec}/{n_fam}", "familias recomendadas"),
        _kpi(f"{n_trades:,}".replace(",", "."), f"operaciones recom. (D+{headline})"),
        _kpi(f"{total:.0f}", "PnL total recom. (puntos)"),
        _kpi(f"{pos}/{n_fam}", "familias con Sharpe > 0"),
    ]
    if best is not None:
        kpis.append(_kpi(_num(best["sharpe"]), f"mejor Sharpe ({FAMILY_LABELS.get(best['family'], best['family'])})"))
    return '<div class="kpis">' + "".join(kpis) + "</div>"


def build_strategy_report(
    ledgers: dict[str, pd.DataFrame],
    stats: pd.DataFrame,
    horizon: int = 25,
    *,
    excluded: pd.DataFrame | None = None,
) -> str:
    """Assemble the self-contained single-page strategy demonstration."""
    if "recommended" in stats.columns:
        recommended = set(stats.loc[stats["recommended"], "family"].unique())
    else:
        recommended = set(ledgers)
    equity = _equity_fig(ledgers, horizon, recommended)
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>CrudeWatch \u00b7 Estrategia de reversi\u00f3n</title>
<script>{_PLOTLYJS}</script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:{BACKGROUND}; color:{TEXT}; font-family:Arial, sans-serif; line-height:1.55; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:32px 22px 80px; }}
  .brand {{ display:flex; align-items:center; gap:12px; color:{ACCENT}; margin-bottom:6px; }}
  .brand svg {{ width:34px; height:34px; }}
  .brand .n {{ font-size:20px; font-weight:700; letter-spacing:.3px; }}
  h1 {{ font-size:30px; margin:14px 0 6px; }}
  .sub {{ color:{SUBTEXT}; font-size:15px; margin-bottom:26px; }}
  h2.cat {{ font-size:21px; margin:40px 0 4px; border-left:3px solid {ACCENT}; padding-left:12px; }}
  .cat-lead {{ color:{SUBTEXT}; font-size:14px; margin:0 0 16px 15px; }}
  .card {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:14px; padding:22px 24px; margin:16px 0; }}
  .card h3 {{ margin:0 0 12px; font-size:17px; color:{TEXT}; }}
  .rule {{ color:{SUBTEXT}; font-size:13.5px; margin:6px 0 14px; }}
  .kpis {{ display:flex; flex-wrap:wrap; gap:14px; margin:18px 0 6px; }}
  .kpi {{ flex:1 1 150px; background:{SURFACE_2}; border:1px solid {BORDER}; border-radius:12px; padding:16px 18px; }}
  .kpi .v {{ font-size:26px; font-weight:700; color:{ACCENT}; }}
  .kpi .l {{ font-size:12.5px; color:{SUBTEXT}; margin-top:4px; }}
  ul.rules {{ margin:6px 0 4px; padding-left:20px; }}
  ul.rules li {{ margin:6px 0; font-size:14px; }}
  ul.rules b {{ color:{ACCENT}; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  .tablewrap {{ overflow-x:auto; }}
  th, td {{ padding:8px 10px; text-align:right; border-bottom:1px solid {BORDER}; white-space:nowrap; }}
  th:first-child, td:first-child {{ text-align:left; }}
  thead th {{ color:{SUBTEXT}; font-weight:600; border-bottom:1px solid {ACCENT}; cursor:help; }}
  tr.hl td {{ background:{SURFACE_2}; }}
  .callout {{ background:{SURFACE_2}; border-left:3px solid {ACCENT}; border-radius:8px; padding:14px 18px; margin:14px 0; font-size:14px; }}
</style></head>
<body><div class="wrap">
  <div class="brand">{BRAND_MARK_SVG}<span class="n">CrudeWatch</span></div>
  <h1>Estrategia de reversi\u00f3n en rango, confirmada</h1>
  <div class="sub">Demostraci\u00f3n fuera de muestra (walk-forward por vintage) de la \u00fanica ventaja robusta del backtest. Todo en puntos, coste descontado.</div>

  <h2 class="cat" id="conclusion">Conclusi\u00f3n del backtest</h2>
  <p class="cat-lead">Lo que aguanta el an\u00e1lisis, en una frase.</p>
  <div class="card">
    <div class="callout">Con ejecuci\u00f3n realista (entrada en el <b>open de t+1</b>) y control de multiple-testing (FDR),
    la ventaja fiable es de <b>reversi\u00f3n a la media dentro del r\u00e9gimen de rango</b>, m\u00e1s clara a <b>horizonte largo
    (D+20\u2013D+30)</b> y en las familias de <b>estructura de calendario</b> (calendars, quarterly, semestral). El predictor
    robusto que sobrevive al FDR es el <b>nivel caro/barato frente a contratos an\u00e1logos</b>: se dispara al llegar a un
    extremo de precio y se confirma con ese nivel.</div>
    <p class="rule">El resto de familias (outrights, cracks, brent-wti, yearly, flies) se excluyen: Sharpe negativo, o
    ventaja solo a un horizonte suelto y muestra fina. Ver el detalle al final.</p>
  </div>

  <h2 class="cat" id="estrategia">La estrategia</h2>
  <p class="cat-lead">Reglas exactas que se replican en el backtest.</p>
  <div class="card"><h3>Definici\u00f3n</h3>
    <ul class="rules">
      <li><b>Universo:</b> quarterly, semestral, yearly y flies (contrato a contrato, sin series continuas).</li>
      <li><b>R\u00e9gimen:</b> solo se opera cuando el <b>Efficiency Ratio</b> est\u00e1 en su tercil bajo (mercado de rango). Terciles fijados en train.</li>
      <li><b>Se\u00f1al:</b> por cada fold se elige el <b>mejor indicador de reversi\u00f3n</b> (IC m\u00e1s negativo en train) entre z-scores, RSI, Bollinger, nivel y precursores de agotamiento.</li>
      <li><b>Entrada:</b> comprar el bucket m\u00e1s <b>barato</b> / vender el m\u00e1s <b>caro</b> de ese indicador.</li>
      <li><b>Confirmaci\u00f3n:</b> solo se ejecuta si el <b>nivel</b> (panel de an\u00e1logos, <i>level_z</i>) est\u00e1 en el mismo extremo (\u201cdoble barato / doble caro\u201d).</li>
      <li><b>Salida:</b> a horizonte fijo (D+h); operaciones no solapadas por contrato; coste de ida y vuelta descontado.</li>
      <li><b>Validaci\u00f3n:</b> walk-forward por vintage \u2014 se entrena en a\u00f1os de expiraci\u00f3n previos y se opera el siguiente, sin mirar al futuro.</li>
    </ul>
  </div>

  {_kpis(stats, horizon)}

  <h2 class="cat" id="pnl">Resultado: PnL acumulado</h2>
  <p class="cat-lead">La demostraci\u00f3n directa de aplicar la estrategia.</p>
  <div class="card">{equity}
    <p class="rule">Cada l\u00ednea es el PnL acumulado real de las operaciones fuera de muestra de una familia; la l\u00ednea
    verde gruesa es la cartera combinada de las familias <b>recomendadas</b>. Las familias no recomendadas se dibujan
    punteadas y atenuadas, a modo de contexto. Suma de operaciones no solapadas, coste descontado, encadenadas en el tiempo.</p>
  </div>

  <h2 class="cat" id="tabla">Resultado por familia y horizonte</h2>
  <p class="cat-lead">La MISMA estrategia aplicada a <b>todas</b> las familias, a D+20/D+25/D+30 (recomendadas primero).</p>
  <div class="card"><h3>Estad\u00edsticas de operaci\u00f3n</h3>
    <p class="rule">Fila resaltada = horizonte principal (D+{horizon}). La columna <i>Rec.</i> marca las familias con ventaja
    robusta y econ\u00f3mica. Sharpe anualizado; PnL y drawdown en puntos.</p>
    {_stats_table(stats, horizon)}
  </div>

  {_excluded_table(excluded, horizon)}

  <div class="sub" style="margin-top:34px">CrudeWatch \u00b7 informe de estrategia \u00b7 generado autom\u00e1ticamente desde el ledger de operaciones.</div>
</div></body></html>"""
