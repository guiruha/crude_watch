"""Per-indicator HTML report across every contract family, with conclusions.

For one :class:`~backtesting.backtest.engine.Spec` we backtest every contract of
every family, then render a single self-contained HTML page: an overall summary,
and per family a conclusion paragraph, a P&L-per-contract chart and the full
per-contract metrics table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from backtesting.backtest.engine import (
    CORE,
    DEFAULT_SPECS,
    ENERGY_INDICATORS,
    EXTRA_INDICATORS,
    FAMILY_LABELS,
    REGIME,
    REVERSION_FAMILY,
    Strategy,
    TREND_MOMENTUM,
    run_frame,
)

# --------------------------------------------------------------------------- #
# CrudeWatch product theme (mirrors app/theme/palette.py) so the offline
# backtest reports look like the rest of the CrudeWatch app.
# --------------------------------------------------------------------------- #
BACKGROUND = "#0B0E0D"    # near-black, faint green tint
SURFACE = "#141A17"       # cards / panels
SURFACE_2 = "#1B221E"     # hover / elevated
BORDER = "#26302A"        # hairline borders
ACCENT = "#10B981"        # emerald (corporate green, matches charts)
ACCENT_MUTED = "#0E9E6E"
TEXT = "#E7ECEA"
SUBTEXT = "#8B9691"
BEAR = "#E5484D"          # restrained red
CHART_GRID = "#1A1F1C"    # subtle green-tinted gridlines

# Brand mark: an oil droplet (crude) with a price-pulse line through it.
BRAND_MARK_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M12 2.2c2.7 3.4 7 7.6 7 11.8a7 7 0 0 1-14 0c0-4.2 4.3-8.4 7-11.8z" '
    'fill="currentColor" fill-opacity="0.13"/>'
    '<path d="M7.5 14.4h2l1.3-2.7 1.7 4 1.1-1.9h2.9" stroke-width="1.7"/>'
    '</svg>'
)

# Back-compat aliases used by the layout helpers and CSS below.
BLACK = BACKGROUND
PANEL = SURFACE
GRID = BORDER
GREEN = ACCENT
RED = BEAR

_PLOTLY_CFG = {"displayModeBar": False, "responsive": True}
# The full Plotly bundle, embedded once per page so reports render with no
# internet connection (fully offline).
_PLOTLYJS = get_plotlyjs()


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _family_summary(metrics: pd.DataFrame) -> dict[str, float | str]:
    """Aggregate one family's per-contract metrics for the summary + conclusion."""
    pnl = metrics["pnl_total"]
    finite_pf = metrics["profit_factor"].replace([np.inf, -np.inf], np.nan)
    best = metrics.loc[pnl.idxmax()]
    worst = metrics.loc[pnl.idxmin()]
    return {
        "n_contracts": int(len(metrics)),
        "pct_profitable": float((pnl > 0).mean()),
        "median_sharpe": float(metrics["sharpe"].median()),
        "median_pf": float(finite_pf.median()),
        "median_win_rate": float(metrics["win_rate"].median()),
        "total_pnl": float(pnl.sum()),
        "avg_trades": float(metrics["n_trades"].mean()),
        "best_contract": str(best["contract"]),
        "best_pnl": float(best["pnl_total"]),
        "worst_contract": str(worst["contract"]),
        "worst_pnl": float(worst["pnl_total"]),
    }


def _conclusion(strategy: Strategy, label: str, s: dict[str, float | str]) -> str:
    """One-paragraph verdict for a family, in Spanish."""
    pct = s["pct_profitable"] * 100
    if pct >= 60:
        verdict = "funciona de forma consistente"
    elif pct >= 45:
        verdict = "da resultados mixtos"
    else:
        verdict = "no aporta ventaja"
    return (
        f"En <b>{label}</b> el indicador <b>{strategy.label}</b> {verdict}: es rentable en "
        f"<b>{pct:.0f}%</b> de {s['n_contracts']} contratos "
        f"(Sharpe mediano {s['median_sharpe']:.2f}, profit factor mediano "
        f"{s['median_pf']:.2f}, win rate mediano {s['median_win_rate'] * 100:.0f}%, "
        f"~{s['avg_trades']:.0f} trades/contrato). "
        f"Mejor: {s['best_contract']} ({s['best_pnl']:+.2f} pts). "
        f"Peor: {s['worst_contract']} ({s['worst_pnl']:+.2f} pts)."
    )


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _bar_layout(title: str, y_title: str, x_title: str = "") -> dict:
    return dict(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(color=GREEN, size=18)),
        template="plotly_dark",
        paper_bgcolor=BLACK,
        plot_bgcolor=BLACK,
        font=dict(color=TEXT, family="Arial"),
        margin=dict(l=60, r=30, t=60, b=90),
        showlegend=False,
        xaxis=dict(title=x_title, gridcolor=CHART_GRID, showline=True, linecolor=ACCENT, linewidth=1),
        yaxis=dict(title=y_title, gridcolor=CHART_GRID, showline=True, linecolor=ACCENT, linewidth=1, zeroline=True, zerolinecolor=CHART_GRID),
    )


def _pnl_bar(metrics: pd.DataFrame, title: str) -> go.Figure:
    """Total P&L (points) per contract, green for winners and red for losers."""
    data = metrics.sort_values("pnl_total", ascending=False)
    colors = [GREEN if v >= 0 else RED for v in data["pnl_total"]]
    fig = go.Figure(go.Bar(
        x=data["contract"], y=data["pnl_total"], marker_color=colors, marker_line_width=0,
        hovertemplate="%{x}<br>P&L: %{y:.2f} pts<extra></extra>",
    ))
    fig.update_layout(**_bar_layout(title, "P&L total (puntos)"))
    return fig


def _summary_bar(summaries: dict[str, dict]) -> go.Figure:
    """% of profitable contracts per family."""
    labels = [FAMILY_LABELS[k] for k in summaries]
    pct = [summaries[k]["pct_profitable"] * 100 for k in summaries]
    colors = [GREEN if v >= 50 else RED for v in pct]
    fig = go.Figure(go.Bar(
        x=labels, y=pct, marker_color=colors, marker_line_width=0,
        hovertemplate="%{x}<br>%{y:.0f}% rentables<extra></extra>",
    ))
    fig.update_layout(**_bar_layout("Contratos rentables por familia", "% rentables"))
    fig.update_yaxes(range=[0, 100])
    return fig


def _div(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config=_PLOTLY_CFG)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _format_table(metrics: pd.DataFrame) -> str:
    """Render the per-contract metrics as a compact HTML table."""
    d = metrics.copy()
    pct = lambda v: "" if pd.isna(v) else f"{v * 100:.0f}%"
    num = lambda v: "" if pd.isna(v) else f"{v:.2f}"
    pf = lambda v: "\u221e" if np.isinf(v) else ("" if pd.isna(v) else f"{v:.2f}")
    view = pd.DataFrame({
        "Contrato": d["contract"],
        "Obs": d["n_obs"].astype(int),
        "Trades": d["n_trades"].astype(int),
        "Win rate": d["win_rate"].map(pct),
        "P&L total": d["pnl_total"].map(num),
        "P&L anual.": d["pnl_annualized"].map(num),
        "Sharpe": d["sharpe"].map(num),
        "Max DD": d["max_drawdown"].map(num),
        "Profit factor": d["profit_factor"].map(pf),
        "Expectancy": d["expectancy"].map(num),
        "Exposici\u00f3n": d["exposure"].map(pct),
    })
    return view.to_html(index=False, border=0, classes="cw-table", escape=False)


# --------------------------------------------------------------------------- #
# HTML assembly
# --------------------------------------------------------------------------- #
_CSS = f"""
  body {{ background:{BACKGROUND}; color:{TEXT}; font-family:Arial,Helvetica,sans-serif; margin:0; padding:32px; }}
  h1 {{ color:{TEXT}; font-weight:700; margin:0 0 4px; border-left:5px solid {ACCENT}; padding-left:14px; }}
  h2 {{ color:{TEXT}; font-weight:700; border-left:5px solid {ACCENT}; padding-left:12px; margin-top:40px; }}
  .sub {{ color:{SUBTEXT}; margin:0 0 24px; }}
  .concl {{ background:{SURFACE}; border-left:3px solid {ACCENT}; padding:12px 16px; margin:16px 0; border-radius:4px; }}
  details {{ background:linear-gradient(180deg,{SURFACE} 0%,{BACKGROUND} 100%); border:1px solid {BORDER}; border-radius:12px; margin:12px 0; padding:8px 14px; }}
  details[open] {{ border-color:{ACCENT}77; box-shadow:inset 3px 0 0 0 {ACCENT}; }}
  summary {{ cursor:pointer; color:{ACCENT}; font-weight:700; padding:6px 2px; }}
  table.cw-table {{ border-collapse:collapse; width:100%; margin-top:12px; font-size:13px; }}
  table.cw-table th {{ color:{ACCENT}; text-align:right; padding:6px 10px; border-bottom:1px solid {BORDER}; position:sticky; top:0; background:{SURFACE}; }}
  table.cw-table td {{ text-align:right; padding:5px 10px; border-bottom:1px solid {SURFACE}; }}
  table.cw-table td:first-child, table.cw-table th:first-child {{ text-align:left; }}
  table.cw-table tr:hover td {{ background:{SURFACE_2}; }}
  .trade {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px; padding:16px 18px; margin:16px 0; }}
  .trade-row {{ display:flex; align-items:flex-start; gap:12px; margin-bottom:10px; }}
  .tag {{ font-size:11px; font-weight:700; letter-spacing:.5px; padding:3px 8px; border-radius:4px; flex:0 0 auto; margin-top:1px; }}
  .tag-in {{ background:{ACCENT}26; color:{ACCENT}; border:1px solid {ACCENT}; }}
  .tag-out {{ background:{BEAR}26; color:{BEAR}; border:1px solid {BEAR}; }}
  .trade-note {{ color:{SUBTEXT}; font-size:12px; margin-top:6px; border-top:1px solid {BORDER}; padding-top:8px; }}
  .kpis {{ display:flex; flex-wrap:wrap; gap:16px; margin:16px 0; }}
  .kpi {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px; padding:12px 18px; min-width:150px; }}
  .kpi .v {{ color:{ACCENT}; font-size:22px; font-weight:700; }}
  .kpi .l {{ color:{SUBTEXT}; font-size:12px; text-transform:uppercase; letter-spacing:.4px; }}
  a {{ color:{ACCENT}; }}
"""


def _kpi(value: str, label: str) -> str:
    return f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>'


def build_report(frames: dict[str, pd.DataFrame], strategy: Strategy) -> str:
    """Return the full HTML page for one indicator across all families."""
    sections: list[str] = []
    summaries: dict[str, dict] = {}
    total_contracts = 0
    total_pnl = 0.0

    for key, label in FAMILY_LABELS.items():
        frame = frames.get(key)
        if frame is None or frame.empty:
            continue
        metrics = run_frame(frame, strategy)
        if metrics.empty:
            continue

        summary = _family_summary(metrics)
        summaries[key] = summary
        total_contracts += summary["n_contracts"]
        total_pnl += summary["total_pnl"]

        bar_title = f"{strategy.label} \u2014 P&L por contrato ({label})"
        conclusion = _conclusion(strategy, label, summary)
        chart = _div(_pnl_bar(metrics, bar_title))
        table = _format_table(metrics)
        n = summary["n_contracts"]
        sections.append(
            f'<h2>{label}</h2>'
            f'<div class="concl">{conclusion}</div>'
            f'{chart}'
            f'<details><summary>Ver tabla por contrato ({n})</summary>{table}</details>'
        )

    n_prof = sum(s["pct_profitable"] * s["n_contracts"] for s in summaries.values())
    overall_pct = (n_prof / total_contracts * 100) if total_contracts else 0.0
    best_family = max(summaries, key=lambda k: summaries[k]["pct_profitable"], default=None)

    kpis = "".join([
        _kpi(f"{total_contracts}", "contratos analizados"),
        _kpi(f"{overall_pct:.0f}%", "rentables (global)"),
        _kpi(f"{total_pnl:+.0f}", "P&L total (puntos)"),
        _kpi(FAMILY_LABELS[best_family] if best_family else "\u2014", "mejor familia"),
    ])
    summary_plot = _div(_summary_bar(summaries)) if summaries else ""

    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Backtest {strategy.label} \u2014 CrudeWatch</title>
<script type="text/javascript">{_PLOTLYJS}</script>
<style>{_CSS}</style></head>
<body>
<h1>Backtest \u00b7 {strategy.label}</h1>
<p class="sub">Long/flat por contrato individual \u00b7 P&L en puntos de precio \u00b7
se\u00f1al desplazada 1 barra (sin look-ahead).</p>
<div class="trade">
  <div class="trade-row"><span class="tag tag-in">ENTRADA</span><span>{strategy.entry_rule}</span></div>
  <div class="trade-row"><span class="tag tag-out">SALIDA</span><span>{strategy.exit_rule}</span></div>
  <div class="trade-note">La se\u00f1al detectada al cierre de un d\u00eda se ejecuta al precio del d\u00eda
  siguiente. Solo se opera al alza (long/flat): entre la entrada y la salida se mantiene el largo;
  el resto del tiempo se est\u00e1 fuera del mercado.</div>
</div>
<div class="kpis">{kpis}</div>
{summary_plot}
{''.join(sections)}
</body></html>"""


# --------------------------------------------------------------------------- #
# Comparative report (all strategies on one page)
# --------------------------------------------------------------------------- #
def _agg_strategy(df: pd.DataFrame) -> pd.Series:
    """Aggregate one strategy's per-contract rows into headline metrics."""
    pnl = df["pnl_total"]
    finite_pf = df["profit_factor"].replace([np.inf, -np.inf], np.nan)
    return pd.Series({
        "contracts": int(len(df)),
        "pct_profitable": float((pnl > 0).mean() * 100),
        "median_sharpe": float(df["sharpe"].median()),
        "mean_sharpe": float(df["sharpe"].mean()),
        "pnl_annualized": float(df["pnl_annualized"].mean()),
        "total_pnl": float(pnl.sum()),
        "median_win": float(df["win_rate"].median() * 100),
        "median_pf": float(finite_pf.median()),
        "avg_trades": float(df["n_trades"].mean()),
        "exposure": float(df["exposure"].mean() * 100),
    })


def _ranking(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per strategy, sorted best-first by median Sharpe."""
    ranking = raw.groupby("indicator", sort=False).apply(_agg_strategy)
    return ranking.sort_values("median_sharpe", ascending=False)


def _rank_bar(ranking: pd.DataFrame) -> go.Figure:
    """Horizontal bar of median Sharpe per strategy (best at the top)."""
    data = ranking.sort_values("median_sharpe")  # ascending so best ends up on top
    colors = [GREEN if v >= 0 else RED for v in data["median_sharpe"]]
    fig = go.Figure(go.Bar(
        x=data["median_sharpe"], y=data.index, orientation="h",
        marker_color=colors, marker_line_width=0,
        hovertemplate="%{y}<br>Sharpe mediano: %{x:.3f}<extra></extra>",
    ))
    layout = _bar_layout("Ranking por Sharpe mediano (por contrato)", "", "Sharpe mediano")
    layout["margin"] = dict(l=180, r=30, t=60, b=50)
    layout["yaxis"]["automargin"] = True
    fig.update_layout(**layout)
    fig.update_layout(height=max(420, 20 * len(data) + 120))
    return fig


def _heatmap(raw: pd.DataFrame, order: list[str]) -> go.Figure:
    """Median Sharpe per strategy × family (diverging colour around zero)."""
    families = [k for k in FAMILY_LABELS if k in raw["family"].unique()]
    pivot = (
        raw.pivot_table(index="indicator", columns="family", values="sharpe", aggfunc="median")
        .reindex(index=order, columns=families)
    )
    fig = go.Figure(go.Heatmap(
        z=pivot.to_numpy(),
        x=[FAMILY_LABELS[c] for c in families],
        y=pivot.index.tolist(),
        zmid=0,
        colorscale=[[0.0, BEAR], [0.5, SURFACE_2], [1.0, ACCENT]],
        colorbar=dict(title="Sharpe", tickfont=dict(color=TEXT)),
        hovertemplate="%{y}<br>%{x}<br>Sharpe mediano: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Sharpe mediano por estrategia y familia", x=0.5, xanchor="center",
                   font=dict(color=GREEN, size=18)),
        template="plotly_dark", paper_bgcolor=BLACK, plot_bgcolor=BLACK,
        font=dict(color=TEXT, family="Arial"),
        margin=dict(l=180, r=30, t=60, b=90),
        height=max(420, 20 * len(order) + 160),
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True, autorange="reversed")
    return fig


def _ranking_table(ranking: pd.DataFrame, name_by_label: dict[str, str], to_anchor: bool = False) -> str:
    """Full ranking as a styled, linkable HTML table.

    When ``to_anchor`` is set, strategy names link to in-page anchors
    (``#NAME``) instead of separate ``NAME.html`` files.
    """
    num = lambda v: "" if pd.isna(v) else f"{v:.2f}"
    num3 = lambda v: "" if pd.isna(v) else f"{v:.3f}"
    pct = lambda v: "" if pd.isna(v) else f"{v:.0f}%"
    pf = lambda v: "\u221e" if np.isinf(v) else ("" if pd.isna(v) else f"{v:.2f}")

    def link(label: str) -> str:
        name = name_by_label.get(label)
        if not name:
            return label
        href = f"#{name}" if to_anchor else f"{name}.html"
        return f'<a href="{href}">{label}</a>'

    rows = []
    for pos, (label, r) in enumerate(ranking.iterrows(), start=1):
        rows.append(
            "<tr>"
            f"<td>{pos}</td>"
            f"<td>{link(label)}</td>"
            f"<td>{int(r['contracts'])}</td>"
            f"<td>{pct(r['pct_profitable'])}</td>"
            f"<td>{num3(r['median_sharpe'])}</td>"
            f"<td>{num(r['pnl_annualized'])}</td>"
            f"<td>{r['total_pnl']:+.0f}</td>"
            f"<td>{pct(r['median_win'])}</td>"
            f"<td>{pf(r['median_pf'])}</td>"
            f"<td>{num(r['avg_trades'])}</td>"
            f"<td>{pct(r['exposure'])}</td>"
            "</tr>"
        )
    head = (
        "<tr><th>#</th><th>Estrategia</th><th>Contratos</th><th>% rentables</th>"
        "<th>Sharpe mediano</th><th>P&L anual. medio</th><th>P&L total</th>"
        "<th>Win rate mediano</th><th>Profit factor mediano</th><th>Trades/contrato</th>"
        "<th>Exposici\u00f3n</th></tr>"
    )
    return f'<table class="cw-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


def build_comparison(raw: pd.DataFrame, strategies: list[Strategy]) -> str:
    """Return a single self-contained HTML page comparing every strategy."""
    ranking = _ranking(raw)
    order = ranking.index.tolist()
    name_by_label = {s.label: s.name for s in strategies}

    best_label = order[0]
    worst_label = order[-1]
    best_pnl_label = ranking["pnl_annualized"].idxmax()
    total_contracts = int(raw.groupby("indicator")["contract"].size().max())

    kpis = "".join([
        _kpi(f"{len(ranking)}", "estrategias comparadas"),
        _kpi(f"{total_contracts}", "contratos por estrategia"),
        _kpi(best_label, "mejor Sharpe mediano"),
        _kpi(best_pnl_label, "mejor P&L anualizado"),
    ])

    note = (
        f"Todas las estrategias son <b>long/flat</b> sobre el mismo universo de contratos, con la "
        f"se\u00f1al desplazada 1 barra (sin look-ahead) y P&L en puntos de precio. El ranking se ordena "
        f"por <b>Sharpe mediano por contrato</b>, m\u00e1s robusto a los pocos contratos extremos que la media. "
        f"Lidera <b>{best_label}</b>; el farolillo rojo es <b>{worst_label}</b>."
    )

    rank_bar = _div(_rank_bar(ranking))
    heatmap = _div(_heatmap(raw, order))
    table = _ranking_table(ranking, name_by_label)

    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Comparativa de estrategias \u2014 CrudeWatch</title>
<script type="text/javascript">{_PLOTLYJS}</script>
<style>{_CSS}</style></head>
<body>
<h1>Comparativa \u00b7 {len(ranking)} estrategias</h1>
<p class="sub">Backtest long/flat por contrato individual \u00b7 P&L en puntos de precio \u00b7
se\u00f1al desplazada 1 barra (sin look-ahead).</p>
<div class="concl">{note}</div>
<div class="kpis">{kpis}</div>
<h2>Ranking</h2>
{table}
<h2>Ranking visual</h2>
{rank_bar}
<h2>Mapa de calor por familia</h2>
<p class="sub">D\u00f3nde funciona cada estrategia: verde = Sharpe mediano positivo, rojo = negativo.</p>
{heatmap}
<p class="sub" style="margin-top:32px">\u2190 Vuelve al <a href="index.html">\u00edndice de informes</a> para el detalle de cada indicador.</p>
</body></html>"""


# --------------------------------------------------------------------------- #
# Combined single-page report (everything, grouped, with a pretty index)
# --------------------------------------------------------------------------- #
_CATEGORY_DEFS = [
    ("ma", "Cruces de medias", "Cruces de medias m\u00f3viles SMA/EMA (tendencia cl\u00e1sica).", DEFAULT_SPECS),
    ("core", "Cl\u00e1sicos (RSI / MACD / divergencias)", "Osciladores de momentum y divergencias de referencia.", CORE),
    ("trend", "Tendencia y momentum", "Seguimiento de tendencia y osciladores de momentum.", TREND_MOMENTUM),
    ("extra", "Indicadores extra", "Adaptativos, momentum avanzado y reversi\u00f3n b\u00e1sica.", EXTRA_INDICATORS),
    ("energy", "Energ\u00eda / WTI", "Los m\u00e1s usados en crudo y futuros de energ\u00eda (versi\u00f3n close-based).", ENERGY_INDICATORS),
    ("reversion", "Familia reversi\u00f3n", "Variantes y alternativas de las reglas de reversi\u00f3n/divergencia que mejor puntuaron.", REVERSION_FAMILY),
    ("regime", "Cambio de r\u00e9gimen", "Meta-estrategia que alterna tendencia y reversi\u00f3n seg\u00fan el r\u00e9gimen.", REGIME),
]

_COMBINED_CSS = f"""
  * {{ box-sizing:border-box; }}
  body {{ background:{BACKGROUND}; color:{TEXT}; font-family:Arial,Helvetica,sans-serif; margin:0;
          font-size:15px; }}
  a {{ color:{ACCENT}; text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
  .layout {{ display:flex; align-items:flex-start; }}

  /* Sidebar (mirrors the app's brand lockup + nav) */
  .side {{ position:sticky; top:0; height:100vh; width:300px; flex:0 0 300px; overflow-y:auto;
           background:linear-gradient(180deg,{SURFACE} 0%,{BACKGROUND} 100%);
           border-right:1px solid {BORDER}; padding:22px 16px; }}
  .cw-side-brand {{ display:flex; align-items:center; gap:10px; margin:0 0 4px 2px; }}
  .cw-side-mark {{ width:46px; height:46px; border-radius:12px; flex:none; display:flex;
                   align-items:center; justify-content:center; color:{ACCENT}; line-height:0;
                   background:radial-gradient(120% 120% at 30% 20%,{SURFACE_2} 0%,{BACKGROUND} 100%);
                   box-shadow:inset 0 0 0 1px {ACCENT}66, 0 6px 18px -8px {ACCENT}; }}
  .cw-side-mark svg {{ width:28px; height:28px; display:block; }}
  .cw-side-word {{ font-size:28px; font-weight:800; letter-spacing:.5px; text-transform:uppercase;
                   color:{ACCENT}; line-height:1; }}
  .cw-side-word span {{ color:{TEXT}; }}
  .cw-side-tag {{ color:{SUBTEXT}; font-size:12px; letter-spacing:.3px; margin:10px 2px 4px; }}
  .cw-divider {{ border:0; border-top:1px solid {BORDER}; margin:14px 0; }}
  .cw-nav-label {{ color:{SUBTEXT}; font-size:12px; font-weight:700; letter-spacing:1.4px;
                   text-transform:uppercase; margin:14px 4px 6px; }}
  .cat-desc {{ color:{SUBTEXT}; font-size:11px; margin:0 4px 8px; line-height:1.4; }}
  .side ul {{ list-style:none; margin:0 0 6px; padding:0; }}
  .side li a {{ display:flex; justify-content:space-between; align-items:center; gap:8px;
                padding:9px 12px; margin:2px 0; border-radius:9px; border:1px solid transparent;
                color:{TEXT}; font-size:13px; transition:background .15s, border-color .15s; }}
  .side li a:hover {{ background:{SURFACE_2}; border-color:{BORDER}; text-decoration:none; }}
  .side li a:target, .side li a.active {{ box-shadow:inset 3px 0 0 0 {ACCENT}; }}
  .badge {{ font-size:11px; font-weight:700; padding:2px 8px; border-radius:10px; flex:0 0 auto; }}
  .badge.pos {{ background:{ACCENT}26; color:{ACCENT}; }}
  .badge.neg {{ background:{BEAR}26; color:{BEAR}; }}
  .cw-side-foot {{ color:{SUBTEXT}; font-size:10.5px; letter-spacing:.3px; margin-top:18px; opacity:.8; }}
  .cw-side-copy {{ color:{SUBTEXT}; font-size:11px; margin:8px 2px 2px; opacity:.7; }}

  /* Main column */
  .main {{ flex:1 1 auto; padding:34px 44px; max-width:1220px; min-width:0; }}
  .hero {{ background:linear-gradient(135deg,{ACCENT}1f,transparent 68%);
           border:1px solid {BORDER}; border-radius:16px; padding:28px 30px; margin-bottom:30px; }}
  .cw-title {{ color:{TEXT}; font-size:32px; font-weight:700; border-left:5px solid {ACCENT};
               padding-left:14px; margin:0 0 6px; }}
  .cw-sub {{ color:{SUBTEXT}; font-size:15px; margin:0 0 0 19px; }}
  h2.cat {{ color:{TEXT}; font-size:23px; font-weight:700; border-left:5px solid {ACCENT};
            padding-left:14px; margin:46px 0 6px; }}
  .cat-lead {{ color:{SUBTEXT}; margin:0 0 18px 19px; font-size:14px; }}

  /* Indicator cards (mirror the app's expander surfaces) */
  .card {{ background:linear-gradient(180deg,{SURFACE} 0%,{BACKGROUND} 100%);
           border:1px solid {BORDER}; border-radius:16px; padding:22px 26px; margin:18px 0;
           scroll-margin-top:16px; transition:border-color .18s, box-shadow .18s, transform .18s; }}
  .card:hover {{ border-color:{ACCENT}55; box-shadow:0 12px 30px -18px {ACCENT}; transform:translateY(-1px); }}
  .card h3 {{ margin:0 0 4px; color:{TEXT}; font-size:20px; }}
  .card .rule {{ color:{SUBTEXT}; margin:0 0 14px; font-size:13px; }}
  .rank-pill {{ display:inline-block; font-size:12px; font-weight:700; padding:3px 10px; border-radius:10px;
                background:{BACKGROUND}; border:1px solid {ACCENT}55; color:{ACCENT}; margin-left:8px;
                vertical-align:middle; }}
  .concl {{ background:{BACKGROUND}; border-left:3px solid {ACCENT}; padding:10px 14px; margin:10px 0;
            border-radius:4px; font-size:13px; }}
  details {{ background:{BACKGROUND}; border:1px solid {BORDER}; border-radius:12px; margin:12px 0;
             padding:6px 14px; transition:border-color .18s, box-shadow .18s; }}
  details:hover {{ border-color:{ACCENT}55; }}
  details[open] {{ border-color:{ACCENT}77; box-shadow:inset 3px 0 0 0 {ACCENT}; }}
  summary {{ cursor:pointer; color:{ACCENT}; font-weight:700; padding:8px 2px; }}
  table.cw-table {{ border-collapse:collapse; width:100%; margin-top:12px; font-size:13px; }}
  table.cw-table th {{ color:{ACCENT}; text-align:right; padding:6px 10px; border-bottom:1px solid {BORDER}; }}
  table.cw-table td {{ text-align:right; padding:5px 10px; border-bottom:1px solid {SURFACE}; }}
  table.cw-table td:first-child, table.cw-table th:first-child {{ text-align:left; }}
  table.cw-table tr:hover td {{ background:{SURFACE_2}; }}
  .trade {{ background:{BACKGROUND}; border:1px solid {BORDER}; border-radius:10px; padding:14px 16px; margin:12px 0; }}
  .trade-row {{ display:flex; align-items:flex-start; gap:12px; margin-bottom:8px; }}
  .tag {{ font-size:11px; font-weight:700; letter-spacing:.5px; padding:3px 8px; border-radius:4px;
          flex:0 0 auto; margin-top:1px; }}
  .tag-in {{ background:{ACCENT}26; color:{ACCENT}; border:1px solid {ACCENT}; }}
  .tag-out {{ background:{BEAR}26; color:{BEAR}; border:1px solid {BEAR}; }}
  .kpis {{ display:flex; flex-wrap:wrap; gap:14px; margin:14px 0; }}
  .kpi {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px; padding:14px 18px; min-width:140px; }}
  .kpi .v {{ color:{ACCENT}; font-size:22px; font-weight:700; }}
  .kpi .l {{ color:{SUBTEXT}; font-size:11px; text-transform:uppercase; letter-spacing:.4px; }}
  .top-link {{ float:right; font-size:12px; color:{SUBTEXT}; }}
  @media (max-width:900px) {{
    .layout {{ flex-direction:column; }}
    .side {{ position:static; height:auto; width:100%; flex:none; border-right:0; border-bottom:1px solid {BORDER}; }}
    .main {{ padding:24px 18px; }}
  }}
"""


def _indicator_summaries(sub: pd.DataFrame) -> dict[str, dict]:
    """Per-family summary dict for one indicator's rows in the tidy metrics table."""
    summaries: dict[str, dict] = {}
    for key in FAMILY_LABELS:
        fam = sub[sub["family"] == key]
        if not fam.empty:
            summaries[key] = _family_summary(fam)
    return summaries


def _family_summary_table(summaries: dict[str, dict]) -> str:
    """Compact per-family aggregate table (one row per family)."""
    num = lambda v: "" if pd.isna(v) else f"{v:.2f}"
    pct = lambda v: "" if pd.isna(v) else f"{v * 100:.0f}%"
    rows = []
    for key, s in summaries.items():
        rows.append(
            "<tr>"
            f"<td>{FAMILY_LABELS[key]}</td>"
            f"<td>{int(s['n_contracts'])}</td>"
            f"<td>{s['pct_profitable'] * 100:.0f}%</td>"
            f"<td>{num(s['median_sharpe'])}</td>"
            f"<td>{num(s['median_pf'])}</td>"
            f"<td>{pct(s['median_win_rate'])}</td>"
            f"<td>{s['total_pnl']:+.0f}</td>"
            f"<td>{num(s['avg_trades'])}</td>"
            "</tr>"
        )
    head = (
        "<tr><th>Familia</th><th>Contratos</th><th>% rentables</th><th>Sharpe mediano</th>"
        "<th>Profit factor mediano</th><th>Win rate mediano</th><th>P&L total</th><th>Trades/contrato</th></tr>"
    )
    return f'<table class="cw-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


def _indicator_section(strategy: Strategy, sub: pd.DataFrame, rank: int | None) -> str:
    """One indicator card: KPIs, entry/exit, per-family summary + conclusions."""
    summaries = _indicator_summaries(sub)
    if not summaries:
        return ""
    total_contracts = sum(s["n_contracts"] for s in summaries.values())
    total_pnl = sum(s["total_pnl"] for s in summaries.values())
    n_prof = sum(s["pct_profitable"] * s["n_contracts"] for s in summaries.values())
    overall_pct = (n_prof / total_contracts * 100) if total_contracts else 0.0
    best_family = max(summaries, key=lambda k: summaries[k]["pct_profitable"])

    kpis = "".join([
        _kpi(f"{total_contracts}", "contratos"),
        _kpi(f"{overall_pct:.0f}%", "rentables"),
        _kpi(f"{total_pnl:+.0f}", "P&L total (pts)"),
        _kpi(FAMILY_LABELS[best_family], "mejor familia"),
    ])
    conclusions = "".join(
        f'<div class="concl">{_conclusion(strategy, FAMILY_LABELS[k], s)}</div>'
        for k, s in summaries.items()
    )
    rank_pill = f'<span class="rank-pill">#{rank} global</span>' if rank else ""
    summary_bar = _div(_summary_bar(summaries))
    return f"""
<div class="card" id="{strategy.name}">
  <a class="top-link" href="#top">\u2191 arriba</a>
  <h3>{strategy.label}{rank_pill}</h3>
  <p class="rule">{strategy.rule}
  &nbsp;\u00b7&nbsp;<a href="{strategy.name}.html">detalle por contrato \u2197</a></p>
  <div class="trade">
    <div class="trade-row"><span class="tag tag-in">ENTRADA</span><span>{strategy.entry_rule}</span></div>
    <div class="trade-row"><span class="tag tag-out">SALIDA</span><span>{strategy.exit_rule}</span></div>
  </div>
  <div class="kpis">{kpis}</div>
  {summary_bar}
  <details><summary>Resumen por familia</summary>{_family_summary_table(summaries)}</details>
  <details><summary>Conclusiones por familia</summary>{conclusions}</details>
</div>"""


def _sidebar(categories: list[tuple], ranking: pd.DataFrame) -> str:
    """Grouped, sticky table of contents with the CrudeWatch brand lockup."""
    n = len(ranking)
    parts = [
        f'<div class="cw-side-brand"><div class="cw-side-mark">{BRAND_MARK_SVG}</div>'
        f'<div class="cw-side-word">Crude<span>Watch</span></div></div>',
        f'<div class="cw-side-tag">Backtests \u00b7 {n} estrategias</div>',
        '<hr class="cw-divider">',
        '<div class="cw-nav-label">Resumen</div>',
        '<ul><li><a href="#comparativa"><span>Comparativa global</span></a></li></ul>',
    ]
    for cid, title, desc, strategies in categories:
        items = []
        for s in strategies:
            if s.label not in ranking.index:
                continue
            sharpe = ranking.loc[s.label, "median_sharpe"]
            cls = "pos" if sharpe >= 0 else "neg"
            items.append(
                f'<li><a href="#{s.name}"><span>{s.label}</span>'
                f'<span class="badge {cls}">{sharpe:.2f}</span></a></li>'
            )
        if not items:
            continue
        parts.append(
            f'<div class="cw-nav-label">{title}</div>'
            f'<div class="cat-desc">{desc}</div><ul>{"".join(items)}</ul>'
        )
    parts.append('<hr class="cw-divider">')
    parts.append('<div class="cw-side-foot">CrudeWatch \u00b7 v0.1 \u00b7 Data: CME / ICE</div>')
    parts.append('<div class="cw-side-copy">\u00a9 guiruha</div>')
    return f'<nav class="side">{"".join(parts)}</nav>'


def build_combined(raw: pd.DataFrame, strategies: list[Strategy]) -> str:
    """Return a single self-contained HTML page with every indicator, grouped."""
    ranking = _ranking(raw)
    order = ranking.index.tolist()
    name_by_label = {s.label: s.name for s in strategies}
    rank_by_label = {label: pos for pos, label in enumerate(order, start=1)}
    strat_by_label = {s.label: s for s in strategies}

    present = set(raw["indicator"].unique())
    categories = [
        (cid, title, desc, [s for s in strats if s.label in present])
        for cid, title, desc, strats in _CATEGORY_DEFS
    ]

    total_contracts = int(raw.groupby("indicator")["contract"].size().max())
    best_label, worst_label = order[0], order[-1]
    best_pnl_label = ranking["pnl_annualized"].idxmax()

    kpis = "".join([
        _kpi(f"{len(ranking)}", "estrategias"),
        _kpi(f"{total_contracts}", "contratos/estrategia"),
        _kpi(best_label, "mejor Sharpe mediano"),
        _kpi(best_pnl_label, "mejor P&L anualizado"),
    ])
    comparison = f"""
<section id="comparativa">
  <h2 class="cat">Comparativa global</h2>
  <p class="cat-lead">Ranking por <b>Sharpe mediano por contrato</b>. Lidera <b>{best_label}</b>;
  el farolillo rojo es <b>{worst_label}</b>. Haz clic en un nombre para saltar a su ficha.</p>
  <div class="kpis">{kpis}</div>
  {_ranking_table(ranking, name_by_label, to_anchor=True)}
  <h3 style="color:{GREEN};margin-top:28px">Ranking visual</h3>
  {_div(_rank_bar(ranking))}
  <h3 style="color:{GREEN};margin-top:28px">Mapa de calor por familia</h3>
  {_div(_heatmap(raw, order))}
</section>"""

    sections = [comparison]
    for cid, title, desc, strats in categories:
        if not strats:
            continue
        cards = []
        for s in strats:
            sub = raw[raw["indicator"] == s.label]
            strat = strat_by_label[s.label]
            cards.append(_indicator_section(strat, sub, rank_by_label.get(s.label)))
        sections.append(
            f'<h2 class="cat" id="cat-{cid}">{title}</h2>'
            f'<p class="cat-lead">{desc}</p>{"".join(cards)}'
        )

    sidebar = _sidebar(categories, ranking)
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backtests CrudeWatch \u2014 informe completo</title>
<script type="text/javascript">{_PLOTLYJS}</script>
<style>{_COMBINED_CSS}</style></head>
<body><a id="top"></a>
<div class="layout">
{sidebar}
<main class="main">
  <div class="hero">
    <div class="cw-title">Backtests \u00b7 {len(ranking)} estrategias</div>
    <div class="cw-sub">Backtest long/flat por contrato individual \u00b7 P&L en puntos de precio \u00b7
    se\u00f1al desplazada 1 barra (sin look-ahead). Todo en una p\u00e1gina, agrupado por familia de indicador.</div>
  </div>
  {''.join(sections)}
</main>
</div>
</body></html>"""


def write_reports(
    frames: dict[str, pd.DataFrame],
    strategies: list[Strategy],
    out_dir: Path,
) -> tuple[list[Path], pd.DataFrame]:
    """Write one HTML per strategy plus an index, and return (paths, raw metrics).

    The returned dataframe is the tidy long table (one row per
    strategy × family × contract) suitable for a CSV dump.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    raw_rows: list[pd.DataFrame] = []

    for strategy in strategies:
        html = build_report(frames, strategy)
        path = out_dir / f"{strategy.name}.html"
        path.write_text(html, encoding="utf-8")
        paths.append(path)

        for key in FAMILY_LABELS:
            frame = frames.get(key)
            if frame is None or frame.empty:
                continue
            metrics = run_frame(frame, strategy)
            if metrics.empty:
                continue
            metrics = metrics.assign(indicator=strategy.label, family=key)
            raw_rows.append(metrics)

    raw = pd.concat(raw_rows, ignore_index=True) if raw_rows else pd.DataFrame()

    if not raw.empty:
        comparison = out_dir / "comparison.html"
        comparison.write_text(build_comparison(raw, strategies), encoding="utf-8")
        paths.append(comparison)

        index = out_dir / "index.html"
        index.write_text(build_combined(raw, strategies), encoding="utf-8")
        paths.append(index)

    return paths, raw
