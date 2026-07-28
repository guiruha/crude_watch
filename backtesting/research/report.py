"""Walk-forward research summary, in the full CrudeWatch report style.

Mirrors the look of the long/flat combined report (brand sidebar, hero, grouped
cards, expandable tables, conclusions) but for the *predictive* backtest: how
well each feature known at ``t`` forecasts the forward outcome after ``t``,
validated out-of-sample by vintage. This is the honest "backtest summary" the
whole research layer feeds into.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from crudewatch.infra.constants import FAMILY_LABELS

# --------------------------------------------------------------------------- #
# CrudeWatch product theme (mirrors backtest/report.py).
# --------------------------------------------------------------------------- #
BACKGROUND = "#0B0E0D"
SURFACE = "#141A17"
SURFACE_2 = "#1B221E"
BORDER = "#26302A"
ACCENT = "#10B981"
TEXT = "#E7ECEA"
SUBTEXT = "#8B9691"
BEAR = "#E5484D"
CHART_GRID = "#1A1F1C"

BRAND_MARK_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M12 2.2c2.7 3.4 7 7.6 7 11.8a7 7 0 0 1-14 0c0-4.2 4.3-8.4 7-11.8z" '
    'fill="currentColor" fill-opacity="0.13"/>'
    '<path d="M7.5 14.4h2l1.3-2.7 1.7 4 1.1-1.9h2.9" stroke-width="1.7"/>'
    '</svg>'
)

_PLOTLY_CFG = {"displayModeBar": False, "responsive": True}
_PLOTLYJS = get_plotlyjs()

# --------------------------------------------------------------------------- #
# Feature vocabulary: label, plain-language description, and category.
# --------------------------------------------------------------------------- #
FEATURE_LABELS: dict[str, str] = {
    "level_z": "Nivel z (panel an\u00e1logos)",
    "level_pct": "Nivel percentil (panel)",
    "z_10": "Z-score 10",
    "z_20": "Z-score 20",
    "z_50": "Z-score 50",
    "pctb_20_2": "Bollinger %B 20/2",
    "pctb_10_1_5": "Bollinger %B 10/1.5",
    "keltner_dist_20": "Distancia Keltner 20",
    "rsi_2": "RSI 2",
    "rsi_14": "RSI 14",
    "rsi_div_14": "Divergencia RSI 14",
    "macd_div": "Divergencia MACD",
    "mom_decel_10": "Desaceleraci\u00f3n momentum",
    "er_drop_20": "Ca\u00edda de ER",
    "vol_ratio": "Ratio de volatilidad",
    "er_20": "Efficiency Ratio 20",
    "slope_20": "Pendiente/ATR 20",
    "macd_hist": "MACD histograma",
}

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "level_z": "Cu\u00e1n caro/barato est\u00e1 el contrato frente a otros contratos del MISMO slot estacional y vintages anteriores, a la misma altura de vida (d\u00edas a vencimiento). Es el bloque de nivel, contrato a contrato, sin series continuas.",
    "level_pct": "Percentil del precio dentro del panel de contratos an\u00e1logos (0 = lo m\u00e1s barato que se ha visto en esa fase de vida, 1 = lo m\u00e1s caro).",
    "z_10": "Desviaciones t\u00edpicas del precio frente a su media m\u00f3vil de 10 barras (extensi\u00f3n a corto).",
    "z_20": "Igual con ventana 20 barras.",
    "z_50": "Igual con ventana 50 barras (extensi\u00f3n a medio plazo).",
    "pctb_20_2": "Posici\u00f3n dentro de las bandas de Bollinger 20/2 (transformaci\u00f3n lineal del z-score 20).",
    "pctb_10_1_5": "Bandas de Bollinger 10/1.5, m\u00e1s r\u00e1pidas (transformaci\u00f3n lineal del z-score 10).",
    "keltner_dist_20": "Distancia a la media de Keltner medida en ATR (sobre-extensi\u00f3n normalizada por volatilidad).",
    "rsi_2": "RSI ultra-corto (2): detecta sobreventa/sobrecompra inmediata.",
    "rsi_14": "RSI cl\u00e1sico (14).",
    "rsi_div_14": "Divergencia entre el RSI(14) y el precio: mide si el momentum acompa\u00f1a o contradice el \u00faltimo movimiento de precio (positivo = divergencia alcista, negativo = bajista). Precursor de agotamiento.",
    "macd_div": "Igual que la divergencia RSI pero usando la l\u00ednea MACD frente al precio.",
    "mom_decel_10": "Cambio del momentum de 10 barras (2\u00aa derivada) normalizado por ATR: negativo = subida perdiendo fuerza (desaceleraci\u00f3n).",
    "er_drop_20": "Variaci\u00f3n del Efficiency Ratio en 5 barras: negativo = la tendencia se est\u00e1 volviendo choppy (favorece reversi\u00f3n).",
    "vol_ratio": "Volatilidad realizada corta (10) sobre larga (50): >1 = expansi\u00f3n de volatilidad, <1 = contracci\u00f3n.",
    "er_20": "Efficiency Ratio (Kaufman): 1 = movimiento limpio y direccional, 0 = ruido lateral. Filtro de r\u00e9gimen.",
    "slope_20": "Pendiente de la regresi\u00f3n de 20 barras normalizada por ATR (gradiente de tendencia).",
    "macd_hist": "Histograma del MACD (momentum direccional).",
}

FEATURE_CATEGORIES: list[tuple[str, str, list[str]]] = [
    ("Nivel / panel", "Caro-barato frente a contratos an\u00e1logos por vintage y fase de vida (Bloque D).",
     ["level_z", "level_pct"]),
    ("Reversi\u00f3n / extensi\u00f3n", "Sobre-extensi\u00f3n frente a la propia media reciente (Bloque E).",
     ["z_10", "z_20", "z_50", "pctb_10_1_5", "pctb_20_2", "keltner_dist_20", "rsi_2", "rsi_14"]),
    ("Agotamiento / precursores", "Divergencias, desaceleraci\u00f3n y cambio de r\u00e9gimen que anticipan la vuelta (Bloque E).",
     ["rsi_div_14", "macd_div", "mom_decel_10", "er_drop_20", "vol_ratio"]),
    ("R\u00e9gimen / direcci\u00f3n", "Tendencia, calidad y momentum direccional (Bloques A/B/C).",
     ["er_20", "slope_20", "macd_hist"]),
]

# --------------------------------------------------------------------------- #
# Metric vocabulary: (column label, plain-language explanation with how to read).
# Used both for the metric glossary and for the <th> hover tooltips.
# --------------------------------------------------------------------------- #
METRIC_DEFS: dict[str, tuple[str, str]] = {
    "feature": (
        "Feature",
        "El indicador continuo evaluado, medido al cierre de t (nunca usa el futuro).",
    ),
    "sesgo": (
        "Sesgo",
        "Direcci\u00f3n del efecto seg\u00fan el signo del IC: 'reversi\u00f3n' si un valor alto de la feature "
        "(caro/sobrecomprado) anticipa un retorno negativo; 'continuaci\u00f3n' si lo prolonga.",
    ),
    "horizon": (
        "Horizonte D+h",
        "N\u00famero de sesiones (barras de trading, no d\u00edas naturales) que se mantiene la operaci\u00f3n. Base "
        "ejecutable: la se\u00f1al se decide al cierre de t, se ENTRA al open de t+1 y se mide hasta el cierre "
        "de t+h. D+25 = entrar en el open siguiente y salir 25 sesiones despu\u00e9s (retorno = close[t+25] \u2212 open[t+1]).",
    ),
    "n_vintages": (
        "Vintages",
        "N\u00famero de contratos-a\u00f1o distintos (p.ej. el spread Dic-Mar de 2018, 2019, 2020\u2026) con datos "
        "suficientes. Es la MUESTRA EFECTIVA: 15 vintages es evidencia mucho m\u00e1s independiente que "
        "'50.000 filas', porque los contratos sint\u00e9ticos solapados est\u00e1n muy correlacionados.",
    ),
    "n_folds": (
        "Folds OOS",
        "N\u00famero de particiones walk-forward fuera de muestra: cada fold entrena con vintages anteriores y "
        "prueba en el siguiente. M\u00e1s folds = el resultado se ha reconfirmado m\u00e1s veces en datos no vistos.",
    ),
    "ic_mean": (
        "IC medio",
        "Coeficiente de Informaci\u00f3n: correlaci\u00f3n de Spearman (por rangos) entre la feature en t y el "
        "retorno futuro, promediada entre folds. Rango \u22121..+1. Signo: negativo = reversi\u00f3n, positivo = "
        "continuaci\u00f3n. Magnitud: en finanzas |IC|\u22480,03-0,05 ya es \u00fatil; |IC|>0,15 es una se\u00f1al fuerte.",
    ),
    "ic_t": (
        "IC t-stat",
        "El IC medio dividido por su error est\u00e1ndar entre folds: mide la ESTABILIDAD (no el tama\u00f1o) del "
        "efecto. |t|\u22652 empieza a ser fiable, |t|\u22653 s\u00f3lido. Un IC alto con |t| bajo depende de pocos a\u00f1os.",
    ),
    "gross_spread": (
        "Spread bruto",
        "Retorno medio (en puntos de precio) del bucket m\u00e1s barato menos el m\u00e1s caro, sin costes. "
        "Positivo = comprar barato y vender caro habr\u00eda ganado.",
    ),
    "net_spread": (
        "Spread neto",
        "El spread bruto tras restar el coste de transacci\u00f3n por familia. Es la ventaja que quedar\u00eda "
        "DESPU\u00c9S de costes. Vac\u00edo = alg\u00fan bucket se qued\u00f3 sin datos en los folds (no se inventa un valor).",
    ),
    "monotonicity": (
        "Monotonicidad",
        "Correlaci\u00f3n de Spearman entre el n\u00famero de bucket (barato\u2192caro) y su retorno medio. \u22121 = "
        "perfectamente ordenado (cuanto m\u00e1s caro, peor: reversi\u00f3n limpia); +1 = lo contrario; 0 = sin orden.",
    ),
    "bucket": (
        "Bucket",
        "Se ordenan las observaciones por el valor de la feature y se parten en 5 grupos (B1 = m\u00e1s bajo/barato, "
        "B5 = m\u00e1s alto/caro). Los cortes se fijan con el train y se aplican al test (out-of-sample).",
    ),
    "cost": (
        "Coste",
        "Coste de transacci\u00f3n por familia (bid/offer + slippage) restado al spread. Hoy es un valor "
        "placeholder por familia; pendiente de calibrar con datos reales de ejecuci\u00f3n.",
    ),
    "mfe_mae": (
        "MFE / MAE",
        "M\u00e1xima excursi\u00f3n favorable / adversa: el mejor y el peor punto que alcanza el precio dentro de la "
        "ventana del horizonte. Miden recorrido y riesgo, no solo el retorno al cierre de D+h.",
    ),
    "n_trades": (
        "N trades",
        "N\u00famero de operaciones NO solapadas generadas fuera de muestra: cada vez que la feature cae en el "
        "bucket extremo (barato o caro) se abre una posici\u00f3n y se bloquea el contrato durante el horizonte, "
        "as\u00ed dos trades nunca se pisan. M\u00e1s trades = estad\u00edstica m\u00e1s robusta.",
    ),
    "win_rate": (
        "Win rate",
        "Fracci\u00f3n de esos trades que acaban en positivo tras coste (0..1). Ojo: un win rate alto con P&L medio "
        "peque\u00f1o puede perder si las pocas p\u00e9rdidas son grandes; l\u00e9elo junto al P&L medio y al MAE.",
    ),
    "avg_pnl": (
        "P&L medio",
        "Beneficio medio por trade en PUNTOS de precio, ya descontado el coste de la familia. El lado de la "
        "operaci\u00f3n (largo el barato / corto el caro) lo fija el signo del IC en el train, nunca el test.",
    ),
    "sharpe": (
        "Sharpe",
        "Sharpe en TIEMPO-CALENDARIO: se contabiliza cada trade en su fecha de entrada, se suman los trades "
        "simult\u00e1neos, se rellenan los d\u00edas ociosos con 0 y se anualiza la serie diaria por \u221a252. As\u00ed refleja "
        "el solapamiento intrad\u00eda y la intensidad real de operaci\u00f3n (no el antiguo \u221a(252/h) por trade). >1 es bueno.",
    ),
    "mfe_mean": (
        "MFE medio",
        "Excursi\u00f3n favorable media por trade, orientada al lado de la operaci\u00f3n (para un corto, lo favorable "
        "es que el precio baje). Cu\u00e1nto recorrido a favor ofrece la se\u00f1al de media, en puntos. Suele ser \u2265 0.",
    ),
    "mae_mean": (
        "MAE medio",
        "Excursi\u00f3n adversa media por trade, orientada al lado de la operaci\u00f3n: cu\u00e1nto llega a ir en contra "
        "de media (drawdown intra-trade), en puntos. Suele ser \u2264 0; \u00fatil para dimensionar el stop.",
    ),
    "p_reversion": (
        "P(reversi\u00f3n)",
        "Probabilidad fuera de muestra de que, estando la feature en el bucket extremo, el precio se mueva HACIA "
        "la media (barato\u2192sube, caro\u2192baja) en el horizonte. >0,5 = sesgo reversivo. Usa el signo del retorno.",
    ),
    "p_continuation": (
        "P(continuaci\u00f3n)",
        "Complemento de P(reversi\u00f3n): probabilidad de que el precio siga alej\u00e1ndose de la media desde el "
        "extremo. >0,5 = el extremo tiende a prolongarse (continuaci\u00f3n).",
    ),
    "confidence": (
        "Confianza",
        "\u00cdndice 0-100 que combina (multiplicando) muestra, estabilidad y consistencia: sample=folds/(folds+3), "
        "stability=min(|t|/3,1), consistencia de signo entre folds, y penalizaci\u00f3n por trades "
        "(min(N/100,1); N=0 \u2192 factor 0). Un IC alto pero inestable, cambiante de signo o que NO llega a operar da confianza baja o nula.",
    ),
    "fwd_vol": (
        "Retorno normalizado",
        "Retorno futuro dividido por la volatilidad conocida en t (ATR close-to-close) y por \u221ah, para que sea "
        "comparable entre contratos y reg\u00edmenes de volatilidad.",
    ),
    "ic_incremental": (
        "IC incremental",
        "IC del residuo de la feature tras quitar la parte explicada linealmente por el representante de su "
        "cl\u00faster. Cercano a 0 = no aporta se\u00f1al m\u00e1s all\u00e1 de su l\u00edder (redundante).",
    ),
    "verdict": (
        "Veredicto",
        "representante = l\u00edder del cl\u00faster (mayor |t|); redundante = correlacionada y con IC incremental \u2248 0; "
        "aporta = correlacionada pero a\u00fan a\u00f1ade se\u00f1al independiente.",
    ),
    "max_abs_corr": (
        "|\u03c1| m\u00e1x",
        "M\u00e1xima correlaci\u00f3n absoluta (Spearman) de la feature con otra de su mismo cl\u00faster.",
    ),
    "subgroup": (
        "Subgrupo",
        "IC descriptivo de la feature titular dentro de una porci\u00f3n de los datos (era, r\u00e9gimen de vol, fase de "
        "vida, mes), con su N. Sirve para ver si la ventaja est\u00e1 repartida o concentrada en una sola porci\u00f3n.",
    ),
    "occupancy": (
        "Ocupaci\u00f3n",
        "Porcentaje de barras que el mercado pasa en ese r\u00e9gimen (rango / zona muerta / tendencia), "
        "seg\u00fan los terciles de Efficiency Ratio.",
    ),
    "mean_run": (
        "Racha media",
        "Duraci\u00f3n media (en barras) de una permanencia continua en ese r\u00e9gimen dentro de un mismo contrato. "
        "Rachas largas = reg\u00edmenes persistentes; cortas = mercado que parpadea.",
    ),
    "up_rate": (
        "P(sube)",
        "Proporci\u00f3n de barras del r\u00e9gimen cuyo retorno futuro (al horizonte) fue positivo.",
    ),
    "abs_fwd": (
        "|Mov.| medio",
        "Magnitud t\u00edpica del movimiento futuro (media del valor absoluto del retorno) en ese r\u00e9gimen, en puntos.",
    ),
    "transition": (
        "Transici\u00f3n",
        "Probabilidad de pasar de un r\u00e9gimen al del d\u00eda siguiente (misma serie de contrato). "
        "La diagonal alta = reg\u00edmenes que se mantienen.",
    ),
    "direction": (
        "Lado (largo/corto)",
        "Desglose de la operativa en su pata larga y su pata corta: N, win rate y P&L medio de cada lado. "
        "Una ventaja sim\u00e9trica funciona en ambos; una unilateral solo en uno.",
    ),
    "er_bin": (
        "Quintil de ER",
        "Quintil del Efficiency Ratio (1 = m\u00e1s choppy, 5 = tendencia m\u00e1s limpia). Sirve para ver si la "
        "reversi\u00f3n mejora con ER bajo y la continuaci\u00f3n con ER alto.",
    ),
    "gross_pnl": (
        "P&L bruto",
        "P&L medio por trade antes de costes (en puntos). Es tambi\u00e9n el coste de equilibrio (break-even).",
    ),
    "breakeven_cost": (
        "Coste de equilibrio",
        "Coste round-trip (puntos) al que el P&L medio por trade se anula. Igual al P&L bruto.",
    ),
    "safety_margin": (
        "Margen de seguridad",
        "Coste de equilibrio dividido por el coste stub actual: cu\u00e1ntas veces el coste placeholder puede "
        "absorber la ventaja antes de morir. >1 = sobrevive al stub; <1 = el stub ya la mata.",
    ),
    "trades_per_year": (
        "Trades / a\u00f1o",
        "N\u00famero de operaciones por a\u00f1o (aprox.), como medida de rotaci\u00f3n y capacidad.",
    ),
    "holding_days": (
        "Tenencia (d\u00edas)",
        "Periodo de mantenimiento de cada trade, igual al horizonte de evaluaci\u00f3n.",
    ),
    "cost_sens": (
        "Sensibilidad a coste",
        "Sharpe y P&L medio de la estrategia a 0\u00d7 / 1\u00d7 / 2\u00d7 / 3\u00d7 el coste stub. Muestra c\u00f3mo se "
        "degrada la ventaja al encarecer la ejecuci\u00f3n.",
    ),
    "regime": (
        "R\u00e9gimen",
        "Estado de mercado clasificado con el Efficiency Ratio (er_20): 'rango' = tercil bajo de er (lateral, "
        "propicio a reversi\u00f3n); 'tendencia' = tercil alto (direccional, propicio a continuaci\u00f3n). El tercio "
        "central es ZONA MUERTA y no se opera. Los cortes se fijan con el train de cada fold.",
    ),
    "range_feature": (
        "Feature en rango",
        "La mejor feature de reversi\u00f3n (IC m\u00e1s negativo en el train) elegida para operar el r\u00e9gimen de rango.",
    ),
    "trend_feature": (
        "Feature en tendencia",
        "La mejor feature de continuaci\u00f3n (IC m\u00e1s positivo en el train) elegida para operar el r\u00e9gimen de tendencia.",
    ),
    "confirm": (
        "Confirmaci\u00f3n por nivel",
        "Filtro extra con level_z (caro/barato vs contratos an\u00e1logos): en reversi\u00f3n solo se opera con 'doble "
        "barato/caro' (la se\u00f1al y el nivel coinciden); en continuaci\u00f3n solo si el precio NO est\u00e1 ya en el "
        "extremo contrario (anti-chasing). Umbral por terciles del train. Reduce trades a cambio de calidad.",
    ),
    "points": (
        "Puntos",
        "Todo el P&L se mide en PUNTOS de precio (diferencias de close), no en %, porque un spread/crack/fly "
        "puede ser negativo y cruzar cero, donde el % no tiene sentido.",
    ),
}


def _flabel(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def _th(key: str, label: str | None = None) -> str:
    """A table header cell with a hover tooltip from METRIC_DEFS."""
    lbl, tip = METRIC_DEFS.get(key, (label or key, ""))
    if label:
        lbl = label
    tip = tip.replace('"', "&quot;")
    return f'<th title="{tip}">{lbl}</th>'


# --------------------------------------------------------------------------- #
# Small formatters
# --------------------------------------------------------------------------- #
_EMDASH = "\u2014"


def _num(v, dp: int = 2) -> str:
    return "" if pd.isna(v) else f"{v:.{dp}f}"


def _kpi(value: str, label: str) -> str:
    return f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>'


def _div(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config=_PLOTLY_CFG)


def _is_edge(row: pd.Series) -> bool:
    strong_ic = abs(row["ic_t"]) >= 3
    economic = row["net_spread"] > 0 if not pd.isna(row["net_spread"]) else False
    # When the multiple-testing q-value is available, require FDR significance so
    # a large |ic_t| that is just one of hundreds of trials is not flagged.
    survives_fdr = True
    if "ic_q" in row.index and not pd.isna(row.get("ic_q")):
        survives_fdr = row["ic_q"] < 0.10
    return strong_ic and economic and survives_fdr


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _base_layout(title: str, x_title: str = "", y_title: str = "") -> dict:
    return dict(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(color=ACCENT, size=17)),
        template="plotly_dark", paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND,
        font=dict(color=TEXT, family="Arial"),
        margin=dict(l=60, r=30, t=56, b=70), showlegend=False,
        xaxis=dict(title=x_title, gridcolor=CHART_GRID, showline=True, linecolor=ACCENT, linewidth=1),
        yaxis=dict(title=y_title, gridcolor=CHART_GRID, showline=True, linecolor=ACCENT, linewidth=1,
                   zeroline=True, zerolinecolor=CHART_GRID),
    )


def _heatmap(results: pd.DataFrame, horizon: int) -> go.Figure:
    """Feature x family heat-map of mean OOS IC (emerald = reversion, red = continuation)."""
    sub = results[(results["group"] == "ALL") & (results["horizon"] == horizon)]
    pivot = sub.pivot_table(index="feature", columns="family", values="ic_mean", aggfunc="mean")
    families = [k for k in FAMILY_LABELS if k in pivot.columns]
    order = [f for _, _, feats in FEATURE_CATEGORIES for f in feats if f in pivot.index]
    pivot = pivot.reindex(index=order, columns=families)

    fig = go.Figure(go.Heatmap(
        z=pivot.to_numpy(),
        x=[FAMILY_LABELS[c] for c in pivot.columns],
        y=[_flabel(i) for i in pivot.index],
        zmid=0,
        colorscale=[[0.0, ACCENT], [0.5, SURFACE_2], [1.0, BEAR]],
        colorbar=dict(title="IC", tickfont=dict(color=TEXT)),
        hovertemplate="%{y}<br>%{x}<br>IC medio: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"IC medio fuera de muestra por feature y familia (D+{horizon})",
                   x=0.5, xanchor="center", font=dict(color=ACCENT, size=17)),
        template="plotly_dark", paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND,
        font=dict(color=TEXT, family="Arial"),
        margin=dict(l=200, r=30, t=56, b=110), height=max(420, 26 * len(pivot) + 170),
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def _bucket_range_labels(edges, n_buckets: int) -> list[str]:
    """Human-readable value range per bucket from the quantile cut-offs."""
    e = [float(v) for v in edges]
    labels = []
    for b in range(n_buckets):
        if b == 0:
            labels.append(f"\u2264 {e[1]:.2f}")
        elif b == n_buckets - 1:
            labels.append(f"> {e[-2]:.2f}")
        else:
            labels.append(f"{e[b]:.2f} \u2013 {e[b + 1]:.2f}")
    return labels


def _bucket_chart(profile: list[float], feature: str, edges=None) -> go.Figure:
    """Mean forward outcome per feature bucket (barato -> caro), signed colours."""
    y = [float(v) for v in profile]
    x = [f"B{i + 1}" for i in range(len(y))]
    colors = [ACCENT if v >= 0 else BEAR for v in y]
    ranges = (
        _bucket_range_labels(edges, len(y))
        if edges is not None and len(edges) == len(y) + 1
        else ["" for _ in y]
    )
    fig = go.Figure(go.Bar(
        x=x, y=y, marker_color=colors, marker_line_width=0,
        customdata=ranges,
        hovertemplate="Bucket %{x}<br>Rango feature: %{customdata}<br>Retorno medio: %{y:.3f} pts<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        f"Retorno futuro por bucket \u00b7 {_flabel(feature)}",
        x_title="Bucket de la feature (bajo \u2192 alto)", y_title="Retorno medio (pts)"))
    fig.update_layout(height=300)
    return fig


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _feature_table(sub: pd.DataFrame) -> str:
    """Ranked feature table for one family at the headline horizon."""
    ranked = sub.reindex(sub["ic_t"].abs().sort_values(ascending=False).index)
    has_prob = "p_reversion" in sub.columns
    rows = []
    for _, r in ranked.iterrows():
        col = ACCENT if _is_edge(r) else SUBTEXT
        edge_kind = "reversi\u00f3n" if r["ic_mean"] < 0 else "continuaci\u00f3n"
        extra = ""
        if has_prob:
            prev = "" if pd.isna(r.get("p_reversion")) else f"{r['p_reversion'] * 100:.0f}%"
            extra = f"<td>{prev}</td><td>{_num(r.get('confidence'), 0)}</td>"
        rows.append(
            "<tr>"
            f"<td style='color:{col}'>{_flabel(r['feature'])}</td>"
            f"<td>{edge_kind}</td>"
            f"<td>{int(r['n_vintages'])}</td>"
            f"<td>{int(r['n_folds'])}</td>"
            f"<td>{_num(r['ic_mean'], 3)}</td>"
            f"<td>{_num(r['ic_t'])}</td>"
            f"<td>{_num(r['net_spread'], 3)}</td>"
            f"<td>{_num(r['monotonicity'])}</td>"
            f"{extra}"
            "</tr>"
        )
    prob_head = _th("p_reversion", "P(rev)") + _th("confidence", "Confianza") if has_prob else ""
    head = (
        "<tr>"
        + _th("feature") + _th("sesgo") + _th("n_vintages", "Vintages") + _th("n_folds", "Folds OOS")
        + _th("ic_mean", "IC medio") + _th("ic_t", "IC t-stat") + _th("net_spread", "Spread neto (pts)")
        + _th("monotonicity", "Monoton.")
        + prob_head
        + "</tr>"
    )
    return f'<table class="cw-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


def _trade_table(sub: pd.DataFrame) -> str:
    """Risk/reward table for one family: trades, win rate, P&L, Sharpe, MFE/MAE."""
    ranked = sub.reindex(sub["ic_t"].abs().sort_values(ascending=False).index)
    rows = []
    for _, r in ranked.iterrows():
        if pd.isna(r.get("n_trades")) or int(r["n_trades"]) == 0:
            continue
        pnl = r["avg_pnl"]
        col = ACCENT if (not pd.isna(pnl) and pnl > 0) else SUBTEXT
        wr = "" if pd.isna(r["win_rate"]) else f"{r['win_rate'] * 100:.0f}%"
        rows.append(
            "<tr>"
            f"<td style='color:{col}'>{_flabel(r['feature'])}</td>"
            f"<td>{int(r['n_trades'])}</td>"
            f"<td>{wr}</td>"
            f"<td>{_num(r['avg_pnl'], 3)}</td>"
            f"<td>{_num(r['sharpe'])}</td>"
            f"<td>{_num(r['mfe_mean'], 3)}</td>"
            f"<td>{_num(r['mae_mean'], 3)}</td>"
            "</tr>"
        )
    if not rows:
        return '<p class="rule">Sin trades fuera de muestra suficientes para esta familia.</p>'
    head = (
        "<tr>"
        + _th("feature") + _th("n_trades", "N trades") + _th("win_rate", "Win rate")
        + _th("avg_pnl", "P&amp;L medio (pts)") + _th("sharpe", "Sharpe")
        + _th("mfe_mean", "MFE medio (pts)") + _th("mae_mean", "MAE medio (pts)")
        + "</tr>"
    )
    return f'<table class="cw-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


def _bucket_ranges_table(sub: pd.DataFrame, n_buckets: int = 5) -> str:
    """Value cut-offs of every bucket, per feature (quintiles on the sample)."""
    order = [f for _, _, feats in FEATURE_CATEGORIES for f in feats]
    sub = sub.set_index("feature")
    rows = []
    for feat in order:
        if feat not in sub.index:
            continue
        edges = sub.loc[feat, "bucket_edges"]
        if not isinstance(edges, (list, tuple)) or len(edges) != n_buckets + 1:
            continue
        labels = _bucket_range_labels(edges, n_buckets)
        cells = "".join(f"<td>{lab}</td>" for lab in labels)
        rows.append(f"<tr><td>{_flabel(feat)}</td>{cells}</tr>")
    if not rows:
        return '<p class="rule">Sin rangos disponibles.</p>'
    _, btip = METRIC_DEFS["bucket"]
    btip = btip.replace('"', "&quot;")
    head = (
        "<tr>" + _th("feature")
        + "".join(f'<th title="{btip}">B{i + 1}</th>' for i in range(n_buckets)) + "</tr>"
    )
    return (
        '<p class="rule">B1 = valor m\u00e1s bajo de la feature (barato/sobreventa), '
        'B5 = m\u00e1s alto (caro/sobrecompra). Cortes = cuantiles sobre la muestra evaluada.</p>'
        f'<table class="cw-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'
    )


def _horizon_table(sub_all_h: pd.DataFrame, horizons: list[int]) -> str:
    """IC medio por feature x horizonte (D+20 / D+25 / D+30)."""
    pivot = sub_all_h.pivot_table(index="feature", columns="horizon", values="ic_mean", aggfunc="mean")
    order = [f for _, _, feats in FEATURE_CATEGORIES for f in feats if f in pivot.index]
    pivot = pivot.reindex(index=order, columns=[h for h in horizons if h in pivot.columns])
    rows = []
    for feat, r in pivot.iterrows():
        cells = "".join(f"<td>{_num(r[h], 3)}</td>" for h in pivot.columns)
        rows.append(f"<tr><td>{_flabel(feat)}</td>{cells}</tr>")
    _, htip = METRIC_DEFS["horizon"]
    _, ictip = METRIC_DEFS["ic_mean"]
    dh = f"{htip} {ictip}".replace('"', "&quot;")
    head = (
        "<tr>" + _th("feature")
        + "".join(f'<th title="{dh}">D+{h}</th>' for h in pivot.columns) + "</tr>"
    )
    return f'<table class="cw-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


def _gated_table(gated: pd.DataFrame) -> str:
    """Assembled gated strategy per family: which features, and its risk/reward."""
    if gated is None or gated.empty:
        return '<p class="rule">Sin estrategia gated disponible.</p>'
    ranked = gated.reindex(gated["sharpe"].fillna(-np.inf).sort_values(ascending=False).index)
    has_raw = "sharpe_raw" in gated.columns
    rows = []
    for pos, (_, r) in enumerate(ranked.iterrows(), start=1):
        pnl = r["avg_pnl"]
        col = ACCENT if (not pd.isna(pnl) and pnl > 0) else SUBTEXT
        wr = "" if pd.isna(r["win_rate"]) else f"{r['win_rate'] * 100:.0f}%"
        raw = ""
        if has_raw:
            n_raw = "" if pd.isna(r.get("n_trades_raw")) else f"{int(r['n_trades_raw'])}"
            wr_raw = "" if pd.isna(r.get("win_rate_raw")) else f"{r['win_rate_raw'] * 100:.0f}%"
            raw = f"<td class='sep'>{n_raw}</td><td>{wr_raw}</td><td>{_num(r.get('sharpe_raw'))}</td>"
        rows.append(
            "<tr>"
            f"<td>{pos}</td>"
            f"<td>{FAMILY_LABELS.get(r['family'], r['family'])}</td>"
            f"<td style='color:{col}'>{_flabel(r['range_feature']) if r['range_feature'] else _EMDASH}</td>"
            f"<td>{_flabel(r['trend_feature']) if r['trend_feature'] else _EMDASH}</td>"
            f"<td>{int(r['n_trades'])}</td>"
            f"<td>{wr}</td>"
            f"<td>{_num(r['avg_pnl'], 3)}</td>"
            f"<td>{_num(r['sharpe'])}</td>"
            f"<td>{_num(r['mfe_mean'], 3)}</td>"
            f"<td>{_num(r['mae_mean'], 3)}</td>"
            f"{raw}"
            "</tr>"
        )
    raw_head = (
        _th("n_trades", "N (sin filtro)") + _th("win_rate", "Win (sin filtro)")
        + _th("sharpe", "Sharpe (sin filtro)")
        if has_raw else ""
    )
    if raw_head:  # mark the first raw column as the visual separator
        raw_head = raw_head.replace("<th", '<th class="sep"', 1)
    head = (
        "<tr><th>#</th><th title='Familia de instrumentos.'>Familia</th>"
        + _th("range_feature", "Rango \u2192 feature") + _th("trend_feature", "Tendencia \u2192 feature")
        + _th("n_trades", "N trades") + _th("win_rate", "Win rate")
        + _th("avg_pnl", "P&amp;L medio (pts)") + _th("sharpe", "Sharpe")
        + _th("mfe_mean", "MFE medio") + _th("mae_mean", "MAE medio")
        + raw_head
        + "</tr>"
    )
    return f'<table class="cw-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


def _regime_diag_table(reg_fam: pd.DataFrame) -> str:
    """Per-feature IC / Sharpe / trades side by side for 'rango' vs 'tendencia'."""
    if reg_fam is None or reg_fam.empty:
        return '<p class="rule">Sin diagn\u00f3stico por r\u00e9gimen para esta familia.</p>'
    order = [f for _, _, feats in FEATURE_CATEGORIES for f in feats]
    by_feat = {feat: sub.set_index("regime") for feat, sub in reg_fam.groupby("feature")}
    rows = []
    for feat in order:
        sub = by_feat.get(feat)
        if sub is None:
            continue

        def cell(regime: str, col: str, dp: int = 3) -> str:
            return _num(sub.loc[regime, col], dp) if regime in sub.index else ""

        rows.append(
            "<tr>"
            f"<td>{_flabel(feat)}</td>"
            f"<td>{cell('range', 'ic_mean')}</td>"
            f"<td>{cell('range', 'sharpe', 2)}</td>"
            f"<td>{cell('range', 'n_trades', 0) if 'range' in sub.index and not pd.isna(sub.loc['range', 'n_trades']) else ''}</td>"
            f"<td class='sep'>{cell('trend', 'ic_mean')}</td>"
            f"<td>{cell('trend', 'sharpe', 2)}</td>"
            f"<td>{cell('trend', 'n_trades', 0) if 'trend' in sub.index and not pd.isna(sub.loc['trend', 'n_trades']) else ''}</td>"
            "</tr>"
        )
    if not rows:
        return '<p class="rule">Sin diagn\u00f3stico por r\u00e9gimen para esta familia.</p>'
    _, ictip = METRIC_DEFS["ic_mean"]
    ictip = ictip.replace('"', "&quot;")
    head = (
        "<tr>" + _th("feature")
        + f'<th title="{ictip}" colspan="3" style="text-align:center">R\u00e9gimen rango</th>'
        + f'<th title="{ictip}" colspan="3" style="text-align:center">R\u00e9gimen tendencia</th></tr>'
        + "<tr><th></th><th>IC</th><th>Sharpe</th><th>N</th>"
        + '<th class="sep">IC</th><th>Sharpe</th><th>N</th></tr>'
    )
    return f'<table class="cw-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


def _ranking_table(headline: pd.DataFrame) -> str:
    """Best feature per family, ranked by |IC t-stat|."""
    best = headline.reindex(headline["ic_t"].abs().sort_values(ascending=False).index)
    best = best.groupby("family", sort=False).head(1)
    best = best.reindex(best["ic_t"].abs().sort_values(ascending=False).index)
    rows = []
    has_prob = "p_reversion" in best.columns
    for pos, (_, r) in enumerate(best.iterrows(), start=1):
        edge_kind = "reversi\u00f3n" if r["ic_mean"] < 0 else "continuaci\u00f3n"
        extra = ""
        if has_prob:
            prev = "" if pd.isna(r.get("p_reversion")) else f"{r['p_reversion'] * 100:.0f}%"
            extra = f"<td>{prev}</td><td>{_num(r.get('confidence'), 0)}</td>"
        rows.append(
            "<tr>"
            f"<td>{pos}</td>"
            f"<td>{FAMILY_LABELS.get(r['family'], r['family'])}</td>"
            f"<td>{_flabel(r['feature'])}</td>"
            f"<td>{edge_kind}</td>"
            f"<td>{_num(r['ic_mean'], 3)}</td>"
            f"<td>{_num(r['ic_t'])}</td>"
            f"<td>{_num(r['net_spread'], 3)}</td>"
            f"<td>{int(r['n_vintages'])}</td>"
            f"{extra}"
            "</tr>"
        )
    prob_head = _th("p_reversion", "P(rev)") + _th("confidence", "Confianza") if has_prob else ""
    head = (
        "<tr><th>#</th><th title='Familia de instrumentos (outrights, spreads, cracks\u2026).'>Familia</th>"
        + _th("feature", "Mejor feature") + _th("sesgo") + _th("ic_mean", "IC medio")
        + _th("ic_t", "IC t-stat") + _th("net_spread", "Spread neto") + _th("n_vintages", "Vintages")
        + prob_head
        + "</tr>"
    )
    return f'<table class="cw-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


# --------------------------------------------------------------------------- #
# Prose blocks
# --------------------------------------------------------------------------- #
def _verdict(best: pd.Series) -> str:
    edge_kind = "reversi\u00f3n" if best["ic_mean"] < 0 else "continuaci\u00f3n"
    if _is_edge(best):
        return (
            f"Ventaja de <b>{edge_kind}</b> clara y estable: <b>{_flabel(best['feature'])}</b> con "
            f"IC {best['ic_mean']:+.3f} (t={best['ic_t']:+.1f}), spread neto {_num(best['net_spread'], 3)} pts "
            f"y monotonicidad {best['monotonicity']:+.2f}, sobre {int(best['n_vintages'])} vintages y "
            f"{int(best['n_folds'])} folds fuera de muestra."
        )
    return (
        f"Sin ventaja estad\u00edstica robusta: el mejor indicador (<b>{_flabel(best['feature'])}</b>) "
        f"solo llega a t={best['ic_t']:+.1f} y spread neto {_num(best['net_spread'], 3)} pts."
    )


def _methodology_html() -> str:
    return f"""
<div class="card">
  <h3>C\u00f3mo leer este informe</h3>
  <p class="rule">Backtest <b>predictivo</b> con <b>base ejecutable</b>: no puntuamos el P&amp;L de una regla long/flat,
  sino si cada se\u00f1al conocida al cierre de <b>t</b> (feature) anticipa el <b>resultado posterior</b>. Clave: la se\u00f1al
  se decide al cierre de t pero la operaci\u00f3n se rellena en el <b>open de t+1</b> (nadie puede operar el cierre que
  acaba de imprimir), as\u00ed que el retorno se mide desde ese open \u2014 sin el look-ahead del mismo cierre.</p>
  <div class="trade">
    <div class="trade-row"><span class="tag tag-in">FEATURE (\u2264 t)</span><span>Valor continuo del indicador
    al cierre de t: z-score, RSI, nivel del panel de an\u00e1logos, etc. Nunca usa informaci\u00f3n futura.</span></div>
    <div class="trade-row"><span class="tag tag-out">RESULTADO (entra en open[t+1])</span><span>Retorno ejecutable en
    puntos a D+20/25/30 (close[t+h] \u2212 open[t+1]), m\u00e1s excursi\u00f3n favorable/adversa. Es la etiqueta; solo existe en backtest.</span></div>
  </div>
  <ul class="notes">
    <li><b>IC (coef. de informaci\u00f3n)</b>: correlaci\u00f3n de Spearman entre la feature y el retorno futuro.
    Negativo = <span style="color:{ACCENT}">reversi\u00f3n</span> (m\u00e1s caro hoy \u2192 peor despu\u00e9s);
    positivo = <span style="color:{BEAR}">continuaci\u00f3n</span>.</li>
    <li><b>IC t-stat</b>: estabilidad del IC a lo largo de los folds. |t| \u2265 3 = se\u00f1al consistente, no un a\u00f1o suelto.</li>
    <li><b>Walk-forward por vintage</b>: se entrena con vintages antiguos y se prueba con el siguiente, avanzando.
    Partir por vintage (no por filas al azar) evita que contratos sint\u00e9ticos solapados filtren entre train y test.</li>
    <li><b>Spread neto</b>: retorno del bucket barato menos el caro, en puntos, con el <b>coste</b> por familia ya descontado.
    Positivo = la reversi\u00f3n paga tras costes.</li>
    <li><b>Muestra efectiva</b>: reportamos <b>vintages</b> y <b>folds</b>, no filas, para no confundir muchos
    contratos correlacionados con evidencia independiente.</li>
    <li><b>Coste</b>: hoy es un <i>stub</i> por familia (placeholder); pendiente calibrar con bid/offer real.</li>
    <li><b>Sharpe en tiempo-calendario</b>: el P&amp;L de cada trade se contabiliza en su fecha de entrada, se suman
    los trades simult\u00e1neos y se anualiza la serie diaria por \u221a252 (honesto con el solapamiento y la intensidad real).</li>
    <li><b>Multiple-testing (FDR)</b>: como se eval\u00faan cientos de celdas (familia \u00d7 feature \u00d7 horizonte), un
    <code>|t|\u22653</code> suelto no basta; se controla la tasa de falsos descubrimientos (Benjamini-Hochberg, columna
    <i>ic_q</i>) y solo se marca como ventaja lo que sobrevive a <code>q&lt;0,10</code>.</li>
    <li><b>Riesgo / recompensa</b>: adem\u00e1s del IC, cada feature se convierte en operativa \u2014 trades
    <b>no solapados</b> en el bucket extremo, con el lado fijado por el signo del IC en train \u2014 y se reporta
    <b>N trades</b>, <b>win rate</b>, <b>P&amp;L medio</b>, <b>Sharpe</b> y <b>MFE/MAE medio</b>
    (recorrido a favor / en contra), todo en puntos y fuera de muestra.</li>
  </ul>
</div>"""


def _metric_glossary_html() -> str:
    """Definition list of every metric / value used, with how to read it."""
    order = [
        "ic_mean", "ic_t", "gross_spread", "net_spread", "monotonicity",
        "n_vintages", "n_folds", "bucket", "sesgo", "horizon", "cost",
        "p_reversion", "p_continuation", "confidence", "fwd_vol",
        "n_trades", "win_rate", "avg_pnl", "sharpe", "mfe_mean", "mae_mean", "mfe_mae",
        "regime", "range_feature", "trend_feature", "confirm", "points",
        "verdict", "ic_incremental", "max_abs_corr", "subgroup",
        "occupancy", "mean_run", "up_rate", "abs_fwd", "transition",
        "direction", "er_bin",
        "gross_pnl", "breakeven_cost", "safety_margin", "trades_per_year",
        "holding_days", "cost_sens",
    ]
    items = "".join(
        f"<li><b>{METRIC_DEFS[k][0]}</b> \u2014 {METRIC_DEFS[k][1]}</li>" for k in order
    )
    return (
        '<div class="card"><h3>Glosario de m\u00e9tricas</h3>'
        '<p class="rule">Qu\u00e9 significa cada n\u00famero de las tablas y c\u00f3mo interpretarlo. '
        '(Tambi\u00e9n aparece al pasar el rat\u00f3n por las cabeceras de cada tabla.)</p>'
        f'<ul class="notes">{items}</ul></div>'
    )


def _feature_glossary_html() -> str:
    blocks = []
    for title, desc, feats in FEATURE_CATEGORIES:
        items = "".join(
            f"<li><b>{_flabel(f)}</b> \u2014 {FEATURE_DESCRIPTIONS.get(f, '')}</li>" for f in feats
        )
        blocks.append(f'<p class="rule" style="margin-top:14px"><b>{title}.</b> {desc}</p><ul class="notes">{items}</ul>')
    return f'<div class="card"><h3>Glosario de features</h3>{"".join(blocks)}</div>'


# --------------------------------------------------------------------------- #
# Family card
# --------------------------------------------------------------------------- #
def _family_card(family: str, headline_fam: pd.DataFrame, allh_fam: pd.DataFrame, horizons: list[int]) -> str:
    label = FAMILY_LABELS.get(family, family)
    best = headline_fam.reindex(headline_fam["ic_t"].abs().sort_values(ascending=False).index).iloc[0]
    n_edge = int(headline_fam.apply(_is_edge, axis=1).sum())

    kpis = "".join([
        _kpi(f"{int(best['n_vintages'])}", "vintages"),
        _kpi(f"{int(best['n_folds'])}", "folds OOS"),
        _kpi(_flabel(best["feature"]), "feature m\u00e1s fuerte"),
        _kpi(_num(best["net_spread"], 3), "mejor spread neto (pts)"),
        _kpi(f"{n_edge}", "features con ventaja"),
    ])

    bucket = ""
    if isinstance(best["bucket_profile"], (list, tuple)) and len(best["bucket_profile"]):
        prof = list(best["bucket_profile"])
        edges = best.get("bucket_edges")
        edges = list(edges) if isinstance(edges, (list, tuple)) else None
        chart = _div(_bucket_chart(prof, best["feature"], edges))
        caption = ""
        if edges is not None and len(edges) == len(prof) + 1:
            labels = _bucket_range_labels(edges, len(prof))
            chips = "".join(
                f'<span class="brange"><b>B{i + 1}</b> {lab}</span>'
                for i, lab in enumerate(labels)
            )
            caption = (
                f'<p class="rule" style="margin-top:8px">Rangos de <b>{_flabel(best["feature"])}</b> '
                f'por bucket (cuantiles sobre la muestra evaluada): {chips}</p>'
            )
        bucket = chart + caption

    return f"""
<div class="card" id="fam-{family}">
  <a class="top-link" href="#top">\u2191 arriba</a>
  <h3>{label}</h3>
  <div class="concl">{_verdict(best)}</div>
  <div class="kpis">{kpis}</div>
  {bucket}
  <details open><summary>Ranking de features (D+{int(headline_fam["horizon"].iloc[0]) if not headline_fam.empty else 10})</summary>{_feature_table(headline_fam)}</details>
  <details><summary>Riesgo / recompensa \u00b7 operativa por bucket extremo</summary>{_trade_table(headline_fam)}</details>
  <details><summary>Rangos de cada bucket por feature</summary>{_bucket_ranges_table(headline_fam)}</details>
  <details><summary>IC por horizonte (D+{'/'.join(str(h) for h in horizons)})</summary>{_horizon_table(allh_fam, horizons)}</details>
</div>"""


# --------------------------------------------------------------------------- #
# Sidebar + CSS
# --------------------------------------------------------------------------- #
def _sidebar(headline: pd.DataFrame) -> str:
    parts = [
        f'<div class="cw-side-brand"><div class="cw-side-mark">{BRAND_MARK_SVG}</div>'
        f'<div class="cw-side-word">Crude<span>Watch</span></div></div>',
        '<div class="cw-side-tag">Backtest walk-forward \u00b7 predictivo</div>',
        '<hr class="cw-divider">',
        '<div class="cw-nav-label">Secciones</div>',
        '<ul>'
        '<li><a href="#conclusiones"><span>Conclusiones</span></a></li>'
        '<li><a href="#metodologia"><span>C\u00f3mo leer esto</span></a></li>'
        '<li><a href="#completo"><span>Metodolog\u00eda completa</span></a></li>'
        '<li><a href="#mapa"><span>Mapa global</span></a></li>'
        '<li><a href="#ranking"><span>Ranking por familia</span></a></li>'
        '<li><a href="#detalle-bucket"><span>Detalle por bucket</span></a></li>'
        '<li><a href="#compuesto"><span>Modelo compuesto</span></a></li>'
        '<li><a href="#regime"><span>Por r\u00e9gimen</span></a></li>'
        '<li><a href="#anatomia"><span>Anatom\u00eda de reg\u00edmenes</span></a></li>'
        '<li><a href="#direccion"><span>Direcci\u00f3n y calidad</span></a></li>'
        '<li><a href="#costes"><span>Costes y operabilidad</span></a></li>'
        '<li><a href="#rejilla"><span>Rejilla condicional</span></a></li>'
        '<li><a href="#robustez"><span>Redundancia y robustez</span></a></li>'
        '<li><a href="#metricas"><span>Glosario de m\u00e9tricas</span></a></li>'
        '<li><a href="#glosario"><span>Glosario de features</span></a></li>'
        '</ul>',
        '<div class="cw-nav-label">Familias</div><ul>',
    ]
    for key, label in FAMILY_LABELS.items():
        fam = headline[headline["family"] == key]
        if fam.empty:
            continue
        best = fam.reindex(fam["ic_t"].abs().sort_values(ascending=False).index).iloc[0]
        cls = "pos" if _is_edge(best) else "neg"
        parts.append(
            f'<li><a href="#fam-{key}"><span>{label}</span>'
            f'<span class="badge {cls}">{best["ic_t"]:+.1f}</span></a></li>'
        )
    parts.append("</ul>")
    parts.append('<hr class="cw-divider">')
    parts.append('<div class="cw-side-foot">CrudeWatch \u00b7 research \u00b7 Data: CME / ICE</div>')
    parts.append('<div class="cw-side-copy">\u00a9 guiruha</div>')
    return f'<nav class="side">{"".join(parts)}</nav>'


_CSS = f"""
  * {{ box-sizing:border-box; }}
  body {{ background:{BACKGROUND}; color:{TEXT}; font-family:Arial,Helvetica,sans-serif; margin:0; font-size:15px; }}
  a {{ color:{ACCENT}; text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
  .layout {{ display:flex; align-items:flex-start; }}
  .side {{ position:sticky; top:0; height:100vh; width:300px; flex:0 0 300px; overflow-y:auto;
           background:linear-gradient(180deg,{SURFACE} 0%,{BACKGROUND} 100%);
           border-right:1px solid {BORDER}; padding:22px 16px; }}
  .cw-side-brand {{ display:flex; align-items:center; gap:10px; margin:0 0 4px 2px; }}
  .cw-side-mark {{ width:46px; height:46px; border-radius:12px; flex:none; display:flex; align-items:center;
                   justify-content:center; color:{ACCENT}; line-height:0;
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
  .side ul {{ list-style:none; margin:0 0 6px; padding:0; }}
  .side li a {{ display:flex; justify-content:space-between; align-items:center; gap:8px;
                padding:9px 12px; margin:2px 0; border-radius:9px; border:1px solid transparent;
                color:{TEXT}; font-size:13px; }}
  .side li a:hover {{ background:{SURFACE_2}; border-color:{BORDER}; text-decoration:none; }}
  .badge {{ font-size:11px; font-weight:700; padding:2px 8px; border-radius:10px; flex:0 0 auto; }}
  .badge.pos {{ background:{ACCENT}26; color:{ACCENT}; }}
  .badge.neg {{ background:{BEAR}26; color:{BEAR}; }}
  .cw-side-foot {{ color:{SUBTEXT}; font-size:10.5px; letter-spacing:.3px; margin-top:18px; opacity:.8; }}
  .cw-side-copy {{ color:{SUBTEXT}; font-size:11px; margin:8px 2px 2px; opacity:.7; }}

  .main {{ flex:1 1 auto; padding:34px 44px; max-width:1220px; min-width:0; }}
  .hero {{ background:linear-gradient(135deg,{ACCENT}1f,transparent 68%);
           border:1px solid {BORDER}; border-radius:16px; padding:28px 30px; margin-bottom:24px; }}
  .cw-title {{ color:{TEXT}; font-size:32px; font-weight:700; border-left:5px solid {ACCENT};
               padding-left:14px; margin:0 0 6px; }}
  .cw-sub {{ color:{SUBTEXT}; font-size:15px; margin:0 0 0 19px; }}
  h2.cat {{ color:{TEXT}; font-size:23px; font-weight:700; border-left:5px solid {ACCENT};
            padding-left:14px; margin:40px 0 6px; }}
  .cat-lead {{ color:{SUBTEXT}; margin:0 0 16px 19px; font-size:14px; }}
  .card {{ background:linear-gradient(180deg,{SURFACE} 0%,{BACKGROUND} 100%); border:1px solid {BORDER};
           border-radius:16px; padding:22px 26px; margin:18px 0; scroll-margin-top:16px; }}
  .card h3 {{ margin:0 0 4px; color:{TEXT}; font-size:20px; }}
  .card .rule {{ color:{SUBTEXT}; margin:6px 0 14px; font-size:13.5px; line-height:1.5; }}
  .concl {{ background:{BACKGROUND}; border-left:3px solid {ACCENT}; padding:10px 14px; margin:10px 0;
            border-radius:4px; font-size:13.5px; line-height:1.5; }}
  ul.notes {{ color:{SUBTEXT}; font-size:13px; line-height:1.6; margin:6px 0 0; padding-left:18px; }}
  ul.notes li {{ margin:3px 0; }}
  details {{ background:{BACKGROUND}; border:1px solid {BORDER}; border-radius:12px; margin:12px 0; padding:6px 14px; }}
  details[open] {{ border-color:{ACCENT}77; box-shadow:inset 3px 0 0 0 {ACCENT}; }}
  summary {{ cursor:pointer; color:{ACCENT}; font-weight:700; padding:8px 2px; }}
  table.cw-table {{ border-collapse:collapse; width:100%; margin-top:12px; font-size:13px; }}
  table.cw-table th {{ color:{ACCENT}; text-align:right; padding:6px 10px; border-bottom:1px solid {BORDER}; }}
  table.cw-table th[title] {{ cursor:help; text-decoration:underline dotted {ACCENT}88; text-underline-offset:3px; }}
  table.cw-table td {{ text-align:right; padding:5px 10px; border-bottom:1px solid {SURFACE}; }}
  table.cw-table td:first-child, table.cw-table th:first-child {{ text-align:left; }}
  table.cw-table tr:hover td {{ background:{SURFACE_2}; }}
  table.cw-table .sep {{ border-left:1px solid {BORDER}; }}
  .trade {{ background:{BACKGROUND}; border:1px solid {BORDER}; border-radius:10px; padding:14px 16px; margin:12px 0; }}
  .trade-row {{ display:flex; align-items:flex-start; gap:12px; margin-bottom:8px; }}
  .tag {{ font-size:11px; font-weight:700; letter-spacing:.5px; padding:3px 8px; border-radius:4px; flex:0 0 auto; margin-top:1px; }}
  .tag-in {{ background:{ACCENT}26; color:{ACCENT}; border:1px solid {ACCENT}; }}
  .tag-out {{ background:{BEAR}26; color:{BEAR}; border:1px solid {BEAR}; }}
  .kpis {{ display:flex; flex-wrap:wrap; gap:14px; margin:14px 0; }}
  .kpi {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px; padding:14px 18px; min-width:150px; }}
  .kpi .v {{ color:{ACCENT}; font-size:20px; font-weight:700; }}
  .kpi .l {{ color:{SUBTEXT}; font-size:11px; text-transform:uppercase; letter-spacing:.4px; }}
  .brange {{ display:inline-block; background:{BACKGROUND}; border:1px solid {BORDER}; border-radius:8px;
             padding:3px 9px; margin:3px 6px 0 0; font-size:12px; color:{TEXT}; }}
  .brange b {{ color:{ACCENT}; margin-right:5px; }}
  .subg-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px; margin-top:6px; }}
  .subg-h {{ color:{SUBTEXT}; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.6px; margin-bottom:6px; }}
  .findings {{ display:flex; flex-direction:column; gap:12px; }}
  .finding {{ display:flex; align-items:flex-start; gap:16px; padding:12px 4px; border-bottom:1px solid {BORDER}; }}
  .finding:last-child {{ border-bottom:0; }}
  .finding-stat {{ flex:0 0 96px; font-size:26px; font-weight:800; text-align:right; line-height:1.1; }}
  .finding-body {{ flex:1; color:{TEXT}; }}
  .top-link {{ float:right; font-size:12px; color:{SUBTEXT}; }}
  @media (max-width:900px) {{
    .layout {{ flex-direction:column; }}
    .side {{ position:static; height:auto; width:100%; flex:none; border-right:0; border-bottom:1px solid {BORDER}; }}
    .main {{ padding:24px 18px; }}
  }}
"""


def _regime_section(gated: pd.DataFrame | None, regime_results: pd.DataFrame | None, horizon: int) -> str:
    """Full 'por régimen' block: intro, gated strategy table, per-family diagnostic."""
    if gated is None and regime_results is None:
        return ""
    gated_h = gated[gated["horizon"] == horizon] if gated is not None and not gated.empty else None
    reg_h = (
        regime_results[regime_results["horizon"] == horizon]
        if regime_results is not None and not regime_results.empty else None
    )

    intro = (
        '<div class="card"><h3>Idea</h3>'
        '<p class="rule">La reversi\u00f3n paga en <b>rango</b> y la continuaci\u00f3n en <b>tendencia</b>; '
        'mezclarlas diluye ambas. Clasificamos cada barra con el <b>Efficiency Ratio (er_20)</b>: tercil bajo '
        '= r\u00e9gimen <b>rango</b>, tercil alto = <b>tendencia</b>, y el tercio central es <b>zona muerta</b> '
        '(no se opera, para no pisar la frontera ruidosa). Los cortes se fijan con el train de cada fold, as\u00ed '
        'que la etiqueta de r\u00e9gimen es point-in-time. La <b>estrategia gated</b> elige, en train, la mejor '
        'feature de reversi\u00f3n para el rango y la mejor de continuaci\u00f3n para la tendencia, y opera cada una '
        'en su r\u00e9gimen (trades no solapados, con coste, fuera de muestra).</p></div>'
    )

    gated_block = ""
    if gated_h is not None and not gated_h.empty:
        gated_block = (
            '<div class="card"><h3>Estrategia gated por familia</h3>'
            '<p class="rule">Feature elegida en cada r\u00e9gimen y riesgo/recompensa de la estrategia combinada, '
            f'ordenada por Sharpe (D+{horizon}). La estrategia titular a\u00f1ade <b>confirmaci\u00f3n por nivel</b> '
            '(<code>level_z</code>): reversi\u00f3n solo con doble barato/caro, continuaci\u00f3n solo si no persigue '
            'extensi\u00f3n. Las dos \u00faltimas columnas muestran la misma estrategia <b>sin</b> ese filtro, para ver '
            'si a\u00f1ade calidad (menos trades, mejor Sharpe).</p>'
            f'{_gated_table(gated_h)}</div>'
        )

    diag_block = ""
    if reg_h is not None and not reg_h.empty:
        cards = []
        for key in FAMILY_LABELS:
            fam = reg_h[reg_h["family"] == key]
            if fam.empty:
                continue
            cards.append(
                f'<details><summary>{FAMILY_LABELS.get(key, key)} \u00b7 IC/Sharpe por r\u00e9gimen</summary>'
                f'{_regime_diag_table(fam)}</details>'
            )
        if cards:
            diag_block = (
                '<div class="card"><h3>Diagn\u00f3stico feature \u00d7 r\u00e9gimen</h3>'
                '<p class="rule">Cada feature medida por separado dentro de cada r\u00e9gimen: compara la misma '
                'se\u00f1al en rango vs tendencia para ver d\u00f3nde tiene la ventaja.</p>'
                f'{"".join(cards)}</div>'
            )

    return (
        '<h2 class="cat" id="regime">Backtest por r\u00e9gimen</h2>'
        '<p class="cat-lead">Tendencia vs rango: operar la se\u00f1al adecuada en cada estado de mercado.</p>'
        f'{intro}{gated_block}{diag_block}'
    )


# --------------------------------------------------------------------------- #
# Diagnostics: redundancy (WS6) + subgroups (WS7)
# --------------------------------------------------------------------------- #
_VERDICT_STYLE = {
    "representante": (ACCENT, "representante"),
    "aporta": (TEXT, "aporta"),
    "redundante": (SUBTEXT, "redundante"),
}


def _redundancy_table(red_fam: pd.DataFrame) -> str:
    """Cluster / representative / incremental IC / verdict for one family."""
    if red_fam is None or red_fam.empty:
        return '<p class="rule">Sin datos de redundancia.</p>'
    ordered = red_fam.sort_values(["cluster", "verdict"], kind="stable")
    rows = []
    for _, r in ordered.iterrows():
        color, _ = _VERDICT_STYLE.get(r["verdict"], (TEXT, r["verdict"]))
        rep = "" if r["feature"] == r["representative"] else _flabel(r["representative"])
        rows.append(
            "<tr>"
            f"<td>{r['cluster']}</td>"
            f"<td>{_flabel(r['feature'])}</td>"
            f"<td>{rep}</td>"
            f"<td>{_num(r['max_abs_corr'])}</td>"
            f"<td>{_num(r['ic_incremental'], 3)}</td>"
            f"<td style='color:{color};font-weight:700'>{r['verdict']}</td>"
            "</tr>"
        )
    head = (
        "<tr><th title='Grupo de features con |\u03c1| por encima del umbral.'>Cl\u00faster</th>"
        + _th("feature") + _th("representative", "Representante")
        + _th("max_abs_corr", "|\u03c1| m\u00e1x") + _th("ic_incremental", "IC incremental")
        + _th("verdict", "Veredicto") + "</tr>"
    )
    return f'<table class="cw-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


_SUBGROUP_DIMS = [
    ("era", "Antes / despu\u00e9s de 2020"),
    ("vol", "R\u00e9gimen de volatilidad"),
    ("fase", "Fase de vida (d\u00edas a vencimiento)"),
    ("mes", "Estacionalidad (mes)"),
]


def _subgroup_tables(sub_fam: pd.DataFrame) -> str:
    """One small IC-by-group table per subgroup dimension for a family."""
    if sub_fam is None or sub_fam.empty:
        return '<p class="rule">Sin desglose por subgrupos.</p>'
    blocks = []
    for dim, title in _SUBGROUP_DIMS:
        part = sub_fam[sub_fam["dimension"] == dim]
        if part.empty:
            continue
        if dim == "mes":
            part = part.assign(_o=part["group"].astype(int)).sort_values("_o")
        rows = []
        for _, r in part.iterrows():
            ic = r["ic"]
            color = SUBTEXT if pd.isna(ic) else (ACCENT if ic < 0 else BEAR)
            rows.append(
                f"<tr><td>{r['group']}</td>"
                f"<td style='color:{color}'>{_num(ic, 3)}</td>"
                f"<td>{int(r['n'])}</td></tr>"
            )
        head = "<tr><th>Grupo</th>" + _th("ic_mean", "IC") + "<th title='N\u00famero de observaciones del subgrupo.'>N</th></tr>"
        blocks.append(
            f'<div class="subg"><div class="subg-h">{title}</div>'
            f'<table class="cw-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'
        )
    return f'<div class="subg-grid">{"".join(blocks)}</div>'


def _diagnostics_section(
    redundancy_results: pd.DataFrame | None,
    subgroup_results: pd.DataFrame | None,
    horizon: int,
) -> str:
    """Full 'Robustez' block: redundancy (Paso 5) + subgroups (Paso 7)."""
    if (redundancy_results is None or redundancy_results.empty) and \
       (subgroup_results is None or subgroup_results.empty):
        return ""

    red_block = ""
    if redundancy_results is not None and not redundancy_results.empty:
        cards = []
        for key in FAMILY_LABELS:
            fam = redundancy_results[redundancy_results["family"] == key]
            if fam.empty:
                continue
            cards.append(
                f'<details><summary>{FAMILY_LABELS.get(key, key)} \u00b7 cl\u00fasteres y redundancia</summary>'
                f'{_redundancy_table(fam)}</details>'
            )
        red_block = (
            '<div class="card"><h3>Redundancia entre features</h3>'
            '<p class="rule">Muchos indicadores dicen lo mismo. Agrupamos las features por correlaci\u00f3n de '
            'Spearman (|\u03c1| \u2265 0.8); en cada cl\u00faster, el <b>representante</b> es el de mayor |t|. El '
            '<b>IC incremental</b> es el IC del residuo de una feature tras quitar la parte explicada por su '
            'representante: cercano a 0 \u2192 <b>redundante</b> (no aporta se\u00f1al nueva).</p>'
            f'{"".join(cards)}</div>'
        )

    sub_block = ""
    if subgroup_results is not None and not subgroup_results.empty:
        cards = []
        for key in FAMILY_LABELS:
            fam = subgroup_results[subgroup_results["family"] == key]
            if fam.empty:
                continue
            feat = fam["feature"].iloc[0]
            cards.append(
                f'<details><summary>{FAMILY_LABELS.get(key, key)} \u00b7 {_flabel(feat)} por subgrupos</summary>'
                f'{_subgroup_tables(fam)}</details>'
            )
        sub_block = (
            '<div class="card"><h3>Robustez por subgrupos</h3>'
            f'<p class="rule">Para la feature titular de cada familia, el IC (descriptivo, D+{horizon}) dentro '
            'de cada porci\u00f3n de los datos con su N. Una ventaja repartida entre \u00e9pocas, reg\u00edmenes de '
            'volatilidad, fases de vida y meses es robusta; una concentrada en una sola porci\u00f3n (t\u00edpicamente '
            '2020) no lo es. Emerald = reversi\u00f3n (IC&lt;0), rojo = continuaci\u00f3n (IC&gt;0).</p>'
            f'{"".join(cards)}</div>'
        )

    return (
        '<h2 class="cat" id="robustez">Redundancia y robustez</h2>'
        '<p class="cat-lead">Qu\u00e9 features son independientes y si la ventaja resiste al trocear los datos.</p>'
        f'{red_block}{sub_block}'
    )


# --------------------------------------------------------------------------- #
# WS4 regime anatomy + WS5 direction/quality + WS8 costs
# --------------------------------------------------------------------------- #
_REGIME_ES = {"range": "Rango", "dead": "Zona muerta", "trend": "Tendencia"}
_REGIME_ORDER = ["range", "dead", "trend"]


def _pct(v, dp: int = 0) -> str:
    return "" if pd.isna(v) else f"{v * 100:.{dp}f}%"


def _regime_anatomy_section(
    profile: pd.DataFrame | None,
    transitions: pd.DataFrame | None,
    horizon: int,
) -> str:
    """WS4: occupancy / persistence / forward profile + transition matrix per family."""
    if profile is None or profile.empty:
        return ""
    cards = []
    for key in FAMILY_LABELS:
        fam = profile[profile["family"] == key]
        if fam.empty:
            continue
        fam = fam.set_index("regime")
        prows = []
        for reg in _REGIME_ORDER:
            if reg not in fam.index:
                continue
            r = fam.loc[reg]
            prows.append(
                "<tr>"
                f"<td>{_REGIME_ES[reg]}</td>"
                f"<td>{_pct(r['occupancy'])}</td>"
                f"<td>{_num(r['mean_run'], 1)}</td>"
                f"<td>{_num(r['mean_fwd'], 3)}</td>"
                f"<td>{_num(r['std_fwd'], 3)}</td>"
                f"<td>{_num(r['abs_fwd'], 3)}</td>"
                f"<td>{_pct(r['up_rate'])}</td>"
                f"<td>{int(r['n'])}</td>"
                "</tr>"
            )
        phead = (
            "<tr>" + _th("regime", "R\u00e9gimen") + _th("occupancy", "Ocupaci\u00f3n")
            + _th("mean_run", "Racha media") + _th("mean_fwd", f"fwd D+{horizon} medio")
            + _th("std_fwd", "vol fwd") + _th("abs_fwd", "|mov.| medio")
            + _th("up_rate", "P(sube)") + "<th title='N\u00famero de barras.'>N</th></tr>"
        )
        ptable = f'<table class="cw-table"><thead>{phead}</thead><tbody>{"".join(prows)}</tbody></table>'

        ttable = ""
        if transitions is not None and not transitions.empty:
            tf = transitions[transitions["family"] == key]
            if not tf.empty:
                piv = tf.pivot_table(index="from", columns="to", values="prob")
                trows = []
                for frm in _REGIME_ORDER:
                    if frm not in piv.index:
                        continue
                    cells = []
                    for to in _REGIME_ORDER:
                        v = piv.loc[frm, to] if to in piv.columns else np.nan
                        col = ACCENT if (frm == to and not pd.isna(v)) else TEXT
                        cells.append(f"<td style='color:{col}'>{_pct(v)}</td>")
                    trows.append(f"<tr><td>{_REGIME_ES[frm]}</td>{''.join(cells)}</tr>")
                thead = (
                    "<tr><th title='R\u00e9gimen de origen \u2192 destino al d\u00eda siguiente.'>de \\ a</th>"
                    + "".join(f"<th>{_REGIME_ES[t]}</th>" for t in _REGIME_ORDER) + "</tr>"
                )
                ttable = (
                    '<div class="subg-h" style="margin-top:12px">Matriz de transici\u00f3n (fila \u2192 d\u00eda siguiente)</div>'
                    f'<table class="cw-table"><thead>{thead}</thead><tbody>{"".join(trows)}</tbody></table>'
                )
        cards.append(
            f'<details><summary>{FAMILY_LABELS.get(key, key)} \u00b7 anatom\u00eda de reg\u00edmenes</summary>'
            f'{ptable}{ttable}</details>'
        )
    return (
        '<h2 class="cat" id="anatomia">Anatom\u00eda de los reg\u00edmenes</h2>'
        '<p class="cat-lead">C\u00f3mo se comporta el mercado en cada estado, con independencia de la se\u00f1al.</p>'
        '<div class="card"><h3>Ocupaci\u00f3n, persistencia y car\u00e1cter</h3>'
        '<p class="rule">Clasificando cada barra por terciles de <b>Efficiency Ratio</b>: cu\u00e1nto tiempo pasa el '
        'mercado en cada r\u00e9gimen, cu\u00e1nto duran las rachas, con qu\u00e9 frecuencia salta de uno a otro y qu\u00e9 '
        'tama\u00f1o/direcci\u00f3n tiene el movimiento posterior. Descriptivo (terciles sobre toda la muestra).</p>'
        f'{"".join(cards)}</div>'
    )


def _direction_quality_section(
    direction: pd.DataFrame | None,
    quality: pd.DataFrame | None,
    horizon: int,
) -> str:
    """WS5: long/short symmetry of the headline edge + IC gradient across ER quintiles."""
    if (direction is None or direction.empty) and (quality is None or quality.empty):
        return ""

    dir_block = ""
    if direction is not None and not direction.empty:
        rows = []
        for _, r in direction.iterrows():
            rows.append(
                "<tr>"
                f"<td>{FAMILY_LABELS.get(r['family'], r['family'])}</td>"
                f"<td>{_flabel(r['feature'])}</td>"
                f"<td>{int(r['n_trades'])}</td>"
                f"<td>{_pct(r['win_rate'])}</td>"
                f"<td class='sep'>{int(r['n_long']) if not pd.isna(r['n_long']) else 0}</td>"
                f"<td>{_pct(r['win_long'])}</td>"
                f"<td>{_num(r['pnl_long'], 3)}</td>"
                f"<td class='sep'>{int(r['n_short']) if not pd.isna(r['n_short']) else 0}</td>"
                f"<td>{_pct(r['win_short'])}</td>"
                f"<td>{_num(r['pnl_short'], 3)}</td>"
                "</tr>"
            )
        head = (
            "<tr><th title='Familia.'>Familia</th>" + _th("feature", "Feature titular")
            + _th("n_trades", "N") + _th("win_rate", "Win")
            + '<th class="sep" title="Operativa del lado largo.">N largo</th><th>Win largo</th><th>P&amp;L largo</th>'
            + '<th class="sep" title="Operativa del lado corto.">N corto</th><th>Win corto</th><th>P&amp;L corto</th>'
            + "</tr>"
        )
        dir_block = (
            '<div class="card"><h3>Direcci\u00f3n: \u00bfsim\u00e9trica o unilateral?</h3>'
            '<p class="rule">La operativa de la feature titular de cada familia, separada en su pata <b>larga</b> '
            '(compra el bucket barato) y su pata <b>corta</b> (vende el caro). Si una pata concentra casi todos los '
            'trades o todo el P&amp;L, la ventaja es unilateral \u2014 clave para dimensionar y para el riesgo.</p>'
            f'{_num_table_wrap(head, rows)}</div>'
        )

    qual_block = ""
    if quality is not None and not quality.empty:
        bins = sorted(quality["er_bin"].unique())
        cards = []
        for key in FAMILY_LABELS:
            fam = quality[quality["family"] == key]
            if fam.empty:
                continue
            rows = []
            for kind in ("reversion", "continuation"):
                sub = fam[fam["kind"] == kind]
                if sub.empty:
                    continue
                by_bin = sub.set_index("er_bin")
                feat = sub["feature"].iloc[0]
                cells = []
                for b in bins:
                    ic = by_bin.loc[b, "ic"] if b in by_bin.index else np.nan
                    col = SUBTEXT if pd.isna(ic) else (ACCENT if ic < 0 else BEAR)
                    cells.append(f"<td style='color:{col}'>{_num(ic, 3)}</td>")
                kind_es = "Reversi\u00f3n" if kind == "reversion" else "Continuaci\u00f3n"
                rows.append(f"<tr><td>{kind_es}</td><td>{_flabel(feat)}</td>{''.join(cells)}</tr>")
            head = (
                "<tr><th>Tipo</th>" + _th("feature", "Feature")
                + "".join(f'<th title="Quintil de Efficiency Ratio (1=choppy, 5=tendencia limpia).">Q{b}</th>' for b in bins)
                + "</tr>"
            )
            cards.append(
                f'<details><summary>{FAMILY_LABELS.get(key, key)} \u00b7 IC por quintil de ER</summary>'
                f'{_num_table_wrap(head, rows)}</details>'
            )
        qual_block = (
            '<div class="card"><h3>Calidad de tendencia: gradiente por Efficiency Ratio</h3>'
            f'<p class="rule">IC (D+{horizon}) de la mejor feature de reversi\u00f3n y de continuaci\u00f3n a lo largo de '
            'los quintiles de ER (Q1 = m\u00e1s choppy, Q5 = tendencia m\u00e1s limpia). La tesis se cumple si la '
            '<span style="color:' + ACCENT + '">reversi\u00f3n</span> es m\u00e1s negativa en Q1\u2013Q2 y la '
            '<span style="color:' + BEAR + '">continuaci\u00f3n</span> m\u00e1s positiva en Q4\u2013Q5.</p>'
            f'{"".join(cards)}</div>'
        )

    return (
        '<h2 class="cat" id="direccion">Direcci\u00f3n y calidad de tendencia</h2>'
        '<p class="cat-lead">\u00bfEs sim\u00e9trica la ventaja y mejora la continuaci\u00f3n en tendencias limpias?</p>'
        f'{dir_block}{qual_block}'
    )


def _costs_section(costs: pd.DataFrame | None, horizon: int) -> str:
    """WS8: break-even, safety margin, turnover and cost sensitivity per family."""
    if costs is None or costs.empty:
        return ""
    rows = []
    ranked = costs.reindex(costs["safety_margin"].fillna(-np.inf).sort_values(ascending=False).index)
    for _, r in ranked.iterrows():
        safety = r.get("safety_margin")
        scol = SUBTEXT if pd.isna(safety) else (ACCENT if safety >= 1 else BEAR)
        rows.append(
            "<tr>"
            f"<td>{FAMILY_LABELS.get(r['family'], r['family'])}</td>"
            f"<td>{int(r['n_trades'])}</td>"
            f"<td>{_num(r['gross_pnl'], 3)}</td>"
            f"<td>{_num(r['stub_cost'], 3)}</td>"
            f"<td>{_num(r['breakeven_cost'], 3)}</td>"
            f"<td style='color:{scol};font-weight:700'>{_num(safety, 1)}\u00d7</td>"
            f"<td>{_num(r['trades_per_year'], 1)}</td>"
            f"<td>{int(r['holding_days'])}</td>"
            f"<td class='sep'>{_num(r.get('sharpe_1x'))}</td>"
            f"<td>{_num(r.get('sharpe_2x'))}</td>"
            f"<td>{_num(r.get('pnl_2x'), 3)}</td>"
            "</tr>"
        )
    head = (
        "<tr><th title='Familia.'>Familia</th>" + _th("n_trades", "N")
        + _th("gross_pnl", "P&amp;L bruto") + "<th title='Coste round-trip placeholder por familia (puntos).'>Coste stub</th>"
        + _th("breakeven_cost", "Break-even") + _th("safety_margin", "Margen seg.")
        + _th("trades_per_year", "Trades/a\u00f1o") + _th("holding_days", "Tenencia")
        + '<th class="sep" title="Sharpe con 1\u00d7 el coste stub.">Sharpe 1\u00d7</th>'
        + '<th title="Sharpe con 2\u00d7 el coste stub.">Sharpe 2\u00d7</th>'
        + '<th title="P&amp;L medio con 2\u00d7 el coste stub.">P&amp;L 2\u00d7</th>'
        + "</tr>"
    )
    return (
        '<h2 class="cat" id="costes">Costes y operabilidad</h2>'
        '<p class="cat-lead">\u00bfCu\u00e1nto coste aguanta la ventaja y cu\u00e1nto rota?</p>'
        '<div class="card"><h3>Break-even, margen de seguridad y sensibilidad</h3>'
        '<p class="rule">Para la <b>estrategia gated confirmada</b> de cada familia (D+' + str(horizon) + '). '
        'El <b>coste de equilibrio</b> (= P&amp;L bruto) es el coste al que la ventaja se anula; el <b>margen de '
        'seguridad</b> es ese coste dividido por el stub actual (>1\u00d7 = sobrevive, <1\u00d7 = el stub ya la mata). '
        'A la derecha, c\u00f3mo cae el Sharpe al duplicar el coste. Sin bid/offer real el nivel absoluto es '
        'orientativo; lo robusto es el <b>margen relativo</b> y la rotaci\u00f3n.</p>'
        f'{_num_table_wrap(head, rows)}</div>'
    )


def _num_table_wrap(head: str, rows: list[str]) -> str:
    if not rows:
        return '<p class="rule">Sin datos.</p>'
    return f'<table class="cw-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


# --------------------------------------------------------------------------- #
# Executive summary (conclusions) + full methodology documentation
# --------------------------------------------------------------------------- #
def _finding(stat: str, title: str, text: str, color: str = ACCENT) -> str:
    return (
        '<div class="finding">'
        f'<div class="finding-stat" style="color:{color}">{stat}</div>'
        f'<div class="finding-body"><b>{title}.</b> {text}</div>'
        '</div>'
    )


_REVERSION_FAMILIES = {"calendars", "quarterly", "semestral", "yearly", "flies"}


def _conclusions_section(
    results: pd.DataFrame,
    gated: pd.DataFrame | None,
    direction: pd.DataFrame | None,
    costs: pd.DataFrame | None,
    redundancy: pd.DataFrame | None,
    horizon: int,
) -> str:
    """Executive summary derived from the actual results (stays in sync with data)."""
    headline = results[(results["group"] == "ALL") & (results["horizon"] == horizon)].copy()
    if headline.empty:
        return ""

    findings: list[str] = []

    # 1) Reversion vs continuation among robust edges.
    edges = headline[headline.apply(_is_edge, axis=1)]
    n_edge = len(edges)
    n_rev = int((edges["ic_mean"] < 0).sum())
    n_cont = int((edges["ic_mean"] > 0).sum())
    if n_edge:
        pct = 100.0 * n_rev / n_edge
        findings.append(_finding(
            f"{pct:.0f}%",
            "La reversi\u00f3n es la ventaja dominante",
            f"De los <b>{n_edge}</b> pares feature\u00d7familia con ventaja robusta (|t|\u22653 y spread neto&gt;0) "
            f"al horizonte D+{horizon}, <b>{n_rev}</b> son de reversi\u00f3n (IC&lt;0) y <b>{n_cont}</b> de "
            "continuaci\u00f3n. Confirma la tesis: no hay que forzar un trend score.",
        ))

    # 2) Where the edge concentrates (families).
    if n_edge:
        fam_edge = edges.groupby("family").size().sort_values(ascending=False)
        listed = ", ".join(
            f"{FAMILY_LABELS.get(f, f)} ({int(c)})" for f, c in fam_edge.items()
        )
        rev_hits = [f for f in fam_edge.index if f in _REVERSION_FAMILIES]
        extra = (
            f" Las estructuras de spread {', '.join(FAMILY_LABELS.get(f, f) for f in rev_hits)} "
            "concentran la ventaja, como anticipaba el documento."
            if rev_hits else ""
        )
        findings.append(_finding(
            f"{fam_edge.size}",
            "La ventaja no es uniforme entre familias",
            f"Familias con al menos una ventaja robusta (n\u00ba de pares): {listed}.{extra} "
            "Por eso el scoring debe calibrarse por familia, no ser universal.",
        ))

    # 3) Strongest single signal.
    best = headline.reindex(headline["ic_t"].abs().sort_values(ascending=False).index).iloc[0]
    kind = "reversi\u00f3n" if best["ic_mean"] < 0 else "continuaci\u00f3n"
    bcol = ACCENT if best["ic_mean"] < 0 else BEAR
    findings.append(_finding(
        f"t={best['ic_t']:+.1f}",
        f"Se\u00f1al m\u00e1s fuerte: {_flabel(best['feature'])} en {FAMILY_LABELS.get(best['family'], best['family'])}",
        f"IC medio {best['ic_mean']:+.3f} ({kind}) sobre {int(best['n_vintages'])} vintages y "
        f"{int(best['n_folds'])} folds fuera de muestra.",
        bcol,
    ))

    # 4) Direction: most one-sided family.
    if direction is not None and not direction.empty:
        d = direction.copy()
        d["long_share"] = d["n_long"] / d["n_trades"].replace(0, np.nan)
        d = d.dropna(subset=["long_share"])
        if not d.empty:
            row = d.iloc[(d["long_share"] - 0.5).abs().argmax()]
            dom = "largo" if row["long_share"] >= 0.5 else "corto"
            share = row["long_share"] if dom == "largo" else 1 - row["long_share"]
            findings.append(_finding(
                f"{share * 100:.0f}%",
                "Hay ventajas unilaterales",
                f"En {FAMILY_LABELS.get(row['family'], row['family'])} el <b>{share * 100:.0f}%</b> de los trades "
                f"de {_flabel(row['feature'])} son del lado <b>{dom}</b>: la ventaja no es sim\u00e9trica y hay que "
                "dimensionar en consecuencia.",
                SUBTEXT,
            ))

    # 5) Costs / operability.
    if costs is not None and not costs.empty:
        c = costs.copy()
        n_ok = int((c["safety_margin"] >= 1).sum())
        best_c = c.reindex(c["safety_margin"].fillna(-np.inf).sort_values(ascending=False).index).iloc[0]
        neg = c[c["safety_margin"] < 0]
        neg_txt = (
            " " + ", ".join(FAMILY_LABELS.get(f, f) for f in neg["family"]) + " no cubren ni el coste stub."
            if not neg.empty else ""
        )
        findings.append(_finding(
            f"{n_ok}/{len(c)}",
            "Margen de coste desigual",
            f"{n_ok} de {len(c)} estrategias gated aguantan al menos 1\u00d7 el coste stub; el mayor margen es "
            f"{FAMILY_LABELS.get(best_c['family'], best_c['family'])} "
            f"(\u00d7{best_c['safety_margin']:.1f}).{neg_txt}",
            ACCENT if n_ok else BEAR,
        ))

    # 6) Redundancy.
    if redundancy is not None and not redundancy.empty:
        n_red = int((redundancy["verdict"] == "redundante").sum())
        n_rep = int((redundancy["verdict"] == "representante").sum())
        total = len(redundancy)
        findings.append(_finding(
            f"{n_red}/{total}",
            "Muchas features dicen lo mismo",
            f"{n_red} de {total} evaluaciones resultan redundantes dentro de su cl\u00faster; solo <b>{n_rep}</b> "
            "act\u00faan como representantes independientes. El scoring no debe sumar diez versiones del mismo concepto.",
            SUBTEXT,
        ))

    reading = (
        '<div class="card"><h3>Lectura para el trader</h3>'
        '<p class="rule">Con ejecuci\u00f3n realista (entrada en el <b>open de t+1</b>) y control de multiple-testing '
        '(FDR), el backtest no pide un seguidor de tendencia. La <b>ventaja robusta es de reversi\u00f3n</b>, y el '
        'predictor que domina los descubrimientos que sobreviven al FDR es el <b>nivel frente a contratos an\u00e1logos</b> '
        '(<i>level_z</i> / <i>level_pct</i>) en las <b>estructuras de calendario</b> (calendars y los spreads '
        'quarterly/semestral), sobre todo a horizonte largo (D+20\u201330). Los osciladores/momentum cl\u00e1sicos apenas '
        'sobreviven. El Sharpe en tiempo-calendario es modesto (mejor en calendars como cartera diversificada), no un '
        'edge grande. Camino recomendado: (1) un motor de reversi\u00f3n por familia anclado en el <b>nivel</b> del panel '
        'de an\u00e1logos, con confirmaci\u00f3n y confianza; (2) combinar familias en cartera antes que perseguir una sola; '
        '(3) calibrar costes reales antes de dar por operable cualquier cifra.</p></div>'
    )

    return (
        '<h2 class="cat" id="conclusiones">Conclusiones</h2>'
        '<p class="cat-lead">Resumen ejecutivo, derivado de los propios resultados de este informe.</p>'
        f'<div class="card"><h3>Hallazgos principales</h3><div class="findings">{"".join(findings)}</div></div>'
        f'{reading}'
    )


def _full_methodology_section() -> str:
    """Self-contained documentation of everything the system does."""
    blocks = [
        ("Objetivo",
         "No es construir un indicador m\u00e1s ni demostrar que existe una tendencia, sino una herramienta "
         "cuantitativa que responda: \u00bfhay oportunidad estad\u00edstica?, \u00bfde tendencia o de reversi\u00f3n?, "
         "\u00bfen qu\u00e9 direcci\u00f3n?, \u00bfbarato o caro?, \u00bfprobabilidad de continuaci\u00f3n/reversi\u00f3n?, "
         "\u00bfqu\u00e9 confianza tiene la se\u00f1al? Si los datos dicen que la reversi\u00f3n funciona mejor, el modelo "
         "prioriza la reversi\u00f3n; no se fuerza ninguna se\u00f1al que los datos no validen."),
        ("1 \u00b7 Instrumentos y ciclo de vida",
         "Nunca se mezclan familias (outrights, calendars, cracks, flies, Brent-WTI\u2026). Cada fila conserva "
         "contrato exacto, mes, a\u00f1o, <b>vintage</b> (a\u00f1o de expiraci\u00f3n con fechas reales de WTI), "
         "d\u00edas a vencimiento, fase de vida y slot estacional. Ese es el eje sobre el que se alinea todo lo dem\u00e1s."),
        ("2 \u00b7 Variables continuas (features)",
         "No se prueban reglas binarias (\u00abRSI&lt;30 = comprar\u00bb): se conservan los valores continuos "
         "as-of\u2011t y se mide qu\u00e9 pasa despu\u00e9s por bucket. Bloques: <b>Nivel</b> (panel de an\u00e1logos "
         "<code>level_z</code>/<code>level_pct</code> por vintage y fase de vida, sin series continuas), "
         "<b>Extensi\u00f3n/reversi\u00f3n</b> (z 10/20/50, Bollinger %B, Keltner, RSI 2/14), "
         "<b>Agotamiento</b> (divergencia RSI/MACD, desaceleraci\u00f3n de momentum, ca\u00edda de ER, ratio de "
         "volatilidad) y <b>R\u00e9gimen/direcci\u00f3n</b> (Efficiency Ratio, pendiente/ATR, MACD). "
         "Ninguna usa informaci\u00f3n futura (verificado por test)."),
        ("3 \u00b7 Objetivo del backtest (labels)",
         "Para cada fecha se mide el resultado <b>posterior</b>: retorno a D+1/3/5/10/20 (tambi\u00e9n normalizado "
         "por volatilidad conocida en t), excursi\u00f3n favorable (MFE) y adversa (MAE), y barras hasta el extremo "
         "(tiempo a target). El objetivo no es el P&amp;L total de una regla."),
        ("4 \u00b7 Separar tendencia y reversi\u00f3n",
         "Cada barra se etiqueta por r\u00e9gimen con terciles de Efficiency Ratio (rango / zona muerta / tendencia), "
         "calibrados en el train de cada fold (point-in-time). La estrategia <b>gated</b> opera reversi\u00f3n en "
         "rango y continuaci\u00f3n en tendencia, y una <b>confirmaci\u00f3n por nivel</b> (<code>level_z</code>) exige "
         "doble barato/caro en reversi\u00f3n y evita perseguir extensi\u00f3n en continuaci\u00f3n."),
        ("5 \u00b7 Eliminar redundancias",
         "Matriz de correlaci\u00f3n de Spearman, clustering (|\u03c1|\u22650.8), representante por |t| e <b>IC "
         "incremental</b> (IC del residuo tras quitar el representante). Una feature solo cuenta si aporta se\u00f1al "
         "m\u00e1s all\u00e1 de su cl\u00faster."),
        ("6 \u00b7 Walk-forward real por vintage",
         "Se entrena con vintages antiguos y se prueba con el siguiente, avanzando. Partir por vintage \u2014y no por "
         "filas al azar\u2014 evita que contratos solapados filtren informaci\u00f3n entre train y test. Los cortes "
         "(buckets, terciles, lado) se fijan siempre en train."),
        ("7 \u00b7 Validaci\u00f3n por subgrupos",
         "El IC de la se\u00f1al titular se desglosa por \u00e9poca (antes/despu\u00e9s de 2020), r\u00e9gimen de "
         "volatilidad, fase de vida y mes, con su N. Una ventaja repartida es robusta; una concentrada en una sola "
         "porci\u00f3n (t\u00edpicamente 2020) no lo es."),
        ("8 \u00b7 Costes y operabilidad",
         "Coste de equilibrio (= P&amp;L bruto por trade), margen de seguridad frente al coste stub, sensibilidad "
         "del Sharpe a 0/1/2/3\u00d7 el coste y rotaci\u00f3n (trades por a\u00f1o, tenencia). Un Sharpe alto con pocas "
         "operaciones no recibe alta confianza (la confianza penaliza el n\u00famero de trades)."),
        ("An\u00e1lisis adicionales del informe",
         "<b>Anatom\u00eda de reg\u00edmenes</b>: ocupaci\u00f3n, persistencia y matriz de transici\u00f3n. "
         "<b>Direcci\u00f3n</b>: desglose largo vs corto de la operativa. <b>Calidad de tendencia</b>: gradiente de "
         "IC por quintil de ER. <b>Confianza</b> (0\u2013100): combina muestra, estabilidad del t-stat, consistencia "
         "de signo entre folds y n\u00famero de trades."),
        ("Qu\u00e9 NO hace el sistema",
         "No fuerza un trend score, no usa el mismo modelo para todas las familias, no clasifica por P&amp;L total, "
         "no asigna pesos por intuici\u00f3n, no optimiza par\u00e1metros sobre toda la muestra, no confunde win rate "
         "con calidad, no considera robusto un Sharpe alto con pocos trades y no da peso a features altamente "
         "correlacionadas."),
    ]
    cards = "".join(
        f'<p class="rule" style="margin-top:14px"><b>{title}.</b> {desc}</p>' for title, desc in blocks
    )
    return (
        '<h2 class="cat" id="completo">Metodolog\u00eda completa</h2>'
        '<p class="cat-lead">Todo lo que hace el sistema, de principio a fin.</p>'
        f'<div class="card"><h3>C\u00f3mo funciona CrudeWatch (research)</h3>{cards}</div>'
    )


def _bucket_matrix(sub: pd.DataFrame, label: str, horizon: int) -> str:
    """Heat-map of mean forward outcome per bucket (B1 cheap -> Bn dear) for every
    indicator of one family. Green = positive future return, red = negative."""
    order = [f for _, _, feats in FEATURE_CATEGORIES for f in feats]
    by_feat = {r["feature"]: r for _, r in sub.iterrows()}
    ynames, profiles = [], []
    for f in order:
        r = by_feat.get(f)
        if r is None:
            continue
        prof = r.get("bucket_profile")
        if not isinstance(prof, (list, tuple)) or not len(prof):
            continue
        profiles.append([float(v) if v == v else np.nan for v in prof])
        ynames.append(_flabel(f))
    if not profiles:
        return '<p class="rule">Sin perfiles de bucket para esta familia.</p>'
    z = np.array(profiles)
    n_buckets = z.shape[1]
    x = [f"B{i + 1}" for i in range(n_buckets)]
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=ynames, zmid=0,
        colorscale=[[0.0, BEAR], [0.5, SURFACE_2], [1.0, ACCENT]],
        colorbar=dict(title="pts", tickfont=dict(color=TEXT)),
        hovertemplate="%{y}<br>Bucket %{x}<br>Retorno medio: %{z:.3f} pts<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Retorno medio por bucket (D+{horizon}) \u00b7 {label}",
                   x=0.5, xanchor="center", font=dict(color=ACCENT, size=15)),
        template="plotly_dark", paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND,
        font=dict(color=TEXT, family="Arial"),
        margin=dict(l=180, r=30, t=48, b=60), height=max(340, 24 * len(ynames) + 150),
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return _div(fig)


def _bucket_detail_section(results: pd.DataFrame, horizon: int) -> str:
    """WS-D1: every indicator x every bucket, per family (the full detail view)."""
    headline = results[(results["group"] == "ALL") & (results["horizon"] == horizon)]
    if headline.empty:
        return ""
    cards = []
    for key in FAMILY_LABELS:
        fam = headline[headline["family"] == key]
        if fam.empty:
            continue
        label = FAMILY_LABELS.get(key, key)
        cards.append(
            f'<details><summary>{label} \u00b7 todos los indicadores \u00d7 bucket</summary>'
            f'{_bucket_matrix(fam, label, horizon)}</details>'
        )
    return (
        '<h2 class="cat" id="detalle-bucket">Detalle por indicador y bucket</h2>'
        '<p class="cat-lead">El retorno futuro medio de cada uno de los 18 indicadores en cada bucket, por familia.</p>'
        '<div class="card"><h3>Perfil completo por bucket</h3>'
        '<p class="rule">Cada fila es un indicador; las columnas son los buckets del m\u00e1s barato (B1) al m\u00e1s caro '
        f'(B{5}), con cuantiles fijados fuera de muestra. La celda es el retorno medio a D+{horizon} en puntos: '
        '<span style="color:' + ACCENT + '">verde</span> = sube despu\u00e9s, '
        '<span style="color:' + BEAR + '">rojo</span> = baja. Un patr\u00f3n de reversi\u00f3n se ve como verde a la '
        'izquierda (barato \u2192 sube) y rojo a la derecha (caro \u2192 baja).</p>'
        f'{"".join(cards)}</div>'
    )


def _composite_section(
    composite: pd.DataFrame | None, results: pd.DataFrame, horizon: int
) -> str:
    """WS-C2: the combined reversion signal per family, vs the best single indicator."""
    if composite is None or composite.empty:
        return ""
    comp_h = composite[composite["horizon"] == horizon]
    if comp_h.empty:
        return ""
    headline = results[(results["group"] == "ALL") & (results["horizon"] == horizon)]

    rows, charts = [], []
    for key in FAMILY_LABELS:
        c = comp_h[comp_h["family"] == key]
        if c.empty:
            continue
        c = c.iloc[0]
        fam_single = headline[headline["family"] == key]
        best_single = fam_single["ic_t"].abs().max() if not fam_single.empty else np.nan
        col = ACCENT if _is_edge(c) else SUBTEXT
        beats = "\u2713" if (not pd.isna(best_single) and abs(c["ic_t"]) >= best_single) else ""
        wr = "" if pd.isna(c["win_rate"]) else f"{c['win_rate'] * 100:.0f}%"
        n_tr = int(c["n_trades"]) if not pd.isna(c["n_trades"]) else 0
        rows.append(
            "<tr>"
            f"<td>{FAMILY_LABELS.get(key, key)}</td>"
            f"<td>{int(c['n_features'])}</td>"
            f"<td style='color:{col}'>{_num(c['ic_mean'], 3)}</td>"
            f"<td>{_num(c['ic_t'])}</td>"
            f"<td>{_num(best_single)} {beats}</td>"
            f"<td>{_num(c['net_spread'], 3)}</td>"
            f"<td>{n_tr}</td>"
            f"<td>{wr}</td>"
            f"<td>{_num(c['sharpe'])}</td>"
            f"<td>{_num(c.get('confidence'), 0)}</td>"
            "</tr>"
        )
        prof = c.get("bucket_profile")
        edges = c.get("bucket_edges")
        if isinstance(prof, (list, tuple)) and len(prof):
            edges = list(edges) if isinstance(edges, (list, tuple)) else None
            charts.append(
                f'<details><summary>{FAMILY_LABELS.get(key, key)} \u00b7 perfil del compuesto por bucket</summary>'
                f'{_div(_bucket_chart(list(prof), "composite", edges))}</details>'
            )
    head = (
        "<tr><th title='Familia.'>Familia</th><th title='N\u00ba de indicadores combinados.'>N feats</th>"
        + _th("ic_mean", "IC medio") + _th("ic_t", "IC t-stat")
        + "<th title='|IC t-stat| de la mejor feature individual (\u2713 si el compuesto la iguala o supera).'>Mejor individual</th>"
        + _th("net_spread", "Spread neto") + _th("n_trades", "N trades")
        + _th("win_rate", "Win rate") + _th("sharpe", "Sharpe") + _th("confidence", "Confianza")
        + "</tr>"
    )
    return (
        '<h2 class="cat" id="compuesto">Modelo compuesto de reversi\u00f3n</h2>'
        '<p class="cat-lead">Una sola se\u00f1al que combina los indicadores de reversi\u00f3n, por familia.</p>'
        '<div class="card"><h3>Se\u00f1al combinada vs. mejor indicador individual</h3>'
        '<p class="rule">Por fold y sobre el train: cada indicador de reversi\u00f3n se estandariza (z-score), se '
        'orienta por el signo de su IC y se promedia (peso igual, sin ajustar) en un <b>compuesto</b>. Luego se '
        f'punt\u00faa fuera de muestra igual que una feature. La columna <i>Mejor individual</i> muestra el |t| del '
        'mejor indicador suelto: un \u2713 indica que el compuesto lo iguala o mejora. Coste descontado, D+'
        f'{horizon}.</p>{_num_table_wrap(head, rows)}{"".join(charts)}</div>'
    )


def _grid_matrix(g: pd.DataFrame, label: str, horizon: int) -> str:
    """5x3 heat-map of mean forward outcome by (primary bucket x confirmator bucket)."""
    primary = g["primary"].iloc[0]
    conf = g["confirmator"].iloc[0]
    piv = g.pivot_table(index="p_bucket", columns="c_bucket", values="mean_fwd")
    npiv = g.pivot_table(index="p_bucket", columns="c_bucket", values="n")
    piv = piv.reindex(sorted(piv.index)).reindex(sorted(piv.columns), axis=1)
    npiv = npiv.reindex(piv.index).reindex(piv.columns, axis=1)
    y = [f"P{i} ({_flabel(primary)})" if i in (piv.index.min(), piv.index.max()) else f"P{i}" for i in piv.index]
    x = [f"C{j}" for j in piv.columns]
    fig = go.Figure(go.Heatmap(
        z=piv.to_numpy(), x=x, y=[f"P{i}" for i in piv.index],
        customdata=npiv.to_numpy(), zmid=0,
        colorscale=[[0.0, BEAR], [0.5, SURFACE_2], [1.0, ACCENT]],
        colorbar=dict(title="pts", tickfont=dict(color=TEXT)),
        hovertemplate="Primario %{y}<br>Confirmador %{x}<br>Retorno medio: %{z:.3f} pts<br>N: %{customdata}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{_flabel(primary)} (P, barato\u2192caro) \u00d7 {_flabel(conf)} (C) \u00b7 {label}",
                   x=0.5, xanchor="center", font=dict(color=ACCENT, size=14)),
        template="plotly_dark", paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND,
        font=dict(color=TEXT, family="Arial"),
        margin=dict(l=60, r=30, t=52, b=50), height=320,
    )
    fig.update_xaxes(automargin=True, title_text=_flabel(conf))
    fig.update_yaxes(automargin=True, title_text=_flabel(primary))
    return _div(fig)


def _grid_section(grid: pd.DataFrame | None, horizon: int) -> str:
    """WS-G3: conditional 2-indicator interaction grid per family."""
    if grid is None or grid.empty:
        return ""
    cards = []
    for key in FAMILY_LABELS:
        g = grid[grid["family"] == key]
        if g.empty:
            continue
        label = FAMILY_LABELS.get(key, key)
        cards.append(
            f'<details><summary>{label} \u00b7 rejilla condicional</summary>{_grid_matrix(g, label, horizon)}</details>'
        )
    if not cards:
        return ""
    return (
        '<h2 class="cat" id="rejilla">Rejilla condicional de 2 indicadores</h2>'
        '<p class="cat-lead">C\u00f3mo cambia el resultado cuando un segundo indicador confirma al primero.</p>'
        '<div class="card"><h3>Interacci\u00f3n indicador primario \u00d7 confirmador</h3>'
        f'<p class="rule">Retorno medio a D+{horizon} por celda: filas = buckets del indicador primario '
        '(P1 barato \u2192 P5 caro), columnas = terciles del confirmador (C1 \u2192 C3). '
        '<span style="color:' + ACCENT + '">Verde</span> = sube despu\u00e9s. La reversi\u00f3n confirmada se ve como '
        'una esquina (p. ej. P1 barato + confirmador en su extremo) m\u00e1s intensa que la fila sola. '
        'Vista descriptiva (agregada), \u00fatil como exploraci\u00f3n.</p>'
        f'{"".join(cards)}</div>'
    )


def build_research_report(
    results: pd.DataFrame,
    horizon: int = 10,
    gated_results: pd.DataFrame | None = None,
    regime_results: pd.DataFrame | None = None,
    redundancy_results: pd.DataFrame | None = None,
    subgroup_results: pd.DataFrame | None = None,
    composite_results: pd.DataFrame | None = None,
    grid_results: pd.DataFrame | None = None,
    regime_profile: pd.DataFrame | None = None,
    regime_transitions: pd.DataFrame | None = None,
    direction_results: pd.DataFrame | None = None,
    quality_results: pd.DataFrame | None = None,
    cost_results: pd.DataFrame | None = None,
) -> str:
    """Return the full self-contained HTML page for the research backtest summary."""
    horizons = sorted(int(h) for h in results["horizon"].unique())
    headline = results[(results["group"] == "ALL") & (results["horizon"] == horizon)].copy()

    best_overall = headline.reindex(headline["ic_t"].abs().sort_values(ascending=False).index).iloc[0]
    n_families = headline["family"].nunique()
    n_robust = int(headline.apply(_is_edge, axis=1).sum())

    kpis = "".join([
        _kpi(f"{n_families}", "familias evaluadas"),
        _kpi(f"D+{horizon}", "horizonte titular"),
        _kpi(_flabel(best_overall["feature"]), "feature m\u00e1s fuerte"),
        _kpi(f"{best_overall['ic_t']:+.1f}", "mejor IC t-stat"),
        _kpi(f"{n_robust}", "pares feature\u00d7familia con ventaja"),
    ])

    family_cards = []
    for key in FAMILY_LABELS:
        fam_h = headline[headline["family"] == key]
        if fam_h.empty:
            continue
        fam_all = results[(results["group"] == "ALL") & (results["family"] == key)]
        family_cards.append(_family_card(key, fam_h, fam_all, horizons))

    sidebar = _sidebar(headline)
    heatmap = _div(_heatmap(results, horizon)) if not headline.empty else ""

    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Resumen de backtest walk-forward \u2014 CrudeWatch</title>
<script type="text/javascript">{_PLOTLYJS}</script>
<style>{_CSS}</style></head>
<body><a id="top"></a>
<div class="layout">
{sidebar}
<main class="main">
  <div class="hero">
    <div class="cw-title">Backtest walk-forward \u00b7 se\u00f1al predictiva</div>
    <div class="cw-sub">Poder de cada feature (conocida al cierre de <b>t</b>) para anticipar el retorno <b>ejecutable</b>
    (entrada en el <b>open de t+1</b>, salida a D+20/25/30), validado fuera de muestra por vintage con control de
    multiple-testing (FDR). IC de Spearman y spread por bucket con coste descontado \u00b7 emerald = reversi\u00f3n, rojo =
    continuaci\u00f3n. Sustituye al screening long/flat: separa medir de operar.</div>
  </div>
  <div class="kpis">{kpis}</div>

  {_conclusions_section(results, gated_results, direction_results, cost_results, redundancy_results, horizon)}

  <h2 class="cat" id="metodologia">Metodolog\u00eda</h2>
  <p class="cat-lead">Qu\u00e9 mide cada n\u00famero y por qu\u00e9 es honesto.</p>
  {_methodology_html()}

  {_full_methodology_section()}

  <h2 class="cat" id="mapa">Mapa global</h2>
  <p class="cat-lead">D\u00f3nde tiene ventaja cada feature: emerald = reversi\u00f3n (IC&lt;0), rojo = continuaci\u00f3n (IC&gt;0).</p>
  {heatmap}

  <h2 class="cat" id="ranking">Ranking por familia</h2>
  <p class="cat-lead">La mejor feature de cada familia al horizonte titular, ordenadas por estabilidad (|t-stat|).</p>
  <div class="card">{_ranking_table(headline)}</div>

  <h2 class="cat">Detalle por familia</h2>
  <p class="cat-lead">Veredicto, perfil de reversi\u00f3n por bucket y tablas por horizonte.</p>
  {''.join(family_cards)}

  {_bucket_detail_section(results, horizon)}

  {_composite_section(composite_results, results, horizon)}

  {_regime_section(gated_results, regime_results, horizon)}

  {_regime_anatomy_section(regime_profile, regime_transitions, horizon)}

  {_direction_quality_section(direction_results, quality_results, horizon)}

  {_costs_section(cost_results, horizon)}

  {_grid_section(grid_results, horizon)}

  {_diagnostics_section(redundancy_results, subgroup_results, horizon)}

  <h2 class="cat" id="metricas">Glosario de m\u00e9tricas</h2>
  <p class="cat-lead">Qu\u00e9 significa cada valor de las tablas (IC, t-stat, spread, monotonicidad\u2026) y c\u00f3mo leerlo.</p>
  {_metric_glossary_html()}

  <h2 class="cat" id="glosario">Glosario de features</h2>
  <p class="cat-lead">Qu\u00e9 es cada feature, en lenguaje llano.</p>
  {_feature_glossary_html()}
</main>
</div>
</body></html>"""
