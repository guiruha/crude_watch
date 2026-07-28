"""Per-component score screens (one block at a time), for a single contract."""
from __future__ import annotations

import html
import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crudewatch.infra import FAMILY_LABELS

from core.scoring import (
    analogous_outcomes_cached,
    calibrator_cached,
    enriched_frame,
    horizon_outcomes_cached,
    score_instrument_dict,
)
from core.selection import Selection
from screens.opportunity import _REGIME_ES, _base_layout, _fmt
from theme.palette import (
    ACCENT,
    BEAR,
    BORDER,
    BULL,
    INFO_MARK_SVG,
    SUBTEXT,
    SURFACE,
    title_block,
)


def _family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, family.title())

COMPONENTS: dict[str, dict] = {
    "regime": {
        "label": "Régimen",
        "kind": "regime",
        "help": (
            "Es la **primera pregunta** del análisis: **¿en qué modo se está comportando el "
            "mercado?** Puede estar en **rango** (el precio oscila y tiende a volver a su media), "
            "en modo **direccional** (el movimiento se autoalimenta y persiste) o en **transición** "
            "(una zona ambigua, sin carácter claro). Esta respuesta condiciona cómo se leen todos "
            "los demás bloques: en rango manda la **reversión**; en direccional, la **continuación**.\n\n"
            "La clasificación es **relativa a la familia**, no a umbrales absolutos: tus indicadores "
            "se comparan con los **terciles históricos** de contratos análogos, así que un valor "
            "modesto en bruto puede ser 'direccional' si para su familia resulta extremo. El resumen "
            "numérico es la *trendiness* (**0–100**), que agrega cuánta **estructura y persistencia** "
            "hay en el precio."
        ),
        "read": (
            "Cuanta **más** *trendiness*, más tendencia. Recuerda que el régimen dice *si* existe "
            "tendencia — no *hacia dónde* (eso es **Dirección**) ni *cómo de limpia* es (eso es "
            "**Fuerza**)."
        ),
        "calc": (
            "**Régimen:** lo fija el **Efficiency Ratio (ER, 20)** = |recorrido neto| / recorrido total "
            "(0 = puro ruido, 1 = línea recta), comparado con los **terciles de la familia**: "
            "ER ≤ tercil bajo → rango; ER ≥ tercil alto → direccional; en medio → transición.\n\n"
            "**Trendiness (0–100):** media de los **percentiles de familia** de ER, *variance ratio* "
            "(varianza a 5 vs 60 sesiones; >1 persistencia, <1 reversión) y *autocorrelación* lag-1 de "
            "los cambios (20)."
        ),
        "features": [
            ("er_20", "Efficiency Ratio (20): fracción del recorrido que es direccional vs ruido. Alto = tendencia."),
            ("variance_ratio_5", "Variance ratio (5/60): >1 persistencia (tendencia), <1 reversión."),
            ("autocorr_20", "Autocorrelación lag-1 de los cambios (20): + persistencia, − reversión."),
        ],
    },
    "direction": {
        "label": "Dirección",
        "kind": "signed",
        "help": (
            "Indica **hacia dónde** se inclina el mercado, en una escala de **−100** (claramente "
            "bajista) a **+100** (claramente alcista), con **0** neutral. Es puro **sesgo**: no dice "
            "si existe tendencia (eso es *Régimen*) ni cómo de buena es (eso es *Fuerza*), solo la "
            "dirección del empuje.\n\n"
            "Se construye promediando varias señales de **momentum y tendencia** —pendiente de "
            "regresión, histograma MACD, alineación de medias móviles y momentum a distintos "
            "plazos—, cada una traducida a su **percentil dentro de la familia**. Así una señal pesa "
            "más cuanto más **rara o extrema** es en su historia, no por su valor bruto."
        ),
        "read": (
            "**Signo** = bando (alcista/bajista); **magnitud** = cuán rotundo es el sesgo. Solo pesa "
            "fuerte en la puntuación final cuando el **régimen es direccional**; en rango, un sesgo "
            "marcado importa menos que el **Nivel**."
        ),
        "calc": (
            "Media de los **percentiles de familia** (mapeados a −100…+100) de cuatro señales de "
            "momentum/tendencia: pendiente de regresión (20) normalizada por ATR, histograma MACD "
            "(línea − señal), alineación de EMA 20/50/100 (distancia del precio en ATR) y momentum a "
            "5/10/20 sesiones. Al promediar percentiles, cada señal aporta según su rareza en la familia."
        ),
        "features": [
            ("slope_20", "Pendiente de regresión (20) normalizada por ATR: gradiente de tendencia."),
            ("macd_hist", "Histograma MACD (línea − señal): momentum; + alcista."),
            ("ema_align", "Alineación de EMA 20/50/100 (distancia del precio en ATR): + apilado alcista."),
            ("mom_5", "Momentum a 5 días normalizado por ATR: + subida reciente."),
            ("mom_10", "Momentum a 10 días normalizado por ATR."),
            ("mom_20", "Momentum a 20 días normalizado por ATR."),
        ],
    },
    "strength": {
        "label": "Fuerza",
        "kind": "unit100",
        "help": (
            "Mide la **calidad** de la tendencia (**0–100**): ¿es una **recta limpia y operable** o "
            "un movimiento **ruidoso y entrecortado**? Responde al *cómo de buena* es, y así completa "
            "a *Dirección* (hacia dónde) y a *Régimen* (si existe).\n\n"
            "Combina la **linealidad** del recorrido —lo bien que una recta explica el precio— con la "
            "**persistencia direccional** —qué proporción de sesiones va en el mismo sentido— y se "
            "**atenúa** cuando no hay un sesgo de Dirección claro: sin rumbo, no hay tendencia que "
            "medir."
        ),
        "read": (
            "**Alta** = tendencia nítida y fiable para operar a favor. **Baja** = movimiento sucio "
            "donde las señales de tendencia valen poco, aunque exista dirección."
        ),
        "calc": (
            "Combina la **linealidad** —R² de la regresión lineal de 20 sesiones (0 = disperso, 1 = "
            "recta perfecta)— y la **persistencia direccional** = |2·(% de sesiones al alza) − 1| "
            "(0 = 50/50 choppy, 1 = todas en la misma dirección), y se **modula por la magnitud de la "
            "Dirección**: sin sesgo claro, la fuerza se atenúa."
        ),
        "features": [
            ("r2_20", "R² de la regresión lineal (20): limpieza/linealidad de la tendencia (0–1)."),
            ("dir_persistence_20", "Persistencia direccional (20): |2·(% días al alza) − 1|; 1 = todas las sesiones en la misma dirección."),
        ],
    },
    "level": {
        "label": "Nivel",
        "kind": "signed",
        "help": (
            "Es la valoración **caro vs. barato**, de **−100** (muy barato) a **+100** (muy caro). "
            "No mira momentum ni dirección: mira **posición de precio** frente a lo que es normal "
            "para contratos comparables.\n\n"
            "El núcleo es la comparación contra un **panel de análogos** —mismo *slot* estacional, "
            "*vintage* y fase de vida del contrato—, reforzada con z-scores propios a varios plazos "
            "y la distancia a su media. Al usar comparables tan afines, es el bloque **más robusto** "
            "del sistema."
        ),
        "read": (
            "En **régimen de rango**, un Nivel extremo (muy caro / muy barato) es lo que **dispara "
            "la reversión**. En direccional pesa menos: la tendencia puede seguir estirando el precio "
            "más allá de lo 'caro'."
        ),
        "calc": (
            "Núcleo: **percentil / z-score de nivel** frente al **panel de análogos** (mismo "
            "slot estacional, vintage y fase de vida del contrato). Se complementa con z-scores del "
            "cierre a 10/20/50 sesiones (nº de desviaciones típicas sobre su media) y la **distancia "
            "Keltner** (cierre − EMA20, en unidades de ATR)."
        ),
        "features": [
            ("level_pct", "Percentil de nivel frente a contratos análogos (mismo slot/vintage/fase)."),
            ("level_z", "Z-score de nivel frente al panel de análogos."),
            ("z_10", "Z-score del cierre sobre 10 días."),
            ("z_20", "Z-score del cierre sobre 20 días."),
            ("z_50", "Z-score del cierre sobre 50 días."),
            ("keltner_dist_20", "Distancia a la media Keltner (EMA20) en unidades de ATR."),
        ],
    },
    "p_reversion": {
        "label": "P(reversión)",
        "kind": "prob",
        "help": (
            "Probabilidad **histórica** de que, partiendo de un **extremo de nivel**, el precio "
            "**revierta** hacia su media dentro del horizonte elegido. Es el juego de **rango**, "
            "calibrado con la propia historia de la familia.\n\n"
            "Se calcula como **tasa condicional** *point-in-time*: de todas las veces que la familia "
            "estuvo cara o barata en rango, ¿qué fracción revirtió a favor a D+h? Esa tasa base se "
            "**modula** con indicadores de agotamiento (RSI, Bollinger %B, divergencias, "
            "desaceleración de momentum): más confirmación acerca la probabilidad a la tasa histórica."
        ),
        "read": (
            "Se lee **junto a Nivel**: Nivel te dice *cuán* extremo estás; esta probabilidad, *con "
            "qué frecuencia* eso revirtió en el pasado. Alta + Nivel extremo = mejor setup de "
            "reversión."
        ),
        "calc": (
            "**Tasa histórica condicional:** de todas las veces que la familia estuvo **barata o cara** "
            "(tercil extremo de nivel) en régimen de rango, ¿qué fracción **revirtió a favor** a D+h? "
            "Estrictamente *point-in-time*. Esa tasa base se **modula por una confirmación** = media "
            "(peso igual) de varios indicadores de agotamiento/oscilador/divergencia orientados al lado "
            "del extremo: RSI(2), RSI(14), Bollinger %B, divergencia RSI y desaceleración de momentum. "
            "Más confirmación → más se acerca la probabilidad a la tasa histórica."
        ),
        "features": [
            ("rsi_2", "RSI(2): sobreventa (<20) o sobrecompra (>80) de muy corto plazo; confirma la reversión."),
            ("rsi_14", "RSI(14): sobreventa (<30) o sobrecompra (>70) de medio plazo."),
            ("pctb_20_2", "Bollinger %B (20,2): >1 por encima de la banda superior (caro), <0 bajo la inferior (barato)."),
            ("rsi_div_14", "Divergencia RSI(14) vs precio: + alcista (precio cae pero RSI aguanta), − bajista."),
            ("mom_decel_10", "Desaceleración de momentum: negativo = el impulso se agota (precursor de reversión)."),
        ],
    },
    "p_continuation": {
        "label": "P(continuación)",
        "kind": "prob",
        "help": (
            "Probabilidad **histórica** de que un movimiento en **régimen direccional continúe** en "
            "el horizonte. Es el juego de **tendencia**, la contraparte de la reversión.\n\n"
            "Igual que aquella, es una **tasa condicional** *point-in-time*: de las veces que la "
            "familia mostró pendiente fuerte en direccional, ¿qué fracción siguió en esa dirección a "
            "D+h? Se pondera por la *trendiness* actual y se mide sobre el **resultado posterior a la "
            "señal**, no vía P&L de cruces."
        ),
        "read": (
            "En crudo suele ser **baja**, coherente con que el *trend-following* puro flojea; no te "
            "extrañe que la reversión domine incluso dentro de una tendencia."
        ),
        "calc": (
            "**Tasa histórica condicional:** de las veces que la familia mostró **pendiente fuerte** "
            "(tercil alto/bajo) en régimen direccional, ¿qué fracción **continuó** en esa dirección a "
            "D+h? Se pondera por la *trendiness* del régimen actual. Point-in-time como P(reversión)."
        ),
        "note": (
            "Probabilidad **derivada**: se calibra a partir de la **dirección** (bloque Dirección) y "
            "la **trendiness** del régimen, más las tasas históricas de continuación de la familia. "
            "No tiene métricas propias que no aparezcan ya en otros bloques."
        ),
        "features": [],
    },
    "probabilities": {
        "label": "Probabilidades",
        "kind": "prob_pair",
        "help": (
            "Las dos caras de **«¿qué pasa después?»**: la probabilidad de **revertir** (juego de "
            "rango, en extremos de nivel) y la de **continuar** (juego de tendencia).\n\n"
            "Son estimaciones **independientes** —cada una calibrada con su propia historia "
            "condicional—, por lo que **no suman 100%**. Pueden ser ambas bajas (setup poco claro) o "
            "ambas apreciables (lecturas en tensión, señales enfrentadas)."
        ),
        "read": (
            "Fíjate en cuál **domina**: si P(reversión) ≫ P(continuación), el sesgo es "
            "**contra-tendencia**; si es al revés, **a favor**. El régimen actual te dice cuál mirar "
            "primero."
        ),
        "features": [],
    },
    "confidence": {
        "label": "Fiabilidad",
        "kind": "reliability",
        "help": (
            "Es el **track record** del setup actual: cuando la familia ha estado en el **mismo "
            "régimen y el mismo nivel** que hoy, ¿con qué frecuencia **acertó seguir la acción** del "
            "modelo, y con qué consistencia?\n\n"
            "No es un número inventado: se toma el **cohorte de casos análogos** con su resultado ya "
            "realizado a D+h (*point-in-time*) y se calcula la **cota inferior de Wilson** del % de "
            "acierto — es decir, el acierto **penalizado por el tamaño de muestra**."
        ),
        "read": (
            "**Alta** solo si hubo muchos casos y salieron bien; con **pocos casos** baja aunque el % "
            "histórico parezca bueno. Trátala como *cuánto fiarte* del resto de bloques."
        ),
        "calc": (
            "Se toma el **cohorte de setups análogos** (mismo régimen + mismo bucket de nivel, calibrado "
            "por familia) con su resultado a D+h. La **Fiabilidad (0–100)** es la **cota inferior de "
            "Wilson** del % de acierto siguiendo la acción: el acierto **penalizado por el tamaño de "
            "muestra** (pocos casos → fiabilidad baja aunque el % sea alto)."
        ),
        "features": [],
    },
}

# Per-metric chart config: human label + reference lines for the time series
# (and the recent-evolution chart). ``bands`` triggers the ER regime zones
# (shared low/high family terciles) on both charts. Every feature listed in a
# block's ``features`` gets the same two charts (family distribution + recent
# evolution) annotated with these references.
FEATURE_CTX: dict[str, dict] = {
    "er_20": {"label": "Efficiency Ratio (20)", "bands": True},
    "variance_ratio_5": {"label": "Variance ratio (5/60)", "reflines": [(1.0, "1.0")]},
    "autocorr_20": {"label": "Autocorrelación lag-1 (20)", "reflines": [(0.0, "0")]},
    "slope_20": {"label": "Pendiente norm. (20)", "reflines": [(0.0, "0")]},
    "macd_hist": {"label": "MACD histograma", "reflines": [(0.0, "0")]},
    "ema_align": {"label": "Alineación EMA 20/50/100", "reflines": [(0.0, "0")]},
    "mom_5": {"label": "Momentum 5d (ATR)", "reflines": [(0.0, "0")]},
    "mom_10": {"label": "Momentum 10d (ATR)", "reflines": [(0.0, "0")]},
    "mom_20": {"label": "Momentum 20d (ATR)", "reflines": [(0.0, "0")]},
    "r2_20": {"label": "R² (20)", "reflines": [(0.3, "0.3"), (0.6, "0.6")]},
    "dir_persistence_20": {"label": "Persistencia direccional (20)", "reflines": [(0.5, "0.5")]},
    "level_pct": {"label": "Percentil de nivel (análogos)", "reflines": [(50.0, "50")]},
    "level_z": {"label": "Z-score de nivel (análogos)", "reflines": [(-2, "−2"), (0, "0"), (2, "+2")]},
    "z_10": {"label": "Z-score cierre (10)", "reflines": [(-2, "−2"), (0, "0"), (2, "+2")]},
    "z_20": {"label": "Z-score cierre (20)", "reflines": [(-2, "−2"), (0, "0"), (2, "+2")]},
    "z_50": {"label": "Z-score cierre (50)", "reflines": [(-2, "−2"), (0, "0"), (2, "+2")]},
    "keltner_dist_20": {"label": "Distancia Keltner (ATR)", "reflines": [(0.0, "0")]},
    "rsi_2": {"label": "RSI(2)", "reflines": [(20.0, "20"), (80.0, "80")]},
    "rsi_14": {"label": "RSI(14)", "reflines": [(30.0, "30"), (70.0, "70")]},
    "pctb_20_2": {"label": "Bollinger %B (20,2)", "reflines": [(0.0, "0"), (1.0, "1")]},
    "rsi_div_14": {"label": "Divergencia RSI(14)", "reflines": [(0.0, "0")]},
    "mom_decel_10": {"label": "Desaceleración de momentum", "reflines": [(0.0, "0")]},
}


def _feature_label(feat: str) -> str:
    return FEATURE_CTX.get(feat, {}).get("label", feat)


def _direction_tag(v: float) -> str:
    if v >= 55:
        return "Fuertemente alcista"
    if v >= 15:
        return "Alcista"
    if v > -15:
        return "Neutral"
    if v > -55:
        return "Bajista"
    return "Fuertemente bajista"


def _level_tag(v: float) -> str:
    if v >= 55:
        return "Muy caro"
    if v >= 15:
        return "Caro"
    if v > -15:
        return "Precio justo"
    if v > -55:
        return "Barato"
    return "Muy barato"


def _strength_tag(v: float) -> str:
    if v >= 66:
        return "Tendencia limpia"
    if v >= 33:
        return "Moderada"
    return "Ruidosa / entrecortada"


# Material-symbol icon per metric for the context sub-menu (clickable pills).
FEATURE_ICON: dict[str, str] = {
    "er_20": ":material/query_stats:",
    "variance_ratio_5": ":material/ssid_chart:",
    "autocorr_20": ":material/autorenew:",
    "slope_20": ":material/trending_up:",
    "macd_hist": ":material/show_chart:",
    "ema_align": ":material/stacked_line_chart:",
    "mom_5": ":material/speed:",
    "mom_10": ":material/speed:",
    "mom_20": ":material/speed:",
    "r2_20": ":material/timeline:",
    "dir_persistence_20": ":material/straighten:",
    "level_pct": ":material/monitoring:",
    "level_z": ":material/functions:",
    "z_10": ":material/functions:",
    "z_20": ":material/functions:",
    "z_50": ":material/functions:",
    "keltner_dist_20": ":material/expand:",
    "rsi_2": ":material/speed:",
    "rsi_14": ":material/speed:",
    "pctb_20_2": ":material/waterfall_chart:",
    "rsi_div_14": ":material/call_split:",
    "macd_div": ":material/call_split:",
    "mom_decel_10": ":material/trending_down:",
}


def _metric_icon(feat: str) -> str:
    return FEATURE_ICON.get(feat, ":material/insights:")


# Directional (signed) features: thresholds for "leve" / "fuerte" magnitude.
# Values are roughly in ATR units except macd_hist (raw price points, so its
# magnitude is instrument-dependent -> we only read its sign, no strength word).
_SIGNED_BANDS: dict[str, tuple[float, float]] = {
    "slope_20": (0.02, 0.10),
    "ema_align": (0.25, 1.00),
    "mom_5": (0.30, 1.00),
    "mom_10": (0.40, 1.50),
    "mom_20": (0.50, 2.00),
}


def _reading(key: str, value) -> str:
    """A plain-language conclusion for a feature value (no number needed)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    v = float(value)

    if key == "er_20":
        if v >= 0.50:
            return "Tendencia clara"
        if v >= 0.35:
            return "Algo direccional"
        return "Rango / ruido"
    if key == "r2_20":
        if v >= 0.60:
            return "Tendencia limpia"
        if v >= 0.30:
            return "Tendencia moderada"
        return "Sin tendencia lineal"
    if key == "variance_ratio_5":
        if v >= 1.15:
            return "Persistente (tendencia)"
        if v <= 0.85:
            return "Reversión a la media"
        return "Aleatorio (sin señal)"
    if key == "autocorr_20":
        if v >= 0.10:
            return "Persistencia"
        if v <= -0.10:
            return "Reversión"
        return "Sin memoria"
    if key == "dir_persistence_20":
        if v >= 0.50:
            return "Muy direccional"
        if v >= 0.25:
            return "Direccional"
        return "Choppy / lateral"
    if key == "macd_hist":  # raw points: sign only
        if v > 0:
            return "Momentum alcista"
        if v < 0:
            return "Momentum bajista"
        return "Plano"
    if key in _SIGNED_BANDS:
        mild, strong = _SIGNED_BANDS[key]
        a = abs(v)
        if a < mild:
            return "Plano / neutral"
        mag = "fuerte" if a >= strong else "leve"
        return f"Alcista {mag}" if v > 0 else f"Bajista {mag}"
    if key in ("level_z", "z_10", "z_20", "z_50", "keltner_dist_20"):
        if v <= -2.0:
            return "Muy barato (sobrevendido)"
        if v <= -1.0:
            return "Barato"
        if v < 1.0:
            return "Normal"
        if v < 2.0:
            return "Caro"
        return "Muy caro (sobrecomprado)"
    if key == "level_pct":  # percentile 0..1
        if v <= 0.10:
            return "Muy barato"
        if v <= 0.30:
            return "Barato"
        if v < 0.70:
            return "Normal"
        if v < 0.90:
            return "Caro"
        return "Muy caro"
    if key == "rsi_2":
        if v <= 5.0:
            return "Sobreventa extrema"
        if v <= 20.0:
            return "Sobreventa"
        if v < 80.0:
            return "Neutral"
        if v < 95.0:
            return "Sobrecompra"
        return "Sobrecompra extrema"
    if key == "rsi_14":
        if v <= 30.0:
            return "Sobreventa"
        if v < 70.0:
            return "Neutral"
        return "Sobrecompra"
    if key == "pctb_20_2":
        if v <= 0.0:
            return "Bajo banda inferior (sobrevendido)"
        if v < 0.2:
            return "Cerca de banda inferior"
        if v < 0.8:
            return "Dentro de bandas"
        if v < 1.0:
            return "Cerca de banda superior"
        return "Sobre banda superior (sobrecomprado)"
    if key in ("rsi_div_14", "macd_div"):
        if v >= 0.5:
            return "Divergencia alcista"
        if v <= -0.5:
            return "Divergencia bajista"
        return "Sin divergencia clara"
    if key == "mom_decel_10":
        if v <= -0.3:
            return "Impulso agotándose"
        if v >= 0.3:
            return "Impulso acelerando"
        return "Impulso estable"
    return "—"


def _reading_relative(pct: float) -> str:
    """Family-relative reading of a feature from its percentile within the family."""
    if pct >= 90:
        return "Extremo alto en su familia"
    if pct >= 70:
        return "Alto vs su familia"
    if pct > 30:
        return "Típico en su familia"
    if pct > 10:
        return "Bajo vs su familia"
    return "Extremo bajo en su familia"


def _regime_sentence(regime: str, er, er_pct, family_label: str) -> str:
    """Frame the regime relative to the family, avoiding an absolute 'tendencia' claim."""
    er_txt = "—" if er is None or er != er else f"{er:.2f}"
    pct_txt = "—" if er_pct is None else f"{er_pct:.0f}"
    if regime == "trend":
        abs_note = (
            f"El Efficiency Ratio ({er_txt}) **no es una tendencia fuerte en absoluto**"
            if (er is not None and er == er and er < 0.5)
            else f"El Efficiency Ratio ({er_txt}) es alto también en términos absolutos"
        )
        return (
            f"{abs_note}, pero está en el **percentil {pct_txt} de {family_label}**: es de lo "
            "más direccional que se ve en esta familia. Por eso, **en relativo a su familia**, "
            "cuenta como direccional — no como una tendencia absoluta."
        )
    if regime == "range":
        return (
            f"Efficiency Ratio ({er_txt}) en el **percentil {pct_txt} de {family_label}**: de lo "
            "menos direccional para esta familia — comportamiento de rango / reversión."
        )
    return (
        f"Efficiency Ratio ({er_txt}) en el **percentil {pct_txt} de {family_label}**: zona "
        "intermedia, sin rango ni dirección claros (transición)."
    )


def _wilson_lower(p: float, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for a proportion (shrinks with small n)."""
    if n <= 0:
        return 0.0
    p = min(max(p, 0.0), 1.0)
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(max(p * (1.0 - p) / n + z * z / (4 * n * n), 0.0))
    return max(0.0, (centre - margin) / denom)


def _reliability_label(rel: float | None, n: int) -> str:
    if rel is None or n == 0:
        return "Insuficiente (sin cohorte)"
    if n < 15:
        return "Insuficiente (muestra pequeña)"
    if rel >= 58:
        return "Alta"
    if rel >= 50:
        return "Media"
    return "Baja"


def _md_inline(text: str) -> str:
    """Escape HTML, then render Markdown **bold** / *italic* as inline tags."""
    out = html.escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"\*(.+?)\*", r"<i>\1</i>", out)
    return out


def _md_paragraphs(text: str) -> str:
    """Render blank-line-separated blocks as <p> paragraphs with inline markdown."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return "".join(f"<p>{_md_inline(p)}</p>" for p in paras)


def _intro_card(text: str, read: str | None = None) -> None:
    """Render a block's explanation as a styled intro panel.

    ``text`` may hold several blank-line-separated paragraphs; ``read`` adds a
    highlighted "Cómo leerlo" footer with the practical interpretation.
    """
    body = _md_paragraphs(text)
    if read:
        body += (
            '<div class="cw-intro-read"><span class="lbl">Cómo leerlo</span>'
            f'<span>{_md_inline(read)}</span></div>'
        )
    st.markdown(
        f'<div class="cw-intro"><span class="ico">{INFO_MARK_SVG}</span>'
        f'<div class="txt">{body}</div></div>',
        unsafe_allow_html=True,
    )


def _risk_chips(risks: list[str]) -> None:
    chips = " ".join(
        f"<span style='display:inline-block;margin:4px 6px 4px 0;padding:4px 10px;"
        f"border-radius:8px;border:1px solid {BORDER};background:{SURFACE};"
        f"color:{SUBTEXT};font-size:13px;'>{r}</span>"
        for r in risks
    )
    st.markdown(chips, unsafe_allow_html=True)


class ComponentScreen:
    """Single score-component view for one contract."""

    def __init__(self, frames: dict[str, pd.DataFrame], key: str) -> None:
        self.frames = frames
        self.key = key
        self.meta = COMPONENTS[key]

    def display(self, selection: Selection) -> None:
        meta = self.meta
        title_block(meta["label"], icon=INFO_MARK_SVG)

        if selection.contract is None:
            st.info("Selecciona una familia y una fecha con contratos activos en el menú superior.")
            return

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

        if self.key == "confidence":
            self._display_reliability(family, contract, horizon, as_of_iso, score)
        elif self.key == "probabilities":
            self._display_probabilities(family, contract, horizon, as_of_iso, scored_date, score)
        else:
            self._display_block(family, contract, horizon, as_of_iso, scored_date, score)

    # -- Probabilidades (reversión + continuación en una sola vista) --------

    def _display_probabilities(
        self, family, contract, horizon, as_of_iso, scored_date, score
    ) -> None:
        meta = self.meta
        blocks = score["blocks"]
        feats = score.get("features", {})
        pcts = score.get("percentiles", {})
        p_rev = blocks.get("p_reversion", float("nan"))
        p_cont = blocks.get("p_continuation", float("nan"))
        regime = blocks.get("regime", "")

        _intro_card(meta["help"], meta.get("read"))

        regime_es = _REGIME_ES.get(regime, regime)
        if regime == "range":
            rel = "manda **P(reversión)**: se busca el retorno a la media desde el extremo de nivel."
        elif regime == "trend":
            rel = "manda **P(continuación)**: interesa que el movimiento siga en su dirección."
        else:
            rel = "ninguna domina: léelas con cautela, la señal es más incierta."
        st.info(f"Régimen actual: **{regime_es}** → {rel}")

        rev_meta = COMPONENTS["p_reversion"]
        cont_meta = COMPONENTS["p_continuation"]

        # Both probabilities in one comparison chart. No side-by-side columns:
        # P(continuación) is a *derived* metric with no context cards of its own,
        # so a second column just left a big empty gap and squeezed P(reversión).
        self._prob_compare_chart(p_rev, p_cont, regime)

        marker = " ⟵ relevante ahora" if regime == "range" else ""
        st.markdown(f"#### P(reversión){marker}")
        st.markdown(rev_meta["help"])
        with st.expander("Cómo se calcula", expanded=False):
            st.markdown(rev_meta["calc"])
        if rev_meta["features"]:
            self._metric_submenu(
                family, contract, horizon, as_of_iso, scored_date,
                rev_meta["features"], feats, pcts, "prob_rev",
            )

        st.divider()

        marker = " ⟵ relevante ahora" if regime == "trend" else ""
        st.markdown(f"#### P(continuación){marker}")
        st.markdown(cont_meta["help"])
        with st.expander("Cómo se calcula", expanded=False):
            st.markdown(cont_meta["calc"])
        if cont_meta.get("note"):
            st.info(cont_meta["note"])

        st.markdown("##### Precio reciente (hasta la fecha)")
        self._price_line(family, contract, scored_date)

    def _display_block(
        self, family, contract, horizon, as_of_iso, scored_date, score
    ) -> None:
        meta = self.meta
        blocks = score["blocks"]
        feats = score.get("features", {})
        pcts = score.get("percentiles", {})
        self._render_score(blocks, feats, pcts, family)

        _intro_card(meta["help"], meta.get("read"))
        if meta.get("calc"):
            with st.expander("Cómo se calcula", expanded=False):
                st.markdown(meta["calc"])

        if meta["features"]:
            self._feature_table(meta["features"], feats, family, as_of_iso)
        elif meta.get("note"):
            st.info(meta["note"])

        if meta["features"]:
            st.markdown("##### Contexto por métrica")
            st.caption(
                "Elige una métrica para ver su contexto: **dónde cae hoy frente a la historia de la "
                "familia** (distribución + percentil) y **cómo ha evolucionado** en el contrato."
            )
            self._metric_submenu(
                family, contract, horizon, as_of_iso, scored_date, meta["features"], feats, pcts, self.key
            )

        st.markdown("##### Precio reciente (hasta la fecha)")
        self._price_line(family, contract, scored_date)

    def _metric_submenu(
        self, family, contract, horizon, as_of_iso, scored_date, features_meta, feats, pcts, key_suffix
    ) -> None:
        """Clickable icon sub-menu (pills) to pick a metric and show its context."""
        options = [f for f, _ in features_meta]
        if not options:
            return
        meanings = dict(features_meta)
        sel = st.pills(
            "Métrica",
            options,
            default=options[0],
            format_func=lambda f: f"{_metric_icon(f)} {_feature_label(f)}",
            key=f"ctxsel_{key_suffix}",
            label_visibility="collapsed",
        )
        if sel is None:
            sel = options[0]
        self._metric_context(
            family, contract, horizon, as_of_iso, scored_date, sel, meanings[sel], feats, pcts
        )

    def _metric_context(
        self, family, contract, horizon, as_of_iso, scored_date, feat, meaning, feats, pcts
    ) -> None:
        """Explanation + the two standard charts (family distribution + recent series)."""
        fmeta = FEATURE_CTX.get(feat, {})
        val = feats.get(feat)

        if val is None or (isinstance(val, float) and np.isnan(val)):
            st.caption("Sin dato para esta métrica en la fecha elegida.")
            st.caption(meaning)
            return

        abs_reading = _reading(feat, val)
        pct = self._family_percentile(family, feat, val, as_of_iso)
        parts = [
            '<div class="cw-mctx-line"><span class="k">Valor</span> '
            f'<b>{float(val):.3f}</b> — <span class="cw-mctx-tag">{html.escape(abs_reading)}</span></div>'
        ]
        if pct is not None:
            w = max(0.0, min(100.0, float(pct)))
            fam_reading = _reading_relative(pct)
            parts.append(
                '<div class="cw-mctx-line"><span class="k">Vs familia</span> · percentil '
                f'<b>{pct:.0f}</b> <span class="cw-mctx-bar"><span style="width:{w:.0f}%"></span></span>'
                f' — <span class="cw-mctx-tag">{html.escape(fam_reading)}</span></div>'
            )
        parts.append(
            f'<div class="cw-mctx-desc"><b>Qué mide:</b> {html.escape(meaning)}</div>'
        )
        st.markdown('<div class="cw-mctx">' + "".join(parts) + "</div>", unsafe_allow_html=True)

        bands = None
        if fmeta.get("bands"):
            try:
                cal = calibrator_cached(family, horizon, as_of_iso)
                bands = (float(cal.er_lo), float(cal.er_hi))
            except Exception:
                bands = None

        if feat not in enriched_frame(family).columns:
            st.caption("Sin serie/distribución disponible para esta métrica.")
            return

        c1, c2 = st.columns(2)
        with c1:
            st.caption("Distribución en la familia (posición actual)")
            self._dist_chart(family, feat, float(val), as_of_iso, bands)
        with c2:
            st.caption("Evolución reciente en el contrato")
            self._ts_chart(family, contract, feat, scored_date, fmeta.get("reflines"), bands)

    def _family_percentile(self, family, feat, value, as_of) -> float | None:
        """On-the-fly family percentile (0–100) of ``value`` for any feature column."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        data = enriched_frame(family)
        if feat not in data.columns:
            return None
        vals = data.loc[pd.to_datetime(data["date"]) <= pd.Timestamp(as_of), feat].dropna()
        if vals.empty:
            return None
        return float((vals <= float(value)).mean() * 100.0)

    def _feature_table(self, features_meta, feats: dict, family: str, as_of) -> None:
        body = []
        for feat_key, meaning in features_meta:
            val = feats.get(feat_key)
            display_val = (
                "—"
                if val is None or (isinstance(val, float) and np.isnan(val))
                else f"{float(val):.4f}"
            )
            pct = self._family_percentile(family, feat_key, val, as_of)
            abs_reading = _reading(feat_key, val)
            fam_reading = _reading_relative(pct) if pct is not None else "—"
            label = _feature_label(feat_key)
            desc = meaning
            if desc.startswith(f"{label}: "):
                desc = desc[len(label) + 2 :]

            if pct is None:
                pct_html = '<span class="f">—</span>'
            else:
                w = max(0.0, min(100.0, float(pct)))
                pct_html = (
                    '<div class="pct">'
                    f'<div class="track"><div class="fill" style="width:{w:.0f}%"></div></div>'
                    f'<div class="num">{w:.0f}</div></div>'
                )
            info = (
                f'<span class="info" title="{html.escape(desc)}">{INFO_MARK_SVG}</span>'
                if desc
                else ""
            )
            body.append(
                "<tr>"
                f'<td class="m">{html.escape(label)}{info}</td>'
                f'<td class="v">{html.escape(display_val)}</td>'
                f'<td class="r">{html.escape(abs_reading)}</td>'
                f'<td class="f">{html.escape(fam_reading)}</td>'
                f"<td>{pct_html}</td>"
                "</tr>"
            )
        st.markdown(
            '<div class="cw-ftab-wrap"><table class="cw-ftab"><thead><tr>'
            "<th>Métrica</th><th>Valor</th><th>Lectura</th><th>Vs familia</th>"
            "<th>Percentil (familia)</th>"
            "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "**Lectura**: qué concluye la métrica por su valor (absoluto). "
            "**Vs familia** y **Percentil**: dónde cae ese valor frente a la historia de la familia. "
            "Pasa el cursor por la **ⓘ** para ver qué mide cada métrica."
        )

    # -- Fiabilidad (empirical reliability) ---------------------------------

    def _display_reliability(self, family, contract, horizon, as_of_iso, score) -> None:
        meta = self.meta
        try:
            coh = analogous_outcomes_cached(family, contract, horizon, as_of_iso)
        except Exception as exc:
            st.error(f"No se pudo calcular el cohorte análogo: {exc}")
            coh = {"n": 0}

        n = int(coh.get("n", 0))
        win = coh.get("aligned_win_rate")
        rel = _wilson_lower(float(win), n) * 100.0 if (n > 0 and win is not None) else None

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Fiabilidad",
            "—" if rel is None else f"{rel:.0f}/100",
            help="Cota inferior de Wilson del % de acierto siguiendo la acción: penaliza muestras pequeñas.",
        )
        c2.metric("Acierto histórico", "—" if win is None else f"{win * 100:.0f}%")
        c3.metric("Casos análogos (n)", f"{n}")
        st.caption(f"Lectura: **{_reliability_label(rel, n)}**.")

        _intro_card(meta["help"], meta.get("read"))
        if meta.get("calc"):
            with st.expander("Cómo se calcula", expanded=False):
                st.markdown(meta["calc"])

        if n == 0:
            st.info(
                "Sin historia análoga suficiente para este setup a la fecha elegida: no hay base "
                "para estimar la fiabilidad. Prueba otro contrato, familia o fecha."
            )
            return

        regime_es = _REGIME_ES.get(coh.get("regime", ""), coh.get("regime", ""))
        st.caption(
            f"Cohorte: contratos de **{_family_label(family)}** en **mismo régimen ({regime_es})** y "
            f"**mismo nivel ({coh.get('level_bin', '—')})**, resultado a **D+{horizon}** en base "
            f"ejecutable y point-in-time. n = {n} observaciones ya realizadas."
        )

        cols = st.columns(4)
        cols[0].metric("PnL medio (alineado)", _fmt(coh.get("avg_aligned", float("nan")), "+.3f"))
        cols[1].metric("% subidas", _fmt(coh.get("up_rate", float("nan")) * 100, ".0f") + "%")
        cols[2].metric("MFE media", _fmt(coh.get("avg_mfe", float("nan")), "+.3f"))
        cols[3].metric("MAE media", _fmt(coh.get("avg_mae", float("nan")), "+.3f"))

        left, right = st.columns(2)
        with left:
            st.caption("Distribución de resultados análogos (alineados con la acción)")
            self._cohort_hist(coh)
        with right:
            st.caption("Fiabilidad por horizonte (acierto siguiendo la acción)")
            self._reliability_by_horizon(family, contract, as_of_iso)

        risks = score.get("risks", [])
        if risks:
            st.markdown("##### Avisos")
            _risk_chips(risks)

    def _cohort_hist(self, coh: dict) -> None:
        samples = coh.get("aligned_samples") or coh.get("fwd_samples") or []
        vals = pd.Series(samples, dtype=float).dropna()
        if vals.empty:
            st.caption("Sin muestras del cohorte.")
            return
        pos = float((vals > 0).mean() * 100.0)
        fig = go.Figure(go.Histogram(x=vals, nbinsx=30, marker_color=SUBTEXT, opacity=0.6))
        fig.add_vline(x=0.0, line_color=ACCENT, line_width=2)
        fig.add_annotation(
            x=0.0, y=1, yref="paper", yanchor="bottom", showarrow=False,
            text=f"{pos:.0f}% a favor", font=dict(color=ACCENT, size=12),
        )
        fig.update_xaxes(title_text="Resultado alineado (pts)")
        fig.update_yaxes(title_text="frecuencia")
        st.plotly_chart(_base_layout(fig, 260), width="stretch")

    def _reliability_by_horizon(self, family, contract, as_of_iso) -> None:
        try:
            hz = horizon_outcomes_cached(family, contract, as_of_iso)
        except Exception as exc:
            st.caption(f"No se pudo calcular por horizonte: {exc}")
            return
        valid = hz[(hz["n"] > 0) & hz["aligned_win_rate"].notna()]
        if valid.empty:
            st.caption("Sin acierto alineado por horizonte (setup sin dirección sugerida).")
            return
        y = valid["aligned_win_rate"] * 100.0
        fig = go.Figure(
            go.Bar(
                x=valid["horizon"], y=y,
                marker_color=[BULL if v >= 50 else BEAR for v in y],
                text=[f"{v:.0f}%" for v in y], textposition="outside",
            )
        )
        fig.add_hline(y=50, line_dash="dot", line_color=SUBTEXT)
        fig.update_xaxes(title_text="Horizonte (D+)")
        fig.update_yaxes(title_text="% acierto", range=[0, 100])
        st.plotly_chart(_base_layout(fig, 260), width="stretch")

    def _render_score(self, blocks: dict, features: dict, percentiles: dict, family: str) -> None:
        kind = self.meta["kind"]
        block_val = blocks[self.key]

        if kind == "regime":
            er = features.get("er_20")
            er_pct = percentiles.get("er_20")
            c1, c2 = st.columns(2)
            c1.metric("Trendiness (vs familia)", f"{blocks['trendiness']:.0f}/100")
            c2.metric("Percentil ER (familia)", "—" if er_pct is None else f"{er_pct:.0f}")
            st.caption(_regime_sentence(blocks["regime"], er, er_pct, _family_label(family)))
        elif kind == "signed":
            if self.key == "level":
                tag = _level_tag(block_val)
                tone = BEAR if block_val >= 15 else BULL if block_val <= -15 else SUBTEXT
                ends = ("−100 · barato", "caro · +100")
            else:
                tag = _direction_tag(block_val)
                tone = BULL if block_val >= 15 else BEAR if block_val <= -15 else SUBTEXT
                ends = ("−100 · bajista", "alcista · +100")
            icon = "▲" if block_val >= 15 else "▼" if block_val <= -15 else "●"
            self._score_flashcard(
                label=self.meta["label"], value_str=f"{block_val:+.0f}", tone=tone,
                tag=tag, icon=icon, vmin=-100, vmax=100, value=block_val, center=0,
                end_left=ends[0], end_right=ends[1],
            )
        elif kind == "unit100":
            tone = ACCENT if block_val >= 33 else SUBTEXT
            icon = "◆" if block_val >= 66 else "◈" if block_val >= 33 else "○"
            self._score_flashcard(
                label=self.meta["label"], value_str=f"{block_val:.0f}/100", tone=tone,
                tag=_strength_tag(block_val), icon=icon, vmin=0, vmax=100, value=block_val,
                center=None, end_left="0 · ruidosa", end_right="limpia · 100",
            )
        elif kind == "prob":
            v = block_val * 100.0
            tone = BULL if v >= 50 else SUBTEXT
            icon = "▲" if v >= 50 else "●"
            self._score_flashcard(
                label=self.meta["label"], value_str=f"{v:.0f}%", tone=tone,
                tag=("Favorable" if v >= 50 else "Sin ventaja"), icon=icon, vmin=0, vmax=100,
                value=v, center=50, end_left="0%", end_right="100%",
            )

    def _score_flashcard(
        self, *, label, value_str, tone, tag, icon, vmin, vmax, value, center, end_left, end_right
    ) -> None:
        """Visual headline card: big value + qualitative tag + gauge with marker.

        ``icon`` is a grayscale-safe glyph (▲/▼/●…) so the reading does not rely on
        colour alone (WCAG 1.4.1: information not conveyed by hue only).
        """
        span = (vmax - vmin) or 1.0
        pos = max(0.0, min(100.0, (value - vmin) / span * 100.0))
        if center is not None:
            c = (center - vmin) / span * 100.0
            lo, hi = min(c, pos), max(c, pos)
            fill = f"left:{lo:.2f}%;width:{hi - lo:.2f}%"
            mid = f'<div class="cw-fc-mid" style="left:{c:.2f}%"></div>'
        else:
            fill = f"left:0;width:{pos:.2f}%"
            mid = ""
        st.markdown(
            f'<div class="cw-fc" style="border-left-color:{tone}">'
            f'<div class="cw-fc-top">'
            f'<span class="cw-fc-label">{html.escape(label)}</span>'
            f'<span class="cw-fc-tag" style="color:{tone};border-color:{tone}66">'
            f'<span class="cw-fc-ico">{icon}</span>{html.escape(tag)}</span></div>'
            f'<div class="cw-fc-val" style="color:{tone}">{html.escape(value_str)}</div>'
            f'<div class="cw-fc-track">{mid}'
            f'<div class="cw-fc-fill" style="{fill};background:{tone}"></div>'
            f'<div class="cw-fc-dot" style="left:{pos:.2f}%;background:{tone}"></div></div>'
            f'<div class="cw-fc-ends"><span>{html.escape(end_left)}</span>'
            f"<span>{html.escape(end_right)}</span></div></div>",
            unsafe_allow_html=True,
        )

    # -- contextual visualizations ------------------------------------------

    def _dist_chart(self, family, feat, value, as_of, bands=None) -> None:
        data = enriched_frame(family)
        vals = data.loc[pd.to_datetime(data["date"]) <= pd.Timestamp(as_of), feat].dropna()
        if vals.empty:
            st.caption("Sin distribución en la familia.")
            return
        fig = go.Figure(go.Histogram(x=vals, nbinsx=40, marker_color=SUBTEXT, opacity=0.55))
        if bands is not None:
            lo, hi = bands
            vmin, vmax = float(vals.min()), float(vals.max())
            fig.add_vrect(x0=vmin, x1=lo, fillcolor=BEAR, opacity=0.10, line_width=0)
            fig.add_vrect(x0=hi, x1=vmax, fillcolor=BULL, opacity=0.10, line_width=0)
        fig.add_vline(x=value, line_color=ACCENT, line_width=2)
        pct = float((vals <= value).mean() * 100.0)
        fig.add_annotation(
            x=value, y=1, yref="paper", yanchor="bottom", showarrow=False,
            text=f"actual {value:.2f} · pct {pct:.0f}", font=dict(color=ACCENT, size=12),
        )
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="frecuencia")
        st.plotly_chart(_base_layout(fig, 260), width="stretch")

    def _ts_chart(self, family, contract, feat, as_of, reflines=None, bands=None) -> None:
        data = enriched_frame(family)
        as_of_ts = pd.Timestamp(as_of)
        sub = data[data["contract"] == contract].sort_values("date")
        sub = sub[pd.to_datetime(sub["date"]) <= as_of_ts].dropna(subset=[feat]).tail(120)
        if sub.empty:
            st.caption("Sin serie del indicador.")
            return

        # Comparison background = ANALOGOUS PRIOR contracts, i.e. the same seasonal
        # slot and strictly-earlier vintages (Bloque D panel definition), aligned by
        # LIFE PHASE (days-to-expiry) instead of calendar date — prior vintages
        # already expired, so a calendar axis would push them out of the window.
        # Point-in-time is automatic: at a given dte an earlier vintage was there on
        # an earlier calendar date than the target, so nothing leaks past ``as_of``.
        has_life = {"slot", "vintage", "dte"}.issubset(data.columns) and sub["dte"].notna().any()
        slot = sub["slot"].iloc[-1] if has_life else None
        vintage = sub["vintage"].iloc[-1] if has_life else None
        dte_win = sub["dte"].dropna() if has_life else pd.Series(dtype=float)
        n_analogs = 0

        fig = go.Figure()
        if has_life and pd.notna(vintage) and not dte_win.empty:
            dte_lo, dte_hi = float(dte_win.min()), float(dte_win.max())
            analogs = data[
                (data["slot"] == slot)
                & (data["vintage"] < vintage)
                & (data["contract"] != contract)
                & (pd.to_datetime(data["date"]) <= as_of_ts)
            ].dropna(subset=[feat, "dte"])
            analogs = analogs[(analogs["dte"] >= dte_lo) & (analogs["dte"] <= dte_hi)]
            for _, g in analogs.groupby("contract", sort=False):
                g = g.sort_values("dte", ascending=False)
                if g.empty:
                    continue
                n_analogs += 1
                fig.add_trace(
                    go.Scatter(
                        x=g["dte"], y=g[feat], mode="lines",
                        line=dict(color=SUBTEXT, width=1), opacity=0.20,
                        showlegend=False, hoverinfo="skip",
                    )
                )
        # Selected contract in bold on top, also on the life-phase axis.
        subp = sub.sort_values("dte", ascending=False) if has_life else sub
        xsel = subp["dte"] if has_life else subp["date"]
        fig.add_trace(
            go.Scatter(
                x=xsel, y=subp[feat], mode="lines",
                line=dict(color=ACCENT, width=3), name=contract,
            )
        )
        # Direct end-of-line label at the most recent point (smallest dte).
        last = subp.iloc[-1]
        fig.add_annotation(
            x=(last["dte"] if has_life else last["date"]), y=last[feat],
            text=f"  {contract}", xanchor="left", yanchor="middle",
            showarrow=False, font=dict(color=ACCENT, size=12),
        )
        lines = list(reflines or [])
        if bands is not None:
            lines += [(bands[0], "rango ≤"), (bands[1], "≥ direccional")]
        for yv, txt in lines:
            fig.add_hline(
                y=yv, line_dash="dot", line_color=SUBTEXT,
                annotation_text=txt, annotation_position="right",
                annotation_font=dict(color=SUBTEXT, size=11),
            )
        if has_life:
            # Reverse so time flows left→right toward expiry (high dte left, 0 right).
            fig.update_xaxes(title_text="días a vencimiento", autorange="reversed")
        fig.update_yaxes(title_text="")
        st.plotly_chart(_base_layout(fig, 260), width="stretch")
        if n_analogs:
            st.caption(
                f"En color: **{contract}**. Líneas tenues: {n_analogs} análogos anteriores "
                "(mismo slot estacional, vintages previos) alineados por días a vencimiento."
            )
        elif has_life:
            st.caption(
                f"En color: **{contract}**. Sin análogos anteriores suficientes en esta fase de vida."
            )

    def _prob_compare_chart(self, p_rev, p_cont, regime: str) -> None:
        """Both probabilities as horizontal bars in a single full-width chart."""
        rows = [
            ("P(reversión)", p_rev, regime == "range"),
            ("P(continuación)", p_cont, regime == "trend"),
        ]
        valid = [
            (lbl, float(p) * 100.0, rel)
            for lbl, p, rel in rows
            if p is not None and p == p
        ]
        if not valid:
            st.caption("Sin probabilidades calibradas para este setup.")
            return
        labels = [f"{lbl}  ⟵" if rel else lbl for lbl, _, rel in valid]
        vals = [v for _, v, _ in valid]
        colors = [BULL if v >= 50 else BEAR for v in vals]
        fig = go.Figure(
            go.Bar(
                x=vals, y=labels, orientation="h", marker_color=colors,
                text=[f"{v:.0f}%" for v in vals], textposition="outside",
            )
        )
        fig.add_vline(x=50, line_dash="dot", line_color=SUBTEXT)
        fig.update_xaxes(range=[0, 100], title_text="Probabilidad (%)")
        fig.update_yaxes(showticklabels=True, autorange="reversed")
        st.plotly_chart(_base_layout(fig, 180), width="stretch")
        st.caption(
            "Frente al **50% de base**: por encima, el histórico de la familia favorece el "
            "escenario; por debajo, no aporta ventaja. La marcada con **⟵** es la que pesa "
            "en el régimen actual."
        )

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
