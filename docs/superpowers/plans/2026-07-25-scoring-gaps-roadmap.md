# CrudeWatch — Análisis de gaps (brief del PM vs. código) y roadmap por fases

> Documento de trabajo. **No modifica código.** Mapea el brief del portfolio
> manager ("Objetivo, metodología y arquitectura del scoring") contra lo que
> ya existe en el repositorio, marca el estado de cada requisito y propone un
> roadmap priorizado (mean-reversion → continuación → selector de régimen).
>
> Cuando se decida ejecutar una fase, cada bloque de trabajo se expandirá a un
> plan de implementación TDD paso a paso (`writing-plans`).

Fecha: 2026-07-25 · Ámbito: `src/crudewatch/scoring`, `src/crudewatch/research`,
`backtesting/research`, `backtesting/backtest`.

---

## 0. Conclusión en una frase

El repositorio **ya implementa** la filosofía del brief (reversión primero,
calibrado por familia, point-in-time, bloques A–G, identidad fija de
contrato/vintage, walk-forward por vintage). Los gaps reales son **acotados y
concretos**, no un rediseño: (1) los pesos no se aprenden por familia, (2) el
scoring es una fórmula ponderada única en lugar de dos modelos + selector,
(3) faltan horizontes cortos D+1/3/5, (4) los costes son *stubs*, y (5) los
bloques de régimen/dirección/nivel usan un subconjunto de las variables que
pide el brief.

---

## 1. Leyenda de estado

- ✅ **Hecho** — implementado y alineado con el brief.
- 🟡 **Parcial** — existe una versión, pero incompleta frente al brief.
- ❌ **Falta** — no existe.

---

## 2. Resumen ejecutivo (estado por requisito)

| # | Requisito del brief | Estado | Dónde | Gap principal |
|---|---------------------|--------|-------|---------------|
| 1 | Herramienta de decisión, no ejecutora | ✅ | `README.md`, `score.py` | — |
| 2 | Reversión priorizada sobre trend | ✅ | `WEIGHTS["range"]` en `score.py:24` | — |
| 3 | Bloques A–G separados | ✅ | `blocks.py`, `score.py:54` | Var. incompletas (ver §4) |
| 4 | Calibración por familia (no universal) | 🟡 | `FamilyCalibrator` `blocks.py:65` | ECDF/probs por familia sí; **pesos universales** |
| 5 | Régimen por ER calibrado (no 0,35 fijo) | ✅ | `regime_label` `blocks.py:187` | Solo ER; faltan var. de régimen |
| 6 | Identidad fija contrato/mes/año/vintage | ✅ | `lifecycle.py`, `panel.py` | — |
| 7 | Variables continuas, no reglas binarias | ✅ | `features.py:124` | — |
| 8 | Nivel por instrumento fijo + vintage | ✅ | `panel.py:30` | — |
| 9 | Backtest simétrico long/short/flat | 🟡 | `evaluate.py:98` (long+short) | Backtest **legacy** sigue long/flat |
| 10 | Objetivo = retornos futuros + MFE/MAE + tiempo | 🟡 | `targets.py:45` | Horizontes 20/25/30; **faltan 1/3/5/10** |
| 11 | Dos modelos (continuación / reversión) + selector | ❌ | `compute_opportunity` `score.py:137` | Fórmula única con pesos por régimen |
| 12 | Pesos aprendidos y validados por familia | ❌ | `WEIGHTS` `score.py:24` | Fijados a mano, universales |
| 13 | Eliminar redundancias (corr/cluster/IC incremental) | ✅ | `diagnostics.py` | — |
| 14 | Walk-forward real (expanding, sin optimizar sobre todo) | ✅ | `evaluate.py:44` | — |
| 15 | Validación por subgrupos | ✅ | `diagnostics.py` (era/vol/DTE/mes) | — |
| 16 | Costes reales (bid/offer, slippage, liquidez) | ❌ | `costs.py`, `dataset.py:23` | **Stub fijo** por familia |
| 17 | Confianza rica (OOS, estabilidad, acuerdo de modelos) | 🟡 | `block_confidence` `blocks.py:259` | Solo nº barras/panel/IC nivel |
| 18 | Score final [-100,+100] + explicación + riesgos | ✅ | `score.py:191`, `:202`, `:219` | — |
| 19 | Salida diaria por instrumento (estado/interp/riesgos) | ✅ | `score_family` `score.py:310` | — |

---

## 3. Bloques A–G: spec del PM vs. implementación

### Bloque A — Régimen 🟡
- **Pide:** Efficiency Ratio, autocorrelación de retornos, variance ratio, R² de
  regresión, persistencia de máx/mín, frecuencia de cruces de la media,
  volatilidad realizada, expansión/contracción de volatilidad. Salida:
  tendencia / rango / transición / incierto.
- **Hay:** `regime_label` con **terciles de ER calibrados por familia**
  (`blocks.py:187`) → range/transition/trend, más `vol_ratio` (short/long vol,
  `features.py`). El umbral 0,35 **no** se usa; los cortes son empíricos.
- **Gap:** faltan variance ratio (Lo-MacKinlay), autocorrelación, R², persistencia
  de máx/mín y frecuencia de cruces. El estado "incierto" se colapsa hoy en
  "transition".

### Bloque B — Dirección 🟡
- **Pide:** pendiente de regresión, pendiente normalizada por vol, posición vs
  EMA 20/50/100, alineación de medias, momentum 5/10/20, máx/mín crecientes,
  dirección de MACD, dirección del carry/estructura.
- **Hay:** `block_direction` promedia percentiles de `slope_20` y `macd_hist`
  (`blocks.py:202`). Salida en `[-100,+100]`.
- **Gap:** faltan EMA 20/50/100 + alineación, momentum 5/10/20, máx/mín
  crecientes/decrecientes y dirección de estructura/carry.

### Bloque C — Fuerza / calidad de tendencia 🟡
- **Pide:** ER, R², pendiente/vol, % sesiones misma dirección, profundidad de
  correcciones, duración, estabilidad del momentum, nº falsas rupturas,
  estabilidad del score.
- **Hay:** `block_strength = trendiness·(0.4+0.6·|dir|)` (`blocks.py:213`).
- **Gap:** proxy simple; faltan R², profundidad de correcciones, duración,
  falsas rupturas y estabilidad temporal del score.

### Bloque D — Nivel / extensión 🟡 (el más importante según el brief)
- **Pide:** z-score 10/20/50, distancia a Bollinger, distancia a Keltner,
  percentil histórico, distancia a estacionalidad, distancia a comparables,
  nivel relativo dentro del ciclo de vida. **Por instrumento fijo y vintage.**
- **Hay:** `block_level` combina `level_pct`, `level_z` (panel análogo por
  slot/vintage/DTE, `panel.py`) y `z_20` (`blocks.py:217`). El panel **ya es por
  vintage/DTE** (cumple lo esencial).
- **Gap:** no entran `z_10`/`z_50`, Bollinger %B, Keltner (que **sí existen** como
  features), distancia explícita a estacionalidad ni a comparables (Brent-WTI,
  cracks).

### Bloque E — Probabilidad de reversión 🟡
- **Pide:** z extremo, Bollinger 20/2.5 y 10/1.5, Keltner 20/2, RSI 2/14,
  divergencias MACD/RSI, desaceleración de momentum, caída de ER, ruptura
  fallida, cambio de vol. Mayor peso en quarterly/semestral/yearly/flies.
- **Hay:** `block_p_reversion` usa la **tasa forward condicional** por barato/caro
  (`p_rev_cheap`/`p_rev_dear`, `blocks.py:233`), calibrada por familia; las
  features de Bollinger/Keltner/RSI/divergencia existen (`features.py:124`).
- **Gap:** las features de sobreextensión **no entran** en el cálculo de
  `p_reversion` (solo `level_pct` decide barato/caro). "Ruptura fallida" no está.

### Bloque F — Continuación 🟡
- **Pide:** analizar directamente el resultado posterior a cada señal:
  rendimiento D+1/D+3/D+5/D+10/D+20, prob. continuación/reversión, MFE, MAE,
  tiempo hasta ruptura. **No** medir vía P&L de cruce de medias.
- **Hay:** `block_p_continuation` usa tasa forward por dirección/pendiente
  (`p_cont_up`/`p_cont_dn`, `blocks.py:246`) — ya es "resultado posterior a la
  señal", correcto. `targets.py` calcula `fwd_h`, `mfe_h`, `mae_h`,
  `bars_to_mfe/mae`.
- **Gap:** horizontes solo **20/25/30**; faltan **D+1/D+3/D+5/D+10** como columnas.

### Bloque G — Confianza 🟡
- **Pide:** nº observaciones, estabilidad por año/vintage, consistencia entre
  subperiodos y out-of-sample, estabilidad ante cambios de parámetros,
  similitud régimen actual vs históricos, dispersión, intervalo de confianza,
  % de modelos que coinciden.
- **Hay:** `block_confidence = 100·f_level·f_bars·f_regime·f_stability`
  (`blocks.py:259`): panel disponible, nº barras del contrato, régimen definido,
  |t| del IC de `level_z`.
- **Gap:** falta estabilidad por año/vintage, consistencia OOS explícita,
  robustez a parámetros, dispersión de análogos e intervalo de confianza, y el
  "% de modelos que coinciden" (que sólo tiene sentido con el diseño de dos
  modelos, ver §5.11).

---

## 4. Metodología de backtest (§8 del brief): pasos 1–8

| Paso | Requisito | Estado | Nota |
|------|-----------|--------|------|
| 1 | Definir instrumentos sin mezclar familias | ✅ | 8 familias con lifecycle propio |
| 2 | Variables continuas (buckets), no binarias | ✅ | `evaluate.py` bucketiza continuas |
| 3 | Objetivo = retorno futuro normalizado, MFE/MAE, DD, tiempo a target, prob. éxito | 🟡 | `targets.py` tiene fwd/MFE/MAE/vol-norm/bars-to; **falta drawdown-antes-de-beneficio y horizontes cortos** |
| 4 | Dos modelos (continuación / reversión) + selector | ❌ | Ver §5.11 |
| 5 | Eliminar redundancias (corr, cluster, IC, contribución incremental) | ✅ | `diagnostics.py` |
| 6 | Walk-forward real (expanding, no optimizar sobre toda la historia) | ✅ | `evaluate.py:44` por vintage |
| 7 | Validación por subgrupos (instrumento/mes/año/vintage/vol/fase/dir/horizonte/pre-post 2020/crisis) | 🟡 | Hay era/vol/DTE/mes; **faltan por año concreto y crisis explícita** |
| 8 | Costes y operabilidad (bid/offer, slippage, liquidez, frecuencia, tamaño, DTE) | 🟡 | `costs.py` da break-even/sensibilidad y trades/año; **coste es stub, sin bid/offer real** |

---

## 5. Gaps priorizados (lo que falta, con detalle técnico)

### 5.1 Pesos aprendidos por familia ❌ (brief §7, §9)
`WEIGHTS` (`score.py:24`) es universal por régimen. El brief exige aprender y
validar pesos **por familia**. Trabajo: estructurar `WEIGHTS` como
`dict[family][regime]` y ajustarlos por optimización walk-forward (maximizando
IC/utilidad neta de coste OOS), no por intuición.

### 5.2 Dos modelos + selector de régimen ❌ (brief §5, Paso 4, Fase 1–3)
Hoy `compute_opportunity` (`score.py:137`) elige una **fórmula** según régimen.
El brief pide un **modelo de reversión** y un **modelo de continuación**
entrenados por separado, y un **selector** que pondere ambos por probabilidad de
régimen (no un corte duro range/trend/transition). Esto también habilita el
"% de modelos que coinciden" del Bloque G.

### 5.3 Horizontes cortos D+1/D+3/D+5/D+10 🟡 (Bloque F, Paso 3)
`HORIZONS = (20, 25, 30)` (`targets.py:45`). Añadir `(1, 3, 5, 10)` propaga
columnas `fwd_h/mfe_h/mae_h/*_vol` y da probabilidades de continuación/reversión
a corto plazo para el timing.

### 5.4 Costes reales ❌ (Paso 8)
`COST_STUB_POINTS` (`dataset.py:23`) es un placeholder con `TODO(fase-0)`.
`costs.py` ya calcula break-even y sensibilidad 0x/1x/2x/3x — sólo falta
alimentarlo con **bid/offer y slippage medidos** por familia/liquidez/DTE.

### 5.5 Variables de régimen y dirección/nivel incompletas 🟡 (Bloques A/B/D/E)
Ampliar el feature set y su uso: variance ratio, autocorrelación, R²,
persistencia (A); EMA 20/50/100 + momentum 5/10/20 (B); `z_10`/`z_50`, Bollinger,
Keltner y distancia a estacionalidad dentro de `block_level`/`p_reversion` (D/E).
Sujeto a **poda de redundancia** (`diagnostics.py`) para no meter diez versiones
de lo mismo (brief §9, Paso 5).

### 5.6 Confianza rica 🟡 (Bloque G)
Extender `block_confidence` con estabilidad por año/vintage, consistencia OOS,
robustez a parámetros, dispersión de análogos (`analogues.py` ya da la cohorte) e
intervalo de confianza.

### 5.7 Backtest legacy long/flat → simétrico 🟡 (brief §4)
`backtesting/research/evaluate.py` ya opera long+short. El motor **legacy**
(`backtesting/backtest/engine.py`) sigue long/flat; decidir si se retira o se
migra a long/short/flat, para no ranquear por P&L long-only (brief §9).

---

## 6. Roadmap por fases (prioridad del PM)

> El PM fija la prioridad: **1º mean-reversion por estructura fija → 2º
> continuación → 3º selector**. El roadmap respeta ese orden y ataca primero las
> familias con ventaja probada (quarterly/semestral/yearly/flies).

### Fase 1 — Excelente motor de mean-reversion por estructura fija
Objetivo: el mejor modelo de reversión posible en quarterly, semestral, yearly y
flies, con pesos aprendidos por familia y costes reales.

1. **F1.1 Horizontes cortos** (§5.3): añadir `1,3,5,10` a `HORIZONS`
   (`targets.py`) + tests en `tests/test_targets.py`.
2. **F1.2 Enriquecer nivel/reversión** (§5.5): incorporar `z_10/z_50`, Bollinger,
   Keltner y estacionalidad a `block_level`/`block_p_reversion`, con poda de
   redundancia previa (`diagnostics.py`).
3. **F1.3 Pesos por familia** (§5.1): `WEIGHTS[family][regime]` + ajuste
   walk-forward que maximice utilidad OOS neta de coste, empezando por las 4
   familias de reversión.
4. **F1.4 Costes reales** (§5.4): medir bid/offer y slippage por familia y
   sustituir `COST_STUB_POINTS`; validar con `costs.py`.
5. **F1.5 Confianza de reversión** (§5.6): estabilidad por vintage/año y
   dispersión de análogos en `block_confidence`.
6. **F1.6 Validación**: walk-forward + subgrupos (añadir año concreto y crisis),
   reporte HTML de reversión por familia.

### Fase 2 — Modelo de continuación
Objetivo: modelo de tendencia **sin** asumir igual importancia que la reversión.

1. **F2.1 Features de tendencia** (§5.5): EMA 20/50/100 + alineación, momentum
   5/10/20, R², persistencia, comportamiento de correcciones, breakouts,
   estabilidad temporal, estructura de mercado.
2. **F2.2 Modelo de continuación**: probabilidades condicionales por horizonte
   (usando F1.1) — resultado posterior a la señal, no P&L de cruces.
3. **F2.3 Fuerza/calidad completa** (Bloque C): R², profundidad de correcciones,
   duración, falsas rupturas.
4. **F2.4 Validación**: walk-forward + subgrupos + costes; confirmar cuánta
   ventaja aporta realmente frente a la reversión.

### Fase 3 — Selector de régimen y score unificado
Objetivo: combinar reversión, continuación, nivel y confianza en un único
Opportunity Score honesto.

1. **F3.1 Probabilidad de régimen** (§5.5, Bloque A): variance ratio,
   autocorrelación, R², persistencia → probabilidad continua de régimen (no corte
   duro).
2. **F3.2 Selector**: `compute_opportunity` pondera modelo de reversión y de
   continuación por probabilidad de régimen; en transición, reducir score y
   exigir confirmación (brief §7).
3. **F3.3 Confianza como acuerdo de modelos** (Bloque G): "% de modelos que
   coinciden", intervalo de confianza.
4. **F3.4 Validación end-to-end**: walk-forward del score completo por familia y
   subgrupo; reporte final.

---

## 7. "Qué no debe hacer el desarrollador" (brief §9) — cumplimiento actual

| Anti-patrón | ¿Se evita hoy? |
|-------------|----------------|
| Buscar obligatoriamente un trend score | ✅ reversión primero |
| Mismo modelo para todas las familias | 🟡 calibración por familia, pero pesos universales (§5.1) |
| Ranquear indicadores sólo por P&L total | ✅ IC/OOS en `evaluate.py` (legacy long/flat aún por P&L, §5.7) |
| Asignar pesos por intuición | ❌ `WEIGHTS` a mano (§5.1) |
| Elegir mejor parámetro sobre toda la muestra | ✅ walk-forward por vintage |
| Confundir win rate con calidad | ✅ métricas múltiples |
| Sharpe alto con pocos trades = robusto | ✅ `costs.py` reporta trades/año; FDR |
| Peso a diez indicadores correlacionados | ✅ `diagnostics.py` (poda) — vigilar al añadir features (§5.5) |
| Recomendación sin nivel de confianza | ✅ `confidence` siempre presente |
| Borrar información histórica al actualizar | ✅ point-in-time, sin sobrescritura destructiva |

---

## 8. Decisiones abiertas (a confirmar con el PM antes de ejecutar)

1. **Backtest legacy long/flat**: ¿se retira o se migra a long/short/flat?
2. **Estacionalidad y comparables** (Bloque D): ¿fuente de referencia estacional
   y qué comparables (Brent-WTI, cracks) para "distancia a comparables"?
3. **Objetivo de optimización de pesos** (F1.3): ¿maximizar IC, utilidad neta de
   coste, o un Sharpe ajustado por confianza?
4. **Medición de costes** (F1.4): ¿hay bid/offer histórico disponible por
   familia, o se estima por liquidez/DTE?
5. **Drawdown-antes-de-beneficio** (Paso 3): ¿se añade como target adicional?
