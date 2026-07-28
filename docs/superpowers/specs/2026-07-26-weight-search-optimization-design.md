# Diseño — Búsqueda de pesos del Opportunity Score por familia (walk-forward)

Fecha: 2026-07-26 · Ámbito: `src/crudewatch/scoring` (`score.py`, nuevo
`weight_search.py`), `scripts/`, `docs/reports/`, `tests/`.

> Aborda el gap **§5.1 / F1.3** del roadmap (`docs/superpowers/plans/2026-07-25-scoring-gaps-roadmap.md`):
> los pesos del composite hoy son **universales y puestos a mano** (`WEIGHTS`,
> `score.py:28`). El brief marca como anti-patrones "asignar pesos por intuición"
> y "elegir el mejor parámetro sobre toda la muestra". Este diseño aprende pesos
> **por familia** con validación **walk-forward** y solo los adopta si generalizan
> fuera de muestra.

---

## 1. Problema y objetivo

Encontrar, **por familia**, los pesos de los dos regímenes del Opportunity Score
que maximizan el rendimiento del backtest "seguir el score", sin sobreajustar.

- **Familias objetivo (Fase 1, reversión):** `quarterly`, `semestral`, `yearly`, `flies`.
- **Función objetivo:** **Sharpe anualizado** de la serie de PnL diaria **agrupada**
  de todos los contratos de la familia, **neta de coste** (`COST_STUB_POINTS`).
- **Validación:** walk-forward por *vintage* (ventana expandiente).
- **Entregable final:** pesos aprendidos aplicados a producción **solo** para las
  familias que baten al baseline equal-weight fuera de muestra.

Fuera de alcance (YAGNI): `transition_shrink` (fijo 0.4), umbrales de entrada/salida
(fijos ±50/±20), horizonte (fijo 25), las otras 4 familias, costes reales.

---

## 2. Espacio de búsqueda

Por familia, dos vectores de peso, cada uno **no-negativo y sumando 1**:

- **Rango:** `[rev_term, lvl_term, timing_term, vol_term]`
- **Tendencia:** `[dir_term, qual_term, cont_term, ext_low]`

Baseline = equal-weight actual (`0.25` cada uno). `transition_shrink` permanece 0.4.

**Método (aprobado): búsqueda aleatoria por símplex (Dirichlet).** Para cada régimen
se muestrean `N` vectores `~ Dirichlet(alpha=1)` (uniforme sobre el símplex). El
conjunto de candidatos incluye además el baseline equal-weight y los 4 vértices
one-hot de cada régimen. Se evalúa el producto cartesiano muestreado de forma
eficiente (ver §4). `N` por defecto ≈ 4000 (configurable, con semilla fija).

---

## 3. Insight de rendimiento: precómputo de términos por barra

Los 8 términos que se combinan **no dependen de los pesos**; solo su combinación
lineal final sí. Reescribiendo `compute_opportunity` (`score.py:135`):

- Rango: `o = clip( m_range · (w_range · range_terms) )`, con
  `range_terms = [_rev_term(p_rev), min(|level|/100,1), timing_term(row,cal), vol_term(row)]`
  y `m_range ∈ {+100, −100, 0}` (según el signo del nivel; 0 si `level == 0`).
- Tendencia: `o = clip( m_trend · (w_trend · trend_terms) )`, con
  `trend_terms = [min(|direction|/100,1), strength/100, _cont_term(p_cont), 1−min(|level|/100,1)]`
  y `m_trend ∈ {+100, −100, 0}` (según el signo de la dirección; 0 si `direction == 0`).
- Transición: `o = clip( 0.4 · m_range · (w_range · range_terms) )`.

Cada término está en `[0,1]` y los pesos suman 1 ⇒ la convicción está en `[0,1]` y
`o ∈ [−100, +100]` sin recorte adicional (se mantiene `clip` por seguridad).

Por tanto, **una sola vez por contrato** (y estrictamente point-in-time, igual que
`score_series`), se precomputa por barra:

```
PrecomputedContract:
  date, close, open, high, low        # para el simulate
  regime_code    ∈ {0=range, 1=trend, 2=transition}
  m_range        ∈ {+100, −100, 0}
  range_terms    (n, 4)
  m_trend        ∈ {+100, −100, 0}
  trend_terms    (n, 4)
```

Con esto, el score de **cualquier** vector de pesos es un producto matriz–vector
(`range_terms @ w_range`, `trend_terms @ w_trend`) + selección por régimen. Barrer
miles de combinaciones es casi instantáneo; la parte cara (refit del calibrador por
barra) se paga una sola vez por contrato.

---

## 4. Arquitectura y módulos

### 4.1 `src/crudewatch/scoring/weight_search.py` (motor puro)

- `precompute_contract(data, family, contract, horizon=25) -> PrecomputedContract`
  — bucle point-in-time por barra (calibrador refit ≤ fecha, `outcome_asof=d`),
  reutilizando `fit_calibrator` + `compute_blocks`; extrae los términos de §3.
- `precompute_family(data, family, horizon=25, min_bars=60) -> list[PrecomputedContract]`
  — todos los contratos con ≥ `min_bars` barras. Cacheado en la app si hiciera falta
  (no en este alcance).
- `opportunity_from_precomputed(pc, w_range, w_trend) -> np.ndarray` — vectorizado.
- `simplex_samples(n, dim=4, seed) -> np.ndarray (n, dim)` — Dirichlet + baseline + one-hot.
- `_fast_pnl(pc, w_range, w_trend, cost, enter=50, exit=20) -> np.ndarray` — PnL diaria
  neta de un contrato: reutiliza `_hysteresis`, lag de ejecución (open[t+1]) y coste
  por pata idéntico a `backtest.simulate` (mismo criterio de puntos/costes).
- `pooled_sharpe(pnl_arrays) -> float` — concatena las series diarias de todos los
  contratos y devuelve `mean/std·√252` (ddof=1; NaN si <2 puntos o std=0).
- `objective(pcs, w_range, w_trend, cost) -> float` — Sharpe agrupado sobre `pcs`.
- `search_weights(pcs, cost, n=4000, seed=0) -> WeightSearchResult` — evalúa candidatos
  y devuelve el mejor `(w_range, w_trend)`, su Sharpe, y el Sharpe del equal-weight.

### 4.2 Walk-forward

- `walk_forward_weights(data, family, horizon=25, n=4000, seed=0) -> WalkForwardResult`
  — usa `walk_forward_splits` (de `backtesting.research.evaluate`) sobre los *vintages*
  presentes. Para cada split: `search_weights` en contratos de entrenamiento →
  evalúa Sharpe **OOS** de esos pesos en los contratos del vintage de test, y también
  el Sharpe OOS del equal-weight. Agrega: `oos_sharpe_opt` y `oos_sharpe_equal`
  (media sobre splits, ponderada por nº de puntos diarios).

### 4.3 Estructura de pesos en `score.py`

- `WEIGHTS` (equal-weight) **se mantiene** como baseline por defecto.
- Nuevo `FAMILY_WEIGHTS: dict[str, dict]` que se **carga desde un JSON**
  (`src/crudewatch/scoring/family_weights.json`) si existe; si no, `{}`.
  Esto evita reescribir código fuente: el script solo escribe el JSON.
- `weights_for(family) -> dict` → `FAMILY_WEIGHTS.get(family, WEIGHTS)`.
- `compute_opportunity(blocks, row, calibrator, weights=None)` — `weights=None`
  usa `WEIGHTS`. Se enhebra por `_range_opportunity` / `_trend_opportunity`.
- `score_instrument` / `score_family` resuelven `weights_for(family)` y lo pasan.
- `backtest.score_series` pasa `weights_for(family)` (el backtest refleja los pesos
  de la familia). El buscador NO usa este camino: usa el precómputo de §3.

### 4.4 Script y reporte

- `scripts/optimize_weights.py` — corre `walk_forward_weights` + `search_weights`
  (full-sample) para las 4 familias, escribe:
  - `docs/reports/weight_search/weights_report.md` — tabla: familia, Sharpe OOS
    equal vs opt, Sharpe in-sample, ¿adoptado?, pesos elegidos (por régimen).
  - `src/crudewatch/scoring/family_weights.json` — **solo** las familias adoptadas.
- **Política de adopción:** el criterio es **maximizar el Sharpe medio**. Para cada
  familia se eligen los pesos que dan el **mayor Sharpe** (objetivo §1) y se **adoptan
  siempre que superen al equal-weight fuera de muestra** (`oos_sharpe_opt >
  oos_sharpe_equal`, margen por defecto > 0). Las familias que no generalizan quedan
  en equal-weight y se marcan en el reporte.

---

## 5. Flujo de datos

```
enriched_frame(family)
  └─ precompute_family ──► [PrecomputedContract]  (caro, 1×, point-in-time)
        ├─ walk_forward_weights ─► (oos_sharpe_opt, oos_sharpe_equal)   [honestidad]
        └─ search_weights(full)  ─► (w_range*, w_trend*, sharpe_is)      [pesos prod]
  └─ política de adopción ──► family_weights.json  +  weights_report.md
```

En producción, `compute_opportunity` lee `weights_for(family)` → los pesos adoptados
o, si no hay, el equal-weight actual. Comportamiento **idéntico** para familias no
optimizadas.

---

## 6. Casos borde y decisiones

- **Contratos cortos:** se excluyen con `min_bars` (60). Si una familia queda con <2
  contratos utilizables, se salta y se marca en el reporte.
- **Sharpe indefinido:** series con <2 puntos o `std==0` → Sharpe NaN; se tratan como
  peor que cualquier candidato válido (no se seleccionan).
- **Empates / ruido:** semilla fija ⇒ reproducible. El baseline equal-weight siempre
  está entre los candidatos, de modo que el "mejor in-sample" nunca es peor que él.
- **Familias con `level==0`/`direction==0`** en muchas barras: `m=0` ⇒ score 0 ⇒ sin
  operación; es correcto (no hay señal). No se penaliza el objetivo salvo por ausencia
  de PnL.
- **Coherencia con el backtest existente:** `_fast_pnl` replica exactamente las
  convenciones de `backtest.simulate` (lag open[t+1], coste `cost/2` por pata). Test de
  paridad lo garantiza (§7).

---

## 7. Testing

- **Paridad de precómputo:** para equal-weight, `opportunity_from_precomputed` ==
  `compute_opportunity` barra a barra (frame sintético) — y para varios vectores de
  pesos aleatorios.
- **Símplex:** `simplex_samples` no-negativo, filas suman 1, incluye equal-weight y
  one-hots; determinista con semilla.
- **Paridad de PnL:** `_fast_pnl` == PnL diaria derivada de `backtest.simulate` para el
  mismo contrato/pesos (reconciliación).
- **Objetivo:** `pooled_sharpe` correcto en un caso a mano; `search_weights` devuelve
  un Sharpe ≥ el del equal-weight en train (por construcción).
- **Point-in-time:** `precompute_contract` no cambia al añadir barras futuras
  (mismo test de no-fuga que `score_series`).
- **`weights_for` / carga JSON:** fallback a `WEIGHTS` cuando no hay override; override
  correcto cuando el JSON está presente.

---

## 8. Riesgos

- **Sobreajuste residual:** aun con walk-forward, un espacio grande + Sharpe ruidoso
  puede "pescar" ruido. Mitigación: baseline siempre presente, gate de adopción OOS,
  reporte in-sample vs OOS visible, y `N` moderado.
- **Coste real ausente:** se usa el *stub* por familia; los pesos podrían cambiar con
  costes reales (gap §5.4, fuera de alcance).
- **Sharpe agrupado por concatenación** (no por calendario) mezcla contratos de
  distinta época; es una elección deliberada (más muestra, menos huecos) y se
  documenta en el reporte.
