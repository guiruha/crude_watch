# Diseño — Búsqueda de indicadores + pesos (subconjuntos, validación OOS honesta)

Fecha: 2026-07-26
Estado: aprobado (brainstorming) — pendiente de plan de implementación.

## 1. Objetivo

Ampliar el optimizador de pesos del Opportunity Score para que, además de buscar
los **pesos** de los 8 términos actuales, considere **indicadores nuevos** como
términos candidatos y busque la mejor **combinación (subconjunto) + pesos** por
familia. Nos quedamos, por familia, con la configuración de mayor **Sharpe
fuera de muestra (OOS) honesto**, siempre que bata al baseline equal-weight
actual. Objetivo: maximizar Sharpe/PnL OOS sin sobreajustar.

Universo: las 4 familias de reversión (`quarterly`, `semestral`, `yearly`,
`flies`).

## 2. Arquitectura actual (recap)

El score combina, por régimen, términos de convicción en `[0,1]`, firmados por
el signo del régimen:

- **Rango** (fade del nivel; `sign = -sign(level)`): `rev_term`, `lvl_term`,
  `timing_term`, `vol_term`.
- **Tendencia** (`sign = sign(direction)`): `dir_term`, `qual_term`,
  `cont_term`, `ext_low`.
- Score de bloque = `±100 · Σ wᵢ·termᵢ`, con `Σ wᵢ = 1` por bloque.
- `transition` = `transition_shrink · score_rango`.

Los features PIT viven en `research/features.py` (`FEATURES`) y ya están en el
dataset enriquecido (sin lookahead, verificado por test). El calibrador
(`FamilyCalibrator`) guarda ECDFs pooled por familia para `ECDF_FEATURES`.

## 3. Indicadores nuevos (set curado aprobado)

Se añaden como términos candidatos adicionales (no sustituyen a los 8). "Apagar"
un indicador = peso 0. Todos los features ya existen en el dataset enriquecido.

| key nuevo        | régimen   | feature base   | tipo                | ¿en score hoy? |
|------------------|-----------|----------------|---------------------|----------------|
| `macd_div_term`  | rango     | `macd_div`     | direccional-acuerdo | no             |
| `rsi_div_term`   | rango     | `rsi_div_14`   | direccional-acuerdo | sí (en p_rev)  |
| `mom_decel_term` | rango     | `mom_decel_10` | calidad/magnitud    | sí (timing)    |
| `er_drop_term`   | rango     | `er_drop_20`   | calidad/magnitud    | no             |
| `autocorr_term`  | rango     | `autocorr_20`  | calidad/magnitud    | sí (trendiness)|
| `ema_align_term` | tendencia | `ema_align`    | direccional-acuerdo | sí (direction) |
| `mom10_term`     | tendencia | `mom_10`       | direccional-acuerdo | sí (direction) |
| `r2_term`        | tendencia | `r2_20`        | calidad/magnitud    | sí (strength)  |
| `dirpers_term`   | tendencia | `dir_persistence_20` | calidad/magnitud | sí (strength) |

Riesgo asumido: **colinealidad / doble conteo** — varios de estos ya alimentan
los bloques agregados. Darles voz directa y ponderable es intencional; la búsqueda
puede anularlos (peso 0) y el gate OOS honesto protege contra sobreajuste. Se
documenta explícitamente en el reporte.

### 3.1 Transformación fija por tipo (NO se busca)

Cada término produce una convicción en `[0,1]`. Normalización PIT y NaN→valor
neutro (0.0 para magnitudes; término inactivo aporta 0).

**Direccional-de-acuerdo** — magnitud del indicador solo cuando su signo coincide
con la dirección del trade del bloque; si no coincide, 0.

- Dirección del trade en **rango**: precio esperado `dir_px = -sign(level)`
  (barato→sube, caro→baja).
- Dirección del trade en **tendencia**: `dir_px = sign(direction)`.
- Magnitud normalizada `m ∈ [0,1]`:
  - Si el feature tiene ECDF: `m = |2·percentile − 1|` (fuerza relativa a la
    familia) y su signo `s = sign(signed_pct(percentile))`.
  - Si no tiene ECDF: se **añade** a `ECDF_FEATURES` (`macd_div`, `rsi_div_14`)
    para normalizarlo igual (pooled, PIT). (Alternativa de respaldo: `s·m =
    tanh(value/scale)`.)
- Término = `m` si `s == dir_px`, si no `0`. (Si `level==0`/`direction==0`, el
  bloque ya devuelve 0 antes; término irrelevante.)

**Calidad/magnitud** — refuerzan el bloque sin importar el signo, en `[0,1]`:

- `mom_decel_term` = `percentile(neg_mom_decel_10, −mom_decel_10)` (exhaustión;
  idéntico a `timing_term` — colinealidad conocida).
- `er_drop_term` = `percentile(neg_er_drop_20, −er_drop_20)` (decaimiento de
  calidad de tendencia → favorece reversión). Requiere ECDF `neg_er_drop_20`.
- `autocorr_term` = `1 − percentile(autocorr_20)` (autocorr bajo/negativo →
  reversión).
- `r2_term` = `clip(r2_20, 0, 1)` (linealidad de tendencia).
- `dirpers_term` = `clip(dir_persistence_20, 0, 1)` (persistencia direccional).

## 4. Cambios de producción (para que los pesos adoptados funcionen en vivo)

### 4.1 Registro de términos en `score.py`

Refactor **conservador de comportamiento**: `_range_opportunity` /
`_trend_opportunity` construyen la convicción sumando sobre un **registro
ordenado** de términos:

```
TERM_REGISTRY: {key: callable(blocks, row, cal, level, direction) -> float[0,1]}
RANGE_TERM_KEYS: tuple  # orden canónico: 8 actuales primero, luego nuevos
TREND_TERM_KEYS: tuple
```

- Los 8 términos actuales pasan al registro con **fórmula byte-idéntica**.
- `conviction = Σ_{k∈KEYS} w["range"].get(k, 0.0) · term_k(...)`.
- `WEIGHTS` por defecto lista solo los 8 (peso 0.25 c/u); los nuevos → peso 0.
  **El comportamiento por defecto en vivo es idéntico al actual.**
- `weights_for(family)` y el loader JSON toleran claves nuevas y claves
  ausentes (peso 0). Loader defensivo ya existente se mantiene.

### 4.2 Calibrador

Extender `ECDF_FEATURES` con `macd_div`, `rsi_div_14` y añadir ECDFs
`neg_er_drop_20` (y `er_drop_20` si hace falta). PIT: los ECDF se ajustan sobre
la historia pooled disponible; no usan futuro.

## 5. Módulo de búsqueda (`weight_search.py`)

### 5.1 Precómputo extendido

`precompute_contract` produce, por barra (PIT, refit del calibrador sobre
`rows ≤ d`), una matriz de términos por bloque con el **orden canónico**
(`RANGE_TERM_KEYS`, `TREND_TERM_KEYS`). Debe cumplir **paridad algebraica**:
`opportunity_from_precomputed(pc, w_range, w_trend, transition_shrink)` ≡
`compute_opportunity(blocks, row, cal, weights)` para cualquier vector de pesos
(incluyendo pesos >0 en términos nuevos), a `atol 1e-9`.

### 5.2 Muestreo: subconjuntos + pesos

- `simplex_samples` (Dirichlet, con equal-weight y one-hots) — ya existe, se
  generaliza a dimensión = nº de términos del bloque.
- **Nuevo** `sparse_simplex_samples(n_terms, n, seed, k_min, k_max)`: por muestra
  elige un tamaño `k ∈ [k_min, k_max]`, un subconjunto aleatorio de `k` términos
  activos, Dirichlet sobre esos `k`, ceros en el resto. Filas ≥0 y suma 1.
- El baseline **equal-weight de los 8 actuales** (nuevos = 0) es un candidato
  garantizado y el patrón de comparación.

### 5.3 Objetivo y PnL

Sin cambios de convención: `fast_pnl` (paridad con `simulate`), `pooled_sharpe`
(ddof=1, √252), `objective`. Solo cambia la dimensión de los vectores de peso.

## 6. Validación OOS honesta (walk-forward anidado)

Para no engañarnos al "quedarnos con el mejor OOS":

- Splits expanding por vintage (`expanding_vintage_splits`, `min_train=3`).
- En **cada split**: la **selección de modelo** (mejor config subconjunto+pesos
  por Sharpe pooled) usa **solo los contratos de train** (vintages anteriores).
  La config seleccionada se evalúa (fast_pnl → Sharpe) en los contratos del
  **vintage retenido**.
- Se agrupa el PnL OOS de la config seleccionada a través de los splits →
  `oos_sharpe_selected`. Se compara contra el equal-weight evaluado en los
  mismos bares OOS → `oos_sharpe_equal`.
- **Config desplegada** = misma selección (subconjunto+pesos) sobre **todo** el
  histórico (train-all). Patrón estándar "gate con walk-forward, refit en todo".

`WeightSearchResult` gana `active_keys_range`/`active_keys_trend` (subconjunto
elegido). `WalkForwardResult` mantiene `oos_sharpe_opt`/`oos_sharpe_equal`/
`n_splits`.

## 7. Script, reporte y adopción

`scripts/optimize_weights.py` (extender):

- Nuevos flags: `--dirichlet` (def. 1200), `--sparse` (def. 800),
  `--k-min`/`--k-max` (tamaño de subconjunto disperso).
- Por familia: precómputo → walk-forward honesto → selección full-sample.
- **Adopción**: se escribe la config de la familia en `family_weights.json`
  **solo si** `oos_sharpe_opt > oos_sharpe_equal + margin`. Si no bate, se
  **elimina** esa familia del JSON (fallback equal-weight). (Demote-on-rerun ya
  implementado.)
- El JSON de pesos incluye claves de los términos nuevos (0 si inactivos).
- Reporte markdown: tabla Sharpe IS/OOS opt/eq, splits, subconjunto elegido por
  bloque, ¿adoptado?, y nota de colinealidad.

Presupuesto por defecto (elección del usuario, "rápido/preview"): 1200 Dirichlet
+ 800 dispersos por split.

## 8. Testing (TDD por tarea)

- **Regresión de comportamiento**: la suite actual (`test_scoring`,
  `test_family_weights`, `test_backtest`, `test_weight_search`) pasa sin cambios
  tras el refactor a registro (comportamiento por defecto idéntico).
- **Paridad**: precómputo extendido ≡ `compute_opportunity` para pesos
  arbitrarios (incl. términos nuevos) a `1e-9`.
- **Términos nuevos (unit)**: cada término ∈ `[0,1]`; regla de acuerdo
  direccional (activo solo si el signo coincide); NaN→neutro; PIT (append de
  barras futuras no cambia valores pasados).
- **Sparse sampler**: filas ≥0, suma 1, exactamente `k` activos, `k∈[k_min,k_max]`.
- **No-fuga anidada**: la selección por split solo ve vintages de train
  (guard con monkeypatch, como en Task 6).
- **Paridad `fast_pnl`↔`simulate`** se mantiene (incrementos diarios).

## 9. Restricciones globales (invariantes)

- PIT estricto (refit `outcome_asof=d` sobre `rows ≤ d`); features ya son as-of-t.
- Pesos por bloque ≥0 y suman 1; equal-weight de los 8 siempre candidato.
- Paridad precómputo↔score y `fast_pnl`↔`simulate` a `1e-9`.
- Selección de modelo dentro de train; evaluación OOS en vintage retenido.
- Comportamiento en vivo por defecto **idéntico** al actual (nuevos términos peso 0).
- Adopción solo si OOS honesto bate equal-weight + margen; demote si no.
- `transition_shrink` = 0.4 fijo; umbrales ±50/±20; horizonte 25.

## 10. Riesgos

- **Sobreajuste** por espacio de búsqueda mayor → mitigado por selección honesta
  dentro de train + gate OOS + preview con pocas muestras.
- **Colinealidad / doble conteo** de términos ya presentes en los bloques →
  documentado; la búsqueda puede anularlos.
- **Mis-especificación de la transformación/polaridad** → tests unitarios por
  término; transformación fija y sencilla.
- **Runtime** dominado por el precómputo PIT (una vez por familia); la selección
  anidada repite búsqueda por split → mantener muestras moderadas en preview.
- **Selección sobre OOS**: evitado por diseño (selección en train, no en OOS).
