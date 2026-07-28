# Clasificación de régimen por Índice de Persistencia de Tendencia (TPI)

Fecha: 2026-07-26
Estado: propuesta (pendiente de revisión del usuario)

## Objetivo

Sustituir la clasificación de régimen **solo por Efficiency Ratio (ER)** por un
**ensemble de varias métricas de persistencia** combinadas en un único índice
0–100 (TPI), manteniendo **coherentes** en la misma definición: (a) la etiqueta
de régimen, (b) las máscaras `range`/`trend` con las que se calibran las
probabilidades de reversión/continuación, y (c) el emparejamiento de cohortes
análogas. Alcance: **solo cierre** (sin ADX/Choppiness, que requerirían OHLC).

Métricas del TPI (todas orientadas a "más alto = más tendencia"):

- `er_20` (ya existe) — recorrido neto / camino.
- `variance_ratio_5` (ya existe) — Lo-MacKinlay 5/60.
- `autocorr_20` (ya existe) — autocorrelación lag-1 de cambios.
- `hurst` (**nuevo**) — exponente de Hurst (R/S) sobre ventana de 100 sesiones.

Fuera de alcance ahora: ADX, Choppiness (OHLC), half-life de reversión. El
backtest offline (`backtesting/`) no se modifica.

## Definición del TPI

Para cada fila, `TPI = media de los percentiles de familia` de las métricas
disponibles (nan-safe, como ya hace `block_trendiness`):

```
TPI(fila) = mean_over_available( percentile(ecdf[m], valor_m) )   ∈ [0, 1]
```

Los cuatro términos son de orientación directa, así que no hace falta invertir
ninguno. `trendiness = TPI * 100` (el bloque A pasa a ser exactamente el TPI).

## Clasificación

Se calculan los **terciles del TPI** sobre la historia de la familia
(`regime_thresholds`, el mismo helper de terciles que hoy usa ER):

```
TPI <= tpi_lo  -> range
TPI >= tpi_hi  -> trend
en medio       -> transition
TPI = NaN      -> transition
```

`tpi_lo`/`tpi_hi` sustituyen a `er_lo`/`er_hi` como umbrales de **clasificación**
y de **máscaras de calibración**. `er_lo`/`er_hi` se conservan en el calibrador
solo para las **bandas del gráfico de distribución de ER** en la app.

Point-in-time: el TPI usa los ECDF de familia (rebanada `<= as_of` que ya hace
`fit_calibrator`) y las features as-of `t`; no mira al futuro.

## Cambios por archivo

1. `src/crudewatch/indicators.py`
   - `hurst_exponent(close, window=100)`: serie rolling del exponente de Hurst
     por R/S sobre log-retornos, look-ahead free, `NaN` en el warmup.

2. `src/crudewatch/research/features.py`
   - Registrar `"hurst": lambda c: hurst_exponent(c, 100)` en `FEATURES`.

3. `src/crudewatch/scoring/blocks.py`
   - `TPI_FEATURES = ("er_20", "variance_ratio_5", "autocorr_20", "hurst")`.
   - `ECDF_FEATURES += ("hurst",)`.
   - `FamilyCalibrator`: nuevos campos `tpi_lo`, `tpi_hi` (se mantienen `er_lo`,
     `er_hi`).
   - `_tpi_vector(data, calibrator) -> np.ndarray` y `trend_persistence_index(row,
     calibrator) -> float` (0..1), nan-safe, compartidos por scoring y análogos.
   - `fit_calibrator`: tras construir los ECDF, calcular `tpi` vectorizado,
     `tpi_lo, tpi_hi = regime_thresholds(tpi_valid)`, y `range_mask = tpi <=
     tpi_lo`, `trend_mask = tpi >= tpi_hi` (sustituyen a las de ER).
   - `regime_label(tpi: float, calibrator)`: clasifica por `tpi_lo/tpi_hi`.
   - `block_trendiness`: devuelve `trend_persistence_index(row, cal) * 100`.

4. `src/crudewatch/scoring/score.py`
   - Calcular `tpi = trend_persistence_index(row, cal)` una vez; `regime =
     regime_label(tpi, cal)`; `trendiness = tpi * 100`.

5. `src/crudewatch/scoring/analogues.py`
   - `_vec_regime`: recibir el vector TPI (vía `_tpi_vector`) y `tpi_lo/tpi_hi`.

6. `app/screens/component.py`
   - `COMPONENTS["regime"]["features"] += ("hurst", …)`; `FEATURE_CTX["hurst"]`
     (label + refline 0.5). Actualizar `help`/`calc` del bloque Régimen para
     describir el TPI y las cuatro métricas.

## Tests

- Actualizar `tests/test_scoring.py::regime_label`: construir un calibrador con
  `tpi_lo/tpi_hi` y comprobar que `regime_label(tpi, cal)` clasifica bien.
- Nuevo: coherencia — para la última fila de un contrato, la etiqueta de
  `score_instrument` coincide con la que produce `_vec_regime` en análogos.
- Nuevo: `hurst_exponent` devuelve valores finitos en [0,1]-ish tras el warmup y
  `NaN` antes.
- Recalcular ECDFs/score (no hay artefactos persistidos salvo el CSV de métricas
  de investigación, que no depende de esto).

## Riesgos

- Los valores del Opportunity Score cambiarán (nueva clasificación → nuevos
  pesos/gating). Es lo esperado.
- Hurst por R/S es algo ruidoso en ventanas cortas; se mitiga con ventana 100 y
  con el promedio nan-safe (si `hurst` es NaN, el TPI usa las otras tres).
