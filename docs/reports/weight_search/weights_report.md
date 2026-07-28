# Optimización de pesos — reporte

Aviso: varios indicadores candidatos ya alimentan los bloques agregados (colinealidad); pesos 0 = indicador inactivo.

Dirichlet: 1200 · sparse: 800 · k: [2, 5] · semilla: 0 · min_bars: 60 · margen OOS: 0.0

| Familia | Sharpe IS | Sharpe IS eq | Sharpe OOS opt | Sharpe OOS eq | Splits | ¿Adoptado? |
|---|---|---|---|---|---|---|
| quarterly | 0.261 | 0.164 | 0.044 | 0.151 | 15 | no |
| semestral | 0.271 | 0.105 | 0.048 | 0.072 | 16 | no |
| yearly | 0.242 | 0.088 | -0.083 | 0.094 | 18 | no |
| flies | 0.356 | 0.356 | -0.303 | 0.348 | 17 | no |

## Pesos elegidos (full-sample)

### quarterly
- **Range activos:** rev_term, lvl_term, timing_term, vol_term, macd_div_term, rsi_div_term, mom_decel_term, er_drop_term, autocorr_term
- **Trend activos:** dir_term, qual_term, cont_term, ext_low, ema_align_term, mom10_term, r2_term, dirpers_term
```json
{
  "range": {
    "rev_term": 0.17329,
    "lvl_term": 0.159317,
    "timing_term": 0.003967,
    "vol_term": 0.162856,
    "macd_div_term": 0.362561,
    "rsi_div_term": 0.016632,
    "mom_decel_term": 0.01524,
    "er_drop_term": 0.005026,
    "autocorr_term": 0.101112
  },
  "trend": {
    "dir_term": 0.036758,
    "qual_term": 0.065922,
    "cont_term": 0.162786,
    "ext_low": 0.354068,
    "ema_align_term": 0.100964,
    "mom10_term": 0.012094,
    "r2_term": 0.182055,
    "dirpers_term": 0.085352
  },
  "transition_shrink": 0.4
}
```
### semestral
- **Range activos:** lvl_term, timing_term, macd_div_term, mom_decel_term
- **Trend activos:** cont_term, ext_low, mom10_term, r2_term, dirpers_term
```json
{
  "range": {
    "rev_term": 0.0,
    "lvl_term": 0.32736,
    "timing_term": 0.168739,
    "vol_term": 0.0,
    "macd_div_term": 0.316435,
    "rsi_div_term": 0.0,
    "mom_decel_term": 0.187465,
    "er_drop_term": 0.0,
    "autocorr_term": 0.0
  },
  "trend": {
    "dir_term": 0.0,
    "qual_term": 0.0,
    "cont_term": 0.210551,
    "ext_low": 0.452431,
    "ema_align_term": 0.0,
    "mom10_term": 0.144309,
    "r2_term": 0.105036,
    "dirpers_term": 0.087673
  },
  "transition_shrink": 0.4
}
```
### yearly
- **Range activos:** rev_term, lvl_term, timing_term, vol_term, macd_div_term, rsi_div_term, mom_decel_term, er_drop_term, autocorr_term
- **Trend activos:** dir_term, qual_term, cont_term, ext_low, ema_align_term, mom10_term, r2_term, dirpers_term
```json
{
  "range": {
    "rev_term": 0.456543,
    "lvl_term": 0.030199,
    "timing_term": 0.060739,
    "vol_term": 0.030397,
    "macd_div_term": 0.01985,
    "rsi_div_term": 0.052433,
    "mom_decel_term": 0.032461,
    "er_drop_term": 0.260906,
    "autocorr_term": 0.056472
  },
  "trend": {
    "dir_term": 0.203234,
    "qual_term": 0.068516,
    "cont_term": 0.331207,
    "ext_low": 0.220226,
    "ema_align_term": 0.088639,
    "mom10_term": 0.019687,
    "r2_term": 0.002887,
    "dirpers_term": 0.065605
  },
  "transition_shrink": 0.4
}
```
### flies
- **Range activos:** rev_term, lvl_term, timing_term, vol_term
- **Trend activos:** dir_term, qual_term, cont_term, ext_low
```json
{
  "range": {
    "rev_term": 0.25,
    "lvl_term": 0.25,
    "timing_term": 0.25,
    "vol_term": 0.25,
    "macd_div_term": 0.0,
    "rsi_div_term": 0.0,
    "mom_decel_term": 0.0,
    "er_drop_term": 0.0,
    "autocorr_term": 0.0
  },
  "trend": {
    "dir_term": 0.25,
    "qual_term": 0.25,
    "cont_term": 0.25,
    "ext_low": 0.25,
    "ema_align_term": 0.0,
    "mom10_term": 0.0,
    "r2_term": 0.0,
    "dirpers_term": 0.0
  },
  "transition_shrink": 0.4
}
```
