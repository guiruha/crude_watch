# Optimización de pesos — reporte

Aviso: varios indicadores candidatos ya alimentan los bloques agregados (colinealidad); pesos 0 = indicador inactivo.

Dirichlet: 1200 · sparse: 800 · k: [2, 5] · semilla: 0 · min_bars: 60 · margen OOS: 0.0

| Familia | Sharpe IS | Sharpe IS eq | Sharpe OOS opt | Sharpe OOS eq | Splits | ¿Adoptado? |
|---|---|---|---|---|---|---|
| quarterly | 0.300 | 0.164 | 0.088 | 0.151 | 15 | no |
| semestral | 0.297 | 0.105 | 0.114 | 0.072 | 16 | sí |
| yearly | 0.232 | 0.088 | 0.077 | 0.094 | 18 | no |
| flies | 0.458 | 0.356 | -0.174 | 0.348 | 17 | no |

## Pesos elegidos (full-sample)

### quarterly
- **Range activos:** rev_term, lvl_term, timing_term, vol_term, macd_div_term, rsi_div_term, mom_decel_term, er_drop_term
- **Trend activos:** dir_term, qual_term, cont_term, ext_low, ema_align_term, mom10_term, dirpers_term
```json
{
  "range": {
    "rev_term": 0.202791,
    "lvl_term": 0.165476,
    "timing_term": 0.048628,
    "vol_term": 0.018388,
    "macd_div_term": 0.335251,
    "rsi_div_term": 0.007131,
    "mom_decel_term": 0.123358,
    "er_drop_term": 0.098976
  },
  "trend": {
    "dir_term": 0.018577,
    "qual_term": 0.348026,
    "cont_term": 0.146505,
    "ext_low": 0.284526,
    "ema_align_term": 0.000629,
    "mom10_term": 0.111328,
    "dirpers_term": 0.09041
  },
  "transition_shrink": 0.4
}
```
### semestral
- **Range activos:** rev_term, lvl_term, timing_term, vol_term, macd_div_term, rsi_div_term, mom_decel_term, er_drop_term
- **Trend activos:** dir_term, qual_term, cont_term, ext_low, ema_align_term, mom10_term, dirpers_term
```json
{
  "range": {
    "rev_term": 0.045854,
    "lvl_term": 0.239523,
    "timing_term": 0.227155,
    "vol_term": 0.039135,
    "macd_div_term": 0.22349,
    "rsi_div_term": 0.025817,
    "mom_decel_term": 0.181002,
    "er_drop_term": 0.018024
  },
  "trend": {
    "dir_term": 0.06723,
    "qual_term": 0.035991,
    "cont_term": 0.219398,
    "ext_low": 0.465609,
    "ema_align_term": 0.155795,
    "mom10_term": 0.041038,
    "dirpers_term": 0.014939
  },
  "transition_shrink": 0.4
}
```
### yearly
- **Range activos:** rev_term, lvl_term, timing_term, vol_term, macd_div_term, rsi_div_term, mom_decel_term, er_drop_term
- **Trend activos:** dir_term, qual_term, cont_term, ext_low, ema_align_term, mom10_term, dirpers_term
```json
{
  "range": {
    "rev_term": 0.220192,
    "lvl_term": 0.134764,
    "timing_term": 0.041263,
    "vol_term": 0.010275,
    "macd_div_term": 0.300626,
    "rsi_div_term": 0.069433,
    "mom_decel_term": 0.065983,
    "er_drop_term": 0.157463
  },
  "trend": {
    "dir_term": 0.07566,
    "qual_term": 0.158873,
    "cont_term": 0.171516,
    "ext_low": 0.222785,
    "ema_align_term": 0.007797,
    "mom10_term": 0.024986,
    "dirpers_term": 0.338383
  },
  "transition_shrink": 0.4
}
```
### flies
- **Range activos:** rev_term, lvl_term, timing_term, vol_term, macd_div_term, rsi_div_term, mom_decel_term, er_drop_term
- **Trend activos:** dir_term, qual_term, cont_term, ext_low, ema_align_term, mom10_term, dirpers_term
```json
{
  "range": {
    "rev_term": 0.442869,
    "lvl_term": 0.064624,
    "timing_term": 0.100341,
    "vol_term": 0.05158,
    "macd_div_term": 0.068952,
    "rsi_div_term": 0.048903,
    "mom_decel_term": 0.170144,
    "er_drop_term": 0.052588
  },
  "trend": {
    "dir_term": 0.151347,
    "qual_term": 0.006943,
    "cont_term": 0.323703,
    "ext_low": 0.123225,
    "ema_align_term": 0.01414,
    "mom10_term": 0.040871,
    "dirpers_term": 0.339769
  },
  "transition_shrink": 0.4
}
```
