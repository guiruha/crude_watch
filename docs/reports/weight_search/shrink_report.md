# Optimización con shrinkage — reporte

Encoge la config seleccionada hacia equal-weight (λ=1 ≡ equal). Se adopta la λ con mejor Sharpe OOS agregado si supera al equal-weight.

Dirichlet: 75000 · sparse: 75000 · k: [1, 9] · min_train: 5 · semilla: 0 · margen: 0.0

| Familia | OOS λ0.10 | OOS λ0.20 | OOS λ0.30 | OOS λ0.40 | OOS λ0.50 | OOS λ0.60 | OOS λ0.70 | OOS λ0.80 | OOS λ0.90 | OOS λ0.95 | OOS eq | Mejor λ | Mejor OOS | Splits | ¿Adoptado? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| quarterly | 0.150 | 0.157 | 0.183 | 0.186 | 0.181 | 0.189 | 0.169 | 0.199 | 0.181 | 0.162 | 0.137 | 0.8 | 0.199 | 13 | sí |
| semestral | 0.052 | 0.031 | 0.001 | 0.030 | 0.080 | 0.047 | 0.060 | 0.096 | 0.092 | 0.034 | 0.077 | 0.8 | 0.096 | 14 | sí |
| yearly | -0.096 | 0.010 | 0.015 | -0.047 | 0.044 | 0.053 | -0.017 | 0.099 | 0.016 | 0.016 | 0.062 | 0.8 | 0.099 | 16 | sí |
| flies | -0.135 | -0.162 | -0.084 | -0.021 | 0.059 | 0.077 | -0.002 | 0.210 | 0.240 | 0.280 | 0.320 | 0.95 | 0.280 | 15 | no |

## Pesos desplegados (full-sample × shrinkage)

### quarterly
```json
{
  "range": {
    "rev_term": 0.241353,
    "lvl_term": 0.24809,
    "timing_term": 0.2,
    "vol_term": 0.205812,
    "macd_div_term": 0.056855,
    "rsi_div_term": 0.0,
    "mom_decel_term": 0.030985,
    "er_drop_term": 0.008133,
    "autocorr_term": 0.008772
  },
  "trend": {
    "dir_term": 0.259098,
    "qual_term": 0.2,
    "cont_term": 0.260328,
    "ext_low": 0.280574,
    "ema_align_term": 0.0,
    "mom10_term": 0.0,
    "r2_term": 0.0,
    "dirpers_term": 0.0
  },
  "transition_shrink": 0.4
}
```
### semestral
```json
{
  "range": {
    "rev_term": 0.208061,
    "lvl_term": 0.217606,
    "timing_term": 0.238778,
    "vol_term": 0.230148,
    "macd_div_term": 0.051776,
    "rsi_div_term": 0.00544,
    "mom_decel_term": 0.036536,
    "er_drop_term": 0.006486,
    "autocorr_term": 0.005169
  },
  "trend": {
    "dir_term": 0.232727,
    "qual_term": 0.215318,
    "cont_term": 0.242589,
    "ext_low": 0.282747,
    "ema_align_term": 0.0131,
    "mom10_term": 0.009833,
    "r2_term": 0.002308,
    "dirpers_term": 0.001379
  },
  "transition_shrink": 0.4
}
```
### yearly
```json
{
  "range": {
    "rev_term": 0.316537,
    "lvl_term": 0.224057,
    "timing_term": 0.2,
    "vol_term": 0.259407,
    "macd_div_term": 0.0,
    "rsi_div_term": 0.0,
    "mom_decel_term": 0.0,
    "er_drop_term": 0.0,
    "autocorr_term": 0.0
  },
  "trend": {
    "dir_term": 0.2,
    "qual_term": 0.212452,
    "cont_term": 0.244904,
    "ext_low": 0.253405,
    "ema_align_term": 0.0,
    "mom10_term": 0.0,
    "r2_term": 0.045494,
    "dirpers_term": 0.043745
  },
  "transition_shrink": 0.4
}
```
### flies
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
