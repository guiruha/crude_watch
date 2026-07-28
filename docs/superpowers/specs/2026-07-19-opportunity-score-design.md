# CrudeWatch — Opportunity Score engine + app (spec v1)

Status: authoritative contract for a first, iterable version. Reversion-first, calibrated per family, honest (features as-of `t`, forward outcomes on the executable basis `open[t+1]`).

## Goal

For each instrument (contract, latest active bar) produce a block-decomposed **Opportunity Score** in `[-100, +100]` plus a PM interpretation and risk flags, exactly as requested in the product brief. Positive = long opportunity, negative = short.

This reuses the existing research engine (`crudewatch.research.*`) — do NOT reimplement features, level panel, forward returns, or regime terciles. The blocks are thin transforms of columns produced by `build_dataset(frame, family)`.

## Available inputs (from `build_dataset`)

Per row (contract, date): `close`, `open`, `dte`, `vintage`, `slot`, plus features
`z_10 z_20 z_50 pctb_20_2 pctb_10_1_5 keltner_dist_20 rsi_2 rsi_14 rsi_div_14 macd_div mom_decel_10 er_drop_20 vol_ratio er_20 slope_20 macd_hist`, panel `level_z level_pct`, and forward outcomes `fwd_h mfe_h mae_h fwd_vol_h ...` for `h in (20,25,30)`.

Regime terciles: `crudewatch.research.regime_thresholds(er_array)` → `(lo, hi)` (33/67% of ER). Convention: `er<=lo` range, `er>=hi` trend, else transition/dead.

## Module: `src/crudewatch/scoring/`

Files: `__init__.py` (exports), `blocks.py` (block calculators + calibrators), `score.py` (composite + PM mapping + public API), and `tests` under repo `tests/test_scoring.py`.

### Dataclasses (exact field names — the UI depends on these)

```python
@dataclass(frozen=True)
class BlockScores:
    regime: str            # "range" | "transition" | "trend"
    trendiness: float      # 0..100  (ER percentile in family history)
    direction: float       # -100..+100 (+ = bullish)
    strength: float        # 0..100  (trend quality)
    level: float           # -100..+100 (+ = expensive/dear, - = cheap)
    p_reversion: float     # 0..1
    p_continuation: float  # 0..1
    confidence: float      # 0..100

@dataclass(frozen=True)
class InstrumentScore:
    family: str
    contract: str
    date: pd.Timestamp
    close: float
    dte: float
    blocks: BlockScores
    opportunity: float          # -100..+100
    action: str                 # PM interpretation (Spanish, from the fixed vocabulary)
    rationale: list[str]        # explanation bullets (Spanish)
    risks: list[str]            # risk flags (Spanish)
    features: dict[str, float]  # snapshot of key raw features for display
```

### Calibrator (fit once per family on pooled history)

```python
@dataclass
class FamilyCalibrator:
    family: str
    horizon: int
    er_lo: float; er_hi: float                 # ER terciles (regime)
    ecdf: dict[str, np.ndarray]                # sorted history per feature, for percentile()
    p_rev_cheap: float                         # P(fwd_h>0 | range & level_pct<=1/3)
    p_rev_dear: float                          # P(fwd_h<0 | range & level_pct>=2/3)
    p_cont_up: float                           # P(fwd_h>0 | trend & slope_20 top tercile)
    p_cont_dn: float                           # P(fwd_h<0 | trend & slope_20 bottom tercile)
    ic_t_level: float                          # |t| of level_z IC at horizon (stability), NaN->default
```

`fit_calibrator(data, family, horizon=25) -> FamilyCalibrator`. `ic_t_level`: if `docs/reports/backtest/research_metrics.csv` exists, read the `group=='all'` row for `(family, feature='level_z', horizon)` and take `abs(ic_t)`; else compute a quick Spearman IC t via `evaluate_feature(data,'level_z',horizon)` or fall back to `NaN`.

Helpers:
- `percentile(ecdf_sorted, value) -> float` in `[0,1]`: fraction `<= value` (`np.searchsorted(..., 'right')/n`); `0.5` if NaN/empty.
- `signed_pct(p) -> float`: `(2p-1)*100`.

### Block formulas (deterministic v1)

Let `row` be the instrument's latest bar; `C` the calibrator.

- **A. Régimen / trendiness**: `regime = "range" if er_20<=er_lo elif er_20>=er_hi "trend" else "transition"`. `trendiness = percentile(ecdf['er_20'], er_20)*100`.
- **B. Dirección** `-100..100`: average of available `signed_pct(percentile(ecdf[f], row[f]))` for `f in ('slope_20','macd_hist')`; `0.0` if all NaN.
- **C. Fuerza** `0..100`: `trendiness * (0.4 + 0.6*min(abs(direction)/100,1))`, clipped `0..100`.
- **D. Nivel** `-100..100` (+ = caro): mean of the available components
  `[signed_pct(percentile(ecdf['level_pct'], level_pct)), 100*tanh(level_z/2), 100*tanh(z_20/2)]`; if none available → `0.0`. (`level_pct`/`level_z` may be NaN when no analogous panel; then rely on `z_20`.)
- **E. P(reversión)** `0..1`: pick conditional by level sign — cheap (`level<0`) → `p_rev_cheap`, dear (`level>0`) → `p_rev_dear`; blend toward `0.5` by extremeness `w = min(abs(level)/60, 1)`: `p_reversion = 0.5 + (p_base-0.5)*w`. If `p_base` NaN → `0.5`.
- **F. P(continuación)** `0..1`: pick by direction sign — up → `p_cont_up`, down → `p_cont_dn`; blend by trendiness `w = trendiness/100`: `p_continuation = 0.5 + (p_base-0.5)*w`. NaN → `0.5`.
- **G. Confianza** `0..100` = `100 * f_level * f_bars * f_regime * f_stability` where
  `f_level = 1.0 if level_pct not NaN else 0.4`; `f_bars = min(n_contract_bars_so_far/40, 1)` (bars of this contract up to and including `date`); `f_regime = 1.0 if regime in (range,trend) else 0.5`; `f_stability = min(ic_t_level/3, 1)` (default `0.6` if NaN).

### Opportunity composite

Sub-terms in `[0,1]`:
`rev_term = clip((p_reversion-0.5)/0.5,0,1)`, `lvl_term = min(abs(level)/100,1)`,
`timing_term = clip(-mom_decel_10 normalized, 0,1)` → use `percentile` of `-mom_decel_10` (deceleration favors reversion); default `0.5` if NaN,
`vol_term = clip(1.2 - vol_ratio, 0, 1)` (contraction favorable); default `0.5` if NaN,
`dir_term = min(abs(direction)/100,1)`, `qual_term = strength/100`,
`cont_term = clip((p_continuation-0.5)/0.5,0,1)`, `ext_low = 1 - lvl_term`, `conf_term = confidence/100`.

- **range** (reversion, fade the extreme): `conviction = 0.40*rev_term + 0.25*lvl_term + 0.15*timing_term + 0.10*vol_term + 0.10*conf_term`; `opportunity = -sign(level) * 100 * conviction` (expensive→short/negative, cheap→long/positive). If `level==0` → `0`.
- **trend** (continuation, follow direction): `conviction = 0.30*dir_term + 0.25*qual_term + 0.20*cont_term + 0.10*ext_low + 0.05*ext_low + 0.10*conf_term`; `opportunity = sign(direction) * 100 * conviction`.
- **transition**: compute the range-style opportunity and shrink: `opportunity = 0.4 * range_opportunity`.

Clip to `[-100,100]`. Weights live in a `WEIGHTS: dict[str, dict[str,float]]` constant so they are easy to iterate on.

### PM action (return one Spanish label)

From `(regime, opportunity, level, confidence)`, using the brief vocabulary
(`mantener, entrar, esperar corrección, reducir, tomar beneficios, evitar, comprar debilidad, vender fortaleza, no perseguir el movimiento`). Suggested mapping:
- `confidence < 30` or `regime=='transition'` → prefix "Esperar confirmación".
- `opportunity >= 50` → range: "Comprar debilidad" ; trend: "Entrar (largo)".
- `20 <= opportunity < 50` → "Sesgo largo — empezar pequeño".
- `-20 < opportunity < 20` → "Sin ventaja — esperar / no perseguir".
- `-50 < opportunity <= -20` → "Sesgo corto — reducir".
- `opportunity <= -50` → range: "Vender fortaleza" ; trend: "Entrar (corto)".
- If trend and `abs(level) >= 80` add/override → "Tomar beneficios — no perseguir".

### Risks (list of Spanish strings; include those that apply)

- `n_contract_bars < 15` → "Señal joven (poca historia del contrato)".
- `abs(level) >= 80 or abs(level_z) >= 2.5` → "Mercado muy extendido".
- `regime == 'transition'` → "Régimen inestable (transición)".
- `isnan(level_pct) or confidence < 30` → "Muestra histórica baja / sin panel análogo".
- range and `sign(direction) == sign(-level)` is False while `abs(direction)>=50 and lvl_term>=0.5` → "Divergencia entre modelos (tendencia vs nivel)".
- `vol_ratio >= 2 or vol_ratio <= 0.5` → "Volatilidad anormal".

### Public API

```python
def score_instrument(data, family, contract, horizon=25, calibrator=None) -> InstrumentScore
def score_family(data, family, horizon=25, active_within_days=45, require_unexpired=True) -> pd.DataFrame
```
- `score_family`: for each contract, take its **latest** bar; keep contracts whose latest `date >= data['date'].max() - active_within_days` and (`dte>0` if `require_unexpired`). Returns tidy columns: `family, contract, date, close, dte, regime, trendiness, direction, strength, level, p_reversion, p_continuation, confidence, opportunity, action` sorted by `opportunity` desc. Fit one calibrator and reuse across contracts.
- `data` is the enriched frame (`build_dataset` output). Both accept a prebuilt `calibrator` to avoid refitting.

### Tests (`tests/test_scoring.py`)

Use small synthetic frames (mirror `tests/test_targets.py` / `test_features.py` style) or `build_dataset` on a tiny constructed family. Assert:
- `percentile`/`signed_pct` bounds and monotonicity.
- All block outputs within their documented ranges; `opportunity` in `[-100,100]`.
- Cheap extreme in range → `opportunity > 0`; expensive extreme in range → `opportunity < 0`.
- `regime` labels partition correctly around ER terciles.
- `score_family` returns one row per active contract with the documented columns and no NaN in `opportunity`.
- NaN robustness: a row with NaN `level_pct`/`level_z` still scores (uses `z_20`), lowers `confidence`.

## App: `Opportunity` screen (Streamlit)

New cache layer `app/core/scoring.py` and screen `app/screens/opportunity.py`, registered FIRST in `app/main.py` `SCREENS` (it is the centerpiece), keeping `Strategy` and `Contract Exploration`.

`app/core/scoring.py` (cached, mirrors `app/core/strategy.py` patterns):
- `enriched_frame(family)` already exists in `core.strategy`; reuse it (import) — do not duplicate.
- `@st.cache_data score_family_cached(family, horizon) -> pd.DataFrame` (calls `enriched_frame` + `scoring.score_family`).
- `@st.cache_data score_instrument_cached(family, contract, horizon) -> InstrumentScore` (or return a dict if dataclass caching is awkward; a dict is fine).

Screen tabs ("ventanas"):
1. **Scanner** — family + horizon selectors; ranked table of active contracts by opportunity (color: green long / red short via `theme.palette` BULL/BEAR), columns regime/level/dirección/confianza/opportunity/acción. Top-N metric cards.
2. **Instrumento** — pick contract; big Opportunity gauge (Plotly `go.Indicator` gauge, −100..100, emerald/red), a horizontal bar breakdown of blocks A–G, the `rationale` bullets, `risks` chips, the PM `action`, and a price line with the recent window. Show `features` snapshot in an expander.
3. **Evidencia** — the existing backtest/research metrics (reuse `load_research_metrics`) framed as "why reversion, per family".
4. **Metodología** — explain blocks A–G, per-family calibration, executable basis, and the caveat that the live snapshot is indicative (pooled-history calibration, not a per-day walk-forward fill).

Use `theme.palette` colors and `title_block`. Validate the screen headlessly with Streamlit `AppTest` (a harness that renders `OpportunityScreen(load_frames()).display()`), asserting `at.exception == []`.

## Non-goals (v1)

- No per-day walk-forward retraining for the live score (calibration is pooled family history; documented as indicative). The backtest evidence tab carries the OOS rigor.
- No trend/continuation model beyond the simple direction/quality/continuation blocks.
- No new persisted CSV artifacts required.
