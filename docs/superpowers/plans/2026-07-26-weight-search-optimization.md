# Búsqueda de pesos del Opportunity Score — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aprender, por familia y con validación walk-forward, los pesos de los dos regímenes del Opportunity Score que maximizan el Sharpe medio del backtest, y aplicarlos a producción solo si generalizan fuera de muestra.

**Architecture:** Se parametriza `compute_opportunity` con un vector de pesos por familia (cargado de un JSON, con *fallback* al equal-weight actual). Un motor puro nuevo (`weight_search.py`) precomputa por barra —una sola vez y point-in-time— los 8 términos que se combinan, de modo que el score de cualquier vector de pesos es un producto matriz–vector; sobre eso corre una búsqueda aleatoria por símplex que maximiza el Sharpe agrupado neto de coste, validada por walk-forward de *vintages*. Un script offline ejecuta las 4 familias de reversión, escribe un reporte y actualiza el JSON de pesos adoptados.

**Tech Stack:** Python, numpy, pandas, pytest. Reutiliza `crudewatch.scoring.blocks.fit_calibrator`, `compute_blocks`, `crudewatch.scoring.backtest._hysteresis`.

## Global Constraints

- Point-in-time estricto: el score en `t` nunca usa filas posteriores a `t` (calibrador refit con `outcome_asof=d` sobre datos `<= d`).
- PnL en **puntos de precio**, **neto de coste** (`COST_STUB_POINTS[familia]`); coste por pata = `cost/2`.
- Pesos por régimen: **no-negativos y suman 1**. `transition_shrink = 0.4` fijo. Umbrales de entrada/salida **±50 / ±20** fijos. Horizonte de calibración **25** fijo.
- Familias objetivo: `quarterly`, `semestral`, `yearly`, `flies`. `min_bars = 60`.
- Objetivo = **Sharpe anualizado** (`mean/std·√252`, `ddof=1`) de la serie de PnL diaria **agrupada** (concatenada) de los contratos.
- Búsqueda reproducible con **semilla fija**. Baseline equal-weight (`0.25` cada uno) siempre entre los candidatos.
- JSON de pesos aprendidos en `src/crudewatch/scoring/family_weights.json`; solo familias que baten al equal-weight OOS.
- Determinismo: `np.random.default_rng(seed)` para todo muestreo.

---

### Task 1: Parametrizar los pesos en `score.py` (por familia, vía JSON)

**Files:**
- Modify: `src/crudewatch/scoring/score.py` (WEIGHTS block `:28-42`, `_range_opportunity` `:97-114`, `_trend_opportunity` `:117-132`, `compute_opportunity` `:135-145`, `score_instrument` `:279-322`, `score_family` `:325-389`)
- Modify: `src/crudewatch/scoring/__init__.py` (export `FAMILY_WEIGHTS`, `weights_for`)
- Test: `tests/test_family_weights.py`

**Interfaces:**
- Produces:
  - `weights_for(family: str) -> dict` — devuelve `FAMILY_WEIGHTS.get(family, WEIGHTS)`.
  - `FAMILY_WEIGHTS: dict[str, dict]` — cargado de `family_weights.json` (o `{}`).
  - `compute_opportunity(blocks, row, calibrator, weights: dict | None = None) -> float` — `weights=None` ⇒ `WEIGHTS`.
  - Forma de un dict de pesos: `{"range": {"rev_term","lvl_term","timing_term","vol_term"}, "trend": {"dir_term","qual_term","cont_term","ext_low"}, "transition_shrink": float}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_family_weights.py
from __future__ import annotations

import numpy as np
import pandas as pd

from crudewatch.scoring import score as S
from crudewatch.scoring.blocks import fit_calibrator
from crudewatch.scoring.score import BlockScores, compute_opportunity, weights_for, WEIGHTS


def _row(**kw):
    base = dict(er_20=0.1, level_pct=0.9, slope_20=0.0, mom_decel_10=0.0, vol_ratio=1.0)
    base.update(kw)
    return pd.Series(base)


def _blocks(regime="range", level=60.0, direction=0.0):
    return BlockScores(
        regime=regime, trendiness=50.0, direction=direction, strength=40.0,
        level=level, p_reversion=0.7, p_continuation=0.6, confidence=50.0,
    )


def test_weights_for_defaults_to_equal_when_no_override():
    assert weights_for("no_such_family") is WEIGHTS


def test_weights_for_uses_override(monkeypatch):
    custom = {
        "range": {"rev_term": 0.7, "lvl_term": 0.1, "timing_term": 0.1, "vol_term": 0.1},
        "trend": {"dir_term": 0.25, "qual_term": 0.25, "cont_term": 0.25, "ext_low": 0.25},
        "transition_shrink": 0.4,
    }
    monkeypatch.setattr(S, "FAMILY_WEIGHTS", {"flies": custom})
    assert weights_for("flies")["range"]["rev_term"] == 0.7


def test_compute_opportunity_none_equals_explicit_default():
    b, r = _blocks(), _row()
    cal = fit_calibrator(pd.DataFrame({"er_20": [0.1, 0.2], "date": pd.to_datetime(["2020-01-01", "2020-01-02"])}), "flies")
    assert compute_opportunity(b, r, cal, None) == compute_opportunity(b, r, cal, WEIGHTS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_family_weights.py -q`
Expected: FAIL (`cannot import name 'weights_for'`).

- [ ] **Step 3: Write minimal implementation**

En `score.py`, tras el bloque `WEIGHTS = {...}` (línea ~42) añadir:

```python
import json
from pathlib import Path

_WEIGHTS_PATH = Path(__file__).with_name("family_weights.json")


def _load_family_weights() -> dict[str, dict]:
    try:
        with open(_WEIGHTS_PATH) as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, dict] = {}
    for fam, reg in raw.items():
        out[fam] = {
            "range": {k: float(v) for k, v in reg["range"].items()},
            "trend": {k: float(v) for k, v in reg["trend"].items()},
            "transition_shrink": float(reg.get("transition_shrink", WEIGHTS["transition_shrink"])),
        }
    return out


FAMILY_WEIGHTS: dict[str, dict] = _load_family_weights()


def weights_for(family: str) -> dict:
    """Learned per-family weights, or the equal-weight default."""
    return FAMILY_WEIGHTS.get(family, WEIGHTS)
```

Cambiar las firmas para aceptar `weights` (mantener comportamiento por defecto):

```python
def _range_opportunity(blocks, row, calibrator, weights=None):
    weights = WEIGHTS if weights is None else weights
    level = 0.0 if blocks.level != blocks.level else blocks.level
    if level == 0.0:
        return 0.0
    w = weights["range"]
    # ... resto idéntico ...


def _trend_opportunity(blocks, row, calibrator, weights=None):
    weights = WEIGHTS if weights is None else weights
    direction = 0.0 if blocks.direction != blocks.direction else blocks.direction
    if direction == 0.0:
        return 0.0
    level = 0.0 if blocks.level != blocks.level else blocks.level
    w = weights["trend"]
    # ... resto idéntico ...


def compute_opportunity(blocks, row, calibrator, weights=None):
    weights = WEIGHTS if weights is None else weights
    if blocks.regime == "range":
        return _range_opportunity(blocks, row, calibrator, weights)
    if blocks.regime == "trend":
        return _trend_opportunity(blocks, row, calibrator, weights)
    return _clip_opportunity(weights["transition_shrink"] * _range_opportunity(blocks, row, calibrator, weights))
```

En `score_instrument` y `score_family`, resolver los pesos de la familia y pasarlos:

```python
# score_instrument: justo antes de compute_opportunity(...)
fam_weights = weights_for(family)
opportunity = compute_opportunity(blocks, row, cal, fam_weights)

# score_family: antes del bucle
fam_weights = weights_for(family)
# dentro del bucle:
opportunity = compute_opportunity(blocks, row, cal, fam_weights)
```

En `__init__.py` añadir a los imports desde `score` y a `__all__`: `FAMILY_WEIGHTS`, `weights_for`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_family_weights.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `uv run pytest -q`
Expected: PASS (todos; el comportamiento por defecto es idéntico).

- [ ] **Step 6: Commit**

```bash
git add src/crudewatch/scoring/score.py src/crudewatch/scoring/__init__.py tests/test_family_weights.py
git commit -m "feat(scoring): per-family weights with equal-weight fallback"
```

---

### Task 2: El backtest refleja los pesos de la familia

**Files:**
- Modify: `src/crudewatch/scoring/backtest.py` (`score_series` `:56-98`, import de `compute_opportunity`)
- Test: `tests/test_backtest.py` (añadir un caso)

**Interfaces:**
- Consumes: `weights_for(family)`, `compute_opportunity(blocks, row, cal, weights)` (Task 1).
- Produces: `score_series` ahora usa los pesos de la familia (sin cambio de firma pública).

- [ ] **Step 1: Write the failing test** (añadir a `tests/test_backtest.py`)

```python
def test_score_series_uses_family_weights(monkeypatch):
    from crudewatch.scoring import score as S
    from crudewatch.scoring.backtest import score_series
    base = _synthetic_family(seed=5)
    s_equal = score_series(base, "outrights", "A", horizon=25)
    heavy_rev = {
        "range": {"rev_term": 1.0, "lvl_term": 0.0, "timing_term": 0.0, "vol_term": 0.0},
        "trend": {"dir_term": 1.0, "qual_term": 0.0, "cont_term": 0.0, "ext_low": 0.0},
        "transition_shrink": 0.4,
    }
    monkeypatch.setattr(S, "FAMILY_WEIGHTS", {"outrights": heavy_rev})
    s_weighted = score_series(base, "outrights", "A", horizon=25)
    # Distinct weights must change at least one bar's opportunity.
    assert not np.allclose(
        s_equal["opportunity"].to_numpy(), s_weighted["opportunity"].to_numpy(), equal_nan=True
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backtest.py::test_score_series_uses_family_weights -q`
Expected: FAIL (los scores son idénticos porque `score_series` aún ignora los pesos de familia).

- [ ] **Step 3: Write minimal implementation**

En `backtest.py`, cambiar el import y el uso en `score_series`:

```python
from crudewatch.scoring.score import compute_blocks, compute_opportunity, weights_for
```

```python
def score_series(data, family, contract, horizon=25):
    df = data.reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])
    dates_all = df["date"].to_numpy()
    csub = df[df["contract"] == contract].sort_values("date").reset_index(drop=True)
    if csub.empty:
        raise KeyError(f"contract {contract!r} not found in data")
    fam_weights = weights_for(family)
    has = {c: (c in csub.columns) for c in ("open", "high", "low", "close")}
    records: list[dict] = []
    for i, row in csub.iterrows():
        d = row["date"]
        window = df[dates_all <= np.datetime64(d)]
        cal = fit_calibrator(window, family, horizon, outcome_asof=d)
        blocks = compute_blocks(row, cal, i + 1)
        opp = compute_opportunity(blocks, row, cal, fam_weights)
        # ... records.append(...) idéntico ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_backtest.py -q`
Expected: PASS (todos los tests del backtest).

- [ ] **Step 5: Commit**

```bash
git add src/crudewatch/scoring/backtest.py tests/test_backtest.py
git commit -m "feat(backtest): score_series honours per-family weights"
```

---

### Task 3: Núcleo de `weight_search.py` — precómputo de términos y score vectorizado

**Files:**
- Create: `src/crudewatch/scoring/weight_search.py`
- Test: `tests/test_weight_search.py`

**Interfaces:**
- Consumes: `fit_calibrator` (blocks), `compute_blocks`, `_rev_term`, `_cont_term` (score), `timing_term`, `vol_term` (blocks), `compute_opportunity` (para parity).
- Produces:
  - `PrecomputedContract` (frozen dataclass): `contract:str`, `vintage:int`, `date:np.ndarray`, `close/open/high/low:np.ndarray`, `regime_code:np.ndarray[int8]` (0 range,1 trend,2 transition), `m_range:np.ndarray`, `range_terms:np.ndarray(n,4)`, `m_trend:np.ndarray`, `trend_terms:np.ndarray(n,4)`.
  - `precompute_contract(data, family, contract, horizon=25) -> PrecomputedContract`.
  - `opportunity_from_precomputed(pc, w_range: np.ndarray, w_trend: np.ndarray) -> np.ndarray` — orden de términos: range `[rev_term, lvl_term, timing_term, vol_term]`, trend `[dir_term, qual_term, cont_term, ext_low]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_weight_search.py
from __future__ import annotations

import numpy as np
import pandas as pd

from crudewatch.scoring.backtest import score_series  # reuse synthetic frame maker
from crudewatch.scoring.score import WEIGHTS, compute_blocks, compute_opportunity
from crudewatch.scoring.blocks import fit_calibrator
from crudewatch.scoring.weight_search import (
    opportunity_from_precomputed,
    precompute_contract,
)
from tests.test_backtest import _synthetic_family

_EQ = np.array([0.25, 0.25, 0.25, 0.25])


def test_precompute_matches_compute_opportunity_equal_weights():
    data = _synthetic_family(seed=1)
    pc = precompute_contract(data, "outrights", "A", horizon=25)
    got = opportunity_from_precomputed(pc, _EQ, _EQ)
    # Reference: the real scorer, bar by bar, point-in-time.
    df = data[data["contract"] == "A"].sort_values("date").reset_index(drop=True)
    dates_all = pd.to_datetime(data["date"]).to_numpy()
    ref = []
    for i, row in df.iterrows():
        d = pd.to_datetime(row["date"])
        win = data[dates_all <= np.datetime64(d)]
        cal = fit_calibrator(win, "outrights", 25, outcome_asof=d)
        blocks = compute_blocks(row, cal, i + 1)
        ref.append(compute_opportunity(blocks, row, cal, WEIGHTS))
    assert np.allclose(got, np.array(ref), atol=1e-9, equal_nan=True)


def test_precompute_matches_for_random_weights():
    data = _synthetic_family(seed=2)
    pc = precompute_contract(data, "outrights", "A", horizon=25)
    rng = np.random.default_rng(0)
    for _ in range(5):
        wr = rng.dirichlet(np.ones(4))
        wt = rng.dirichlet(np.ones(4))
        got = opportunity_from_precomputed(pc, wr, wt)
        df = data[data["contract"] == "A"].sort_values("date").reset_index(drop=True)
        dates_all = pd.to_datetime(data["date"]).to_numpy()
        ref = []
        for i, row in df.iterrows():
            d = pd.to_datetime(row["date"])
            win = data[dates_all <= np.datetime64(d)]
            cal = fit_calibrator(win, "outrights", 25, outcome_asof=d)
            blocks = compute_blocks(row, cal, i + 1)
            w = {"range": dict(zip(("rev_term", "lvl_term", "timing_term", "vol_term"), wr)),
                 "trend": dict(zip(("dir_term", "qual_term", "cont_term", "ext_low"), wt)),
                 "transition_shrink": 0.4}
            ref.append(compute_opportunity(blocks, row, cal, w))
        assert np.allclose(got, np.array(ref), atol=1e-9, equal_nan=True)


def test_precompute_is_point_in_time():
    base = _synthetic_family(seed=3)
    pc_base = precompute_contract(base, "outrights", "A", horizon=25)
    future = base[base["contract"] == "B"].copy()
    future["contract"] = "C"
    future["date"] = pd.to_datetime(future["date"]) + pd.Timedelta(days=400)
    ext = pd.concat([base, future], ignore_index=True)
    pc_ext = precompute_contract(ext, "outrights", "A", horizon=25)
    assert np.allclose(
        opportunity_from_precomputed(pc_base, _EQ, _EQ),
        opportunity_from_precomputed(pc_ext, _EQ, _EQ),
        equal_nan=True,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_weight_search.py -q`
Expected: FAIL (`No module named 'crudewatch.scoring.weight_search'`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/crudewatch/scoring/weight_search.py
"""Search per-family Opportunity-Score weights that maximise backtest Sharpe.

The 8 terms combined by ``compute_opportunity`` do not depend on the weights;
only their linear combination does. So we precompute, once per contract and
strictly point-in-time, the per-bar term values and regime sign, after which the
score for any weight vector is a matrix-vector product. See
``docs/superpowers/specs/2026-07-26-weight-search-optimization-design.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crudewatch.scoring.blocks import fit_calibrator, timing_term, vol_term
from crudewatch.scoring.score import _cont_term, _rev_term, compute_blocks

_REGIME_CODE = {"range": 0, "trend": 1, "transition": 2}


@dataclass(frozen=True)
class PrecomputedContract:
    contract: str
    vintage: int
    date: np.ndarray
    close: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    regime_code: np.ndarray
    m_range: np.ndarray
    range_terms: np.ndarray
    m_trend: np.ndarray
    trend_terms: np.ndarray


def _col(csub: pd.DataFrame, name: str) -> np.ndarray:
    if name in csub.columns:
        return csub[name].to_numpy(dtype=float)
    return np.full(len(csub), np.nan)


def precompute_contract(data, family, contract, horizon: int = 25) -> PrecomputedContract:
    df = data.reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])
    dates_all = df["date"].to_numpy()
    csub = df[df["contract"] == contract].sort_values("date").reset_index(drop=True)
    if csub.empty:
        raise KeyError(f"contract {contract!r} not found in data")
    n = len(csub)
    regime = np.zeros(n, dtype=np.int8)
    m_range = np.zeros(n)
    m_trend = np.zeros(n)
    range_terms = np.zeros((n, 4))
    trend_terms = np.zeros((n, 4))
    for i, row in csub.iterrows():
        d = row["date"]
        cal = fit_calibrator(df[dates_all <= np.datetime64(d)], family, horizon, outcome_asof=d)
        b = compute_blocks(row, cal, i + 1)
        regime[i] = _REGIME_CODE[b.regime]
        level = 0.0 if b.level != b.level else b.level
        direction = 0.0 if b.direction != b.direction else b.direction
        lvl_term = min(abs(level) / 100.0, 1.0)
        m_range[i] = 100.0 if level < 0 else (-100.0 if level > 0 else 0.0)
        m_trend[i] = 100.0 if direction > 0 else (-100.0 if direction < 0 else 0.0)
        range_terms[i] = (_rev_term(b.p_reversion), lvl_term, timing_term(row, cal), vol_term(row))
        trend_terms[i] = (min(abs(direction) / 100.0, 1.0), b.strength / 100.0, _cont_term(b.p_continuation), 1.0 - lvl_term)
    vintage = int(csub["vintage"].iloc[0]) if "vintage" in csub.columns else 0
    return PrecomputedContract(
        contract=str(contract), vintage=vintage,
        date=csub["date"].to_numpy(),
        close=_col(csub, "close"), open=_col(csub, "open"),
        high=_col(csub, "high"), low=_col(csub, "low"),
        regime_code=regime, m_range=m_range, range_terms=range_terms,
        m_trend=m_trend, trend_terms=trend_terms,
    )


def opportunity_from_precomputed(pc: PrecomputedContract, w_range, w_trend) -> np.ndarray:
    w_range = np.asarray(w_range, dtype=float)
    w_trend = np.asarray(w_trend, dtype=float)
    o_range = np.clip(pc.m_range * (pc.range_terms @ w_range), -100.0, 100.0)
    o_trend = np.clip(pc.m_trend * (pc.trend_terms @ w_trend), -100.0, 100.0)
    return np.where(pc.regime_code == 1, o_trend, np.where(pc.regime_code == 2, 0.4 * o_range, o_range))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_weight_search.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/crudewatch/scoring/weight_search.py tests/test_weight_search.py
git commit -m "feat(weight-search): PIT per-bar term precompute + vectorised score"
```

---

### Task 4: Muestreo de símplex, PnL rápido y Sharpe agrupado

**Files:**
- Modify: `src/crudewatch/scoring/weight_search.py`
- Test: `tests/test_weight_search.py`

**Interfaces:**
- Consumes: `_hysteresis` (backtest), `opportunity_from_precomputed` (Task 3).
- Produces:
  - `simplex_samples(n:int, dim:int=4, seed:int=0) -> np.ndarray` — filas: equal-weight, `dim` one-hots, luego `n` Dirichlet; todas suman 1.
  - `fast_pnl(pc, w_range, w_trend, cost, enter=50.0, exit=20.0) -> np.ndarray` — PnL diaria neta (reproduce las convenciones de `backtest.simulate`: lag open[t+1], mark close-a-close con fill al open, coste `cost/2` por pata).
  - `pooled_sharpe(pnl_arrays: list[np.ndarray]) -> float`.
  - `objective(pcs: list[PrecomputedContract], w_range, w_trend, cost) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# añadir a tests/test_weight_search.py
from crudewatch.scoring.backtest import simulate
from crudewatch.scoring.weight_search import (
    fast_pnl, objective, pooled_sharpe, simplex_samples,
)


def test_simplex_samples_valid():
    s = simplex_samples(50, dim=4, seed=0)
    assert s.shape == (55, 4)                       # equal + 4 one-hots + 50
    assert np.all(s >= -1e-12)
    assert np.allclose(s.sum(axis=1), 1.0)
    assert np.allclose(s[0], 0.25)                  # equal-weight first
    assert np.allclose(s[1:5], np.eye(4))           # one-hots


def test_fast_pnl_matches_simulate():
    # Build a score_df directly and compare fast_pnl to simulate's equity increments.
    import pandas as pd
    n = 6
    sdf = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "close": [100, 101, 103, 104, 102, 101.0],
        "open": [100, 101, 102, 103, 104, 105.0],
        "high": [100, 101, 103, 105, 104, 105.0],
        "low": [100, 100, 101, 102, 101, 100.0],
        "opportunity": [0, 60, 30, 10, 0, 0.0],
        "regime": ["range"] * n,
    })
    _, equity, _, _ = simulate(sdf, cost=0.10, enter_at=50.0, exit_at=20.0)
    # A tiny PrecomputedContract whose opportunity_from_precomputed returns sdf["opportunity"].
    from crudewatch.scoring.weight_search import PrecomputedContract
    opp = sdf["opportunity"].to_numpy()
    pc = PrecomputedContract(
        contract="X", vintage=0, date=sdf["date"].to_numpy(),
        close=sdf["close"].to_numpy(), open=sdf["open"].to_numpy(),
        high=sdf["high"].to_numpy(), low=sdf["low"].to_numpy(),
        regime_code=np.zeros(n, np.int8),
        m_range=np.where(opp >= 0, 1.0, -1.0) * 0.0,  # placeholder, unused below
        range_terms=np.zeros((n, 4)), m_trend=np.zeros(n), trend_terms=np.zeros((n, 4)),
    )
    # Monkeypatch-free: call fast_pnl on a pc whose score we override by injecting opp.
    net = fast_pnl(pc, np.zeros(4), np.zeros(4), cost=0.10, opp_override=opp)
    assert abs(net.sum() - equity.iloc[-1]) < 1e-9


def test_pooled_sharpe_handles_degenerate():
    assert np.isnan(pooled_sharpe([np.array([1.0])]))       # <2 points
    assert np.isnan(pooled_sharpe([np.zeros(5)]))           # std == 0
    val = pooled_sharpe([np.array([1.0, -0.5, 0.3, 0.2])])
    assert np.isfinite(val)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_weight_search.py -q`
Expected: FAIL (`cannot import name 'simplex_samples'`).

- [ ] **Step 3: Write minimal implementation** (añadir a `weight_search.py`)

```python
from crudewatch.scoring.backtest import _hysteresis


def simplex_samples(n: int, dim: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.dirichlet(np.ones(dim), size=n)
    fixed = np.vstack([np.full(dim, 1.0 / dim), np.eye(dim)])
    return np.vstack([fixed, draws])


def fast_pnl(pc, w_range, w_trend, cost, enter=50.0, exit=20.0, opp_override=None) -> np.ndarray:
    """Daily net PnL reproducing ``backtest.simulate`` (open[t+1] fills, cost/2 per leg)."""
    opp = pc_opportunity if (pc_opportunity := opp_override) is not None else opportunity_from_precomputed(pc, w_range, w_trend)
    n = len(opp)
    if n < 2:
        return np.zeros(0)
    desired = _hysteresis(opp, enter, exit)
    effective = np.empty(n, dtype=int)
    effective[0] = 0
    effective[1:] = desired[:-1]
    prev_eff = np.empty(n, dtype=int)
    prev_eff[0] = 0
    prev_eff[1:] = effective[:-1]
    close = pc.close
    open_eff = np.where(np.isnan(pc.open), close, pc.open)
    close_prev = np.empty(n)
    close_prev[0] = close[0]
    close_prev[1:] = close[:-1]
    pnl = prev_eff * (open_eff - close_prev) + effective * (close - open_eff)
    pnl[0] = 0.0
    cost_daily = np.abs(effective - prev_eff) * (cost / 2.0)
    return pnl - cost_daily


def pooled_sharpe(pnl_arrays) -> float:
    pooled = np.concatenate([a for a in pnl_arrays if len(a)]) if pnl_arrays else np.array([])
    if len(pooled) < 2:
        return float("nan")
    sd = float(np.std(pooled, ddof=1))
    if not sd or sd <= 0:
        return float("nan")
    return float(np.mean(pooled) / sd * np.sqrt(252.0))


def objective(pcs, w_range, w_trend, cost) -> float:
    return pooled_sharpe([fast_pnl(pc, w_range, w_trend, cost) for pc in pcs])
```

Nota: la doble asignación con walrus en `fast_pnl` es solo para permitir `opp_override` en el test; en producción `opp_override=None` y se calcula desde el precómputo.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_weight_search.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/crudewatch/scoring/weight_search.py tests/test_weight_search.py
git commit -m "feat(weight-search): simplex sampling, fast PnL, pooled Sharpe"
```

---

### Task 5: Precómputo por familia y búsqueda de pesos

**Files:**
- Modify: `src/crudewatch/scoring/weight_search.py`
- Test: `tests/test_weight_search.py`

**Interfaces:**
- Consumes: `precompute_contract`, `objective`, `simplex_samples` (Tasks 3-4); `COST_STUB_POINTS`.
- Produces:
  - `precompute_family(data, family, horizon=25, min_bars=60) -> list[PrecomputedContract]`.
  - `WeightSearchResult` (frozen dataclass): `w_range:np.ndarray`, `w_trend:np.ndarray`, `sharpe:float`, `equal_sharpe:float`.
  - `search_weights(pcs, cost, n=4000, seed=0) -> WeightSearchResult`.

- [ ] **Step 1: Write the failing test**

```python
# añadir a tests/test_weight_search.py
from crudewatch.scoring.weight_search import precompute_family, search_weights


def test_precompute_family_filters_short_contracts():
    data = _synthetic_family(seed=1)              # A and B, 60 bars each
    pcs = precompute_family(data, "outrights", horizon=25, min_bars=60)
    assert {p.contract for p in pcs} == {"A", "B"}
    pcs2 = precompute_family(data, "outrights", horizon=25, min_bars=1000)
    assert pcs2 == []


def test_search_weights_beats_or_ties_equal_in_sample():
    data = _synthetic_family(seed=7)
    pcs = precompute_family(data, "outrights", horizon=25, min_bars=60)
    res = search_weights(pcs, cost=0.02, n=200, seed=0)
    # Equal-weight is always a candidate, so the best is never worse than equal
    # (treating NaN as -inf).
    best = res.sharpe if res.sharpe == res.sharpe else -np.inf
    eq = res.equal_sharpe if res.equal_sharpe == res.equal_sharpe else -np.inf
    assert best >= eq - 1e-9
    assert np.isclose(res.w_range.sum(), 1.0) and np.isclose(res.w_trend.sum(), 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_weight_search.py -q`
Expected: FAIL (`cannot import name 'search_weights'`).

- [ ] **Step 3: Write minimal implementation** (añadir a `weight_search.py`)

```python
from crudewatch.research.dataset import COST_STUB_POINTS

_EQUAL = np.full(4, 0.25)


@dataclass(frozen=True)
class WeightSearchResult:
    w_range: np.ndarray
    w_trend: np.ndarray
    sharpe: float
    equal_sharpe: float


def precompute_family(data, family, horizon: int = 25, min_bars: int = 60) -> list[PrecomputedContract]:
    counts = data.groupby("contract").size()
    keep = counts[counts >= min_bars].index
    return [precompute_contract(data, family, c, horizon) for c in sorted(keep)]


def _nan_to_neg_inf(x: float) -> float:
    return x if x == x else float("-inf")


def search_weights(pcs, cost, n: int = 4000, seed: int = 0) -> WeightSearchResult:
    equal_sharpe = objective(pcs, _EQUAL, _EQUAL, cost)
    Wr = simplex_samples(n, 4, seed)
    Wt = simplex_samples(n, 4, seed + 1)
    best_score = _nan_to_neg_inf(equal_sharpe)
    best_wr, best_wt = _EQUAL, _EQUAL
    for wr, wt in zip(Wr, Wt):
        s = _nan_to_neg_inf(objective(pcs, wr, wt, cost))
        if s > best_score:
            best_score, best_wr, best_wt = s, wr, wt
    return WeightSearchResult(
        w_range=np.asarray(best_wr, dtype=float),
        w_trend=np.asarray(best_wt, dtype=float),
        sharpe=(best_score if best_score != float("-inf") else float("nan")),
        equal_sharpe=equal_sharpe,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_weight_search.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/crudewatch/scoring/weight_search.py tests/test_weight_search.py
git commit -m "feat(weight-search): family precompute + random-search over simplex"
```

---

### Task 6: Validación walk-forward por vintage

**Files:**
- Modify: `src/crudewatch/scoring/weight_search.py`
- Test: `tests/test_weight_search.py`

**Interfaces:**
- Consumes: `precompute_family`, `search_weights`, `objective`, `_EQUAL`.
- Produces:
  - `expanding_vintage_splits(vintages, min_train=3) -> list[tuple[list[int], int]]` — ventana expandiente sobre vintages ordenados distintos (misma lógica que `backtesting.research.evaluate.walk_forward_splits`, definida localmente para no acoplar `crudewatch` a `backtesting`).
  - `WalkForwardResult` (frozen dataclass): `oos_sharpe_opt:float`, `oos_sharpe_equal:float`, `n_splits:int`.
  - `walk_forward_weights(pcs, cost, n=4000, seed=0, min_train=3) -> WalkForwardResult`.

- [ ] **Step 1: Write the failing test**

```python
# añadir a tests/test_weight_search.py
from crudewatch.scoring.weight_search import expanding_vintage_splits, walk_forward_weights


def test_expanding_vintage_splits():
    assert expanding_vintage_splits([2020, 2021, 2022, 2023], min_train=2) == [
        ([2020, 2021], 2022), ([2020, 2021, 2022], 2023),
    ]
    assert expanding_vintage_splits([2020, 2021], min_train=3) == []


def test_walk_forward_runs_and_reports_both(monkeypatch):
    # Give contracts distinct vintages so splits exist.
    data = _synthetic_family(seed=4)
    data["vintage"] = np.where(data["contract"] == "A", 2020, 2021)
    pcs = precompute_family(data, "outrights", horizon=25, min_bars=60)
    res = walk_forward_weights(pcs, cost=0.02, n=50, seed=0, min_train=1)
    assert res.n_splits >= 1
    assert isinstance(res.oos_sharpe_opt, float)
    assert isinstance(res.oos_sharpe_equal, float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_weight_search.py -q`
Expected: FAIL (`cannot import name 'expanding_vintage_splits'`).

- [ ] **Step 3: Write minimal implementation** (añadir a `weight_search.py`)

```python
@dataclass(frozen=True)
class WalkForwardResult:
    oos_sharpe_opt: float
    oos_sharpe_equal: float
    n_splits: int


def expanding_vintage_splits(vintages, min_train: int = 3):
    vs = sorted({int(v) for v in vintages})
    return [(vs[:i], vs[i]) for i in range(min_train, len(vs))]


def walk_forward_weights(pcs, cost, n: int = 4000, seed: int = 0, min_train: int = 3) -> WalkForwardResult:
    vintages = [pc.vintage for pc in pcs]
    splits = expanding_vintage_splits(vintages, min_train)
    opt_pnls: list[np.ndarray] = []
    eq_pnls: list[np.ndarray] = []
    used = 0
    for train_vs, test_v in splits:
        train = [pc for pc in pcs if pc.vintage in set(train_vs)]
        test = [pc for pc in pcs if pc.vintage == test_v]
        if not train or not test:
            continue
        res = search_weights(train, cost, n=n, seed=seed)
        for pc in test:
            opt_pnls.append(fast_pnl(pc, res.w_range, res.w_trend, cost))
            eq_pnls.append(fast_pnl(pc, _EQUAL, _EQUAL, cost))
        used += 1
    return WalkForwardResult(
        oos_sharpe_opt=pooled_sharpe(opt_pnls),
        oos_sharpe_equal=pooled_sharpe(eq_pnls),
        n_splits=used,
    )
```

Nota: agrupar el PnL OOS de todos los splits y calcular un solo Sharpe equivale a la media ponderada por nº de puntos diarios (cumple "Sharpe medio" del objetivo).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_weight_search.py -q`
Expected: PASS.

- [ ] **Step 5: Run full suite**

Run: `uv run pytest -q`
Expected: PASS (todos).

- [ ] **Step 6: Commit**

```bash
git add src/crudewatch/scoring/weight_search.py tests/test_weight_search.py
git commit -m "feat(weight-search): walk-forward validation over vintages"
```

---

### Task 7: Script offline — optimizar, reportar y aplicar

**Files:**
- Create: `scripts/optimize_weights.py`
- Create (salida en tiempo de ejecución): `docs/reports/weight_search/weights_report.md`, `src/crudewatch/scoring/family_weights.json`
- Export: añadir `precompute_family`, `search_weights`, `walk_forward_weights`, `WeightSearchResult`, `WalkForwardResult` a `src/crudewatch/scoring/__init__.py`

**Interfaces:**
- Consumes: `precompute_family`, `search_weights`, `walk_forward_weights` (Tasks 5-6); `COST_STUB_POINTS`; `crudewatch.data_preparation.build_all`, `crudewatch.infra.load_raw`, `crudewatch.research.build_dataset`.
- Produces: JSON `{familia: {"range": {...}, "trend": {...}, "transition_shrink": 0.4}}` solo para familias adoptadas; reporte markdown.

- [ ] **Step 1: Añadir exports** en `src/crudewatch/scoring/__init__.py`

```python
from crudewatch.scoring.weight_search import (
    WalkForwardResult,
    WeightSearchResult,
    precompute_family,
    search_weights,
    walk_forward_weights,
)
# y añadir esos nombres a __all__
```

Run: `uv run python -c "import crudewatch.scoring as s; print(s.search_weights)"`
Expected: imprime la función sin error.

- [ ] **Step 2: Escribir el script** (`scripts/optimize_weights.py`)

```python
"""Optimise per-family Opportunity-Score weights and (optionally) apply them.

Usage:
    uv run python scripts/optimize_weights.py            # all reversion families
    uv run python scripts/optimize_weights.py --families flies --samples 2000
    uv run python scripts/optimize_weights.py --dry-run  # report only, no JSON write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from crudewatch.data_preparation import build_all  # noqa: E402
from crudewatch.infra import load_raw  # noqa: E402
from crudewatch.research import build_dataset  # noqa: E402
from crudewatch.research.dataset import COST_STUB_POINTS  # noqa: E402
from crudewatch.scoring.weight_search import (  # noqa: E402
    precompute_family, search_weights, walk_forward_weights,
)

REVERSION_FAMILIES = ["quarterly", "semestral", "yearly", "flies"]
_RANGE_KEYS = ("rev_term", "lvl_term", "timing_term", "vol_term")
_TREND_KEYS = ("dir_term", "qual_term", "cont_term", "ext_low")
_JSON_PATH = _SRC / "crudewatch" / "scoring" / "family_weights.json"
_REPORT_PATH = _ROOT / "docs" / "reports" / "weight_search" / "weights_report.md"


def _load_enriched(family: str) -> pd.DataFrame:
    processed = _ROOT / "data" / "processed" / f"{family}.parquet"
    if processed.exists():
        frame = pd.read_parquet(processed)
    else:
        frame = build_all(load_raw(_ROOT / "data" / "raw_files.xlsx"))[family]
    return build_dataset(frame, family)


def _weights_dict(w_range, w_trend) -> dict:
    return {
        "range": {k: round(float(v), 6) for k, v in zip(_RANGE_KEYS, w_range)},
        "trend": {k: round(float(v), 6) for k, v in zip(_TREND_KEYS, w_trend)},
        "transition_shrink": 0.4,
    }


def _fmt(x: float) -> str:
    return "—" if x != x else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=REVERSION_FAMILIES)
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-bars", type=int, default=60)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--min-train", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    adopted: dict[str, dict] = {}
    rows: list[dict] = []
    for family in args.families:
        cost = COST_STUB_POINTS.get(family, 0.0)
        print(f"[{family}] precomputing…", flush=True)
        pcs = precompute_family(_load_enriched(family), family, min_bars=args.min_bars)
        if len(pcs) < 2:
            rows.append({"family": family, "note": f"skipped ({len(pcs)} contracts)"})
            continue
        wf = walk_forward_weights(pcs, cost, n=args.samples, seed=args.seed, min_train=args.min_train)
        full = search_weights(pcs, cost, n=args.samples, seed=args.seed)
        beats = (wf.oos_sharpe_opt == wf.oos_sharpe_opt) and (
            wf.oos_sharpe_opt > (wf.oos_sharpe_equal if wf.oos_sharpe_equal == wf.oos_sharpe_equal else float("-inf")) + args.margin
        )
        if beats:
            adopted[family] = _weights_dict(full.w_range, full.w_trend)
        rows.append({
            "family": family, "note": "",
            "is_sharpe": full.sharpe, "eq_sharpe": full.equal_sharpe,
            "oos_opt": wf.oos_sharpe_opt, "oos_eq": wf.oos_sharpe_equal,
            "splits": wf.n_splits, "adopted": beats,
            "weights": _weights_dict(full.w_range, full.w_trend),
        })
        print(f"[{family}] IS={_fmt(full.sharpe)} eq={_fmt(full.equal_sharpe)} "
              f"OOS opt={_fmt(wf.oos_sharpe_opt)} eq={_fmt(wf.oos_sharpe_equal)} adopt={beats}", flush=True)

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Optimización de pesos — reporte", "",
             f"Muestras: {args.samples} · semilla: {args.seed} · min_bars: {args.min_bars} · margen OOS: {args.margin}", "",
             "| Familia | Sharpe IS | Sharpe IS eq | Sharpe OOS opt | Sharpe OOS eq | Splits | ¿Adoptado? |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("note"):
            lines.append(f"| {r['family']} | {r['note']} | | | | | |")
            continue
        lines.append(f"| {r['family']} | {_fmt(r['is_sharpe'])} | {_fmt(r['eq_sharpe'])} | "
                     f"{_fmt(r['oos_opt'])} | {_fmt(r['oos_eq'])} | {r['splits']} | {'sí' if r['adopted'] else 'no'} |")
    lines += ["", "## Pesos elegidos (full-sample)", ""]
    for r in rows:
        if r.get("note"):
            continue
        lines.append(f"### {r['family']}")
        lines.append("```json")
        lines.append(json.dumps(r["weights"], indent=2, ensure_ascii=False))
        lines.append("```")
    _REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Report -> {_REPORT_PATH}")

    if args.dry_run:
        print("dry-run: no JSON written.")
        return
    existing = {}
    if _JSON_PATH.exists():
        existing = json.loads(_JSON_PATH.read_text())
    existing.update(adopted)
    _JSON_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n")
    print(f"Adopted {sorted(adopted)} -> {_JSON_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-run en seco sobre una familia pequeña**

Run: `uv run python scripts/optimize_weights.py --families flies --samples 300 --dry-run`
Expected: imprime línea `[flies] IS=… eq=… OOS opt=… eq=… adopt=…`, escribe `docs/reports/weight_search/weights_report.md`, y NO escribe el JSON.

- [ ] **Step 4: Ejecutar de verdad las 4 familias y aplicar**

Run: `uv run python scripts/optimize_weights.py --samples 4000`
Expected: reporte actualizado + `family_weights.json` con las familias adoptadas (0..4). Puede tardar varios minutos (precómputo PIT por barra).

- [ ] **Step 5: Verificar que producción carga los pesos**

Run: `uv run python -c "import crudewatch.scoring.score as s; print(sorted(s.FAMILY_WEIGHTS))"`
Expected: lista de familias adoptadas (o `[]` si ninguna batió al equal-weight).

- [ ] **Step 6: Full suite + commit**

Run: `uv run pytest -q`
Expected: PASS.

```bash
git add scripts/optimize_weights.py src/crudewatch/scoring/__init__.py docs/reports/weight_search/weights_report.md src/crudewatch/scoring/family_weights.json
git commit -m "feat(weight-search): offline optimiser, report, and applied family weights"
```

---

## Notas de auto-revisión (cobertura del spec)

- §2 espacio de búsqueda → Tasks 3-5 (símplex, precómputo, search).
- §3 precómputo de términos → Task 3 (parity + PIT).
- §4.1 motor puro → Tasks 3-5. §4.2 walk-forward → Task 6. §4.3 estructura de pesos → Task 1. §4.4 script/reporte/adopción → Task 7.
- §6 casos borde → `min_bars` (Task 5), Sharpe NaN (Task 4/5), baseline siempre presente (Task 5), paridad con `simulate` (Task 4).
- §7 testing → cada Task trae sus tests; paridad, símplex, PnL, objetivo, PIT, `weights_for`.
- Desviación menor vs spec: `expanding_vintage_splits` se define local en `weight_search.py` (misma lógica que `walk_forward_splits`) para no acoplar `crudewatch` → `backtesting`.
