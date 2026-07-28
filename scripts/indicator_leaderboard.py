"""Fixed-config indicator leaderboard: which single indicators and pairs give
good Sharpe / PnL, evaluated honestly (no weight search, no selection).

For each family we precompute the point-in-time per-bar term values once, then
score a battery of FIXED equal-weighted configs:
  - the equal-weight baseline (4 core range + 4 core trend terms),
  - each single range term (trend held at equal-weight), and each range pair,
  - each single trend term (range held at equal-weight), and each trend pair.
Because nothing is fitted to the outcome, evaluating over the full history is a
fair out-of-sample read of each indicator/combination's standalone edge.

Metrics per config: pooled annualised Sharpe and total net PnL (points), plus
annualised PnL. Results per family and pooled across families -> JSON.

Usage:
    uv run python scripts/indicator_leaderboard.py
    uv run python scripts/indicator_leaderboard.py --families quarterly flies
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from crudewatch.data_preparation import build_all  # noqa: E402
from crudewatch.infra import load_raw  # noqa: E402
from crudewatch.research import build_dataset  # noqa: E402
from crudewatch.research.dataset import COST_STUB_POINTS  # noqa: E402
from crudewatch.scoring.score import RANGE_TERM_KEYS, TREND_TERM_KEYS  # noqa: E402
from crudewatch.scoring.weight_search import (  # noqa: E402
    _EQUAL_RANGE, _EQUAL_TREND, _fast_pnl_matrix, _opportunity_matrix,
    precompute_family,
)

FAMILIES = ["quarterly", "semestral", "yearly", "flies", "calendars"]
_REPORT_DIR = _ROOT / "docs" / "reports" / "weight_search"
_JSON_OUT = _REPORT_DIR / "indicator_leaderboard.json"

_DR = len(RANGE_TERM_KEYS)
_DT = len(TREND_TERM_KEYS)


def _onehot(dim: int, idx) -> np.ndarray:
    v = np.zeros(dim)
    for i in idx:
        v[i] = 1.0
    return v / v.sum()


def _load_enriched(family: str) -> pd.DataFrame:
    processed = _ROOT / "data" / "processed" / f"{family}.parquet"
    frame = pd.read_parquet(processed) if processed.exists() else \
        build_all(load_raw(_ROOT / "data" / "raw_files.xlsx"))[family]
    return build_dataset(frame, family)


def _build_configs():
    """Return list of (block, terms, w_range, w_trend)."""
    cfgs = [("baseline", ("equal-weight (8 core)",), _EQUAL_RANGE, _EQUAL_TREND)]
    for i, k in enumerate(RANGE_TERM_KEYS):
        cfgs.append(("range", (k,), _onehot(_DR, [i]), _EQUAL_TREND))
    for i, j in combinations(range(_DR), 2):
        cfgs.append(("range", (RANGE_TERM_KEYS[i], RANGE_TERM_KEYS[j]), _onehot(_DR, [i, j]), _EQUAL_TREND))
    for i, k in enumerate(TREND_TERM_KEYS):
        cfgs.append(("trend", (k,), _EQUAL_RANGE, _onehot(_DT, [i])))
    for i, j in combinations(range(_DT), 2):
        cfgs.append(("trend", (TREND_TERM_KEYS[i], TREND_TERM_KEYS[j]), _EQUAL_RANGE, _onehot(_DT, [i, j])))
    return cfgs


def _pooled_pnl_matrix(pcs, w_range, w_trend, cost) -> np.ndarray:
    """(N_bars_total, C) net PnL stacked across a family's contracts."""
    mats = [
        _fast_pnl_matrix(pc, _opportunity_matrix(pc, w_range, w_trend), cost)
        for pc in pcs if len(pc.close) >= 2
    ]
    return np.vstack(mats) if mats else np.zeros((0, w_range.shape[1]))


def _metrics(pooled: np.ndarray):
    n = pooled.shape[0]
    if n < 2:
        c = pooled.shape[1]
        return np.full(c, np.nan), np.zeros(c), np.full(c, np.nan)
    total = pooled.sum(axis=0)
    mean = pooled.mean(axis=0)
    sd = pooled.std(axis=0, ddof=1)
    sharpe = np.full(pooled.shape[1], np.nan)
    good = sd > 0
    sharpe[good] = mean[good] / sd[good] * np.sqrt(252.0)
    ann_pnl = mean * 252.0
    return sharpe, total, ann_pnl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    ap.add_argument("--min-bars", type=int, default=60)
    args = ap.parse_args()

    cfgs = _build_configs()
    w_range = np.column_stack([c[2] for c in cfgs])  # (dr, C)
    w_trend = np.column_stack([c[3] for c in cfgs])   # (dt, C)
    c = len(cfgs)
    print(f"Configs: {c} ({sum(x[0]=='range' for x in cfgs)} range, "
          f"{sum(x[0]=='trend' for x in cfgs)} trend, 1 baseline)", flush=True)

    per_family = {}
    pooled_by_family = []
    for family in args.families:
        cost = COST_STUB_POINTS.get(family, 0.0)
        print(f"[{family}] precomputing…", flush=True)
        pcs = precompute_family(_load_enriched(family), family, min_bars=args.min_bars, progress=True)
        if len(pcs) < 1:
            print(f"[{family}] skipped (no contracts)", flush=True)
            continue
        pooled = _pooled_pnl_matrix(pcs, w_range, w_trend, cost)
        pooled_by_family.append(pooled)
        sharpe, total, ann = _metrics(pooled)
        per_family[family] = {
            "cost": cost, "n_contracts": len(pcs), "n_days": int(pooled.shape[0]),
            "rows": [
                {"block": cfgs[k][0], "terms": list(cfgs[k][1]),
                 "sharpe": None if np.isnan(sharpe[k]) else round(float(sharpe[k]), 4),
                 "total_pnl": round(float(total[k]), 4),
                 "ann_pnl": None if np.isnan(ann[k]) else round(float(ann[k]), 6)}
                for k in range(c)
            ],
        }
        top = sorted(
            [(cfgs[k][0], cfgs[k][1], sharpe[k]) for k in range(c)],
            key=lambda r: (-np.inf if r[2] != r[2] else r[2]), reverse=True,
        )[:5]
        base_sh = sharpe[0]
        print(f"[{family}] eq Sharpe={base_sh:.3f} | top: " +
              " ; ".join(f"{'+'.join(t)}={s:.3f}" for _, t, s in top), flush=True)

    if pooled_by_family:
        allp = np.vstack(pooled_by_family)
        sharpe, total, ann = _metrics(allp)
        per_family["_pooled_all"] = {
            "n_days": int(allp.shape[0]),
            "rows": [
                {"block": cfgs[k][0], "terms": list(cfgs[k][1]),
                 "sharpe": None if np.isnan(sharpe[k]) else round(float(sharpe[k]), 4),
                 "total_pnl": round(float(total[k]), 4),
                 "ann_pnl": None if np.isnan(ann[k]) else round(float(ann[k]), 6)}
                for k in range(c)
            ],
        }

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _JSON_OUT.write_text(json.dumps(per_family, indent=2, ensure_ascii=False) + "\n")
    print(f"Leaderboard -> {_JSON_OUT}", flush=True)


if __name__ == "__main__":
    main()
