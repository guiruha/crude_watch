"""Backtest the Opportunity Score with strictly non-overlapping trades.

    uv run python scripts/run_nonoverlap_backtest.py
    uv run python scripts/run_nonoverlap_backtest.py --families quarterly --hold 10 20

One position per family at a time: enter when the point-in-time score clears
``--enter``, fill at the next bar's open, hold a fixed number of bars, exit, and
only then become eligible again. Holding periods are therefore pairwise
disjoint, every trade is an independent observation, and the Sharpe can be
annualised by the realised trades-per-year without an overlap correction.

Sweeps the hold length and the entry threshold so the result is read as a
surface rather than a single cherry-picked configuration — a signal that only
works at one (hold, threshold) pair is a fit, not an edge.

Writes:
    docs/reports/nonoverlap/<family>_trades.csv   the trade ledger per config
    docs/reports/nonoverlap/summary.csv           one row per family x config
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # the offline `backtesting` package lives at repo root

from crudewatch.data_preparation import build_all  # noqa: E402
from crudewatch.infra import load_raw  # noqa: E402
from crudewatch.research.dataset import COST_STUB_POINTS, build_dataset  # noqa: E402
from crudewatch.scoring.score import RANGE_TERM_KEYS, TREND_TERM_KEYS, weights_for  # noqa: E402
from crudewatch.scoring.weight_search import precompute_family  # noqa: E402
from backtesting.research.nonoverlap import (  # noqa: E402
    nonoverlapping_trades,
    panel_from_precomputed,
    summarize,
)

FAMILIES = ["quarterly", "semestral", "yearly", "flies"]
RAW = ROOT / "data" / "raw_files.xlsx"
OUT_DIR = ROOT / "docs" / "reports" / "nonoverlap"


def weight_vectors(family: str) -> tuple[np.ndarray, np.ndarray]:
    """The family's fitted weights as dense vectors in canonical term order."""
    w = weights_for(family)
    r = np.array([w["range"].get(k, 0.0) for k in RANGE_TERM_KEYS], dtype=float)
    t = np.array([w["trend"].get(k, 0.0) for k in TREND_TERM_KEYS], dtype=float)
    return r, t


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    p.add_argument("--hold", nargs="+", type=int, default=[5, 10, 20],
                   help="holding periods in bars to sweep")
    p.add_argument("--enter", nargs="+", type=float, default=[40.0, 50.0, 60.0],
                   help="entry thresholds on |score| to sweep")
    p.add_argument("--min-bars", type=int, default=60)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if not RAW.exists():
        raise SystemExit(f"Missing raw workbook: {RAW}")

    print(f"Loading raw workbook: {RAW}", flush=True)
    frames = build_all(load_raw(RAW))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    started = time.time()
    for family in args.families:
        cost = COST_STUB_POINTS.get(family, 0.0)
        print(f"[{family}] enriching + precomputing (cost stub {cost} pts) ...", flush=True)
        enriched = build_dataset(frames[family], family)
        pcs = precompute_family(enriched, family, min_bars=args.min_bars)
        if len(pcs) < 2:
            print(f"[{family}] skipped: only {len(pcs)} usable contracts")
            continue

        w_range, w_trend = weight_vectors(family)
        panel = panel_from_precomputed(pcs, w_range, w_trend)

        ledgers = []
        for hold in args.hold:
            for enter in args.enter:
                t = nonoverlapping_trades(panel, hold_bars=hold, enter_at=enter, cost=cost)
                s = summarize(t)
                s.update(family=family, hold=hold, enter=enter, contracts=len(pcs))
                rows.append(s)
                if not t.empty:
                    ledgers.append(t.assign(hold=hold, enter=enter))
                print(f"  hold={hold:>3} enter={enter:>5.0f} -> "
                      f"{s['trades']:>4} trades  net={s['total_pnl']:>8.2f} pts  "
                      f"hit={s['hit_rate'] if s['hit_rate']==s['hit_rate'] else float('nan'):.3f}  "
                      f"sharpe={s['sharpe'] if s['sharpe']==s['sharpe'] else float('nan'):.2f}",
                      flush=True)
        if ledgers:
            pd.concat(ledgers, ignore_index=True).to_csv(
                args.out_dir / f"{family}_trades.csv", index=False)

    if not rows:
        print("No results produced.")
        return
    summary = pd.DataFrame(rows)
    cols = ["family", "hold", "enter", "contracts", "trades", "trades_per_year", "years",
            "total_pnl", "pnl_per_year", "mean_pnl", "hit_rate", "sharpe", "worst", "best"]
    summary = summary[[c for c in cols if c in summary.columns]]
    path = args.out_dir / "summary.csv"
    summary.to_csv(path, index=False)
    print(f"\nWrote {path}  ({len(summary)} configs, {time.time()-started:.0f}s)")


if __name__ == "__main__":
    main()
