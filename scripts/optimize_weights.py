"""Optimise per-family Opportunity-Score weights and (optionally) apply them.

Usage:
    uv run python scripts/optimize_weights.py            # all reversion families
    uv run python scripts/optimize_weights.py --families flies --dirichlet 1200 --sparse 800
    uv run python scripts/optimize_weights.py --dry-run  # report only; never writes JSON
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
from crudewatch.scoring.score import (  # noqa: E402
    RANGE_TERM_KEYS as _RANGE_KEYS,
    TREND_TERM_KEYS as _TREND_KEYS,
)
from crudewatch.scoring.weight_search import (  # noqa: E402
    precompute_family, search_weights, walk_forward_weights,
)

REVERSION_FAMILIES = ["quarterly", "semestral", "yearly", "flies"]
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
    ap.add_argument("--dirichlet", type=int, default=1200)
    ap.add_argument("--sparse", type=int, default=800)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=5)
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
        pcs = precompute_family(_load_enriched(family), family, min_bars=args.min_bars, progress=True)
        if len(pcs) < 2:
            rows.append({"family": family, "note": f"skipped ({len(pcs)} contracts)"})
            continue
        wf = walk_forward_weights(
            pcs, cost,
            dirichlet_n=args.dirichlet, sparse_n=args.sparse,
            k_min=args.k_min, k_max=args.k_max,
            seed=args.seed, min_train=args.min_train,
        )
        full = search_weights(
            pcs, cost,
            dirichlet_n=args.dirichlet, sparse_n=args.sparse,
            k_min=args.k_min, k_max=args.k_max,
            seed=args.seed,
        )
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
            "active_range": full.active_range,
            "active_trend": full.active_trend,
        })
        print(f"[{family}] IS={_fmt(full.sharpe)} eq={_fmt(full.equal_sharpe)} "
              f"OOS opt={_fmt(wf.oos_sharpe_opt)} eq={_fmt(wf.oos_sharpe_equal)} adopt={beats}", flush=True)

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Optimización de pesos — reporte",
        "",
        "Aviso: varios indicadores candidatos ya alimentan los bloques agregados (colinealidad); pesos 0 = indicador inactivo.",
        "",
        f"Dirichlet: {args.dirichlet} · sparse: {args.sparse} · k: [{args.k_min}, {args.k_max}] · "
        f"semilla: {args.seed} · min_bars: {args.min_bars} · margen OOS: {args.margin}",
        "",
        "| Familia | Sharpe IS | Sharpe IS eq | Sharpe OOS opt | Sharpe OOS eq | Splits | ¿Adoptado? |",
        "|---|---|---|---|---|---|---|",
    ]
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
        lines.append(f"- **Range activos:** {', '.join(r['active_range']) or '—'}")
        lines.append(f"- **Trend activos:** {', '.join(r['active_trend']) or '—'}")
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
    for fam in args.families:
        existing.pop(fam, None)
    existing.update(adopted)
    _JSON_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n")
    print(f"Adopted {sorted(adopted)} -> {_JSON_PATH}")


if __name__ == "__main__":
    main()
