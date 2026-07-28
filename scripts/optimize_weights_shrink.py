"""Anti-overfit weight optimisation via shrinkage toward equal-weight.

Per family, the honest nested walk-forward selects a config on each train
vintage and evaluates it OOS after shrinking it toward the 8-term equal-weight
baseline by a factor ``lambda``. We sweep ``lambda`` and adopt the value whose
pooled OOS Sharpe beats equal-weight (else keep equal-weight). Deployment uses
the same shrinkage applied to the full-sample selected config.

Usage:
    uv run python scripts/optimize_weights_shrink.py --dry-run
    uv run python scripts/optimize_weights_shrink.py            # writes JSON
"""
from __future__ import annotations

import argparse
import json
import sys
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
from crudewatch.scoring.score import (  # noqa: E402
    RANGE_TERM_KEYS as _RANGE_KEYS,
    TREND_TERM_KEYS as _TREND_KEYS,
)
from crudewatch.scoring.weight_search import (  # noqa: E402
    _EQUAL_RANGE, _EQUAL_TREND, apply_shrinkage,
    precompute_family, search_weights, walk_forward_sweep,
)

REVERSION_FAMILIES = ["quarterly", "semestral", "yearly", "flies"]
_JSON_PATH = _SRC / "crudewatch" / "scoring" / "family_weights.json"
_REPORT_PATH = _ROOT / "docs" / "reports" / "weight_search" / "shrink_report.md"


def _load_enriched(family: str) -> pd.DataFrame:
    processed = _ROOT / "data" / "processed" / f"{family}.parquet"
    frame = pd.read_parquet(processed) if processed.exists() else \
        build_all(load_raw(_ROOT / "data" / "raw_files.xlsx"))[family]
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
    ap.add_argument("--lambdas", type=float, nargs="*", default=[0.3, 0.5, 0.7, 0.85, 0.95])
    ap.add_argument("--dirichlet", type=int, default=1200)
    ap.add_argument("--sparse", type=int, default=800)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-bars", type=int, default=60)
    ap.add_argument("--min-train", type=int, default=5)
    ap.add_argument("--margin", type=float, default=0.0)
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
        sweep = walk_forward_sweep(
            pcs, cost, lambdas=tuple(args.lambdas),
            dirichlet_n=args.dirichlet, sparse_n=args.sparse,
            k_min=args.k_min, k_max=args.k_max, seed=args.seed, min_train=args.min_train,
        )
        full = search_weights(
            pcs, cost, dirichlet_n=args.dirichlet, sparse_n=args.sparse,
            k_min=args.k_min, k_max=args.k_max, seed=args.seed,
        )
        oos = list(sweep.oos_sharpe_by_lambda)
        best_i = int(np.nanargmax(oos)) if any(v == v for v in oos) else -1
        best_lam = args.lambdas[best_i] if best_i >= 0 else float("nan")
        best_oos = oos[best_i] if best_i >= 0 else float("nan")
        eq = sweep.oos_sharpe_equal
        beats = (best_oos == best_oos) and (
            best_oos > (eq if eq == eq else float("-inf")) + args.margin
        )
        w_range = apply_shrinkage(full.w_range, _EQUAL_RANGE, best_lam) if beats else _EQUAL_RANGE
        w_trend = apply_shrinkage(full.w_trend, _EQUAL_TREND, best_lam) if beats else _EQUAL_TREND
        if beats:
            adopted[family] = _weights_dict(w_range, w_trend)
        rows.append({
            "family": family, "note": "",
            "lambdas": args.lambdas, "oos": oos, "oos_eq": eq,
            "best_lam": best_lam, "best_oos": best_oos, "splits": sweep.n_splits,
            "adopted": beats, "weights": _weights_dict(w_range, w_trend),
        })
        curve = " ".join(f"λ{lam:.2f}={_fmt(s)}" for lam, s in zip(args.lambdas, oos))
        print(f"[{family}] {curve} | eq={_fmt(eq)} -> best λ={best_lam if best_lam==best_lam else '—'} "
              f"OOS={_fmt(best_oos)} adopt={beats}", flush=True)

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Optimización con shrinkage — reporte",
        "",
        "Encoge la config seleccionada hacia equal-weight (λ=1 ≡ equal). Se adopta la λ "
        "con mejor Sharpe OOS agregado si supera al equal-weight.",
        "",
        f"Dirichlet: {args.dirichlet} · sparse: {args.sparse} · k: [{args.k_min}, {args.k_max}] · "
        f"min_train: {args.min_train} · semilla: {args.seed} · margen: {args.margin}",
        "",
        "| Familia | " + " | ".join(f"OOS λ{lam:.2f}" for lam in args.lambdas) +
        " | OOS eq | Mejor λ | Mejor OOS | Splits | ¿Adoptado? |",
        "|---" * (len(args.lambdas) + 6) + "|",
    ]
    for r in rows:
        if r.get("note"):
            lines.append(f"| {r['family']} | {r['note']} |" + " |" * (len(args.lambdas) + 5))
            continue
        cells = " | ".join(_fmt(s) for s in r["oos"])
        lines.append(
            f"| {r['family']} | {cells} | {_fmt(r['oos_eq'])} | "
            f"{r['best_lam'] if r['best_lam']==r['best_lam'] else '—'} | {_fmt(r['best_oos'])} | "
            f"{r['splits']} | {'sí' if r['adopted'] else 'no'} |"
        )
    lines += ["", "## Pesos desplegados (full-sample × shrinkage)", ""]
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
    for fam in args.families:
        existing.pop(fam, None)
    existing.update(adopted)
    _JSON_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n")
    print(f"Adopted {sorted(adopted)} -> {_JSON_PATH}")


if __name__ == "__main__":
    main()
