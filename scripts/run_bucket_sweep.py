"""Sweep forward outcomes across every combination of indicator bucket states.

    uv run python scripts/run_bucket_sweep.py
    uv run python scripts/run_bucket_sweep.py --families quarterly flies --max-k 2
    uv run python scripts/run_bucket_sweep.py --jobs 4 --resume

For each contract family: bucket all 24 indicators into terciles whose cutoffs
are expanding quantiles over *strictly prior dates* (no look-ahead), form every
cross-theme combination of up to --max-k indicators (at most one indicator per
theme, so near-duplicates like z_20 with z_50 are never paired), and report the
distribution of forward price differences (close[t+h] - open[t+1]) inside each
joint bucket cell, for h in 1, 2, 3, 5, 10, 15, 20.

Writes, per family, to docs/reports/bucket_sweep/:

    <family>.parquet              one row per surviving cell x horizon
    <family>_cutoffs.parquet      the per-date edge series behind every bucket
    <family>_cutoffs_latest.csv   the edges in force on the final date

plus a pooled top_cells.csv of the strongest cells by |t_stat|.

Descriptive only. Every input is point-in-time, but cells are chosen by
inspection: with ~322k cells per family thousands clear |t| > 3 on noise alone,
and overlapping forward windows inflate it further. Rank with it, then confirm
on held-out dates before believing it.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # the offline `backtesting` package lives at repo root

from crudewatch.data_preparation import build_all  # noqa: E402
from crudewatch.infra import load_raw  # noqa: E402
from crudewatch.research.features import FEATURE_NAMES  # noqa: E402
from backtesting.research.bucket_sweep import (  # noqa: E402
    HORIZONS,
    THEMES,
    InsufficientData,
    count_theme_combinations,
    run_family,
)

FRAME_NAMES = [
    "outrights", "calendars", "cracks", "brent_wti",
    "quarterly", "semestral", "yearly", "flies",
]
RAW = ROOT / "data" / "raw_files.xlsx"
OUT_DIR = ROOT / "docs" / "reports" / "bucket_sweep"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--families", nargs="+", default=FRAME_NAMES, choices=FRAME_NAMES)
    p.add_argument("--max-k", type=int, default=4, help="combination depth (default 4)")
    p.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    p.add_argument("--n-buckets", type=int, default=3, help="buckets per indicator")
    p.add_argument("--min-samples", type=int, default=30, help="minimum rows per cell")
    p.add_argument("--min-history", type=int, default=2000,
                   help="prior rows required before a date can be bucketed")
    p.add_argument("--top-n", type=int, default=500, help="rows in top_cells.csv")
    p.add_argument("--jobs", type=int, default=1, help="worker processes")
    p.add_argument("--resume", action="store_true", help="skip families already written")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--force", action="store_true", help="allow --max-k above 4")
    return p.parse_args(argv)


def validate(args: argparse.Namespace) -> None:
    """Fail fast on inputs that would waste hours or blow up the output."""
    if args.max_k < 1:
        raise SystemExit("--max-k must be at least 1")
    if args.max_k > len(THEMES):
        raise SystemExit(
            f"--max-k {args.max_k} exceeds the {len(THEMES)} themes. A combination "
            f"holds at most one indicator per theme, so no combination can be "
            f"larger than {len(THEMES)}."
        )
    if args.max_k > 4 and not args.force:
        cells = 0
        for k in range(1, args.max_k + 1):
            n_k = (count_theme_combinations(FEATURE_NAMES, k)
                   - count_theme_combinations(FEATURE_NAMES, k - 1))
            cells += n_k * args.n_buckets**k
        raise SystemExit(
            f"--max-k {args.max_k} would produce {cells:,} cells per family. "
            f"Re-run with --force if that is genuinely what you want."
        )
    if not RAW.exists():
        raise SystemExit(f"Missing raw workbook: {RAW}")


def sweep_one(family: str, frame: pd.DataFrame, args: argparse.Namespace):
    """Run one family, returning (family, results, cutoffs, error_message)."""
    try:
        results, cutoffs = run_family(
            family,
            frame,
            horizons=args.horizons,
            n_buckets=args.n_buckets,
            max_k=args.max_k,
            min_samples=args.min_samples,
            min_history=args.min_history,
        )
    except InsufficientData as exc:
        return family, None, None, str(exc)
    return family, results, cutoffs, None


def write_family(out_dir: Path, family: str, results: pd.DataFrame,
                 cutoffs: pd.DataFrame) -> Path:
    """Write one family's outputs as soon as it finishes, so --resume can skip it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{family}.parquet"
    results.to_parquet(path, index=False)
    cutoffs.to_parquet(out_dir / f"{family}_cutoffs.parquet", index=False)
    latest = cutoffs[cutoffs["date"] == cutoffs["date"].max()]
    latest.to_csv(out_dir / f"{family}_cutoffs_latest.csv", index=False)
    return path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate(args)

    families = list(args.families)
    if args.resume:
        skipped = [f for f in families if (args.out_dir / f"{f}.parquet").exists()]
        for f in skipped:
            print(f"[resume] {f}: already written, skipping")
        families = [f for f in families if f not in skipped]
    if not families:
        print("Nothing to do.")
        return

    print(f"Loading raw workbook: {RAW}")
    frames = build_all(load_raw(RAW))

    written: list[Path] = []
    started = time.time()

    def report(family: str, results: pd.DataFrame, path: Path) -> None:
        print(f"[done] {family}: {len(results):,} cells -> {path}  "
              f"({time.time() - started:.0f}s elapsed)")

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(sweep_one, f, frames[f], args): f for f in families}
            for fut in as_completed(futures):
                family, results, cutoffs, err = fut.result()
                if err:
                    print(f"[skip] {err}")
                    continue
                path = write_family(args.out_dir, family, results, cutoffs)
                written.append(path)
                report(family, results, path)
    else:
        for family in families:
            print(f"[run ] {family} ...")
            _, results, cutoffs, err = sweep_one(family, frames[family], args)
            if err:
                print(f"[skip] {err}")
                continue
            path = write_family(args.out_dir, family, results, cutoffs)
            written.append(path)
            report(family, results, path)

    if not written:
        print("No results produced.")
        return

    pooled = pd.concat([pd.read_parquet(p) for p in written], ignore_index=True)
    finite = np.isfinite(pooled["t_stat"])
    n_excluded = int((~finite).sum())
    if n_excluded:
        print(f"Excluded {n_excluded} cells with non-finite t_stat (zero-variance) "
              f"from the ranking")
    rankable = pooled[finite]
    top = rankable.reindex(
        rankable["t_stat"].abs().sort_values(ascending=False).index
    ).head(args.top_n)
    top_path = args.out_dir / "top_cells.csv"
    top.to_csv(top_path, index=False)
    print(f"Wrote {top_path}  ({len(top)} rows)")


if __name__ == "__main__":
    main()
