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
import os
import sys
import tempfile
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
from backtesting.research.bucket_sweep import (  # noqa: E402
    HORIZONS,
    SWEEP_INDICATORS,
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
    p.add_argument("--n-buckets", type=int, default=3,
                   help="buckets per indicator (terciles only -- 3 is the only "
                        "supported value)")
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
    if args.n_buckets != 3:
        raise SystemExit(
            f"--n-buckets {args.n_buckets} is not supported: BUCKET_LABELS is a "
            f"hard-coded ('low', 'mid', 'high') tercile tuple, so bucket labels "
            f"are only defined for --n-buckets 3."
        )
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
            n_k = (count_theme_combinations(SWEEP_INDICATORS, k)
                   - count_theme_combinations(SWEEP_INDICATORS, k - 1))
            cells += n_k * args.n_buckets**k
        raise SystemExit(
            f"--max-k {args.max_k} would produce {cells:,} cells per family. "
            f"Re-run with --force if that is genuinely what you want."
        )
    if not RAW.exists():
        raise SystemExit(f"Missing raw workbook: {RAW}")


def sweep_one(family: str, frame: pd.DataFrame, args: argparse.Namespace, progress=None):
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
            progress=progress,
        )
    except InsufficientData as exc:
        return family, None, None, str(exc)
    return family, results, cutoffs, None


def _atomic_write(final_path: Path, write_fn) -> None:
    """Write to a temp file in the same directory, then atomically rename it onto
    final_path. Guards against a crash mid-write leaving a truncated file at the
    exact path --resume's existence check looks for. The temp name uses a distinct
    suffix so a leftover from a killed run is never mistaken for a completed
    output."""
    fd, tmp_name = tempfile.mkstemp(
        dir=final_path.parent, prefix=f".{final_path.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, final_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_family(out_dir: Path, family: str, results: pd.DataFrame,
                 cutoffs: pd.DataFrame) -> Path:
    """Write one family's outputs as soon as it finishes, so --resume can skip it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{family}.parquet"
    _atomic_write(path, lambda tmp: results.to_parquet(tmp, index=False))

    cutoffs_path = out_dir / f"{family}_cutoffs.parquet"
    _atomic_write(cutoffs_path, lambda tmp: cutoffs.to_parquet(tmp, index=False))

    latest = cutoffs[cutoffs["date"] == cutoffs["date"].max()]
    latest_path = out_dir / f"{family}_cutoffs_latest.csv"
    _atomic_write(latest_path, lambda tmp: latest.to_csv(tmp, index=False))

    return path


def _progress_printer(family: str):
    """A sparse, single-line ``sweep`` progress callback: updates on ~5% steps."""
    last_pct: list[int] = [-1]

    def progress(done: int, total: int) -> None:
        pct = 0 if total == 0 else int(done * 100 / total)
        pct = (pct // 5) * 5
        if pct == last_pct[0] and done != total:
            return
        last_pct[0] = pct
        end = "\n" if done == total else ""
        print(f"\r[run ] {family}: {done:,}/{total:,} combinations ({pct}%)",
              end=end, flush=True)

    return progress


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

    started = time.time()

    def finish(family: str, results, cutoffs, err: str | None) -> None:
        if err:
            print(f"[skip] {err}")
            return
        path = write_family(args.out_dir, family, results, cutoffs)
        print(f"[done] {family}: {len(results):,} cells -> {path}  "
              f"({time.time() - started:.0f}s elapsed)")

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(sweep_one, f, frames[f], args): f for f in families}
            for fut in as_completed(futures):
                family, results, cutoffs, err = fut.result()
                finish(family, results, cutoffs, err)
    else:
        for family in families:
            print(f"[run ] {family} ...")
            family, results, cutoffs, err = sweep_one(
                family, frames[family], args, progress=_progress_printer(family)
            )
            finish(family, results, cutoffs, err)

    # Pool the ranking from every family requested in this invocation, not just
    # the ones freshly written this run -- otherwise a `--resume` that skips
    # already-completed families silently ranks only the newly written subset.
    all_paths = [args.out_dir / f"{f}.parquet" for f in args.families]
    existing_paths = [p for p in all_paths if p.exists()]
    if not existing_paths:
        print("No results produced.")
        return

    # Loads every family's full results into memory at once (tens of millions of
    # rows at the documented full scale: ~322k cells x 7 horizons x 8 families).
    # A full run wants enough RAM; use --families to batch it if that's tight.
    pooled = pd.concat([pd.read_parquet(p) for p in existing_paths], ignore_index=True)
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
