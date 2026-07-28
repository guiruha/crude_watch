"""Generate one HTML backtest report per moving-average crossover indicator.

    python scripts/run_backtests.py

Reads the processed parquet cache when present (``data/processed/*.parquet``),
otherwise rebuilds the frames from ``data/raw_files.xlsx``. Writes:

    docs/reports/backtest/<INDICATOR>.html   one page per indicator
    docs/reports/backtest/index.html         links to all of them
    docs/reports/backtest/metrics.csv        raw per-contract metrics
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # the offline `backtesting` package lives at repo root

from backtesting.backtest.engine import DEFAULT_STRATEGIES  # noqa: E402
from backtesting.backtest.report import write_reports  # noqa: E402
from crudewatch.data_preparation import build_all  # noqa: E402
from crudewatch.infra import load_raw  # noqa: E402

FRAME_NAMES = [
    "outrights", "calendars", "cracks", "brent_wti",
    "quarterly", "semestral", "yearly", "flies",
]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw_files.xlsx"
OUT_DIR = ROOT / "docs" / "reports" / "backtest"


def load_frames() -> dict[str, pd.DataFrame]:
    """Prefer the parquet cache; fall back to rebuilding from the raw workbook."""
    parquet = {name: PROCESSED / f"{name}.parquet" for name in FRAME_NAMES}
    if all(p.exists() for p in parquet.values()):
        print(f"Reading parquet cache from {PROCESSED}")
        return {name: pd.read_parquet(p) for name, p in parquet.items()}
    print(f"No parquet cache; building frames from {RAW}")
    return build_all(load_raw(RAW))


def main() -> None:
    frames = load_frames()
    paths, raw = write_reports(frames, DEFAULT_STRATEGIES, OUT_DIR)

    if not raw.empty:
        csv_path = OUT_DIR / "metrics.csv"
        raw.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}  ({len(raw)} rows)")

    print(f"\nGenerated {len(DEFAULT_STRATEGIES)} reports in {OUT_DIR}:")
    for path in paths:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
