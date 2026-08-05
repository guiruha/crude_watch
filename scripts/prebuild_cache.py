"""Pre-build the parquet cache so the packaged app boots instantly.

Reads ``data/raw_files.xlsx`` and writes ``data/processed/*.parquet``. Run this
before packaging (``pyinstaller CrudeWatch.spec``) so the parquet files get
baked into the executable and no rebuild is needed on the target machine.

    python scripts/prebuild_cache.py
    python scripts/prebuild_cache.py --enriched
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crudewatch.data_preparation import build_all  # noqa: E402
from crudewatch.infra import load_raw, save_frames  # noqa: E402
from crudewatch.research import build_dataset  # noqa: E402

RAW = ROOT / "data" / "raw_files.xlsx"
OUT = ROOT / "data" / "processed"
ENRICHED_OUT = ROOT / "data" / "enriched"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--enriched",
        action="store_true",
        help="also prebuild data/enriched/*.parquet for low-memory deployments",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not RAW.exists():
        raise SystemExit(f"Raw workbook not found: {RAW}")
    OUT.mkdir(parents=True, exist_ok=True)
    frames = build_all(load_raw(RAW))
    save_frames(frames, OUT, "parquet")
    print(f"Wrote {len(frames)} frames to {OUT}")
    if args.enriched:
        ENRICHED_OUT.mkdir(parents=True, exist_ok=True)
        for family, frame in frames.items():
            path = ENRICHED_OUT / f"{family}.parquet"
            build_dataset(frame, family).to_parquet(path, index=False)
            print(f"  wrote {path}")
        print(f"Wrote {len(frames)} enriched frames to {ENRICHED_OUT}")


if __name__ == "__main__":
    main()
