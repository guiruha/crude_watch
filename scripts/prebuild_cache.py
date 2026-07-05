"""Pre-build the parquet cache so the packaged app boots instantly.

Reads ``data/raw_files.xlsx`` and writes ``data/processed/*.parquet``. Run this
before packaging (``pyinstaller CrudeWatch.spec``) so the parquet files get
baked into the executable and no rebuild is needed on the target machine.

    python scripts/prebuild_cache.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crudewatch.data_preparation import build_all  # noqa: E402
from crudewatch.infra import load_raw, save_frames  # noqa: E402

RAW = ROOT / "data" / "raw_files.xlsx"
OUT = ROOT / "data" / "processed"


def main() -> None:
    if not RAW.exists():
        raise SystemExit(f"Raw workbook not found: {RAW}")
    OUT.mkdir(parents=True, exist_ok=True)
    frames = build_all(load_raw(RAW))
    save_frames(frames, OUT, "parquet")
    print(f"Wrote {len(frames)} frames to {OUT}")


if __name__ == "__main__":
    main()
