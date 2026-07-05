"""Reading the raw feed and writing the built dataframes to disk."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw GLBX Excel export and attach a clean daily ``date`` column.

    ``ts_event`` is tz-aware UTC (e.g. "2010-06-06T00:00:00.000000000Z"); we
    convert it once into a naive, midnight-normalized daily date.
    """
    df = pd.read_excel(path)
    df["date"] = pd.to_datetime(df["ts_event"], utc=True).dt.tz_localize(None).dt.normalize()
    return df


def save_frames(frames: dict[str, pd.DataFrame], outdir: Path, fmt: str = "parquet") -> None:
    """Write each frame to ``outdir`` as ``<name>.<fmt>``.

    Parquet is preferred (smaller, dtype-preserving); if no parquet engine is
    installed we fall back to CSV per-frame.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        if fmt == "parquet":
            try:
                path = outdir / f"{name}.parquet"
                frame.to_parquet(path, index=False)
            except Exception as exc:  # pragma: no cover - depends on optional engine
                path = outdir / f"{name}.csv"
                frame.to_csv(path, index=False)
                print(f"  (parquet unavailable: {exc}; wrote CSV instead)")
        else:
            path = outdir / f"{name}.csv"
            frame.to_csv(path, index=False)
        print(f"  wrote {path}")
