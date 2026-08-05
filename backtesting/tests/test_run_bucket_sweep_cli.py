"""Tests for the ``scripts/run_bucket_sweep.py`` CLI.

The script lives in ``scripts/``, not a package, so it is loaded by file path
via ``importlib``. Only ``parse_args``, ``validate``, and ``write_family`` are
exercised here -- nothing that would trigger an actual sweep (which needs the
raw workbook and takes real time).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_bucket_sweep.py"
_spec = importlib.util.spec_from_file_location("run_bucket_sweep", _SCRIPT_PATH)
run_bucket_sweep = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = run_bucket_sweep
_spec.loader.exec_module(run_bucket_sweep)


def test_validate_rejects_max_k_above_the_theme_count():
    """7 themes total, so --max-k 8 can never form a combination."""
    args = run_bucket_sweep.parse_args(["--all-cross-theme", "--max-k", "8"])
    with pytest.raises(SystemExit, match="exceeds the 7 themes"):
        run_bucket_sweep.validate(args)


def test_validate_rejects_max_k_above_4_without_force():
    args = run_bucket_sweep.parse_args(["--all-cross-theme", "--max-k", "5"])
    with pytest.raises(SystemExit, match="--force"):
        run_bucket_sweep.validate(args)


def test_validate_allows_max_k_above_4_with_force(monkeypatch, tmp_path):
    """--force bypasses the cell-count guard; the raw-workbook check further
    down still needs a real path, so point RAW at something that exists.
    """
    dummy_raw = tmp_path / "raw.xlsx"
    dummy_raw.touch()
    monkeypatch.setattr(run_bucket_sweep, "RAW", dummy_raw)

    args = run_bucket_sweep.parse_args(["--all-cross-theme", "--max-k", "5", "--force"])
    run_bucket_sweep.validate(args)  # must not raise


def test_validate_rejects_n_buckets_other_than_three():
    args = run_bucket_sweep.parse_args(["--n-buckets", "4"])
    with pytest.raises(SystemExit, match="terciles|not supported"):
        run_bucket_sweep.validate(args)

    args_low = run_bucket_sweep.parse_args(["--n-buckets", "2"])
    with pytest.raises(SystemExit):
        run_bucket_sweep.validate(args_low)


def test_write_family_writes_all_three_outputs_atomically(tmp_path):
    results = pd.DataFrame({
        "family": ["fam"],
        "k": [1],
        "indicators": ["a1"],
        "buckets": ["low"],
        "horizon": [1],
        "n": [10],
        "mean": [0.1],
        "median": [0.1],
        "std": [0.2],
        "hit_rate": [0.5],
        "t_stat": [1.5],
    })
    cutoffs = pd.DataFrame({
        "family": ["fam", "fam"],
        "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "indicator": ["a1", "a1"],
        "edge_index": [0, 0],
        "value": [1.0, 2.0],
    })

    path = run_bucket_sweep.write_family(tmp_path, "fam", results, cutoffs)

    assert path == tmp_path / "fam.parquet"
    assert path.exists()
    assert (tmp_path / "fam_cutoffs.parquet").exists()
    assert (tmp_path / "fam_cutoffs_latest.csv").exists()

    # Nothing else -- no leftover temp files from the atomic-write dance.
    leftover = [p for p in tmp_path.iterdir()
                if p.name not in {"fam.parquet", "fam_cutoffs.parquet",
                                  "fam_cutoffs_latest.csv"}]
    assert leftover == []

    written = pd.read_parquet(path)
    pd.testing.assert_frame_equal(written, results)

    latest = pd.read_csv(tmp_path / "fam_cutoffs_latest.csv", parse_dates=["date"])
    assert (latest["date"] == pd.Timestamp("2020-01-02")).all()
