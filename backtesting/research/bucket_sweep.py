"""Indicator bucket sweep (offline): forward outcomes per joint indicator state.

Buckets each continuous indicator into quantile terciles, forms every
combination of up to ``max_k`` indicators, and reports the distribution of
forward price differences inside each joint bucket cell.

**No look-ahead.** Bucket cutoffs are expanding quantiles over *strictly prior
dates*: the edges applied to a row dated ``d`` come from rows dated ``< d``
only. A plain ``panel[col].quantile(q)`` would silently calibrate on the whole
history, including the future of every early row — that is the mistake this
module exists to avoid. Features are as-of ``t`` by construction and forward
outcomes never enter bucketing, so the appended-future-bars test in the suite
covers the whole chain.

This module performs no file IO and no printing — ``scripts/run_bucket_sweep.py``
owns all of that.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

BUCKET_LABELS: tuple[str, ...] = ("low", "mid", "high")
MISSING_CODE: int = -1


class InsufficientData(RuntimeError):
    """Raised when a family has too few usable rows to sweep meaningfully."""


def _interior_quantiles(n_buckets: int) -> list[float]:
    if n_buckets < 2:
        raise ValueError(f"n_buckets must be at least 2, got {n_buckets}")
    return [i / n_buckets for i in range(1, n_buckets)]


def expanding_cutoffs(
    panel: pd.DataFrame,
    indicators: Sequence[str],
    n_buckets: int = 3,
    min_history: int = 2000,
    date_col: str = "date",
) -> pd.DataFrame:
    """Per-date interior quantile edges computed from strictly prior dates.

    The panel is sorted by date, so for the first row of any date every earlier
    row is strictly earlier in time. Reading an expanding quantile at the index
    just *before* that first row therefore yields a quantile over ``date < d``
    only — which also means sibling contracts trading on the same date cannot
    influence one another's edges.

    Returns a tidy ``date, indicator, edge_index, value`` frame. ``value`` is
    NaN while fewer than ``min_history`` prior rows exist.
    """
    quantiles = _interior_quantiles(n_buckets)
    ordered = panel.sort_values(date_col, kind="mergesort")
    dates = ordered[date_col].to_numpy()

    if len(dates) == 0:
        return pd.DataFrame(
            {
                "date": pd.Series(dtype=panel[date_col].dtype),
                "indicator": pd.Series(dtype=object),
                "edge_index": pd.Series(dtype=int),
                "value": pd.Series(dtype=float),
            }
        )

    # Index of the first row of each date, and the dates themselves.
    is_first = np.r_[True, dates[1:] != dates[:-1]]
    first_idx = np.flatnonzero(is_first)
    unique_dates = dates[first_idx]

    # Rows strictly before date d end at first_idx - 1. The very first date has
    # no prior history at all.
    has_prior = first_idx > 0
    prior_end = first_idx - 1

    frames: list[pd.DataFrame] = []
    for name in indicators:
        series = ordered[name].reset_index(drop=True)
        for edge_index, q in enumerate(quantiles):
            expanding = series.expanding(min_periods=min_history).quantile(q).to_numpy()
            values = np.full(len(first_idx), np.nan)
            values[has_prior] = expanding[prior_end[has_prior]]
            frames.append(
                pd.DataFrame(
                    {
                        "date": unique_dates,
                        "indicator": name,
                        "edge_index": edge_index,
                        "value": values,
                    }
                )
            )

    return pd.concat(frames, ignore_index=True)


def bucketize(
    panel: pd.DataFrame,
    indicators: Sequence[str],
    n_buckets: int = 3,
    min_history: int = 2000,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign each row a bucket code per indicator using that date's prior edges.

    A row's code is the number of its date's edges the value meets or exceeds
    (0 = low, 1 = mid, 2 = high for terciles). Rows whose date has no usable
    edges — still inside the ``min_history`` warmup, or a prior window so
    degenerate the edges are not strictly increasing — get ``MISSING_CODE``
    rather than a fabricated bucket. Rows whose own indicator value is NaN
    also get ``MISSING_CODE``, since ``value >= edge`` is always False for
    NaN and would otherwise be silently coded as bucket 0 ("low").

    Returns ``(codes, cutoffs)``; ``codes`` is indexed like the date-sorted
    ``panel`` so it can be aligned straight back onto it.
    """
    cutoffs = expanding_cutoffs(panel, indicators, n_buckets, min_history, date_col)
    ordered = panel.sort_values(date_col, kind="mergesort")
    row_dates = ordered[date_col]

    wide = cutoffs.pivot(index="date", columns=["indicator", "edge_index"], values="value")

    code_cols: dict[str, np.ndarray] = {}
    for name in indicators:
        values = ordered[name].to_numpy(dtype=float)
        edges = [
            wide[(name, j)].reindex(row_dates).to_numpy(dtype=float)
            for j in range(n_buckets - 1)
        ]

        usable = ~np.isnan(values)
        usable &= ~np.isnan(edges[0])
        for j in range(1, n_buckets - 1):
            usable &= ~np.isnan(edges[j])
            usable &= edges[j] > edges[j - 1]  # strictly increasing, else no real split

        code = np.zeros(len(values), dtype=np.int16)
        for edge in edges:
            code += (values >= edge).astype(np.int16)

        code_cols[name] = np.where(usable, code, MISSING_CODE).astype(np.int8)

    codes = pd.DataFrame(code_cols, index=ordered.index)
    return codes, cutoffs
