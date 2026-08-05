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

from collections.abc import Callable, Iterator, Mapping, Sequence
from itertools import combinations, product

import numpy as np
import pandas as pd

from crudewatch.research.features import FEATURE_NAMES, add_features
from crudewatch.research.targets import add_forward_returns

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
    NaN and would otherwise be silently coded as bucket 0 ("low"). This
    exclusion is per indicator, one column at a time; callers that need a
    single complete-case sample across every indicator (as ``run_family``
    does) drop rows with any ``MISSING_CODE`` from all combinations, not just
    that one indicator's.

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


RESULT_COLUMNS: tuple[str, ...] = (
    "k", "themes", "indicators", "buckets", "horizon",
    "n", "mean", "median", "std", "hit_rate", "t_stat",
)


def decode_cell(cell: int, k: int, n_buckets: int) -> str:
    """Turn a joint integer cell code back into ``|``-joined bucket labels.

    Inverse of the mixed-radix encoding ``b0 + n*b1 + n^2*b2 + ...`` used by the
    sweep, so position ``i`` of the result is the bucket of the ``i``-th
    indicator of the combination.
    """
    labels = [BUCKET_LABELS[(cell // n_buckets**i) % n_buckets] for i in range(k)]
    return "|".join(labels)


def cell_stats(
    data: pd.DataFrame,
    code: pd.Series,
    horizons: Sequence[int],
    min_samples: int,
) -> pd.DataFrame:
    """Per-cell forward-outcome statistics for one combination, all horizons.

    ``data`` holds ``fwd_{h}`` (forward price difference) and ``hit_{h}``
    (1.0 where ``fwd_{h} > 0``) columns; ``code`` is the joint integer cell per
    row. Cells with fewer than ``min_samples`` rows are omitted entirely rather
    than emitted with NaN statistics.
    """
    agg_map: dict[str, list[str]] = {}
    for h in horizons:
        agg_map[f"fwd_{h}"] = ["count", "mean", "median", "std"]
        agg_map[f"hit_{h}"] = ["mean"]

    desc = data.groupby(code, observed=True, sort=False).agg(agg_map)

    frames: list[pd.DataFrame] = []
    for h in horizons:
        keep = desc[(f"fwd_{h}", "count")] >= min_samples
        if not keep.any():
            continue
        kept = desc.loc[keep]
        n = kept[(f"fwd_{h}", "count")].to_numpy(dtype=float)
        mean = kept[(f"fwd_{h}", "mean")].to_numpy(dtype=float)
        std = kept[(f"fwd_{h}", "std")].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = mean / (std / np.sqrt(n))
        frames.append(
            pd.DataFrame(
                {
                    "cell": kept.index.to_numpy(),
                    "horizon": h,
                    "n": n.astype(np.int64),
                    "mean": mean,
                    "median": kept[(f"fwd_{h}", "median")].to_numpy(dtype=float),
                    "std": std,
                    "hit_rate": kept[(f"hit_{h}", "mean")].to_numpy(dtype=float),
                    "t_stat": t_stat,
                }
            )
        )

    if not frames:
        int_cols = ("cell", "horizon", "n")
        float_cols = ("mean", "median", "std", "hit_rate", "t_stat")
        return pd.DataFrame(
            {
                **{c: pd.Series(dtype="int64") for c in int_cols},
                **{c: pd.Series(dtype="float64") for c in float_cols},
            }
        )[list(int_cols + float_cols)]
    return pd.concat(frames, ignore_index=True)


# Indicator -> theme. Mirrors the scoring engine's blocks (block_trendiness,
# block_direction, block_strength, block_level in crudewatch.scoring.blocks) and
# the app's component screens, so the sweep speaks the project's vocabulary.
#
# A combination takes at most one indicator per theme. Two indicators from the
# same theme mostly restate each other — z_20 and z_50 are one measurement at
# two windows — so pairing them buys a narrower cell, not a new question.
THEMES: dict[str, tuple[str, ...]] = {
    "level": ("z_10", "z_20", "z_50", "keltner_dist_20"),
    "direction": ("slope_20", "macd_hist", "ema_align", "mom_5", "mom_10", "mom_20"),
    "exhaustion": ("rsi_div_14", "macd_div", "mom_decel_10", "er_drop_20"),
    "regime": ("er_20", "variance_ratio_5", "autocorr_20"),
    "quality": ("r2_20", "dir_persistence_20"),
    "oscillator": ("rsi_2", "rsi_14"),
    "volatility": ("vol_ratio",),
}

CORE_BLOCK_ORDER: tuple[str, ...] = ("regime", "direction", "strength", "level")
CORE_BLOCK_THEMES: dict[str, tuple[str, ...]] = {
    "regime": ("er_20", "variance_ratio_5", "autocorr_20"),
    "direction": ("slope_20", "macd_hist", "ema_align", "mom_5", "mom_10", "mom_20"),
    "strength": ("r2_20", "dir_persistence_20", "vol_ratio"),
    "level": ("z_10", "z_20", "z_50", "keltner_dist_20"),
}
CORE_BLOCK_THEME: dict[str, str] = {
    name: theme for theme, names in CORE_BLOCK_THEMES.items() for name in names
}

# Excluded from the sweep, not from the project. Bollinger %B expands to
# (z + k) / 2k, an exact affine transform of the z-score over the same window,
# so with rank-based quantile bucketing these produce byte-identical bucket
# codes to z_10 / z_20 and cost compute for zero information. They stay in
# ``FEATURES`` and on the dashboard, where %B is the more legible presentation.
EXCLUDED_INDICATORS: dict[str, str] = {
    "pctb_10_1_5": "affine transform of z_10",
    "pctb_20_2": "affine transform of z_20",
}

INDICATOR_THEME: dict[str, str] = {
    name: theme for theme, names in THEMES.items() for name in names
}

# The indicators the sweep actually runs on: FEATURE_NAMES minus the exclusions,
# in FEATURE_NAMES order so combination enumeration stays deterministic.
SWEEP_INDICATORS: tuple[str, ...] = tuple(
    n for n in FEATURE_NAMES if n not in EXCLUDED_INDICATORS
)


def _by_theme(
    indicators: Sequence[str], theme_of: Mapping[str, str]
) -> list[tuple[str, list[str]]]:
    """Group ``indicators`` by theme, preserving first-seen order for determinism."""
    grouped: dict[str, list[str]] = {}
    for name in indicators:
        grouped.setdefault(theme_of[name], []).append(name)
    return list(grouped.items())


def theme_combinations(
    indicators: Sequence[str],
    max_k: int,
    theme_of: Mapping[str, str] = INDICATOR_THEME,
) -> Iterator[tuple[str, ...]]:
    """Yield every cross-theme combination of size 1..``max_k``.

    A combination picks k distinct themes and one indicator from each, so no two
    of its indicators share a theme. Themes are chosen with ``combinations`` (so
    unordered, each set once) and the per-theme picks are enumerated in order,
    making the whole sequence deterministic.
    """
    grouped = _by_theme(indicators, theme_of)

    for k in range(1, max_k + 1):
        for theme_group in combinations(grouped, k):
            yield from product(*(names for _, names in theme_group))


def count_theme_combinations(
    indicators: Sequence[str],
    max_k: int,
    theme_of: Mapping[str, str] = INDICATOR_THEME,
) -> int:
    """How many combinations ``theme_combinations`` will yield, without building them.

    This is the sum of the elementary symmetric polynomials ``e_1..e_max_k`` of
    the theme sizes, accumulated by expanding ``prod(1 + size_i * x)``.
    """
    sizes = [len(names) for _, names in _by_theme(indicators, theme_of)]
    poly = [1]  # coefficients of x^0, x^1, ...
    for size in sizes:
        poly = poly + [0]
        for i in range(len(poly) - 1, 0, -1):
            poly[i] += size * poly[i - 1]
    return sum(poly[1 : max_k + 1])


def core_block_combinations(
    indicators: Sequence[str],
    top_n_per_theme: int = 3,
    block_order: Sequence[str] = CORE_BLOCK_ORDER,
    theme_sets: Mapping[str, Sequence[str]] = CORE_BLOCK_THEMES,
) -> Iterator[tuple[str, ...]]:
    """Yield exact one-per-core-block combinations.

    Indicators are capped per block in the order supplied by ``theme_sets``.
    This keeps the default offline search focused on the PM structure:
    Régimen + Dirección + Fuerza + Nivel.
    """
    available = set(indicators)
    grouped: list[tuple[str, ...]] = []
    for theme in block_order:
        names = tuple(name for name in theme_sets[theme] if name in available)
        names = names[: max(int(top_n_per_theme), 0)]
        if not names:
            return
        grouped.append(names)
    yield from product(*grouped)


def count_core_block_combinations(
    indicators: Sequence[str],
    top_n_per_theme: int = 3,
    block_order: Sequence[str] = CORE_BLOCK_ORDER,
    theme_sets: Mapping[str, Sequence[str]] = CORE_BLOCK_THEMES,
) -> int:
    total = 1
    available = set(indicators)
    for theme in block_order:
        n = sum(1 for name in theme_sets[theme] if name in available)
        n = min(n, max(int(top_n_per_theme), 0))
        if n == 0:
            return 0
        total *= n
    return total


def sweep(
    codes: pd.DataFrame,
    forwards: pd.DataFrame,
    max_k: int = 4,
    min_samples: int = 30,
    n_buckets: int = 3,
    theme_of: Mapping[str, str] = INDICATOR_THEME,
    core_blocks: bool = False,
    top_n_per_theme: int = 3,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Statistics for every joint bucket cell of every cross-theme combination.

    Combinations come from :func:`theme_combinations`, so each is unordered,
    appears once, and never contains two indicators of the same theme. The joint
    cell of a combination is the mixed-radix code ``b0 + n*b1 + n^2*b2 + ...``,
    which collapses a k-way grouping into a single integer key — grouping on one
    integer column is markedly faster in pandas than grouping on k separate keys.
    """
    names = list(codes.columns)
    horizons = [int(c.split("_")[1]) for c in forwards.columns]

    hits = (forwards > 0).astype(float)
    hits.columns = [f"hit_{h}" for h in horizons]
    data = pd.concat([forwards, hits], axis=1)

    if core_blocks:
        combos = core_block_combinations(names, top_n_per_theme)
        total = count_core_block_combinations(names, top_n_per_theme)
        combo_theme = CORE_BLOCK_THEME
    else:
        combos = theme_combinations(names, max_k, theme_of)
        total = count_theme_combinations(names, max_k, theme_of)
        combo_theme = theme_of
    done = 0
    parts: list[pd.DataFrame] = []

    for combo in combos:
        k = len(combo)
        valid = (codes[list(combo)] != MISSING_CODE).all(axis=1)
        code = codes[combo[0]].astype(np.int32)
        for i, col in enumerate(combo[1:], start=1):
            code = code + codes[col].astype(np.int32) * (n_buckets**i)

        stats = cell_stats(data.loc[valid], code.loc[valid], horizons, min_samples)
        done += 1
        if progress is not None:
            progress(done, total)
        if stats.empty:
            continue

        stats["k"] = k
        stats["themes"] = "|".join(combo_theme[name] for name in combo)
        stats["indicators"] = "|".join(combo)
        stats["buckets"] = [decode_cell(int(c), k, n_buckets) for c in stats["cell"]]
        parts.append(stats.drop(columns="cell"))

    if not parts:
        int_cols = ("k", "horizon", "n")
        object_cols = ("themes", "indicators", "buckets")
        float_cols = ("mean", "median", "std", "hit_rate", "t_stat")
        return pd.DataFrame(
            {
                **{c: pd.Series(dtype="int64") for c in int_cols},
                **{c: pd.Series(dtype=str) for c in object_cols},
                **{c: pd.Series(dtype="float64") for c in float_cols},
            }
        )[list(RESULT_COLUMNS)]
    return pd.concat(parts, ignore_index=True)[list(RESULT_COLUMNS)]


HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 10, 15, 20)


def run_family(
    family: str,
    frame: pd.DataFrame,
    horizons: Sequence[int] = HORIZONS,
    n_buckets: int = 3,
    max_k: int = 4,
    min_samples: int = 30,
    min_history: int = 2000,
    core_blocks: bool = False,
    top_n_per_theme: int = 3,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full sweep for one contract family: features -> outcomes -> buckets -> cells.

    Bucketing runs on the feature-complete panel (every indicator non-NaN),
    pooling as much prior history as possible into each date's quantile
    cutoffs. Forward columns must NOT gate that pool: whether row ``r`` has a
    valid ``fwd_20`` depends on data 20 bars after ``r``, so filtering on it
    before bucketing would let a date's edges depend on data after that date.
    Only once cutoffs and codes are fixed is the panel narrowed to the
    forward-complete subset the sweep actually runs on. What survives is a
    single complete-case panel for the sweep, so every combination is
    measured on an identical sample and cells are comparable across
    combinations.
    """
    featured = add_features(frame)
    full = add_forward_returns(featured, horizons=tuple(horizons))

    fwd_cols = [f"fwd_{h}" for h in horizons]
    featured_ok = full.dropna(subset=list(SWEEP_INDICATORS))

    if len(featured_ok) < min_history:
        raise InsufficientData(
            f"{family}: {len(featured_ok)} feature-complete rows, fewer than "
            f"the min_history={min_history} needed before any date can be "
            f"bucketed"
        )

    codes, cutoffs = bucketize(featured_ok, SWEEP_INDICATORS, n_buckets, min_history)
    codeable = (codes != MISSING_CODE).all(axis=1)
    codes = codes.loc[codeable]
    panel = featured_ok.loc[codes.index].dropna(subset=fwd_cols)
    codes = codes.loc[panel.index]
    # Not an assert: a silent misalignment here would pair every row's buckets
    # with a different row's forward outcomes, corrupting every statistic while
    # still looking structurally valid. This must hold under ``python -O`` too.
    if not codes.index.equals(panel.index):
        raise RuntimeError(
            f"{family}: bucket codes and forward outcomes are misaligned "
            f"({len(codes)} vs {len(panel)} rows)"
        )

    required = min_samples * n_buckets**max_k
    if len(codes) < required:
        raise InsufficientData(
            f"{family}: {len(codes)} codeable rows after warmup, need at least "
            f"{required} for max_k={max_k}, n_buckets={n_buckets}, "
            f"min_samples={min_samples}"
        )

    results = sweep(
        codes, panel[fwd_cols], max_k=max_k, min_samples=min_samples,
        n_buckets=n_buckets, core_blocks=core_blocks,
        top_n_per_theme=top_n_per_theme, progress=progress,
    )

    results.insert(0, "family", family)
    cutoffs.insert(0, "family", family)
    return results, cutoffs
