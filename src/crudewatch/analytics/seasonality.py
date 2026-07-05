"""Seasonality as a *context* layer for fixed-date spreads (Chapter 2).

Per the v1 spec, seasonality is **not** a mean-reversion signal: we never say
"high vs history -> sell" or "low vs history -> buy". The primary signal is the
current trend; seasonality only qualifies its *timing and trajectory*. It
produces several **separate** readings — never one opaque score:

* **Seasonal directional bias** — in this window of the life, has the spread
  historically tended up, down, or nowhere? (Bullish / Bearish / Neutral)
* **Consistency** — is that tendency tight or dispersed? (Strong / Medium /
  Weak / Mixed), from the agreement count (e.g. 9/12 years).
* **Alignment with the current trend** — the key dashboard read: does
  seasonality push *with* the trend (Tailwind), *against* it (Headwind) or add
  nothing (Mixed)? A headwind never blocks a strong trend; it flags that manual
  flow/fundamental confirmation is needed.
* **Lifecycle phase** — Quiet / Build-up / Acceleration / Late / Expiry-risk: a
  label that qualifies the trend score rather than a hard filter.

Path-vs-seasonal-path comparison is a drill-down *visual* only, not a formal
score input in v1. Every function is pure over the long stack frame with columns
``vintage``, ``date``, ``day_of_year``, ``days_to_expiry`` and ``close``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Dominant seasonal driver by calendar month (drill-down context panel).
SEASONAL_DRIVERS: dict[int, str] = {
    1: "Heating demand / year-end; liquidity concentrates in Dec.",
    2: "Winter demand tail; refinery run-cut planning begins.",
    3: "Refinery maintenance season — pressure on near contracts.",
    4: "Refinery maintenance continues; pre-driving-season build.",
    5: "Transition into driving season; gasoline demand firms.",
    6: "Driving season (gasoline demand) — tendency toward backwardation.",
    7: "Peak driving season; near-month strength.",
    8: "Driving season tail; hurricane-season supply risk.",
    9: "Shoulder season; autumn refinery maintenance.",
    10: "Shoulder season; inventory rebuild ahead of winter.",
    11: "Heating demand ramps; positioning into December.",
    12: "Heating demand / year-end; peak December liquidity.",
}

# Lifecycle phase descriptions (qualify the trend score; never a hard filter).
LIFECYCLE_NOTES: dict[str, str] = {
    "Quiet": "Early life, market asleep — little seasonal signal yet.",
    "Build-up": "The slope starts to build; positioning begins.",
    "Acceleration": "Window where the move has historically tended to extend.",
    "Late": "Mature phase — higher risk of exhaustion.",
    "Expiry-risk": "Convergence / roll / thin liquidity — beware artificial spikes.",
}

AXIS = "days_to_expiry"

# Seasonal analysis is confined to the final 330 days before expiry: the liquid,
# economically-comparable window that still reaches into the 12–10 month maturity
# bucket. Anything older is dropped. The smoothing buffer is a fixed ±3 days (not
# user-tunable) so the read is stable and reproducible.
MAX_DAYS_TO_EXPIRY = 330
BUFFER_DAYS = 3

# Maturity split by months-to-expiry (≈30.4 days/month). Every screen can slice
# the analysis to one of these regimes; "All" is the full window. Ranges are in
# days-to-expiry, inclusive on both ends.
_BIG_DTE = 100_000
MATURITY_BUCKETS: dict[str, tuple[int, int] | None] = {
    "All": None,
    "12–10m": (305, 366),
    "10–6m": (183, 305),
    "6–1m": (30, 183),
    "1–0m": (0, 30),
}


def bucket_bounds(label: str) -> tuple[int, int]:
    """(min, max) days-to-expiry for a maturity bucket label; full range for 'All'."""
    rng = MATURITY_BUCKETS.get(label)
    return (0, _BIG_DTE) if rng is None else rng


_BULL, _BEAR, _NEUTRAL = "Bullish", "Bearish", "Neutral"
_TAILWIND, _HEADWIND, _MIXED = "Tailwind", "Headwind", "Mixed"
# Map any trend vocabulary onto the seasonal Bullish/Bearish/Neutral vocabulary.
_TREND_NORM = {
    "Up": _BULL, "Down": _BEAR, "Flat": _NEUTRAL,
    _BULL: _BULL, _BEAR: _BEAR, _NEUTRAL: _NEUTRAL,
}


@dataclass(frozen=True)
class SeasonalBias:
    """Directional bias + consistency over a forward ``horizon`` from this point in the life."""
    direction: str        # Bullish | Bearish | Neutral
    up: int
    down: int
    total: int
    horizon: int

    @property
    def hit_rate(self) -> float:
        """Share of past vintages agreeing with the modal direction (0..1)."""
        return 0.0 if self.total == 0 else max(self.up, self.down) / self.total

    @property
    def count_label(self) -> str:
        """Agreement count in the modal direction, e.g. '9/12'."""
        return f"{max(self.up, self.down)}/{self.total}"

    @property
    def consistency(self) -> str:
        """Strong / Medium / Weak / Mixed — dispersed or small samples read Mixed."""
        if self.total < 5 or self.direction == _NEUTRAL:
            return _MIXED
        r = self.hit_rate
        if r >= 0.75:
            return "Strong"
        if r >= 0.60:
            return "Medium"
        if r >= 0.55:
            return "Weak"
        return _MIXED


@dataclass(frozen=True)
class Lifecycle:
    """Where the active vintage sits in its life, as a qualifying label."""
    phase: str            # Quiet | Build-up | Acceleration | Late | Expiry-risk
    days_to_expiry: int | None
    elapsed: float        # 0 (new) .. 1 (at expiry)

    @property
    def note(self) -> str:
        return LIFECYCLE_NOTES.get(self.phase, "")


# -- helpers ------------------------------------------------------------------

def cap_to_year(stack: pd.DataFrame, max_dte: int = MAX_DAYS_TO_EXPIRY, axis: str = AXIS,
                min_dte: int = 0) -> pd.DataFrame:
    """Restrict a stack to the ``[min_dte, max_dte]`` days-to-expiry window."""
    if stack.empty:
        return stack
    return stack[(stack[axis] <= max_dte) & (stack[axis] >= min_dte)]


# -- confidence cone & heatmap (visual layers) --------------------------------

def buffer_cone(stack: pd.DataFrame, buffer_days: int = BUFFER_DAYS, exclude_vintage: int | None = None,
                axis: str = AXIS, max_dte: int = MAX_DAYS_TO_EXPIRY, min_dte: int = 0) -> pd.DataFrame:
    """Rolling mean ± 1σ over a ±``buffer_days`` window on the seasonal ``axis``.

    Returns a frame indexed by the axis value with ``mean``, ``std``, ``lower``,
    ``upper``, ``median`` and ``n``. ``exclude_vintage`` leaves the active year
    out so it can be compared against its own history. Restricted to the
    ``[min_dte, max_dte]`` days-to-expiry window.
    """
    stack = cap_to_year(stack, max_dte, axis, min_dte)
    hist = stack if exclude_vintage is None else stack[stack["vintage"] != exclude_vintage]
    if hist.empty:
        return pd.DataFrame(columns=["mean", "std", "lower", "upper", "median", "n"])

    x = hist[axis].to_numpy()
    val = hist["close"].to_numpy()
    order = np.argsort(x)
    x_sorted = x[order]
    val_sorted = val[order]

    rows = {}
    for d in range(int(x_sorted.min()), int(x_sorted.max()) + 1):
        lo = np.searchsorted(x_sorted, d - buffer_days, side="left")
        hi = np.searchsorted(x_sorted, d + buffer_days, side="right")
        sample = val_sorted[lo:hi]
        if sample.size == 0:
            continue
        m = float(sample.mean())
        s = float(sample.std(ddof=0))
        rows[d] = (m, s, m - s, m + s, float(np.median(sample)), int(sample.size))

    if not rows:
        return pd.DataFrame(columns=["mean", "std", "lower", "upper", "median", "n"])
    out = pd.DataFrame.from_dict(
        rows, orient="index", columns=["mean", "std", "lower", "upper", "median", "n"],
    )
    out.index.name = axis
    return out


def current_percentile(stack: pd.DataFrame, current_vintage: int, buffer_days: int = BUFFER_DAYS,
                       axis: str = AXIS, max_dte: int = MAX_DAYS_TO_EXPIRY, min_dte: int = 0) -> float | None:
    """Percentile of the active value vs history at the same point in life (context only).

    Deliberately *not* a trade trigger — shown as background context, never as a
    "rich = sell / cheap = buy" signal.
    """
    stack = cap_to_year(stack, max_dte, axis, min_dte)
    active = stack[stack["vintage"] == current_vintage].sort_values("date")
    if active.empty:
        return None
    anchor = float(active[axis].iloc[-1])
    current = float(active["close"].iloc[-1])

    hist = stack[stack["vintage"] != current_vintage]
    window = hist[(hist[axis] - anchor).abs() <= buffer_days]
    if window.empty:
        return None
    return float((window["close"] <= current).mean() * 100)


def monthly_heatmap(stack: pd.DataFrame, max_dte: int = MAX_DAYS_TO_EXPIRY, axis: str = AXIS) -> pd.DataFrame:
    """Each vintage's spread move per calendar month (last close - first close)."""
    stack = cap_to_year(stack, max_dte, axis)
    if stack.empty:
        return pd.DataFrame()
    df = stack.copy()
    df["month"] = df["date"].dt.month
    df = df.sort_values("date")
    grp = df.groupby(["month", "vintage"])["close"]
    change = grp.last() - grp.first()
    return change.unstack("vintage").sort_index()


# -- separate seasonal readings -----------------------------------------------

def seasonal_bias(stack: pd.DataFrame, current_vintage: int, buffer_days: int = BUFFER_DAYS,
                  horizon: int = 20, axis: str = AXIS, max_dte: int = MAX_DAYS_TO_EXPIRY,
                  min_dte: int = 0) -> SeasonalBias:
    """Historical forward tendency from this point in the life (directional bias).

    Anchors at the active vintage's current point on the seasonal ``axis``. For
    each *other* vintage, finds the nearest observation (within the buffer),
    looks ``horizon`` trading rows forward and records the sign of the move.
    Reported as directional bias + consistency — context, not a trade trigger.
    """
    stack = cap_to_year(stack, max_dte, axis, min_dte)
    active = stack[stack["vintage"] == current_vintage].sort_values("date")
    if active.empty:
        return SeasonalBias(_NEUTRAL, 0, 0, 0, horizon)
    anchor = float(active[axis].iloc[-1])

    up = down = 0
    for vintage, grp in stack.groupby("vintage"):
        if vintage == current_vintage:
            continue
        grp = grp.sort_values("date").reset_index(drop=True)
        dist = (grp[axis] - anchor).abs()
        if dist.min() > buffer_days:
            continue
        idx = int(dist.idxmin())
        if idx + horizon >= len(grp):
            continue
        move = grp["close"].iloc[idx + horizon] - grp["close"].iloc[idx]
        if move > 0:
            up += 1
        elif move < 0:
            down += 1

    total = up + down
    direction = _BULL if up > down else _BEAR if down > up else _NEUTRAL
    return SeasonalBias(direction, up, down, total, horizon)


def forward_tendency(stack: pd.DataFrame, current_vintage: int, buffer_days: int = BUFFER_DAYS,
                     horizons: tuple[int, ...] = (10, 20, 30), axis: str = AXIS,
                     max_dte: int = MAX_DAYS_TO_EXPIRY, min_dte: int = 0) -> dict[int, SeasonalBias]:
    """Seasonal directional bias at several forward horizons (10 / 20 / 30d)."""
    return {h: seasonal_bias(stack, current_vintage, buffer_days, h, axis, max_dte, min_dte) for h in horizons}


def alignment(seasonal_direction: str, trend_direction: str) -> str:
    """Tailwind / Headwind / Mixed from the seasonal bias vs the current trend.

    Same direction -> Tailwind; opposite -> Headwind; either side neutral ->
    Mixed (no clear seasonal edge). A headwind is a *caution*, not a veto.
    """
    s = _TREND_NORM.get(seasonal_direction, seasonal_direction)
    t = _TREND_NORM.get(trend_direction, trend_direction)
    if s == _NEUTRAL or t == _NEUTRAL:
        return _MIXED
    return _TAILWIND if s == t else _HEADWIND


def lifecycle_phase(stack: pd.DataFrame, current_vintage: int, expiry_days: int = 15,
                    axis: str = AXIS, max_dte: int = MAX_DAYS_TO_EXPIRY) -> Lifecycle:
    """Classify the active vintage's life stage from its position on the axis.

    Uses how far the vintage has travelled through the 1-year window (a fraction
    of the ``max_dte`` span), with a dedicated Expiry-risk band near convergence.
    This qualifies the trend score rather than filtering it out.
    """
    stack = cap_to_year(stack, max_dte, axis)
    active = stack[stack["vintage"] == current_vintage].sort_values("date")
    if active.empty or stack.empty:
        return Lifecycle("Quiet", None, 0.0)

    dte = int(active[axis].iloc[-1])
    dmax = int(stack[axis].max())
    dmin = int(stack[axis].min())
    span = max(dmax - dmin, 1)
    elapsed = float(np.clip((dmax - dte) / span, 0.0, 1.0))

    if dte <= expiry_days:
        phase = "Expiry-risk"
    elif elapsed < 0.25:
        phase = "Quiet"
    elif elapsed < 0.50:
        phase = "Build-up"
    elif elapsed < 0.78:
        phase = "Acceleration"
    else:
        phase = "Late"
    return Lifecycle(phase, dte, elapsed)


# -- validation backtest (§8: does tailwind actually help?) -------------------

def _trend_signs(closes: np.ndarray, lookback: int) -> np.ndarray:
    """Sign of the trailing ``lookback``-row change at each index (0 before warmup)."""
    signs = np.zeros(closes.shape[0], dtype=int)
    if closes.shape[0] > lookback:
        diff = closes[lookback:] - closes[:-lookback]
        signs[lookback:] = np.sign(diff).astype(int)
    return signs


def alignment_backtest(stack: pd.DataFrame, horizon: int = 20, buffer_days: int = BUFFER_DAYS,
                       trend_lookback: int = 20, axis: str = AXIS,
                       max_dte: int = MAX_DAYS_TO_EXPIRY, min_dte: int = 0) -> pd.DataFrame:
    """Does seasonal alignment improve trend continuation? (§8 minimum test.)

    For every historical (vintage, date), classify the trailing trend, derive the
    seasonal bias from *other* vintages at that point in the life, label the
    alignment (Tailwind / Headwind / Mixed) and measure forward continuation *in
    the trend's direction* over ``horizon`` rows. Returns one row per alignment
    bucket with ``n``, ``avg_continuation``, ``hit_rate`` and ``median_return``.
    """
    cols = ["alignment", "n", "avg_continuation", "hit_rate", "median_return"]
    stack = cap_to_year(stack, max_dte, axis, min_dte)
    if stack.empty or stack["vintage"].nunique() < 2:
        return pd.DataFrame(columns=cols)

    vintages = {
        v: g.sort_values("date").reset_index(drop=True)
        for v, g in stack.groupby("vintage")
    }
    arrays = {v: (g["close"].to_numpy(), g[axis].to_numpy()) for v, g in vintages.items()}

    align_col: list[str] = []
    cont_col: list[float] = []
    for v, (closes, dtes) in arrays.items():
        n = closes.shape[0]
        signs = _trend_signs(closes, trend_lookback)
        for i in range(trend_lookback, n - horizon):
            trend_sign = signs[i]
            if trend_sign == 0:
                continue
            anchor = dtes[i]
            up = down = 0
            for v2, (c2, d2) in arrays.items():
                if v2 == v:
                    continue
                dist = np.abs(d2 - anchor)
                j = int(dist.argmin())
                if dist[j] > buffer_days or j + horizon >= c2.shape[0]:
                    continue
                mv = c2[j + horizon] - c2[j]
                if mv > 0:
                    up += 1
                elif mv < 0:
                    down += 1
            s_dir = _BULL if up > down else _BEAR if down > up else _NEUTRAL
            t_dir = _BULL if trend_sign > 0 else _BEAR
            align = alignment(s_dir, t_dir)
            # Continuation measured along the trend direction (positive = trend extended).
            cont = (closes[i + horizon] - closes[i]) * trend_sign
            align_col.append(align)
            cont_col.append(float(cont))

    if not align_col:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame({"alignment": align_col, "cont": cont_col})
    out = []
    for bucket in (_TAILWIND, _HEADWIND, _MIXED):
        sub = df.loc[df["alignment"] == bucket, "cont"]
        out.append({
            "alignment": bucket,
            "n": int(sub.size),
            "avg_continuation": float(sub.mean()) if sub.size else np.nan,
            "hit_rate": float((sub > 0).mean()) if sub.size else np.nan,
            "median_return": float(sub.median()) if sub.size else np.nan,
        })
    return pd.DataFrame(out, columns=cols)


__all__ = [
    "SEASONAL_DRIVERS", "LIFECYCLE_NOTES", "AXIS", "MAX_DAYS_TO_EXPIRY", "BUFFER_DAYS",
    "MATURITY_BUCKETS", "bucket_bounds",
    "SeasonalBias", "Lifecycle",
    "cap_to_year", "buffer_cone", "current_percentile", "monthly_heatmap",
    "seasonal_bias", "forward_tendency", "alignment", "lifecycle_phase",
    "alignment_backtest",
]
