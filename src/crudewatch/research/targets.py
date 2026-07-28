"""Forward outcomes per contract: the *labels* the scoring model is trained on.

For every date of every contract we look **forward** and record what actually
happened next. These are targets, never features: they use information strictly
after ``t`` (via ``shift(-k)``), so they must never enter the feature matrix and
the last rows of each contract (no full future window) are left as ``NaN``.

Everything is in **price points** (price differences), not percent returns,
because a spread / crack / fly can be negative and cross zero — points are the
only measure that behaves uniformly across every family (same convention as the
long/flat engine's P&L).

**Executable basis.** The signal is decided at the close of ``t`` (features use
information up to and including ``close[t]``), but no one can trade that print.
The tradable fill is the **next bar's open**, ``open[t+1]``. Every forward
outcome is therefore anchored at that entry price rather than at ``close[t]`` —
this removes the same-bar look-ahead that inflates close-to-close backtests. If
a frame has no ``open`` column we fall back to the legacy ``close[t]`` anchor.

Recorded per horizon ``h`` (trading bars, not calendar days):

- ``fwd_h``  — point change from the executable entry to ``t + h``
  (``close[t+h] - open[t+1]``).
- ``fwd_vol_h`` / ``mfe_vol_h`` / ``mae_vol_h`` — the same, normalised by a
  point-in-time close-to-close ATR (``fwd`` also by ``√h``) for cross-contract
  comparability.
- ``mfe_h``  — maximum favourable excursion for a *long*: best move over
  ``[t+1, t+h]`` measured from the entry. For a short, the adverse excursion is ``-mfe_h``.
- ``mae_h``  — maximum adverse excursion for a *long* (``<= 0`` unless price only
  rose): worst move over ``[t+1, t+h]`` from the entry. For a short, favourable is ``-mae_h``.

Plus, over the longest horizon, ``bars_to_mfe`` / ``bars_to_mae`` — how many bars
it took to reach that best / worst point (a "time-to-target" primitive).

Direction-agnostic on purpose: whether a given move is "good" depends on the
signal's side, which lives in the evaluation layer. Normalising these point
outcomes (e.g. by the analogous-panel dispersion at the same slot/dte) is also
left to the evaluation layer, so nothing here needs a rolling volatility window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20, 25, 30)


def _atr_cc(close: pd.Series, window: int) -> pd.Series:
    """Close-to-close 'ATR' as-of ``t``: Wilder-smoothed absolute daily change.

    Uses only information up to and including ``t`` (an EWM of ``|Δclose|``), so it
    is safe to normalise forward outcomes by ``sigma_t`` without leaking the future.
    Kept local (not imported from ``features``) to keep ``targets`` dependency-free.
    """
    return close.diff().abs().ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def _forward_shift_frame(close: pd.Series, max_h: int) -> pd.DataFrame:
    """Columns ``1..max_h`` where column ``k`` is ``close`` shifted so row ``t``
    holds ``close[t + k]`` (i.e. the value ``k`` bars into the future)."""
    return pd.concat(
        {k: close.shift(-k) for k in range(1, max_h + 1)}, axis=1
    )


def _contract_targets(
    close: pd.Series,
    entry: pd.Series,
    horizons: tuple[int, ...],
    atr_window: int = 10,
) -> pd.DataFrame:
    """Forward-return, MFE/MAE and time-to-extreme columns for one contract.

    ``close`` must already be sorted by date. ``entry`` is the **executable entry
    price** aligned to the same index — normally ``open.shift(-1)`` (next-bar
    open), or ``close`` itself for the legacy same-bar anchor. All forward
    outcomes are measured relative to ``entry``. The returned frame shares
    ``close``'s index so it can be joined straight back onto the source rows.

    Vol-normalised variants (``fwd_vol_h``, ``mfe_vol_h``, ``mae_vol_h``) divide by
    a point-in-time close-to-close ATR known at ``t`` (``fwd`` also by ``√h``), so
    outcomes are comparable across contracts and volatility regimes.
    """
    max_h = max(horizons)
    fwd = _forward_shift_frame(close, max_h)  # row t -> [close[t+1] .. close[t+max_h]]
    entry_np = entry.to_numpy(dtype=float)
    sigma = _atr_cc(close, atr_window).replace(0.0, np.nan)  # as-of t; avoid /0

    cols: dict[str, np.ndarray | pd.Series] = {}
    for h in horizons:
        window = fwd.loc[:, 1:h]                    # the next h future closes
        fwd_h = fwd[h] - entry                      # close[t+h] - entry (open[t+1])
        mfe_h = window.max(axis=1) - entry
        mae_h = window.min(axis=1) - entry
        cols[f"fwd_{h}"] = fwd_h
        cols[f"mfe_{h}"] = mfe_h
        cols[f"mae_{h}"] = mae_h
        cols[f"fwd_vol_{h}"] = fwd_h / (sigma * np.sqrt(h))
        cols[f"mfe_vol_{h}"] = mfe_h / sigma
        cols[f"mae_vol_{h}"] = mae_h / sigma

    # Bars-to-extreme over the longest horizon (a time-to-target primitive).
    # NaN where the full window (or the entry) is unavailable (contract tail).
    long_window = fwd.loc[:, 1:max_h].to_numpy(dtype=float)  # shape (n, max_h)
    full = ~np.isnan(long_window).any(axis=1) & ~np.isnan(entry_np)
    bars_to_mfe = np.full(len(close), np.nan)
    bars_to_mae = np.full(len(close), np.nan)
    if full.any():
        rel = long_window[full] - entry_np[full, None]
        bars_to_mfe[full] = rel.argmax(axis=1) + 1
        bars_to_mae[full] = rel.argmin(axis=1) + 1
    cols["bars_to_mfe"] = bars_to_mfe
    cols["bars_to_mae"] = bars_to_mae

    return pd.DataFrame(cols, index=close.index)


def add_forward_returns(
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
    price_col: str = "close",
    contract_col: str = "contract",
    date_col: str = "date",
    open_col: str = "open",
) -> pd.DataFrame:
    """Return ``frame`` (sorted by contract, date) with forward-outcome columns.

    Outcomes use an **executable basis**: the entry price is ``open[t+1]`` (the
    next bar's open), so a signal decided at ``close[t]`` is filled at a price
    nobody has yet observed. If ``open_col`` is absent the entry falls back to
    ``close[t]`` (legacy same-bar anchor).

    Outcomes are computed independently per contract so no future leaks across a
    contract boundary. The result is ordered by ``[contract_col, date_col]``;
    forward columns are ``NaN`` on the last ``h`` rows of each contract (and the
    very last row, which has no next open).
    """
    for col in (price_col, contract_col, date_col):
        if col not in frame.columns:
            raise KeyError(f"frame is missing required column {col!r}")

    ordered = frame.sort_values([contract_col, date_col])
    has_open = open_col in ordered.columns
    parts = []
    for _, group in ordered.groupby(contract_col, sort=False):
        close = group[price_col].astype(float)
        # Executable entry = next bar's open; legacy fallback = same-bar close.
        entry = group[open_col].astype(float).shift(-1) if has_open else close
        parts.append(_contract_targets(close, entry, horizons))
    targets = pd.concat(parts).reindex(ordered.index)
    return pd.concat([ordered, targets], axis=1)
