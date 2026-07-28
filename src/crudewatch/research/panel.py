"""Bloque D — level (cheap / expensive) from a panel of ANALOGOUS contracts.

The level of a contract is not judged against its own thin history nor against
any stitched / rolling continuous series (we don't build those). It is judged
against **other individual contracts of the same seasonal slot, at the same
point of their life**, using only vintages that already existed.

For a target row (slot ``s``, vintage ``v``, days-to-expiry ``dte``, close ``c``):

* reference = every contract with the same ``slot`` and a **strictly earlier
  vintage** (``vintage < v``), sampled in the same ``dte`` bin (life phase);
* ``level_pct`` = fraction of those reference closes at or below ``c`` (0 = the
  cheapest this slot has ever been at this life phase, 1 = the most expensive);
* ``level_z`` = ``(c - mean_reference) / std_reference``.

Point-in-time is automatic: an earlier vintage expires earlier, so its same-life
observations occurred on an earlier calendar date than the target's — no future
information leaks in. Each vintage contributes one summary point per life-phase
bin, so the reference is a clean cross-section of "where analogous contracts sat
at this stage", exactly what the proposal's level block asks for.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("close", "slot", "vintage", "dte")


def add_level_panel(
    frame: pd.DataFrame,
    dte_bin_days: int = 21,
    min_prior: int = 3,
    price_col: str = "close",
) -> pd.DataFrame:
    """Return ``frame`` with ``level_pct`` and ``level_z`` from the analogous panel.

    ``dte_bin_days`` sets the life-phase bin width (default ~one trading month).
    Rows with fewer than ``min_prior`` prior-vintage anchors in their
    (slot, life-phase) cell are left ``NaN`` — no reference, no level.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"frame is missing panel columns {missing}; run add_lifecycle first")

    out = frame.reset_index(drop=True).copy()
    n = len(out)
    level_pct = np.full(n, np.nan)
    level_z = np.full(n, np.nan)

    valid_dte = out["dte"].notna()
    out["_dte_bin"] = np.where(valid_dte, np.floor(out["dte"] / dte_bin_days), np.nan)

    # One anchor point per (slot, life-phase bin, vintage): the mean close there.
    anchors = (
        out[valid_dte]
        .groupby(["slot", "_dte_bin", "vintage"], as_index=False)[price_col]
        .mean()
        .rename(columns={price_col: "anchor"})
    )
    anchor_map: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    for (slot, dte_bin), sub in anchors.groupby(["slot", "_dte_bin"]):
        sub = sub.sort_values("vintage")
        anchor_map[(slot, dte_bin)] = (sub["vintage"].to_numpy(), sub["anchor"].to_numpy())

    # For each (slot, bin, vintage) block of rows, rank against strictly-earlier vintages.
    for (slot, dte_bin, vintage), idx in out[valid_dte].groupby(["slot", "_dte_bin", "vintage"]).groups.items():
        vints, closes = anchor_map[(slot, dte_bin)]
        k = int(np.searchsorted(vints, vintage, side="left"))  # anchors with vintage < target
        if k < min_prior:
            continue
        prior = closes[:k]
        mean = float(prior.mean())
        std = float(prior.std(ddof=1))
        sorted_prior = np.sort(prior)

        pos = np.asarray(idx, dtype=int)
        row_closes = out.loc[pos, price_col].to_numpy(dtype=float)
        level_pct[pos] = np.searchsorted(sorted_prior, row_closes, side="right") / k
        if std > 0:
            level_z[pos] = (row_closes - mean) / std

    out = out.drop(columns="_dte_bin")
    out["level_pct"] = level_pct
    out["level_z"] = level_z
    return out


PANEL_FEATURES: list[str] = ["level_z", "level_pct"]
