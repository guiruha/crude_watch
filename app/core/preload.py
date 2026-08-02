"""Background cache warm-up for the Streamlit dashboard."""
from __future__ import annotations

import threading

from core.scoring import (
    active_contract_scores_cached,
    analogous_outcomes_cached,
    contracts_on_date,
    enriched_frame,
    family_date_bounds,
    horizon_outcomes_cached,
    score_instrument_dict,
)
from core.selection import Selection
from core.validation import validation_table

_LOCK = threading.Lock()
_RUNNING: set[str] = set()
_DONE: set[str] = set()


def _warm_selected(selection: Selection) -> None:
    if selection.contract is None:
        return
    score_instrument_dict(
        selection.family,
        selection.contract,
        selection.horizon,
        selection.as_of_iso,
    )
    analogous_outcomes_cached(
        selection.family,
        selection.contract,
        selection.horizon,
        selection.as_of_iso,
    )
    horizon_outcomes_cached(selection.family, selection.contract, selection.as_of_iso)


def _warm_family(family: str, as_of_iso: str, horizon: int, top_n: int) -> None:
    enriched_frame(family)
    family_date_bounds(family)
    contracts_on_date(family, as_of_iso)

    panel = active_contract_scores_cached(family, horizon, as_of_iso)

    if panel.empty:
        return
    contracts = panel["contract"].astype(str).tolist()
    for contract in contracts:
        analogous_outcomes_cached(family, contract, horizon, as_of_iso)
    if top_n <= 0:
        return
    for contract in contracts[:top_n]:
        horizon_outcomes_cached(family, contract, as_of_iso)


def _worker(key: str, families: tuple[str, ...], selection: Selection, top_n: int) -> None:
    try:
        validation_table()
        _warm_selected(selection)
        for family in families:
            _warm_family(family, selection.as_of_iso, selection.horizon, top_n)
    finally:
        with _LOCK:
            _RUNNING.discard(key)
            _DONE.add(key)


def start_preload(selection: Selection, families: list[str], top_n: int = 3) -> None:
    """Start one background warm-up for the current date/horizon selection."""
    key = "|".join(
        [
            selection.as_of_iso,
            str(selection.horizon),
            selection.family,
        ]
    )
    ordered = [selection.family] + [family for family in families if family != selection.family]
    family_tuple = tuple(ordered)
    with _LOCK:
        if _RUNNING or key in _DONE:
            return
        _RUNNING.add(key)
    thread = threading.Thread(
        target=_worker,
        args=(key, family_tuple, selection, int(top_n)),
        name=f"cw-preload-{selection.as_of_iso}-{selection.horizon}",
        daemon=True,
    )
    thread.start()
