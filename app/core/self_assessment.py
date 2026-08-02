"""Background sign-vs-evidence agreement audit for the dashboard."""
from __future__ import annotations

import math
import threading

import numpy as np
import pandas as pd

from core.evidence import MIN_EFFECTIVE_N
from core.scoring import active_contract_scores_cached, enriched_frame

_LOCK = threading.Lock()
_RUNNING: set[str] = set()
_RESULTS: dict[str, dict] = {}


def _key(family: str, horizon: int) -> str:
    return f"{family}|{int(horizon)}"


def _ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = 1.96
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return centre - half, centre + half


WARMUP_YEARS = 3


def _sample_dates(family: str, max_dates: int, warmup_years: int = WARMUP_YEARS) -> list[str]:
    dates = pd.Series(pd.to_datetime(enriched_frame(family)["date"]).dropna().unique()).sort_values()
    if dates.empty:
        return []
    cutoff = pd.Timestamp(dates.iloc[0]) + pd.DateOffset(years=int(warmup_years))
    dates = dates[dates >= cutoff]
    if dates.empty:
        return []
    if len(dates) <= max_dates:
        sample = dates
    else:
        idx = np.linspace(0, len(dates) - 1, max_dates).round().astype(int)
        sample = dates.iloc[np.unique(idx)]
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in sample]


def _worker(family: str, horizon: int, max_dates: int) -> None:
    key = _key(family, horizon)
    try:
        usable = 0
        agree = 0
        inspected = 0
        for as_of in _sample_dates(family, max_dates):
            board = active_contract_scores_cached(family, horizon, as_of)
            if board.empty:
                continue
            top = board.iloc[0]
            signed = float(top.get("signed_composite", float("nan")))
            if signed == 0 or signed != signed:
                continue
            inspected += 1
            n = int(top.get("cohort_n", 0))
            median = float(top.get("cohort_median", float("nan")))
            if n < MIN_EFFECTIVE_N or median != median:
                continue
            usable += 1
            agree += int(median > 0)
        lo, hi = _ci(agree, usable)
        result = {
            "state": "done",
            "family": family,
            "horizon": int(horizon),
            "claim": "top_1_by_date",
            "warmup_years": WARMUP_YEARS,
            "agreement": agree / usable if usable else float("nan"),
            "ci_low": lo,
            "ci_high": hi,
            "usable": usable,
            "inspected": inspected,
            "dates": max_dates,
        }
    except Exception as exc:  # noqa: BLE001 - background audit should not break the app
        result = {"state": "error", "error": str(exc)}
    with _LOCK:
        _RUNNING.discard(key)
        _RESULTS[key] = result


def start_self_assessment(family: str, horizon: int, max_dates: int = 200) -> None:
    key = _key(family, horizon)
    with _LOCK:
        if _RUNNING or key in _RESULTS:
            return
        _RUNNING.add(key)
    thread = threading.Thread(
        target=_worker,
        args=(family, int(horizon), int(max_dates)),
        name=f"cw-self-assessment-{family}-{horizon}",
        daemon=True,
    )
    thread.start()


def self_assessment_status(family: str, horizon: int) -> dict:
    key = _key(family, horizon)
    with _LOCK:
        if key in _RESULTS:
            return dict(_RESULTS[key])
        if key in _RUNNING:
            return {"state": "running", "claim": "top_1_by_date", "warmup_years": WARMUP_YEARS}
    return {"state": "pending"}
