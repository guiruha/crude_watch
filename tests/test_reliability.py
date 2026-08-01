"""The reliability table must not drift from the weights the app actually runs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from crudewatch.scoring.reliability import (
    HEADLINE,
    RELIABILITY,
    UNMEASURED,
    reliability_for,
)

_WEIGHTS = Path(__file__).resolve().parents[1] / "src" / "crudewatch" / "scoring" / "family_weights.json"


def _fitted_families() -> set[str]:
    if not _WEIGHTS.exists():
        return set()
    return set(json.loads(_WEIGHTS.read_text()))


def test_weighting_matches_the_fitted_weights_file():
    """A family claiming 'fitted' must have weights on disk, and vice versa.

    This is the guard that matters: re-running the weight search can adopt or
    drop a family, and a stale reliability claim in the UI would then be a
    statement about a configuration that is no longer running.
    """
    fitted = _fitted_families()
    for family, rel in RELIABILITY.items():
        if rel.weighting == "fitted":
            assert family in fitted, f"{family} claims fitted weights but has none on disk"
        else:
            assert family not in fitted, (
                f"{family} is recorded as equal-weighted but family_weights.json "
                f"has fitted weights for it — re-run the weight search and update "
                f"crudewatch.scoring.reliability"
            )


def test_every_fitted_family_has_a_reliability_record():
    for family in _fitted_families():
        assert family in RELIABILITY, f"{family} has fitted weights but no reliability record"


def test_bands_follow_the_measured_sharpe():
    assert reliability_for("flies").band == "modest"      # 0.348
    assert reliability_for("quarterly").band == "weak"    # 0.151
    assert reliability_for("semestral").band == "weak"    # 0.114
    assert reliability_for("yearly").band == "negligible" # 0.094


def test_unevaluated_families_are_explicitly_unmeasured():
    """Absence of a number must not read as a good number."""
    for family in ("outrights", "calendars", "cracks", "brent_wti"):
        rel = reliability_for(family)
        assert rel is UNMEASURED
        assert rel.sharpe is None
        assert rel.measured is False
        assert rel.band == "unmeasured"


def test_recorded_sharpes_are_the_live_configuration():
    """`sharpe` is the OOS figure for the weighting the family actually runs."""
    for family, rel in RELIABILITY.items():
        assert rel.sharpe is not None
        if rel.weighting == "equal":
            # equal-weighted families are, by definition, running the equal config
            assert rel.sharpe == pytest.approx(rel.equal_sharpe)
        else:
            # fitted weights were only adopted because they beat equal OOS
            assert rel.sharpe > rel.equal_sharpe


def test_headline_states_the_limitation():
    assert "atención" in HEADLINE
    assert "0.07" in HEADLINE and "0.35" in HEADLINE
