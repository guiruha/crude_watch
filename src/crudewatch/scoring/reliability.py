"""Measured, out-of-sample reliability of the Opportunity Score, per family.

Single source of truth for "how much should anyone believe this number?", so the
app, the offline reports and any future analysis quote the same figures instead
of each carrying its own optimistic copy.

Provenance
----------
Walk-forward evaluation run by ``scripts/optimize_weights.py`` on 2026-08-01
(vintage-split, out-of-sample Sharpe of the score's own long/flat rule, net of
the ``COST_STUB_POINTS`` stub). ``sharpe`` is the figure for the weighting the
family **actually runs today**, which is what a user sees.

What the numbers said, and why three families lost their fitted weights: every
family's optimised weights beat equal weighting in-sample by 2-3x, and three of
four were then *beaten by equal weighting* out-of-sample. ``flies`` was the
starkest — in-sample 0.458, out-of-sample -0.174, against 0.348 for plain equal
weights. The weight search's walk-forward guard refused to adopt them, so those
families fall back to equal weights.

Re-run ``scripts/optimize_weights.py`` after any change to the term set and
update this table; the test suite checks it stays consistent with
``family_weights.json``.
"""
from __future__ import annotations

from dataclasses import dataclass

MEASURED_ON: str = "2026-08-01"


@dataclass(frozen=True)
class FamilyReliability:
    """Out-of-sample read on one family's live scoring configuration."""

    sharpe: float | None          # walk-forward OOS Sharpe of the live config
    weighting: str                # "fitted" | "equal"
    equal_sharpe: float | None    # OOS Sharpe of plain equal weighting
    note: str = ""

    @property
    def measured(self) -> bool:
        return self.sharpe is not None

    @property
    def band(self) -> str:
        """Coarse confidence band, used to colour the UI."""
        if self.sharpe is None:
            return "unmeasured"
        if self.sharpe >= 0.30:
            return "modest"
        if self.sharpe >= 0.10:
            return "weak"
        return "negligible"


# Families the weight search covers. The four not listed (outrights, calendars,
# cracks, brent_wti) have never been walk-forward evaluated — absence of a
# number is not a good number, and the UI says so rather than staying silent.
RELIABILITY: dict[str, FamilyReliability] = {
    "flies": FamilyReliability(
        sharpe=0.348, weighting="equal", equal_sharpe=0.348,
        note="Pesos ajustados rechazados: dentro de muestra 0.458, fuera de muestra −0.174.",
    ),
    "quarterly": FamilyReliability(
        sharpe=0.151, weighting="equal", equal_sharpe=0.151,
        note="Pesos ajustados rechazados: fuera de muestra 0.088 frente a 0.151 con pesos iguales.",
    ),
    "semestral": FamilyReliability(
        sharpe=0.114, weighting="fitted", equal_sharpe=0.072,
        note="Única familia cuyos pesos ajustados superaron a los pesos iguales fuera de muestra.",
    ),
    "yearly": FamilyReliability(
        sharpe=0.094, weighting="equal", equal_sharpe=0.094,
        note="Pesos ajustados rechazados: fuera de muestra 0.077 frente a 0.094 con pesos iguales.",
    ),
}

UNMEASURED = FamilyReliability(
    sharpe=None, weighting="equal", equal_sharpe=None,
    note="Sin evaluación walk-forward. No hay medida de fiabilidad para esta familia.",
)


def reliability_for(family: str) -> FamilyReliability:
    """Reliability record for ``family``; an explicit 'unmeasured' if untested."""
    return RELIABILITY.get(family, UNMEASURED)


HEADLINE: str = (
    "El Opportunity Score es una herramienta de **atención**, no una señal de entrada. "
    "Medido walk-forward fuera de muestra, su Sharpe por familia va de **0.07 a 0.35**: "
    "positivo, pero débil. Inclina las probabilidades; no dimensiona una posición."
)
