"""The fixed-date spread universe: 19 structures across 3 levels.

Every structure is defined once, declaratively, as a linear combination of
*outright* legs. A leg is ``(month_code, year_offset, coefficient)`` where
``year_offset`` is added to the vintage's base year (so the December-March
quarterly and the two year-crossing butterflies are expressed cleanly). Building
the price series is then a single dot product against the outright close matrix
(see :mod:`crudewatch.analytics.data_layer`), which handles monthly spreads,
quarterly spreads and butterflies (``A - 2B + C``) uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Calendar-ordered month codes (Jan -> Dec) used to enumerate the monthly ladder.
_ORDER = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]
_NAME = {
    "F": "Jan", "G": "Feb", "H": "Mar", "J": "Apr", "K": "May", "M": "Jun",
    "N": "Jul", "Q": "Aug", "U": "Sep", "V": "Oct", "X": "Nov", "Z": "Dec",
}


class Level(str, Enum):
    """The structure tiers, ordered by decreasing volatility."""
    MONTHLY = "Monthly"
    QUARTERLY = "Quarterly"
    SEMESTRAL = "Semestral"
    BUTTERFLY = "Butterfly"


# A single outright leg of a structure.
@dataclass(frozen=True)
class Leg:
    month_code: str      # CME month letter, e.g. "Z"
    year_offset: int     # years added to the vintage base year
    coefficient: float   # weight in the linear combination


@dataclass(frozen=True)
class Structure:
    """One tradable fixed-date structure (independent of vintage year)."""
    key: str             # stable id, e.g. "fly_UZH"
    label: str           # UI label, e.g. "(Sep-Dec)-(Dec-Mar)"
    level: Level
    legs: tuple[Leg, ...]
    fill: bool = False   # area-fill charts to zero (never, for spreads)

    @property
    def span_months(self) -> int:
        """Calendar months from the earliest to the latest leg (for expiry logic)."""
        months = [
            leg.year_offset * 12 + _ORDER.index(leg.month_code) for leg in self.legs
        ]
        return max(months) - min(months)


def _cal(near: str, far: str, near_off: int = 0, far_off: int = 0) -> tuple[Leg, ...]:
    """A calendar spread: long near, short far."""
    return (Leg(near, near_off, 1.0), Leg(far, far_off, -1.0))


def _fly(a: str, b: str, c: str, offs: tuple[int, int, int]) -> tuple[Leg, ...]:
    """A butterfly A - 2B + C, centred on the shared middle leg B."""
    return (Leg(a, offs[0], 1.0), Leg(b, offs[1], -2.0), Leg(c, offs[2], 1.0))


def _monthly_ladder() -> list[Structure]:
    """The 11 consecutive-month spreads Jan-Feb ... Nov-Dec (within one year)."""
    out: list[Structure] = []
    for near, far in zip(_ORDER, _ORDER[1:]):
        out.append(Structure(
            key=f"m_{near}{far}",
            label=f"{_NAME[near]}-{_NAME[far]}",
            level=Level.MONTHLY,
            legs=_cal(near, far),
        ))
    return out


# Level 2 — quarterly spreads (3-month gap). Dec-Mar rolls into the next year.
_QUARTERLY = [
    Structure("q_HM", "Mar-Jun", Level.QUARTERLY, _cal("H", "M")),
    Structure("q_MU", "Jun-Sep", Level.QUARTERLY, _cal("M", "U")),
    Structure("q_UZ", "Sep-Dec", Level.QUARTERLY, _cal("U", "Z")),
    Structure("q_ZH", "Dec-Mar", Level.QUARTERLY, _cal("Z", "H", far_off=1)),
]

# Level 3 — semestral spreads (6-month gap) on the quarterly grid. Sep-Mar and
# Dec-Jun roll into the next year.
_SEMESTRAL = [
    Structure("s_HU", "Mar-Sep", Level.SEMESTRAL, _cal("H", "U")),
    Structure("s_MZ", "Jun-Dec", Level.SEMESTRAL, _cal("M", "Z")),
    Structure("s_UH", "Sep-Mar", Level.SEMESTRAL, _cal("U", "H", far_off=1)),
    Structure("s_ZM", "Dec-Jun", Level.SEMESTRAL, _cal("Z", "M", far_off=1)),
]

# Level 4 — quarterly butterflies, each the difference of two adjacent quarterly
# spreads sharing their central leg: (A-B)-(B-C) = A - 2B + C.
_BUTTERFLY = [
    Structure("f_HMU", "(Mar-Jun)-(Jun-Sep)", Level.BUTTERFLY, _fly("H", "M", "U", (0, 0, 0))),
    Structure("f_MUZ", "(Jun-Sep)-(Sep-Dec)", Level.BUTTERFLY, _fly("M", "U", "Z", (0, 0, 0))),
    Structure("f_UZH", "(Sep-Dec)-(Dec-Mar)", Level.BUTTERFLY, _fly("U", "Z", "H", (0, 0, 1))),
    Structure("f_ZHM", "(Dec-Mar)-(Mar-Jun)", Level.BUTTERFLY, _fly("Z", "H", "M", (0, 1, 1))),
]

# The full universe: 11 monthly + 4 quarterly + 4 semestral + 4 butterfly = 23 structures.
STRUCTURES: tuple[Structure, ...] = (
    tuple(_monthly_ladder()) + tuple(_QUARTERLY) + tuple(_SEMESTRAL) + tuple(_BUTTERFLY)
)

# Lookups.
BY_KEY: dict[str, Structure] = {s.key: s for s in STRUCTURES}
BY_LEVEL: dict[Level, list[Structure]] = {level: [] for level in Level}
for _s in STRUCTURES:
    BY_LEVEL[_s.level].append(_s)


def structures_for(level: Level) -> list[Structure]:
    """All structures belonging to a level, in canonical order."""
    return BY_LEVEL[level]
