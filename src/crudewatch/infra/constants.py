"""Shared reference data and configuration for the WTI dataset build.

Everything here is constant across the pipeline: symbol vocabulary, the keys
that identify a physical contract, the regexes for each listed symbol family,
the synthetic spread tenors, and default paths.
"""
from __future__ import annotations

from pathlib import Path

# CME single-letter month codes -> calendar month number.
MONTH_CODES: dict[str, int] = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}
# Reverse map, month number -> single-letter code, for labelling synthetic legs.
CODE_BY_MONTH: dict[int, str] = {v: k for k, v in MONTH_CODES.items()}

# A single physical contract is identified by (instrument_id, symbol). We must
# key on BOTH because GLBX recycles instrument_id across unrelated symbols over
# the years (e.g. the same id is "CL:BZ Q9-U9" in 2019 and "CLQ0-BZU0" in 2020),
# while a symbol like "CLZ9" is itself reused every decade for a new contract.
CONTRACT_KEYS: list[str] = ["instrument_id", "symbol"]

# Symbol patterns for each exchange-listed family.
#
# The year token is 1 OR 2 digits: near-dated contracts carry a single,
# decade-ambiguous digit (the "9" in "CLZ9"), while deep-deferred contracts are
# quoted with an explicit two-digit year (e.g. "CLZ29" = Dec 2029). See
# `resolve_expiry_year` for how each width is turned into a full year.
OUTRIGHT_RE = r"^CL(?P<month_code>[FGHJKMNQUVXZ])(?P<year_digit>\d{1,2})$"
CAL_RE = (
    r"^CL(?P<m1>[FGHJKMNQUVXZ])(?P<y1>\d{1,2})-"   # near leg
    r"CL(?P<m2>[FGHJKMNQUVXZ])(?P<y2>\d{1,2})$"    # far leg
)
CRACK_RE = (
    r"^CL:C1 (?P<product>[A-Z]{2})-CL "
    r"(?P<month_code>[FGHJKMNQUVXZ])(?P<year_digit>\d{1,2})$"
)
# Brent (BZ) vs WTI (CL) inter-commodity spread, e.g. "CLZ2-BZZ2" (same-month
# arb) or "CLF3-BZG3" (cross-month). Listed price is WTI - Brent; the builder
# flips it to the conventional Brent - WTI premium.
BRENT_WTI_RE = (
    r"^CL(?P<wti_m>[FGHJKMNQUVXZ])(?P<wti_y>\d{1,2})-"
    r"BZ(?P<bz_m>[FGHJKMNQUVXZ])(?P<bz_y>\d{1,2})$"
)

# Synthetic two-leg calendar spreads: name -> gap in months between the legs.
SPREAD_STRUCTURES: dict[str, int] = {
    "quarterly": 3,
    "semestral": 6,
    "yearly": 12,
}

# Front-leg months for which same-month butterflies are built. A fly needs its
# 2-year-deferred back leg to trade, and WTI liquidity that far out collapses to
# essentially December (Z) and, secondarily, June (M); every other month yields
# only a handful of overlapping days. Narrow to ("Z",) for December-only flies.
FLY_MONTHS: tuple[str, ...] = ("M", "Z")

# Default filesystem locations (relative to the repo root).
RAW_DEFAULT = Path("data/raw_files.xlsx")
PROCESSED_DEFAULT = Path("data/processed")
