"""Descriptive audit helpers for one market read."""
from __future__ import annotations

from core.evidence import MIN_EFFECTIVE_N, evidence_read


def _fmt(value, spec: str = ".2f", dash: str = "-") -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return dash
    if value != value:
        return dash
    return format(value, spec)


def descriptive_bias(score: dict) -> str:
    """Compact market-state bias; descriptive, not executable."""
    blocks = score.get("blocks", {})
    regime = blocks.get("regime", "")
    level = float(blocks.get("level", 0.0))
    direction = float(blocks.get("direction", 0.0))
    if regime == "range":
        if level >= 60:
            return "caro / reversión potencial"
        if level <= -60:
            return "barato / reversión potencial"
        return "rango sin extremo"
    if regime == "trend":
        if direction > 20:
            return "direccional alcista"
        if direction < -20:
            return "direccional bajista"
        return "direccional sin pendiente clara"
    return "transición"


def pm_description(score: dict, cohort: dict, rank_row: dict | None = None) -> str:
    """Non-operational PM read: prioritises review, never prescribes a trade."""
    read = evidence_read(score, cohort)
    n = int(cohort.get("n", 0))
    side = float(cohort.get("side", 0.0))
    rank_row = rank_row or {}
    if rank_row.get("liquidity") == "baja":
        return "Baja prioridad: liquidez baja."
    if rank_row.get("vol_regime") == "expansión":
        return "Revisar: volatilidad en expansión."
    if n < MIN_EFFECTIVE_N:
        return "Muestra insuficiente; revisar solo muestras raw."
    if side == 0.0:
        return "Sin signo interno activo; lectura informativa."
    if read["state"] == "supports":
        regime = score.get("blocks", {}).get("regime", "")
        return "Vigilar reversión" if regime == "range" else "Vigilar continuación"
    if read["state"] == "opposes":
        return "Evidencia en contra."
    return "Cohorte sin dirección histórica clara."


def audit_rows(family: str, horizon: int, score: dict, cohort: dict, rank_row: dict) -> dict[str, str]:
    """Rows for the dashboard audit panel.

    This intentionally reports coverage and assumptions instead of a synthetic
    robustness score.
    """
    n = int(cohort.get("n", 0))
    n_raw = int(cohort.get("n_raw", n))
    side = float(cohort.get("side", 0.0))
    cost = cohort.get("cost", rank_row.get("cost_points", float("nan")))
    cost_usd = rank_row.get("cost_usd", float("nan"))
    active = n >= MIN_EFFECTIVE_N and side != 0.0
    return {
        "Estado de lectura": "con signo y muestra suficiente" if active else "solo descriptiva",
        "Cobertura estadística": (
            f"n efectivo {n}; raw {n_raw}; "
            f"vintages {int(cohort.get('vintage_count', 0))}; "
            f"años {int(cohort.get('year_count', 0))}"
        ),
        "Cohorte usado": (
            f"régimen {cohort.get('regime', '-')}; nivel {cohort.get('level_bin', '-')}; "
            f"tenor {cohort.get('tenor_bucket', '-')}; mes {cohort.get('month', '-')}; "
            f"far {cohort.get('far_leg', '-')}"
        ),
        "Contexto instrumento": (
            f"slot {rank_row.get('slot', '-')}; vintage {_fmt(rank_row.get('vintage'), '.0f')}; "
            f"vida {rank_row.get('life_phase', '-')}; vol {rank_row.get('vol_regime', '-')}"
        ),
        "Coste aplicado": f"{_fmt(cost, '.3f')} pts; ${_fmt(cost_usd, '.0f')}",
        "Sharpe cohorte anual.": _fmt(cohort.get("sharpe_aligned"), "+.2f") if active else "oculto",
        "Lectura PM descriptiva": pm_description(score, cohort, rank_row),
        "Sesgo descriptivo": descriptive_bias(score),
        "Límite pendiente": "coste asumido; bid/offer real no disponible en los datos actuales",
        "Horizonte": f"{family} D+{int(horizon)}; cálculo point-in-time con forwards resueltos",
    }
