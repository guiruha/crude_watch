"""Small view helpers for empirical cohort evidence."""
from __future__ import annotations

MIN_EFFECTIVE_N = 15


def _finite(value: float) -> bool:
    return value is not None and value == value


def evidence_read(score: dict, cohort: dict) -> dict:
    """Conclusion-level read of the cohort against the composite sign."""
    n = int(cohort.get("n", 0))
    n_raw = int(cohort.get("n_raw", n))
    median = float(cohort.get("median_aligned", float("nan")))
    signed = float(score.get("opportunity", cohort.get("opportunity", float("nan"))))
    if n < MIN_EFFECTIVE_N:
        return {
            "state": "insufficient",
            "label": "Evidencia insuficiente",
            "detail": f"n efectivo {n}; raw {n_raw}. Se muestran muestras, no medianas.",
            "n": n,
            "n_raw": n_raw,
            "median": median,
            "tone": "neutral",
        }
    if not _finite(median) or not _finite(signed) or signed == 0:
        return {
            "state": "neutral",
            "label": "Evidencia sin signo claro",
            "detail": f"n={n}, mediana no concluyente.",
            "n": n,
            "n_raw": n_raw,
            "median": median,
            "tone": "neutral",
        }
    if median > 0:
        return {
            "state": "supports",
            "label": "Evidencia a favor",
            "detail": f"n={n}, mediana {median:+.2f}.",
            "n": n,
            "n_raw": n_raw,
            "median": median,
            "tone": "positive",
        }
    if median < 0:
        return {
            "state": "opposes",
            "label": "Evidencia en contra",
            "detail": f"n={n}, mediana {median:+.2f}.",
            "n": n,
            "n_raw": n_raw,
            "median": median,
            "tone": "negative",
        }
    return {
        "state": "flat",
        "label": "Evidencia plana",
        "detail": f"n={n}, mediana {median:+.2f}.",
        "n": n,
        "n_raw": n_raw,
        "median": median,
        "tone": "neutral",
    }
