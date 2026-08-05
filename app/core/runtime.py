"""Runtime switches for constrained deployments."""
from __future__ import annotations

import os


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def low_memory_mode() -> bool:
    """True on Render/free-style deployments unless explicitly disabled."""
    if "CRUDEWATCH_LOW_MEMORY" in os.environ:
        return _truthy(os.environ.get("CRUDEWATCH_LOW_MEMORY"))
    return _truthy(os.environ.get("RENDER"))
