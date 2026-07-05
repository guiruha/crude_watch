"""Chapter contract shared by the three analytical chapters.

A chapter is a self-contained view over the *active* structure/vintage. The
screen builds a :class:`ChapterContext` once (series + seasonal stack, all
cached) and hands it to whichever chapter is selected. In Phase 1 the chapters
render their shell — header, roadmap note and the active price series — so the
whole app is navigable before the indicator work begins.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from crudewatch.analytics import Structure


@dataclass(frozen=True)
class ChapterContext:
    """Everything a chapter needs about the currently selected spread."""
    structure: Structure
    base_year: int
    series: pd.DataFrame       # date, close, volume for the active vintage
    seasonal: pd.DataFrame     # vintage, date, day_of_year, close across vintages

    @property
    def title(self) -> str:
        return f"{self.structure.label} · {self.base_year}"


class Chapter(ABC):
    """Base class for a navigable analytical chapter."""

    #: Short label shown in the chapter navigation.
    name: str = "Chapter"
    #: One-line description under the chapter header.
    subtitle: str = ""
    #: Roadmap phase that fleshes this chapter out (shown while it is a stub).
    phase: str = ""

    @abstractmethod
    def render(self, ctx: ChapterContext) -> None:
        """Draw the chapter for the active structure/vintage."""
        raise NotImplementedError
