"""The analytical chapters of the Spread Analytics dashboard."""
from chapters.base import Chapter, ChapterContext
from chapters.bollinger import BollingerChapter
from chapters.score import ScoreChapter
from chapters.seasonality import SeasonalityChapter
from chapters.technical import TechnicalChapter

# Analytical chapters, each surfaced as its own left-menu screen. The composite
# ScoreChapter is rendered on the main page rather than as a navigation entry.
CHAPTERS: list[Chapter] = [
    TechnicalChapter(),
    SeasonalityChapter(),
    BollingerChapter(),
]

__all__ = [
    "Chapter",
    "ChapterContext",
    "TechnicalChapter",
    "SeasonalityChapter",
    "BollingerChapter",
    "ScoreChapter",
    "CHAPTERS",
]
