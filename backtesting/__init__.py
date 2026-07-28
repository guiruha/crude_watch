"""Offline backtesting & research for CrudeWatch.

This package is deliberately kept OUT of the installable ``crudewatch`` source
tree (``src/``): ``src`` holds only the code the Streamlit app runs, while this
sibling package holds the offline machinery — the legacy long/flat indicator
backtests (:mod:`backtesting.backtest`) and the walk-forward research /
strategy simulation and their HTML reports (:mod:`backtesting.research`).

It depends one-way on the shared, app-facing pipeline in ``crudewatch`` (data
preparation, the feature/dataset enrichment, indicator math and constants) and
is exercised by the ``scripts/run_*.py`` entry points and ``backtesting/tests``.
"""
