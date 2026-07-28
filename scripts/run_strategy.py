"""Generate the single-page demonstration report for the selected strategy.

    python scripts/run_strategy.py

Replays the range-regime, level-confirmed reversion strategy out-of-sample on
the tradeable calendar-structure families, writes the trade ledgers and a
self-contained HTML tearsheet with the cumulative-PnL curve.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # the offline `backtesting` package lives at repo root

from crudewatch.data_preparation import build_all  # noqa: E402
from crudewatch.infra import load_raw  # noqa: E402
from backtesting.research.evaluate import COST_STUB_POINTS, build_dataset  # noqa: E402
from backtesting.research.strategy import (  # noqa: E402
    TRADEABLE_FAMILIES,
    simulate_strategy,
    strategy_stats,
)
from backtesting.research.strategy_report import build_strategy_report  # noqa: E402

FRAME_NAMES = [
    "outrights", "calendars", "cracks", "brent_wti",
    "quarterly", "semestral", "yearly", "flies",
]
HORIZONS = (20, 25, 30)
HEADLINE_HORIZON = 25
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw_files.xlsx"
OUT_DIR = ROOT / "docs" / "reports" / "backtest"

EXCLUDED_REASONS = {
    "outrights": "Sharpe negativo en todos los horizontes: el plano direccional no tiene ventaja ejecutable.",
    "cracks": "El nivel es significativo (FDR) pero el trading da Sharpe negativo tras coste: no econ\u00f3mica.",
    "brent_wti": "Ventaja solo a horizonte largo (negativa en D+20); poco consistente entre horizontes.",
    "yearly": "Positiva solo en D+20; se vuelve negativa en D+25/D+30 y la muestra es fina (~120 trades).",
    "flies": "Positiva solo en D+20; negativa en D+25/D+30 y muy pocos trades (~85).",
}


def load_frames() -> dict[str, pd.DataFrame]:
    parquet = {name: PROCESSED / f"{name}.parquet" for name in FRAME_NAMES}
    if all(p.exists() for p in parquet.values()):
        print(f"Reading parquet cache from {PROCESSED}")
        return {name: pd.read_parquet(p) for name, p in parquet.items()}
    print(f"No parquet cache; building frames from {RAW}")
    return build_all(load_raw(RAW))


def main() -> None:
    frames = load_frames()
    stats_rows: list[dict] = []
    ledgers: dict[str, pd.DataFrame] = {}
    excluded_rows: list[dict] = []

    for name in FRAME_NAMES:
        frame = frames.get(name)
        if frame is None or frame.empty:
            continue
        data = build_dataset(frame, name)
        cost = COST_STUB_POINTS.get(name, 0.0)
        recommended = name in TRADEABLE_FAMILIES

        # Apply the SAME strategy to every family (even the weak ones) so the
        # report can show the full picture, not just the recommended universe.
        print(f"Simulating strategy on {name} ...", flush=True)
        for h in HORIZONS:
            led = simulate_strategy(data, name, h, cost=cost, confirm=True)
            st = strategy_stats(led, h)
            stats_rows.append({
                "family": name, "horizon": h, "recommended": recommended, **st,
            })
            if h == HEADLINE_HORIZON:
                ledgers[name] = led
                if not led.empty:
                    led.to_csv(OUT_DIR / f"strategy_ledger_{name}.csv", index=False)
                if not recommended:
                    excluded_rows.append({
                        "family": name,
                        "sharpe": st["sharpe"],
                        "reason": EXCLUDED_REASONS.get(name, ""),
                    })

    stats = pd.DataFrame(stats_rows)
    excluded = pd.DataFrame(excluded_rows)

    # Attach the headline single-feature IC for the excluded families, if available.
    metrics_path = OUT_DIR / "research_metrics.csv"
    if metrics_path.exists() and not excluded.empty:
        m = pd.read_csv(metrics_path)
        h = m[(m["group"] == "ALL") & (m["horizon"] == HEADLINE_HORIZON)]
        best_ic = {
            fam: g.reindex(g["ic_t"].abs().sort_values(ascending=False).index)["ic_mean"].iloc[0]
            for fam, g in h.groupby("family") if not g.empty
        }
        excluded["ic_mean"] = excluded["family"].map(best_ic)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not stats.empty:
        stats.to_csv(OUT_DIR / "strategy_stats.csv", index=False)
        print(f"Wrote {OUT_DIR / 'strategy_stats.csv'}  ({len(stats)} rows)")

    html_path = OUT_DIR / "strategy.html"
    html_path.write_text(
        build_strategy_report(
            ledgers, stats, horizon=HEADLINE_HORIZON,
            excluded=excluded if not excluded.empty else None,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
