"""Generate the walk-forward research backtest summary.

    python scripts/run_research.py

Reads the processed parquet cache when present (``data/processed/*.parquet``),
otherwise rebuilds the frames from ``data/raw_files.xlsx``. Evaluates every
feature (including the analogous-panel level) out-of-sample by vintage across
all contract families, then writes:

    docs/reports/backtest/research.html          the themed summary page
    docs/reports/backtest/research_metrics.csv   the tidy walk-forward metrics
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # the offline `backtesting` package lives at repo root

from crudewatch.data_preparation import build_all  # noqa: E402
from crudewatch.infra import load_raw  # noqa: E402
from backtesting.research.evaluate import COST_STUB_POINTS, EVAL_FEATURES, build_dataset, evaluate_family  # noqa: E402
from backtesting.research.regime import (  # noqa: E402
    REVERSION_CANDIDATES,
    evaluate_gated_strategy,
    evaluate_regime_family,
    regime_profile,
)
from backtesting.research.diagnostics import conditional_grid, redundancy_report, subgroup_report  # noqa: E402
from backtesting.research.quality import direction_breakdown, trend_quality_gradient  # noqa: E402
from backtesting.research.costs import cost_operability  # noqa: E402
from backtesting.research.composite import evaluate_composite_family  # noqa: E402
from backtesting.research.report import build_research_report  # noqa: E402

FRAME_NAMES = [
    "outrights", "calendars", "cracks", "brent_wti",
    "quarterly", "semestral", "yearly", "flies",
]
# Full horizon grid for the predictive tables (forward returns / IC by horizon,
# Bloque F). The heavier regime + gated loops stay on the trade horizons only.
HORIZONS = (20, 25, 30)
TRADE_HORIZONS = (20, 25, 30)
HEADLINE_HORIZON = 25
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw_files.xlsx"
OUT_DIR = ROOT / "docs" / "reports" / "backtest"


def add_fdr(raw: pd.DataFrame) -> pd.DataFrame:
    """Attach a per-row IC p-value and a Benjamini-Hochberg FDR q-value.

    The whole grid (family x feature x horizon) is a large simultaneous search,
    so a raw ``|ic_t| >= 3`` is not by itself significant. We turn each fold-based
    ``ic_t`` into a two-sided p-value (Student-t, ``n_folds - 1`` df) and control
    the false-discovery rate across the grid, so the report can flag only the
    features that survive multiple-testing (``sig_fdr``).
    """
    from scipy import stats

    raw = raw.copy()
    t = raw["ic_t"].to_numpy(dtype=float)
    df = (raw["n_folds"].to_numpy(dtype=float) - 1.0)
    p = np.full(len(raw), np.nan)
    ok = np.isfinite(t) & (df > 0)
    p[ok] = 2.0 * stats.t.sf(np.abs(t[ok]), df[ok])
    raw["ic_p"] = p

    q = np.full(len(raw), np.nan)
    idx = np.where(np.isfinite(p))[0]
    if len(idx):
        m = len(idx)
        order = idx[np.argsort(p[idx])]
        ranked = p[order]
        bh = ranked * m / (np.arange(1, m + 1))
        bh = np.minimum.accumulate(bh[::-1])[::-1]  # enforce monotone non-decreasing q
        q_sorted = np.clip(bh, 0.0, 1.0)
        q[order] = q_sorted
    raw["ic_q"] = q
    raw["sig_fdr"] = (raw["ic_q"] < 0.10) & raw["ic_q"].notna()
    return raw


def load_frames() -> dict[str, pd.DataFrame]:
    parquet = {name: PROCESSED / f"{name}.parquet" for name in FRAME_NAMES}
    if all(p.exists() for p in parquet.values()):
        print(f"Reading parquet cache from {PROCESSED}")
        return {name: pd.read_parquet(p) for name, p in parquet.items()}
    print(f"No parquet cache; building frames from {RAW}")
    return build_all(load_raw(RAW))


def main() -> None:
    frames = load_frames()
    results = []
    regime_rows = []
    gated_rows = []
    redundancy_rows = []
    subgroup_rows = []
    regime_profile_rows = []
    regime_transition_rows = []
    direction_rows = []
    quality_rows = []
    cost_rows = []
    composite_rows = []
    grid_rows = []
    for name in FRAME_NAMES:
        frame = frames.get(name)
        if frame is None or frame.empty:
            continue
        print(f"Evaluating {name} ...", flush=True)
        fam_res = evaluate_family(frame, name, horizons=HORIZONS)
        results.append(fam_res)

        # Regime-gated backtest (shares the enriched dataset to avoid rebuilding).
        data = build_dataset(frame, name)
        cost = COST_STUB_POINTS.get(name, 0.0)
        feats = [f for f in EVAL_FEATURES if f in data.columns]

        # WS6 redundancy + WS7 subgroups at the headline horizon.
        headline = fam_res[(fam_res["group"] == "ALL") & (fam_res["horizon"] == HEADLINE_HORIZON)]
        if not headline.empty:
            ic_strength = {
                r["feature"]: abs(r["ic_t"]) if pd.notna(r["ic_t"]) else 0.0
                for _, r in headline.iterrows()
            }
            redundancy_rows.append(
                redundancy_report(data, name, feats, HEADLINE_HORIZON, ic_strength)
            )
            best = headline.reindex(headline["ic_t"].abs().sort_values(ascending=False).index).iloc[0]
            subgroup_rows.append(subgroup_report(data, name, best["feature"], HEADLINE_HORIZON))

            # WS4 regime anatomy + WS5 direction / trend-quality gradient.
            prof, trans = regime_profile(data, name, HEADLINE_HORIZON)
            if not prof.empty:
                regime_profile_rows.append(prof)
                regime_transition_rows.append(trans)
            direction = direction_breakdown(data, best["feature"], HEADLINE_HORIZON, cost=cost)
            if direction is not None:
                direction_rows.append({"family": name, **direction})
            quality_rows.append(trend_quality_gradient(data, name, HEADLINE_HORIZON))

            # Composite reversion signal (all horizons) + conditional 2-indicator grid.
            composite_rows.append(evaluate_composite_family(data, name, HORIZONS, cost=cost))
            rev_head = headline[headline["feature"].isin(REVERSION_CANDIDATES) & (headline["ic_mean"] < 0)]
            pick_src = rev_head if not rev_head.empty else headline
            primary = pick_src.reindex(pick_src["ic_t"].abs().sort_values(ascending=False).index)["feature"].iloc[0]
            confirmator = "level_z" if primary != "level_z" else "z_20"
            grid_rows.append(conditional_grid(data, name, primary, confirmator, HEADLINE_HORIZON))

        # WS8 costs / operability: derive from the confirmed gated strategy at the
        # headline horizon, using the family's calendar span for trades-per-year.
        span_days = (data["date"].max() - data["date"].min()).days
        years = span_days / 365.25 if span_days else 0.0

        for h in TRADE_HORIZONS:
            regime_rows.append(evaluate_regime_family(data, name, feats, h, cost=cost))
            # Confirmed strategy is the headline; the unconfirmed run is kept
            # alongside so the report can show whether the level filter helps.
            gated = evaluate_gated_strategy(data, h, cost=cost, confirm=True)
            base = evaluate_gated_strategy(data, h, cost=cost, confirm=False)
            if gated is None:
                gated = base
            if gated is not None:
                row = {"family": name, **gated}
                if base is not None:
                    row["n_trades_raw"] = base["n_trades"]
                    row["win_rate_raw"] = base["win_rate"]
                    row["sharpe_raw"] = base["sharpe"]
                    row["avg_pnl_raw"] = base["avg_pnl"]
                gated_rows.append(row)
                if h == HEADLINE_HORIZON:
                    co = cost_operability(
                        gated["avg_pnl"], gated["sharpe"], gated["n_trades"],
                        cost, years, h,
                    )
                    cost_rows.append({"family": name, **co})

    raw = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if not raw.empty:
        raw = add_fdr(raw)  # multiple-testing control across the whole grid
    regime_raw = pd.concat(regime_rows, ignore_index=True) if regime_rows else pd.DataFrame()
    gated_raw = pd.DataFrame(gated_rows) if gated_rows else pd.DataFrame()
    redundancy_raw = pd.concat(redundancy_rows, ignore_index=True) if redundancy_rows else pd.DataFrame()
    subgroup_raw = pd.concat(subgroup_rows, ignore_index=True) if subgroup_rows else pd.DataFrame()
    profile_raw = pd.concat(regime_profile_rows, ignore_index=True) if regime_profile_rows else pd.DataFrame()
    transition_raw = pd.concat(regime_transition_rows, ignore_index=True) if regime_transition_rows else pd.DataFrame()
    direction_raw = pd.DataFrame(direction_rows) if direction_rows else pd.DataFrame()
    quality_raw = pd.concat(quality_rows, ignore_index=True) if quality_rows else pd.DataFrame()
    cost_raw = pd.DataFrame(cost_rows) if cost_rows else pd.DataFrame()
    composite_raw = pd.concat(composite_rows, ignore_index=True) if composite_rows else pd.DataFrame()
    grid_raw = pd.concat(grid_rows, ignore_index=True) if grid_rows else pd.DataFrame()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not raw.empty:
        csv_path = OUT_DIR / "research_metrics.csv"
        raw.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}  ({len(raw)} rows)")

        if not regime_raw.empty:
            regime_raw.to_csv(OUT_DIR / "research_regime.csv", index=False)
            print(f"Wrote {OUT_DIR / 'research_regime.csv'}  ({len(regime_raw)} rows)")
        if not gated_raw.empty:
            gated_raw.to_csv(OUT_DIR / "research_gated.csv", index=False)
            print(f"Wrote {OUT_DIR / 'research_gated.csv'}  ({len(gated_raw)} rows)")
        if not redundancy_raw.empty:
            redundancy_raw.to_csv(OUT_DIR / "research_redundancy.csv", index=False)
            print(f"Wrote {OUT_DIR / 'research_redundancy.csv'}  ({len(redundancy_raw)} rows)")
        if not subgroup_raw.empty:
            subgroup_raw.to_csv(OUT_DIR / "research_subgroups.csv", index=False)
            print(f"Wrote {OUT_DIR / 'research_subgroups.csv'}  ({len(subgroup_raw)} rows)")
        for df, fname in [
            (profile_raw, "research_regime_profile.csv"),
            (transition_raw, "research_regime_transitions.csv"),
            (direction_raw, "research_direction.csv"),
            (quality_raw, "research_trend_quality.csv"),
            (cost_raw, "research_costs.csv"),
            (composite_raw, "research_composite.csv"),
            (grid_raw, "research_grid.csv"),
        ]:
            if not df.empty:
                df.to_csv(OUT_DIR / fname, index=False)
                print(f"Wrote {OUT_DIR / fname}  ({len(df)} rows)")

        html_path = OUT_DIR / "research.html"
        html_path.write_text(
            build_research_report(
                raw, horizon=HEADLINE_HORIZON,
                gated_results=gated_raw if not gated_raw.empty else None,
                regime_results=regime_raw if not regime_raw.empty else None,
                redundancy_results=redundancy_raw if not redundancy_raw.empty else None,
                subgroup_results=subgroup_raw if not subgroup_raw.empty else None,
                composite_results=composite_raw if not composite_raw.empty else None,
                grid_results=grid_raw if not grid_raw.empty else None,
                regime_profile=profile_raw if not profile_raw.empty else None,
                regime_transitions=transition_raw if not transition_raw.empty else None,
                direction_results=direction_raw if not direction_raw.empty else None,
                quality_results=quality_raw if not quality_raw.empty else None,
                cost_results=cost_raw if not cost_raw.empty else None,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {html_path}")
    else:
        print("No results produced.")


if __name__ == "__main__":
    main()
