"""Cost sensitivity and operability (WS8).

Without a measured bid/offer we cannot state a P&L after real costs, but we can
state the two things that actually matter for a go/no-go:

* **Break-even cost** — the round-trip cost (in price points) at which the
  average trade P&L hits zero. Because every trade pays a *constant* cost, this
  is simply the **gross** average P&L per trade. Dividing by the current cost
  *stub* gives a **safety margin**: how many times the placeholder cost the edge
  can absorb before it dies.

* **Cost sensitivity** — the Sharpe and mean P&L at 0x / 1x / 2x / 3x the stub.
  Subtracting a constant does not change the trade-to-trade dispersion, so the
  Sharpe scales linearly with the (shrinking) mean; we derive it in closed form
  from the 1x figures rather than re-simulating.

Plus light **operability**: trades per year and the holding period (= the
horizon), so the reader can judge turnover and rough capacity.
"""
from __future__ import annotations

import numpy as np

DEFAULT_MULTIPLIERS: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)


def cost_operability(
    avg_pnl_1x: float,
    sharpe_1x: float,
    n_trades: int,
    stub_cost: float,
    years: float,
    horizon: int,
    *,
    multipliers: tuple[float, ...] = DEFAULT_MULTIPLIERS,
) -> dict:
    """Break-even, cost sensitivity and turnover from the 1x-cost trade stats.

    ``avg_pnl_1x`` / ``sharpe_1x`` are the mean P&L and Sharpe already computed
    with one unit of ``stub_cost`` subtracted per trade. Returns a flat dict with
    the gross figures, break-even cost, safety margin, per-multiplier P&L/Sharpe,
    and trades-per-year.
    """
    gross = avg_pnl_1x + stub_cost if not np.isnan(avg_pnl_1x) else np.nan
    breakeven = gross  # cost at which net avg P&L == 0
    safety = float(gross / stub_cost) if stub_cost and not np.isnan(gross) else np.nan
    trades_per_year = float(n_trades / years) if years and years > 0 else np.nan

    out = {
        "n_trades": int(n_trades),
        "stub_cost": float(stub_cost),
        "gross_pnl": float(gross) if not np.isnan(gross) else np.nan,
        "breakeven_cost": float(breakeven) if not np.isnan(breakeven) else np.nan,
        "safety_margin": safety,
        "trades_per_year": trades_per_year,
        "holding_days": int(horizon),
    }
    for m in multipliers:
        net = gross - m * stub_cost if not np.isnan(gross) else np.nan
        # Sharpe scales with the mean (dispersion is cost-invariant).
        if not np.isnan(sharpe_1x) and not np.isnan(avg_pnl_1x) and avg_pnl_1x != 0:
            sharpe_m = sharpe_1x * net / avg_pnl_1x
        else:
            sharpe_m = np.nan
        tag = f"{m:g}x"
        out[f"pnl_{tag}"] = float(net) if not np.isnan(net) else np.nan
        out[f"sharpe_{tag}"] = float(sharpe_m) if not np.isnan(sharpe_m) else np.nan
    return out
