"""Per-indicator bucket outcomes for PM evidence."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st

from crudewatch.research import COST_STUB_POINTS
from crudewatch.scoring.score import FEATURE_SNAPSHOT

from core.evidence import MIN_EFFECTIVE_N
from core.scoring import HORIZONS, enriched_frame

BUCKET_LABELS = ("muy bajo", "bajo", "medio", "alto", "muy alto")

INDICATOR_LABELS = {
    "z_10": "Z-score 10",
    "z_20": "Z-score 20",
    "z_50": "Z-score 50",
    "pctb_20_2": "Bollinger 20/2",
    "pctb_10_1_5": "Bollinger 10/1.5",
    "keltner_dist_20": "Keltner 20",
    "rsi_2": "RSI 2",
    "rsi_14": "RSI 14",
    "rsi_div_14": "RSI divergence",
    "macd_div": "MACD divergence",
    "mom_decel_10": "Momentum decel",
    "er_drop_20": "ER drop",
    "vol_ratio": "Vol ratio",
    "slope_20": "Slope 20",
    "macd_hist": "MACD hist",
    "ema_align": "EMA align",
    "mom_5": "Momentum 5",
    "mom_10": "Momentum 10",
    "mom_20": "Momentum 20",
    "er_20": "Efficiency Ratio 20",
    "r2_20": "R2 trend 20",
    "variance_ratio_5": "Variance ratio 5",
    "autocorr_20": "Autocorr 20",
    "dir_persistence_20": "Dir persistence 20",
    "level_z": "Level z",
    "level_pct": "Level percentile",
}

COMBINATION_INDICATORS = (
    "z_20",
    "z_50",
    "pctb_20_2",
    "pctb_10_1_5",
    "keltner_dist_20",
    "rsi_2",
    "rsi_14",
    "rsi_div_14",
    "macd_div",
    "mom_decel_10",
    "er_drop_20",
    "vol_ratio",
    "slope_20",
    "macd_hist",
    "ema_align",
    "mom_10",
    "er_20",
    "r2_20",
    "variance_ratio_5",
    "autocorr_20",
    "dir_persistence_20",
    "level_z",
    "level_pct",
)

TRIPLE_COMBINATION_INDICATORS = (
    "z_20",
    "pctb_20_2",
    "keltner_dist_20",
    "rsi_2",
    "rsi_14",
    "rsi_div_14",
    "macd_div",
    "mom_decel_10",
    "er_drop_20",
    "slope_20",
    "macd_hist",
    "ema_align",
    "er_20",
    "vol_ratio",
    "level_z",
)


def _bucket_edges(values: np.ndarray) -> np.ndarray:
    finite = values[~np.isnan(values)]
    if len(finite) < MIN_EFFECTIVE_N:
        return np.array([], dtype=float)
    return np.unique(np.nanpercentile(finite, [20, 40, 60, 80]))


def _bucket_id(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, values, side="right")


def _non_overlapping(indexed: pd.DataFrame, horizon: int) -> np.ndarray:
    if {"date", "_target_date"}.issubset(indexed.columns):
        ordered = indexed.reset_index(drop=True).reset_index(names="_pos").sort_values(["date", "contract"])
        keep = np.zeros(len(indexed), dtype=bool)
        last_exit: pd.Timestamp | None = None
        for _, row in ordered.iterrows():
            entry = pd.Timestamp(row["date"])
            exit_date = pd.Timestamp(row["_target_date"])
            if pd.isna(entry) or pd.isna(exit_date):
                continue
            if last_exit is None or entry > last_exit:
                keep[int(row["_pos"])] = True
                last_exit = exit_date
        return keep

    keep = np.zeros(len(indexed), dtype=bool)
    last_by_contract: dict[str, int] = {}
    for i, (contract, bar_idx) in enumerate(indexed[["contract", "_bar_idx"]].to_numpy()):
        contract = str(contract)
        bar_idx = int(bar_idx)
        last = last_by_contract.get(contract)
        if last is None or bar_idx - last >= horizon:
            keep[i] = True
            last_by_contract[contract] = bar_idx
    return keep


def _annual_sharpe(samples: np.ndarray, horizon: int) -> float:
    clean = samples[~np.isnan(samples)]
    if len(clean) < 2:
        return float("nan")
    sd = float(np.std(clean, ddof=1))
    if sd <= 0 or sd != sd:
        return float("nan")
    return float((np.mean(clean) / sd) * np.sqrt(252.0 / max(int(horizon), 1)))


def _conservative_sharpe(sharpe: float, n: int, horizon: int, trials: int) -> float:
    if sharpe != sharpe or n < 2:
        return float("nan")
    horizon = max(int(horizon), 1)
    ann = np.sqrt(252.0 / horizon)
    period_sharpe = sharpe / ann
    se_period = np.sqrt((1.0 + 0.5 * period_sharpe * period_sharpe) / max(int(n), 1))
    # Lower confidence bound for the measured no-overlap Sharpe. The number of
    # tested combinations is shown separately in the UI; it is not used as a
    # blanket kill-switch because that made the PM read unusably conservative.
    return float(sharpe - 1.96 * se_period * ann)


def _side_label(side: float) -> str:
    if side > 0:
        return "subidas"
    if side < 0:
        return "bajadas"
    return "plano"


def _bucket_snapshot(data: pd.DataFrame, current: pd.Series, indicators: list[str]) -> dict[str, dict]:
    snapshot: dict[str, dict] = {}
    for indicator in indicators:
        values = data[indicator].to_numpy(dtype=float)
        current_value = float(current.get(indicator, np.nan))
        edges = _bucket_edges(values)
        if current_value != current_value or len(edges) == 0:
            continue
        current_bucket = int(_bucket_id(np.asarray([current_value], dtype=float), edges)[0])
        snapshot[indicator] = {
            "label": INDICATOR_LABELS.get(indicator, indicator),
            "value": current_value,
            "bucket_id": current_bucket,
            "bucket": BUCKET_LABELS[min(current_bucket, len(BUCKET_LABELS) - 1)],
            "all_buckets": _bucket_id(values, edges),
            "finite": ~np.isnan(values),
        }
    return snapshot


@st.cache_data(show_spinner=False, max_entries=1024)
def indicator_bucket_outcomes_cached(
    family: str,
    contract: str,
    as_of: str,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """Historical outcomes for the selected instrument's current indicator buckets."""
    data = enriched_frame(family).sort_values(["contract", "date"]).copy()
    dates = pd.to_datetime(data["date"])
    stamp = pd.Timestamp(as_of)
    data = data.loc[dates <= stamp].copy()
    if data.empty:
        return pd.DataFrame()

    data["_bar_idx"] = data.groupby("contract", sort=False).cumcount()
    date_values = pd.to_datetime(data["date"]).to_numpy()
    date_order = np.argsort(date_values)
    sub = data.loc[data["contract"].astype(str) == str(contract)].sort_values("date")
    if sub.empty:
        return pd.DataFrame()
    current = sub.iloc[-1]

    available = [name for name in FEATURE_SNAPSHOT if name in data.columns]
    snapshot = _bucket_snapshot(data, current, available)
    rows: list[dict] = []
    cost = float(COST_STUB_POINTS.get(family, 0.0))
    target_dates_by_horizon = {
        int(horizon): data.groupby("contract", sort=False)["date"].shift(-int(horizon))
        for horizon in horizons
        if f"fwd_{int(horizon)}" in data.columns
    }
    resolved_by_horizon = {
        horizon: pd.to_datetime(target_dates).to_numpy() <= np.datetime64(stamp)
        for horizon, target_dates in target_dates_by_horizon.items()
    }
    target_values_by_horizon = {
        horizon: pd.to_datetime(target_dates).to_numpy()
        for horizon, target_dates in target_dates_by_horizon.items()
    }
    fwd_by_horizon = {
        horizon: data[f"fwd_{horizon}"].to_numpy(dtype=float)
        for horizon in target_dates_by_horizon
    }

    def effective_indices(match: np.ndarray, horizon: int) -> np.ndarray:
        target_values = target_values_by_horizon[horizon]
        out: list[int] = []
        last_exit: np.datetime64 | None = None
        for idx in date_order[match[date_order]]:
            entry = date_values[idx]
            exit_date = target_values[idx]
            if np.isnat(entry) or np.isnat(exit_date):
                continue
            if last_exit is None or entry > last_exit:
                out.append(int(idx))
                last_exit = exit_date
        return np.asarray(out, dtype=int)

    for indicator, item in snapshot.items():
        base_match = item["finite"] & (item["all_buckets"] == item["bucket_id"])

        for horizon in target_dates_by_horizon:
            target = f"fwd_{horizon}"
            mfe_col = f"mfe_{horizon}"
            mae_col = f"mae_{horizon}"
            resolved = resolved_by_horizon[horizon]
            fwd = fwd_by_horizon[horizon]
            match = base_match & resolved & ~np.isnan(fwd)
            n_raw = int(match.sum())
            idx = effective_indices(match, horizon) if n_raw else np.array([], dtype=int)
            n = int(len(idx))
            record = {
                "indicator": indicator,
                "indicator_label": item["label"],
                "value": item["value"],
                "bucket": item["bucket"],
                "horizon": horizon,
                "n": n,
                "n_raw": n_raw,
                "median_fwd": float("nan"),
                "avg_aligned": float("nan"),
                "avg_long": float("nan"),
                "avg_short": float("nan"),
                "hit_rate": float("nan"),
                "historical_side": "oculto",
                "sharpe_bucket": float("nan"),
                "sharpe_long": float("nan"),
                "sharpe_short": float("nan"),
                "mae_p80": float("nan"),
            }
            if n >= MIN_EFFECTIVE_N:
                fwd_c = fwd[idx]
                median = float(np.nanmedian(fwd_c))
                side = 1.0 if median > 0 else -1.0 if median < 0 else 0.0
                aligned = side * fwd_c - cost if side else np.full(len(fwd_c), np.nan)
                long_net = fwd_c - cost
                short_net = -fwd_c - cost
                record["median_fwd"] = median
                record["avg_aligned"] = float(np.nanmean(aligned)) if side else float("nan")
                record["avg_long"] = float(np.nanmean(long_net))
                record["avg_short"] = float(np.nanmean(short_net))
                record["hit_rate"] = float(np.mean(fwd_c > 0))
                record["historical_side"] = _side_label(side)
                record["sharpe_bucket"] = _annual_sharpe(aligned, horizon) if side else float("nan")
                record["sharpe_long"] = _annual_sharpe(long_net, horizon)
                record["sharpe_short"] = _annual_sharpe(short_net, horizon)
                if side and {mfe_col, mae_col}.issubset(data.columns):
                    mfe_c = data[mfe_col].to_numpy(dtype=float)[idx]
                    mae_c = data[mae_col].to_numpy(dtype=float)[idx]
                    adverse = np.where(side > 0, mae_c, -mfe_c)
                    record["mae_p80"] = float(np.nanpercentile(np.abs(np.minimum(adverse, 0.0)), 80))
            rows.append(record)
    out = pd.DataFrame.from_records(rows)
    if not out.empty:
        trials = out.groupby("horizon")["indicator"].transform("nunique").clip(lower=1)
        out["sharpe_bucket_cons"] = [
            _conservative_sharpe(s, n, h, t)
            for s, n, h, t in zip(out["sharpe_bucket"], out["n"], out["horizon"], trials)
        ]
    return out


@st.cache_data(show_spinner=False, max_entries=512)
def indicator_bucket_combinations_cached(
    family: str,
    contract: str,
    as_of: str,
    horizons: tuple[int, ...] = HORIZONS,
    focus_horizon: int | None = None,
    max_evolution_pairs: int | None = None,
    combo_size: int = 2,
) -> pd.DataFrame:
    """Historical outcomes for current indicator-bucket combinations."""
    data = enriched_frame(family).sort_values(["contract", "date"]).copy()
    dates = pd.to_datetime(data["date"])
    stamp = pd.Timestamp(as_of)
    data = data.loc[dates <= stamp].copy()
    if data.empty:
        return pd.DataFrame()

    data["_bar_idx"] = data.groupby("contract", sort=False).cumcount()
    date_values = pd.to_datetime(data["date"]).to_numpy()
    date_order = np.argsort(date_values)
    sub = data.loc[data["contract"].astype(str) == str(contract)].sort_values("date")
    if sub.empty:
        return pd.DataFrame()
    current = sub.iloc[-1]

    indicator_universe = TRIPLE_COMBINATION_INDICATORS if combo_size >= 3 else COMBINATION_INDICATORS
    available = [name for name in indicator_universe if name in data.columns]
    snapshot = _bucket_snapshot(data, current, available)
    combo_size = int(combo_size)
    if combo_size < 2 or len(snapshot) < combo_size:
        return pd.DataFrame()

    cost = float(COST_STUB_POINTS.get(family, 0.0))
    target_dates_by_horizon = {
        int(horizon): data.groupby("contract", sort=False)["date"].shift(-int(horizon))
        for horizon in horizons
        if f"fwd_{int(horizon)}" in data.columns
    }
    resolved_by_horizon = {
        horizon: pd.to_datetime(target_dates).to_numpy() <= np.datetime64(stamp)
        for horizon, target_dates in target_dates_by_horizon.items()
    }
    target_values_by_horizon = {
        horizon: pd.to_datetime(target_dates).to_numpy()
        for horizon, target_dates in target_dates_by_horizon.items()
    }
    fwd_by_horizon = {
        horizon: data[f"fwd_{horizon}"].to_numpy(dtype=float)
        for horizon in target_dates_by_horizon
    }

    def effective_indices(match: np.ndarray, horizon: int) -> np.ndarray:
        target_values = target_values_by_horizon[horizon]
        out: list[int] = []
        last_exit: np.datetime64 | None = None
        for idx in date_order[match[date_order]]:
            entry = date_values[idx]
            exit_date = target_values[idx]
            if np.isnat(entry) or np.isnat(exit_date):
                continue
            if last_exit is None or entry > last_exit:
                out.append(int(idx))
                last_exit = exit_date
        return np.asarray(out, dtype=int)

    rows: list[dict] = []
    combo_defs = []
    for names in combinations(snapshot, combo_size):
        items = tuple(snapshot[name] for name in names)
        base_match = np.ones(len(data), dtype=bool)
        for item in items:
            base_match &= item["finite"] & (item["all_buckets"] == item["bucket_id"])
        combo_defs.append(
            {
                "key": tuple(names),
                "key_label": "|".join(names),
                "names": tuple(names),
                "items": items,
                "base_match": base_match,
            }
        )

    def build_record(combo: dict, horizon: int) -> dict:
        names = combo["names"]
        items = combo["items"]
        first = items[0]
        second = items[1]
        third = items[2] if len(items) > 2 else None
        target = f"fwd_{horizon}"
        mfe_col = f"mfe_{horizon}"
        mae_col = f"mae_{horizon}"
        resolved = resolved_by_horizon[horizon]
        fwd = fwd_by_horizon[horizon]
        match = combo["base_match"] & resolved & ~np.isnan(fwd)
        n_raw = int(match.sum())
        idx = effective_indices(match, horizon) if n_raw else np.array([], dtype=int)
        n = int(len(idx))
        labels = [str(item["label"]) for item in items]
        buckets = [str(item["bucket"]) for item in items]
        record = {
            "combo_size": len(items),
            "combo_key": combo["key_label"],
            "left": names[0],
            "left_label": first["label"],
            "left_value": first["value"],
            "left_bucket": first["bucket"],
            "right": names[1],
            "right_label": second["label"],
            "right_value": second["value"],
            "right_bucket": second["bucket"],
            "third": names[2] if len(names) > 2 else "",
            "third_label": third["label"] if third else "",
            "third_value": third["value"] if third else float("nan"),
            "third_bucket": third["bucket"] if third else "",
            "combo_label": " + ".join(labels),
            "combo_bucket": " + ".join(buckets),
            "horizon": horizon,
            "n": n,
            "n_raw": n_raw,
            "median_fwd": float("nan"),
            "avg_aligned": float("nan"),
            "avg_long": float("nan"),
            "avg_short": float("nan"),
            "hit_rate": float("nan"),
            "historical_side": "oculto",
            "sharpe_combo": float("nan"),
            "sharpe_long": float("nan"),
            "sharpe_short": float("nan"),
            "mae_p80": float("nan"),
        }
        if n >= MIN_EFFECTIVE_N:
            fwd_c = fwd[idx]
            median = float(np.nanmedian(fwd_c))
            side = 1.0 if median > 0 else -1.0 if median < 0 else 0.0
            aligned = side * fwd_c - cost if side else np.full(len(fwd_c), np.nan)
            long_net = fwd_c - cost
            short_net = -fwd_c - cost
            record["median_fwd"] = median
            record["avg_aligned"] = float(np.nanmean(aligned)) if side else float("nan")
            record["avg_long"] = float(np.nanmean(long_net))
            record["avg_short"] = float(np.nanmean(short_net))
            record["hit_rate"] = float(np.mean(fwd_c > 0))
            record["historical_side"] = _side_label(side)
            record["sharpe_combo"] = _annual_sharpe(aligned, horizon) if side else float("nan")
            record["sharpe_long"] = _annual_sharpe(long_net, horizon)
            record["sharpe_short"] = _annual_sharpe(short_net, horizon)
            if side and {mfe_col, mae_col}.issubset(data.columns):
                mfe_c = data[mfe_col].to_numpy(dtype=float)[idx]
                mae_c = data[mae_col].to_numpy(dtype=float)[idx]
                adverse = np.where(side > 0, mae_c, -mfe_c)
                record["mae_p80"] = float(np.nanpercentile(np.abs(np.minimum(adverse, 0.0)), 80))
        return record

    focus = int(focus_horizon) if focus_horizon is not None else None
    if max_evolution_pairs is not None and focus in target_dates_by_horizon:
        focus_rows = [build_record(combo, focus) for combo in combo_defs]
        rows.extend(focus_rows)
        focus_df = pd.DataFrame.from_records(focus_rows)
        ranked = focus_df[
            (focus_df["n"] >= MIN_EFFECTIVE_N) & focus_df["sharpe_combo"].notna()
        ].copy()
        ranked["_abs_sharpe"] = ranked["sharpe_combo"].abs()
        keep_keys = set(
            str(x)
            for x in ranked.sort_values(["_abs_sharpe", "n"], ascending=[False, False])
            ["combo_key"]
            .head(int(max_evolution_pairs))
            .tolist()
        )
        combo_defs = [combo for combo in combo_defs if combo["key_label"] in keep_keys]
        horizons_to_run = [h for h in target_dates_by_horizon if h != focus]
    else:
        horizons_to_run = list(target_dates_by_horizon)

    for combo in combo_defs:
        for horizon in horizons_to_run:
            rows.append(build_record(combo, horizon))
    out = pd.DataFrame.from_records(rows)
    if not out.empty:
        trials = out.groupby("horizon")["combo_key"].transform("nunique").clip(lower=1)
        out["sharpe_combo_cons"] = [
            _conservative_sharpe(s, n, h, t)
            for s, n, h, t in zip(out["sharpe_combo"], out["n"], out["horizon"], trials)
        ]
    return out


@st.cache_data(show_spinner=False, max_entries=1024)
def bucket_signal_windows_cached(
    family: str,
    contract: str,
    as_of: str,
    horizon: int,
    indicators: tuple[str, ...],
    side_mode: str = "historical",
) -> pd.DataFrame:
    """Entry/exit windows that feed one current bucket or bucket-combo Sharpe."""
    indicators = tuple(str(x) for x in indicators if str(x))
    if not indicators:
        return pd.DataFrame()

    data = enriched_frame(family).sort_values(["contract", "date"]).copy()
    dates = pd.to_datetime(data["date"])
    stamp = pd.Timestamp(as_of)
    data = data.loc[dates <= stamp].copy()
    target = f"fwd_{int(horizon)}"
    if data.empty or target not in data.columns:
        return pd.DataFrame()

    data["_bar_idx"] = data.groupby("contract", sort=False).cumcount()
    if "open" in data.columns:
        data["_entry_price"] = data.groupby("contract", sort=False)["open"].shift(-1)
    else:
        data["_entry_price"] = data["close"]
    date_values = pd.to_datetime(data["date"]).to_numpy()
    date_order = np.argsort(date_values)
    target_dates = data.groupby("contract", sort=False)["date"].shift(-int(horizon))
    target_values = pd.to_datetime(target_dates).to_numpy()
    resolved = target_values <= np.datetime64(stamp)
    fwd = data[target].to_numpy(dtype=float)

    sub = data.loc[data["contract"].astype(str) == str(contract)].sort_values("date")
    if sub.empty:
        return pd.DataFrame()
    snapshot = _bucket_snapshot(data, sub.iloc[-1], [name for name in indicators if name in data.columns])
    if len(snapshot) != len(indicators):
        return pd.DataFrame()

    match = resolved & ~np.isnan(fwd)
    for name in indicators:
        item = snapshot[name]
        match &= item["finite"] & (item["all_buckets"] == item["bucket_id"])
    n_raw = int(match.sum())
    if n_raw == 0:
        return pd.DataFrame()

    idx: list[int] = []
    last_exit: np.datetime64 | None = None
    for pos in date_order[match[date_order]]:
        entry = date_values[pos]
        exit_date = target_values[pos]
        if np.isnat(entry) or np.isnat(exit_date):
            continue
        if last_exit is None or entry > last_exit:
            idx.append(int(pos))
            last_exit = exit_date
    if not idx:
        return pd.DataFrame()

    idx_arr = np.asarray(idx, dtype=int)
    fwd_c = fwd[idx_arr]
    entry_price = data["_entry_price"].to_numpy(dtype=float)[idx_arr]
    exit_price = entry_price + fwd_c
    historical_side = 1.0 if float(np.nanmedian(fwd_c)) > 0 else -1.0 if float(np.nanmedian(fwd_c)) < 0 else 0.0
    side_mode = str(side_mode)
    if side_mode == "long":
        side = 1.0
    elif side_mode == "short":
        side = -1.0
    else:
        side = historical_side

    cost = float(COST_STUB_POINTS.get(family, 0.0))
    net = side * fwd_c - cost if side else np.full(len(idx_arr), np.nan)
    out = data.iloc[idx_arr].copy()
    out = out.assign(
        entry_date=pd.to_datetime(out["date"]).to_numpy(),
        exit_date=pd.to_datetime(target_values[idx_arr]),
        side="Largo" if side > 0 else "Corto" if side < 0 else "Plano",
        historical_side=_side_label(historical_side),
        signal_close=out["close"].to_numpy(dtype=float),
        entry_price=entry_price,
        exit_price=exit_price,
        fwd_points=fwd_c,
        net_points=net,
        result=np.where(net > 0, "positivo", np.where(net < 0, "negativo", "plano")),
        n_raw=n_raw,
        signal_label=" + ".join(snapshot[name]["label"] for name in indicators),
        signal_bucket=" + ".join(snapshot[name]["bucket"] for name in indicators),
    )
    keep_cols = [
        "contract",
        "entry_date",
        "exit_date",
        "side",
        "historical_side",
        "close",
        "signal_close",
        "entry_price",
        "exit_price",
        "fwd_points",
        "net_points",
        "result",
        "signal_label",
        "signal_bucket",
        "n_raw",
    ]
    if "volume" in out.columns:
        keep_cols.insert(6, "volume")
    return out[keep_cols].reset_index(drop=True)
