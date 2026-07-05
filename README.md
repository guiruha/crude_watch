# CrudeWatch

A WTI crude-futures dataset pipeline and a **Calendar Spread Analytics** dashboard.

The project turns a proprietary history of WTI outright prices (2012–2026) into
fixed-date calendar-spread structures and a three-chapter Streamlit dashboard for
systematic spread trading — technical, seasonal and statistical context in one place.

> Decision-support only. The tool does not execute or size trades; sizing and
> execution remain under human control and the firm's risk policies.

## Layout

```
src/crudewatch/
  infra/            # constants, raw I/O
  data_preparation/ # build outrights, calendars, cracks, brent-wti, synthetic spreads/flies
  plots/            # black & emerald Plotly theme + figures
  analytics/        # the dashboard engine (pure, testable)
    structures.py   # the 23 fixed-date structures across 4 tiers
    data_layer.py   # price matrix, fixed-date series (incl. butterflies A-2B+C), seasonal stack
    indicators/     # trend, momentum, volatility, statistical (QUANT), + signal panel
    regime.py       # contango/backwardation, Bollinger layers, dynamic TP/SL, risk metrics
    seasonality.py  # buffer cone, percentile, monthly heatmap, setup score
    scoring.py      # regime-weighted composite score
app/                # Streamlit UI (theme, screens, chapters)
tests/              # unit + integration tests
```

## Structure universe (23 fixed-date spreads)

| Tier | Count | Members |
|------|-------|---------|
| Monthly consecutive | 11 | Jan-Feb … Nov-Dec |
| Quarterly | 4 | Mar-Jun, Jun-Sep, Sep-Dec, Dec-Mar |
| Semestral | 4 | Mar-Sep, Jun-Dec, Sep-Mar, Dec-Jun |
| Butterfly | 4 | (Mar-Jun)-(Jun-Sep) … (Dec-Mar)-(Mar-Jun) |

Each structure is a linear combination of outright legs, so every series
(including butterflies, `A − 2B + C`, and the year-crossing quarterly/semestral
spreads) is built uniformly from the outright close matrix.

## Dashboard chapters

1. **Technical** — traffic-light signal panel (Trend / Momentum / Volatility /
   Statistical) plus activatable layers: MA ribbon, Bollinger, Supertrend, RSI,
   MACD, Z-Score.
2. **Seasonality** — every vintage aligned on a **days-to-expiry** axis with a
   ±N-day buffer cone, current percentile, Seasonal Setup Score, monthly heatmap
   and fundamental context.
3. **Bollinger** — regime detection (contango → mean-reversion, backwardation →
   breakout; butterflies always mean-revert), ±1/2/3σ bands, dynamic TP/SL, risk
   metrics and a simulated position tracker.
4. **Score** — regime-weighted composite of every vote into a −100…+100 gauge,
   with the vote breakdown and the QUANT mean-reversion diagnostics (Hurst,
   half-life, ADF, variance ratio).

All indicators are **close-based**: a fixed-date spread is a difference of leg
closes and has no meaningful intraday high/low, so ATR/ADX/Supertrend use the
close-to-close true range.

## Setup

```bash
uv sync --extra app --group dev     # app + test dependencies
```

## Run the dashboard

```bash
uv run streamlit run app/main.py
```

First launch builds the processed parquet cache from the raw workbook (~30s);
later launches read the cache.

## Tests

```bash
uv run pytest
```

The integration tests build and score every structure from the processed data;
they skip automatically when the processed parquet is absent.

## Packaging a standalone Windows executable

For recipients who cannot install Python or Docker, CrudeWatch can be bundled
into a single, fully-offline `CrudeWatch.exe`. Double-clicking it starts the app
and opens the browser — the Python runtime, all libraries, and the market data
are baked in.

> Reality check: because it embeds the whole scientific stack (pandas, scipy,
> statsmodels, plotly, streamlit), the exe is large (roughly 300–500 MB). That
> is the price of "no dependencies, fully offline" — there is no way to make it
> small while shipping its own runtime.

PyInstaller cannot cross-compile, so a **Windows** `.exe` must be built on
Windows. Two ways:

### Option A — GitHub Actions (recommended if you have no Windows machine)

Push the repo (keep it **private** — the data workbook is baked in) and run the
**Build Windows executable** workflow (`.github/workflows/build-windows.yml`)
from the Actions tab, or push a `v*` tag. Download `CrudeWatch.exe` from the run
artifacts. Requires `data/raw_files.xlsx` to be committed.

### Option B — Build locally on a Windows machine

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e ".[app,build]"
python scripts\prebuild_cache.py    # bakes data/processed/*.parquet for instant startup
pyinstaller CrudeWatch.spec --noconfirm
```

The result is `dist\CrudeWatch.exe`. That single file is what you share.

Notes:
- `run_app.py` is the launcher; `CrudeWatch.spec` controls the bundle.
- The parquet cache is written next to the exe on first run if it wasn't baked.
- To see logs while debugging a build, set `console=True` in `CrudeWatch.spec`.
