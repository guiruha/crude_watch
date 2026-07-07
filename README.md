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

## Hosting privately (free) with Google login

The app can run on **Streamlit Community Cloud** (free) behind a native Google
sign-in, restricted to an email allowlist. Login is enforced only when an
`[auth]` block exists in secrets, so local runs stay ungated.

1. **Google OAuth client** — in the Google Cloud Console create an *OAuth 2.0
   Client ID* (type: Web application). Add authorised redirect URIs:
   - `http://localhost:8501/oauth2callback` (local testing)
   - `https://<your-app>.streamlit.app/oauth2callback` (deployed)
   Copy the client ID and secret.
2. **Deploy** — push the repo (keep it private), go to
   [share.streamlit.io](https://share.streamlit.io), *New app*, pick the repo and
   set the main file to `app/main.py`.
3. **Secrets** — in the app's *Settings → Secrets*, paste the contents of
   `.streamlit/secrets.toml.example`, filling in `redirect_uri` (the deployed
   URL), a random `cookie_secret`, the Google `client_id`/`client_secret`, and
   the `allowed_emails` list. Only those addresses can open the app.

Data note: the app rebuilds the parquet cache from `data/raw_files.xlsx` on the
first cold start (~30s). Commit `data/processed/*.parquet` if you'd rather skip
that (they're small); otherwise leave them ignored.

## Private always-on hosting (no public URL) via Tailscale

Run the app on a machine **you** control and expose it **only to your private
Tailscale network** — there is no public URL, so only people you invite can
reach it. Access control is handled by Tailscale, so the Google-login gate is
optional here (leave secrets unset to run ungated).

### 1. Get an always-on host (free)

Any machine that stays on works: a home server / Raspberry Pi, or a free cloud
VM such as **Oracle Cloud Always Free** (Ubuntu, Ampere). SSH into it.

### 2. Get the app onto the host

```bash
git clone https://github.com/guiruha/crude_watch.git
cd crude_watch          # ensure data/raw_files.xlsx is present
```

Then run it one of two ways:

**Docker (simplest):**

```bash
docker build -t crudewatch .
docker run -d --restart unless-stopped -p 127.0.0.1:8501:8501 --name crudewatch crudewatch
```

**Native + systemd:**

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python scripts/prebuild_cache.py
sudo cp deploy/crudewatch.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now crudewatch
```

Either way the app listens on `127.0.0.1:8501` (not exposed publicly).

### 3. Publish it to your tailnet only

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
sudo tailscale serve --bg 8501     # tailnet-only HTTPS (NOT `funnel`, which is public)
```

`tailscale serve` prints a private URL like
`https://<machine>.<your-tailnet>.ts.net` reachable **only** by devices in your
tailnet.

### 4. Invite your few people

In the Tailscale admin console, invite them as users (or share this node with
them). They install Tailscale, sign in, and open that URL. Anyone not in your
tailnet cannot reach the app at all — there is no public address to find.

> Belt-and-suspenders: you can still enable the Google-login gate (see above) on
> top of Tailscale by adding the `[auth]` secrets to `.streamlit/secrets.toml`.

### Oracle Cloud Always Free — exact steps

One-time, in a browser:

1. Sign up at [oracle.com/cloud/free](https://www.oracle.com/cloud/free/) (card
   used for identity check only; Always-Free resources never charge).
2. **Create instance** → Image **Ubuntu 22.04**, Shape **VM.Standard.A1.Flex**
   (Ampere — Always Free; 1 OCPU / 6 GB is plenty). Upload your SSH public key.
   No ingress ports are needed (Tailscale tunnels out), so leave the security
   list default.
3. In the [Tailscale admin console](https://login.tailscale.com/admin/dns),
   enable **MagicDNS** and **HTTPS certificates** (required by `tailscale serve`).

Then on the VM:

```bash
ssh ubuntu@<your-vm-public-ip>

# Clone the private repo (use a GitHub token or deploy key for auth)
git clone https://github.com/guiruha/crude_watch.git
cd crude_watch

# One command sets up Docker + Tailscale + serves it privately
bash deploy/setup_oracle.sh
```

Follow the Tailscale login URL it prints. When it finishes it shows your private
`https://<machine>.<tailnet>.ts.net` URL. To ship updates later:

```bash
cd crude_watch && bash deploy/update.sh
```

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
