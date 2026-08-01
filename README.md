# CrudeWatch

A WTI crude-futures **dataset pipeline** and a lightweight Streamlit **contract
exploration** view.

The project turns a proprietary history of WTI outright prices (2012–2026) into
a set of published contract families — outrights, calendar spreads, cracks,
Brent–WTI and synthetic quarterly/semestral/yearly spreads and flies — and lets
you browse any single contract's price history in one screen.

> Decision-support only. The tool does not execute or size trades.

## Layout

```
src/crudewatch/       # installable, app-facing package (what the Streamlit app runs)
  infra/              # constants (incl. FAMILY_LABELS), raw I/O
  data_preparation/   # build outrights, calendars, cracks, brent-wti, synthetic spreads/flies
  indicators.py       # shared indicator math (used by features + backtesting)
  research/           # feature/dataset pipeline: lifecycle, features, level panel, targets, build_dataset
  scoring/            # live Opportunity Score engine
  plots/              # black & emerald Plotly theme + figures
app/                  # Streamlit UI (theme, opportunity + contract-exploration screens)

backtesting/          # OFFLINE only — kept out of src, run in place from the repo root
  backtest/           # legacy long/flat indicator backtests + HTML reports
  research/           # walk-forward evaluation, regime-gated backtest, strategy simulation + reports
  tests/              # tests for the offline backtesting/research code
scripts/              # run_backtests.py / run_research.py / run_strategy.py / run_bucket_sweep.py (offline report generators)
```

## Published contract families

`build_all` (in `data_preparation/pipeline.py`) produces one dataframe per
family from the raw feed:

| Family | Description |
|--------|-------------|
| `outrights` | Exchange-listed WTI outright closes |
| `calendars` | Consecutive-month calendar spreads |
| `cracks` | HO and RB crack spreads vs WTI |
| `brent_wti` | Brent − WTI inter-commodity premium |
| `quarterly` / `semestral` / `yearly` | Synthetic 3/6/12-month calendar spreads |
| `flies` | Same-month butterflies (`A − 2B + C`) |

The **Contract Exploration** screen picks a family and a contract, then charts
its price history with summary stats and the underlying table.

## Indicator bucket sweep (offline)

Descriptive study of what followed each *joint indicator state*. Buckets all 24
indicators into terciles, forms every combination of up to four of them, and
reports forward price differences (`close[t+h] − open[t+1]`, for
h = 1, 2, 3, 5, 10, 15, 20) per bucket cell.

Indicators are grouped into 7 themes (level, direction, exhaustion, regime,
quality, oscillator, volatility) and a combination takes **at most one indicator
per theme** — so themes are crossed against each other, but `z_20` is never
paired with `z_50`. That is 5,015 combinations and 321,975 cells per family.

Cutoffs are **expanding quantiles over strictly prior dates**, so a row's bucket
never depends on its own date or any later one — "low" means low relative to
what was knowable at the time.

```bash
uv run python scripts/run_bucket_sweep.py --families flies --max-k 2   # quick look
uv run python scripts/run_bucket_sweep.py --jobs 4 --resume            # full run
```

Output lands in `docs/reports/bucket_sweep/`. A full `--max-k 4` run is ~322k
cells per family and takes hours — use `--jobs` and `--resume`. The pooled
`top_cells.csv` step holds every family's full results in memory at once, so a
full 8-family run wants enough RAM (or run families in batches with
`--families`).

> Every input is point-in-time, but cells are picked by inspection, so `t_stat`
> is still selection-biased and thousands of cells clear |t| > 3 by chance.
> Rank candidates with it; confirm them on held-out dates.

## Setup

```bash
uv sync --extra app --group dev     # app dependencies
```

## Run the app

```bash
uv run streamlit run app/main.py
```

First launch builds the processed parquet cache from the raw workbook (~30s);
later launches read the cache.

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
