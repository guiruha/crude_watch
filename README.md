# CrudeWatch

Dashboard privado para leer futuros de crudo desde una vista PM: curva, extremo
histórico, buckets de indicadores, evidencia por cohortes y estructura de WTI.

> Herramienta de soporte discrecional. No ejecuta, no dimensiona y no sustituye
> validación de coste/liquidez real.

## Qué Incluye

- **PM**: resumen accionable para el gestor, con conclusiones documentadas,
  cohortes, buckets individuales y combinaciones de buckets por horizonte.
- **Curva**: WTI strip, calendar matrix y fly grid con explicación integrada en
  cada plot.
- **Detalle**: régimen, dirección, fuerza, nivel, probabilidades y evidencia.
- **Datos cacheados**: `data/processed` y `data/enriched` pueden preconstruirse
  para que el arranque sea rápido.
- **Google login opcional**: solo se activa si existe `.streamlit/secrets.toml`.

## Arranque Local

```bash
uv sync --extra app --group dev
./run.sh
```

Sin `uv`:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```

La app abre en `http://127.0.0.1:8501`.

## Release ZIP

Genera un paquete limpio con código, datos, cachés parquet y guía de arranque:

```bash
./scripts/build_release.sh
```

Salida:

- `release/CrudeWatch-YYYY.MM.DD/`
- `release/CrudeWatch-YYYY.MM.DD.zip`

Ese zip es el paquete que puedes guardar o pasar a otra máquina con Python.

## Docker

```bash
docker compose up -d --build
```

La app queda en:

```text
http://127.0.0.1:8501
```

Parar:

```bash
docker compose down
```

## Windows Ejecutable

El workflow `.github/workflows/build-windows.yml` genera un `CrudeWatch.exe`
offline con PyInstaller.

Desde GitHub:

1. Push a un repo privado.
2. Actions -> **Build Windows executable** -> Run workflow.
3. Descargar el artifact `CrudeWatch-windows`.

En Windows local:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e ".[app,build]"
python scripts\prebuild_cache.py
pyinstaller CrudeWatch.spec --noconfirm
```

Salida:

```text
dist\CrudeWatch.exe
```

## Google Login

Localmente no hay login si no existe `.streamlit/secrets.toml`.

Para activarlo:

```bash
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
```

Rellena:

- `client_id`
- `client_secret`
- `cookie_secret`
- `allowed_emails`
- `redirect_uri`

`.streamlit/secrets.toml` está ignorado por git.

## Estructura

```text
app/                  Streamlit UI
app/core/             cachés, scoring view-models, auth, buckets, preload
app/screens/          PM, Curva y pantallas de detalle
src/crudewatch/       paquete de datos, indicadores, scoring y plots
backtesting/          investigación offline
scripts/              prebuild, research y release
data/raw_files.xlsx   input propietario
data/processed/       cache de familias publicadas
data/enriched/        cache de features/targets/scoring
deploy/               systemd/Tailscale para servidor privado
```

## Comandos Útiles

```bash
uv run python scripts/prebuild_cache.py
uv run pytest
./scripts/build_release.sh
docker compose up -d --build
```

## Nota De Entrega

Si el objetivo es no tocar más la aplicación:

1. Ejecuta `uv run pytest`.
2. Ejecuta `./scripts/build_release.sh`.
3. Guarda el zip de `release/`.
4. Si necesitas Windows, lanza el workflow de GitHub Actions y guarda el `.exe`.
