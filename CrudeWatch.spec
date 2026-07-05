# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the CrudeWatch Windows executable.

Build (on Windows, inside the project venv with build deps installed):

    pyinstaller CrudeWatch.spec --noconfirm

Produces ``dist/CrudeWatch.exe`` — a self-contained, offline executable that
launches the Streamlit app and opens the browser. The market-data workbook and
(if present) the pre-built parquet cache are baked in.
"""
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_submodules,
    copy_metadata,
)

PROJECT_ROOT = Path(SPECPATH)

datas = []
binaries = []
hiddenimports = []

# --- Bundle the app source, Streamlit config, and baked data ----------------
# The app script is executed by Streamlit from disk, so ship the tree verbatim.
datas += [(str(PROJECT_ROOT / "app"), "app")]

_raw = PROJECT_ROOT / "data" / "raw_files.xlsx"
if _raw.exists():
    datas += [(str(_raw), "data")]

# Pre-built parquet cache is optional; bake it in when present for instant boot.
_processed = PROJECT_ROOT / "data" / "processed"
if _processed.exists() and any(_processed.glob("*.parquet")):
    datas += [(str(_processed), "data/processed")]

# --- Our own packages -------------------------------------------------------
hiddenimports += collect_submodules("crudewatch")
for pkg in ("core", "screens", "chapters", "theme"):
    hiddenimports += collect_submodules(pkg)

# --- Heavy third-party deps that need full collection -----------------------
for pkg in ("streamlit", "plotly", "pyarrow", "scipy", "statsmodels", "openpyxl", "altair"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Streamlit (and friends) read their own dist metadata at runtime.
for pkg in (
    "streamlit",
    "plotly",
    "pyarrow",
    "pandas",
    "numpy",
    "scipy",
    "statsmodels",
    "openpyxl",
    "altair",
):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

hiddenimports += [
    "crudewatch",
    "core.data",
    "pandas",
    "numpy",
]


a = Analysis(
    ["run_app.py"],
    pathex=[str(PROJECT_ROOT / "app"), str(PROJECT_ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "pytest", "ipykernel", "nbconvert"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="CrudeWatch",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / "app" / "assets" / "icon.ico")
    if (PROJECT_ROOT / "app" / "assets" / "icon.ico").exists()
    else None,
)
