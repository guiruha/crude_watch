"""CrudeWatch desktop launcher.

Boots the Streamlit app in-process and opens the default browser. This is the
entry point used both for local runs (``python run_app.py``) and for the frozen
Windows executable produced by PyInstaller (see ``CrudeWatch.spec``).

When frozen, PyInstaller unpacks the bundled files into ``sys._MEIPASS``; the
app script and baked data live there. The parquet cache, if it needs to be
built, is written next to the executable so it persists between launches.
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

DEFAULT_PORT = 8501


def resource_base() -> Path:
    """Directory that holds bundled, read-only resources (app code + data)."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def writable_base() -> Path:
    """Directory we may write to (parquet cache). Persists between launches."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _free_port(preferred: int = DEFAULT_PORT) -> int:
    """Return ``preferred`` if free, otherwise an OS-assigned free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _open_browser_later(url: str, delay: float = 2.5) -> None:
    def _open() -> None:
        time.sleep(delay)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()


def main() -> None:
    base = resource_base()
    # Let the app locate baked data / writable cache regardless of CWD.
    os.environ.setdefault("CRUDEWATCH_RESOURCE_DIR", str(base))
    os.environ.setdefault("CRUDEWATCH_CACHE_DIR", str(writable_base()))

    main_script = str(base / "app" / "main.py")
    port = _free_port(DEFAULT_PORT)
    url = f"http://localhost:{port}"

    flag_options = {
        "server_headless": True,
        "server_port": port,
        "server_address": "127.0.0.1",
        "server_fileWatcherType": "none",
        "browser_gatherUsageStats": False,
        "global_developmentMode": False,
    }

    from streamlit.web import bootstrap

    bootstrap.load_config_options(flag_options)
    _open_browser_later(url)
    bootstrap.run(main_script, is_hello=False, args=[], flag_options=flag_options)


if __name__ == "__main__":
    main()
