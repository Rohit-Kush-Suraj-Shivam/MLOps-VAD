"""
VAD App Launcher
----------------
Bundles the FastAPI server into a single executable.
On launch:
  1. Copies bundled model/scaler to a writable temp dir
  2. Starts uvicorn in a background thread
  3. Opens the browser at the Swagger UI
  4. Keeps running until the user closes the terminal / window
"""

import os
import sys
import shutil
import time
import threading
import webbrowser
import tempfile
import signal
from pathlib import Path


# ── Resolve bundle root (works both frozen and plain Python) ──────────────
if getattr(sys, "frozen", False):
    BUNDLE_DIR = Path(sys._MEIPASS)          # PyInstaller temp extraction dir
else:
    BUNDLE_DIR = Path(__file__).resolve().parent

# Writable working dir (next to the .exe, or cwd when running as script)
EXE_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path.cwd()
APP_DIR = EXE_DIR / "vad_data"
APP_DIR.mkdir(exist_ok=True)

# ── Copy assets from bundle → writable dir (first run only) ──────────────
ASSETS = ["model.pkl", "scaler.pkl", "models"]

def sync_assets():
    for asset in ASSETS:
        src = BUNDLE_DIR / asset
        dst = APP_DIR / asset
        if src.exists() and not dst.exists():
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

sync_assets()

# ── Patch sys.path so our api package is importable ──────────────────────
sys.path.insert(0, str(BUNDLE_DIR))
os.chdir(str(APP_DIR))          # FastAPI resolves model paths from cwd

HOST = "127.0.0.1"
PORT = 8765

# ── Start uvicorn in a daemon thread ─────────────────────────────────────
def start_server():
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=HOST,
        port=PORT,
        log_level="warning",
    )

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

# ── Wait until the server is accepting connections ────────────────────────
import socket

def wait_for_server(host, port, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False

print("=" * 50)
print("  VAD — Voice Activity Detection")
print("  Starting server, please wait...")
print("=" * 50)

if wait_for_server(HOST, PORT):
    url = f"http://{HOST}:{PORT}"
    print(f"\n  ✅  Server running at {url}")
    print(f"  📊  API docs:   {url}/docs")
    print(f"  🔍  Model info: {url}/model/info")
    print("\n  Press Ctrl+C to stop.\n")
    webbrowser.open(f"{url}/docs")
else:
    print("  ❌  Server failed to start within 15 s.")
    sys.exit(1)

# ── Keep alive until Ctrl-C ───────────────────────────────────────────────
def _handle_signal(sig, frame):
    print("\n  Shutting down. Goodbye!")
    sys.exit(0)

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

while True:
    time.sleep(1)
