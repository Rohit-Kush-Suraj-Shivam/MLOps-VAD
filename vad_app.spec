# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for VAD App
Build with:   pyinstaller vad_app.spec
Output:       dist/VAD-App.exe  (Windows)
              dist/VAD-App      (Linux/Mac)
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files

ROOT = Path(SPECPATH)   # noqa: F821 — injected by PyInstaller

# ── Collect heavy ML library data files ──────────────────────────────────
datas     = []
hiddenimports = []

for pkg in ["librosa", "sklearn", "scipy", "numba", "llvmlite",
            "soundfile", "audioread", "resampy", "pooch", "joblib",
            "mlflow", "fastapi", "uvicorn", "starlette", "anyio",
            "h11", "httptools", "watchfiles", "websockets"]:
    try:
        d, b, h = collect_all(pkg)
        datas          += d
        hiddenimports  += h
    except Exception:
        pass

# ── Add soundfile binary backends ────────────────────────────────────────
datas += collect_data_files("soundfile")

# ── Bundle our own source files ───────────────────────────────────────────
datas += [
    (str(ROOT / "api"),                "api"),
    (str(ROOT / "model.pkl"),          "."),
    (str(ROOT / "scaler.pkl"),         "."),
    (str(ROOT / "models" / "active"),  "models/active"),
]

# ── Analysis ──────────────────────────────────────────────────────────────
a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports + [
        "api.main", "api.combined", "api.mfcc", "api.energy",
        "sklearn.utils._cython_blas",
        "sklearn.neighbors._partition_nodes",
        "scipy.special.cython_special",
        "uvicorn.logging", "uvicorn.loops", "uvicorn.loops.auto",
        "uvicorn.protocols", "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.lifespan", "uvicorn.lifespan.on",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "jupyter",
              "notebook", "PyQt5", "PyQt6", "wx"],
    noarchive=False,
)

pyz = PYZ(a.pure)   # noqa: F821

exe = EXE(   # noqa: F821
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="VAD-App",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,           # keep console so users see status messages
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,              # replace with path to a .ico file if desired
)
