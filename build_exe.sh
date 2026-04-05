#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────
# build_exe.sh  —  Build the VAD .exe (or binary) locally
# Run from the repo root: bash build_exe.sh
# ────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "================================================"
echo " VAD App — Local Build Script"
echo "================================================"

# 1. Run the MLflow pipeline to get fresh models
echo ""
echo "Step 1/3 — Training models (MLflow pipeline)..."
python mlflow_pipeline/train_branches.py

# 2. Install PyInstaller if not present
echo ""
echo "Step 2/3 — Checking PyInstaller..."
pip install pyinstaller --quiet

# 3. Build the executable
echo ""
echo "Step 3/3 — Building executable..."
pyinstaller vad_app.spec --clean --noconfirm

echo ""
echo "================================================"
echo " Build complete!"

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    EXE="dist/VAD-App.exe"
else
    EXE="dist/VAD-App"
fi

if [ -f "$EXE" ]; then
    SIZE=$(du -sh "$EXE" | cut -f1)
    echo " Output: $EXE  ($SIZE)"
    echo ""
    echo " To run:  ./$EXE"
    echo " Then open: http://localhost:8765/docs"
else
    echo " ❌  Build failed — exe not found."
    exit 1
fi

echo "================================================"
