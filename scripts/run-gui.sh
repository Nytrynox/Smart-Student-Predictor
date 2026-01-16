#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/out"
LIB_DIR="$ROOT_DIR/lib"

cd "$ROOT_DIR"

# Build first
"$ROOT_DIR/scripts/build-gui.sh"

# Run the GUI with MongoDB support
echo "Starting GUI..."
java -cp "$OUT_DIR:$LIB_DIR/*" com.smartstudent.gui.SimpleGUI
