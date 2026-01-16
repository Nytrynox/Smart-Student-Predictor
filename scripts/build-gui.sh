#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/out"
LIB_DIR="$ROOT_DIR/lib"

cd "$ROOT_DIR"

# Compile with MongoDB dependencies
echo "Compiling with MongoDB support..."
find src -name "*.java" -print0 | xargs -0 javac -cp "$LIB_DIR/*" -d "$OUT_DIR"

echo "Build complete -> $OUT_DIR"
