#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/out"

if [ ! -d "$OUT_DIR" ]; then
  echo "Project not built. Building now..."
  "$ROOT_DIR/scripts/build.sh"
fi

# Prefer JAVA_HOME if present
if [[ -n "${JAVA_HOME:-}" && -x "$JAVA_HOME/bin/java" ]]; then
  JAVA_BIN="$JAVA_HOME/bin/java"
else
  JAVA_BIN="java"
fi

if ! "$JAVA_BIN" -version >/dev/null 2>&1; then
  echo "Error: No working Java runtime found. Please install JDK 21+ and/or set JAVA_HOME." >&2
  exit 1
fi

CMD=${1:-help}
shift || true

# Build classpath with all JARs in lib directory
CLASSPATH="$OUT_DIR"
for jar in "$ROOT_DIR"/lib/*.jar; do
    CLASSPATH="$CLASSPATH:$jar"
done

"$JAVA_BIN" -cp "$CLASSPATH" com.smartstudent.cli.App "$CMD" "$@"
