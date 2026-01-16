#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/out"
SRC_DIR="$ROOT_DIR/src"

mkdir -p "$OUT_DIR"

# Prefer JAVA_HOME if present
if [[ -n "${JAVA_HOME:-}" && -x "$JAVA_HOME/bin/javac" ]]; then
	JAVAC="$JAVA_HOME/bin/javac"
else
	JAVAC="javac"
fi

if ! "$JAVAC" -version >/dev/null 2>&1; then
	echo "Error: No working JDK found. Please install JDK 21+ and/or set JAVA_HOME." >&2
	echo "On macOS with Homebrew: brew install openjdk@21" >&2
	exit 1
fi


cd "$ROOT_DIR"

# Build classpath with all JARs in lib directory
CLASSPATH="$OUT_DIR"
for jar in lib/*.jar; do
    CLASSPATH="$CLASSPATH:$jar"
done

find "src" -name "*.java" -print0 | xargs -0 "$JAVAC" -cp "$CLASSPATH" -d "$OUT_DIR"

echo "Build complete -> $OUT_DIR"
