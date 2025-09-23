#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/build.sh [--clean] [--build-type <TYPE>]

Options:
  --clean              Remove the existing build directory before configuring.
  --build-type <TYPE>  Set the CMake build type (default: Release).
  -h, --help           Show this help message and exit.

Positional compatibility:
  A single positional argument is treated as --build-type for backward
  compatibility with existing CI scripts.

The script automatically wipes the build directory when the cached
CMAKE_SYSTEM_NAME differs from the current host to avoid reusing
artifacts across host/Docker builds.
EOF
}

CLEAN=0
BUILD_TYPE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      CLEAN=1
      shift
      ;;
    --build-type)
      if [[ $# -lt 2 ]]; then
        echo "error: --build-type requires an argument" >&2
        exit 1
      fi
      BUILD_TYPE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -n "$BUILD_TYPE" ]]; then
        echo "error: unexpected argument '$1'" >&2
        usage
        exit 1
      fi
      BUILD_TYPE="$1"
      shift
      ;;
  esac
done

if [[ -z "$BUILD_TYPE" ]]; then
  BUILD_TYPE="Release"
fi

BUILD_DIR="build/${BUILD_TYPE,,}" # lowercase

DEFAULT_PREFIX="${PREFIX:-$HOME/.local}"
ENV_HINT="$DEFAULT_PREFIX/share/multi_agent_solver/environment.sh"
if [[ -z "${CMAKE_PREFIX_PATH:-}" ]]; then
  if [[ -f "$ENV_HINT" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_HINT"
    echo "📚 Loaded CMAKE_PREFIX_PATH hints from $ENV_HINT"
  elif [[ -d "$DEFAULT_PREFIX/lib/cmake" || -d "$DEFAULT_PREFIX/lib64/cmake" || -d "$DEFAULT_PREFIX/share/cmake" ]]; then
    export CMAKE_PREFIX_PATH="$DEFAULT_PREFIX"
    echo "📚 Defaulting CMAKE_PREFIX_PATH to $DEFAULT_PREFIX"
  fi
fi

clean_build_dir=0
if [[ $CLEAN -eq 1 ]]; then
  clean_build_dir=1
elif [[ -d "$BUILD_DIR" && -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  current_system=$(uname -s)
  cached_system=$(grep -E '^CMAKE_SYSTEM_NAME:' "$BUILD_DIR/CMakeCache.txt" | head -n1 | cut -d= -f2 || true)
  # Trim whitespace
  cached_system=${cached_system//[[:space:]]/}
  if [[ -n "$cached_system" && "$cached_system" != "$current_system" ]]; then
    echo "⚠️  Detected host '${current_system}' differs from cached CMAKE_SYSTEM_NAME '${cached_system}'."
    clean_build_dir=1
  fi
fi

if [[ $clean_build_dir -eq 1 && -d "$BUILD_DIR" ]]; then
  echo "🧹 Removing stale build directory '${BUILD_DIR}'"
  rm -rf "$BUILD_DIR"
fi

echo "🛠️  Building MultiAgentSolver in ${BUILD_TYPE} mode..."

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" ../..
cmake --build . --target all -j"$(nproc)"

echo "✅ Build complete. Binaries are in ${BUILD_DIR}"
