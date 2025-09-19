#!/bin/bash

set -euo pipefail

BUILD_TYPE="RelWithDebInfo"
BUILD_DIR="build/${BUILD_TYPE,,}"
EXE_NAME=${1:-multi_agent_lqr}
shift || true

if ! command -v perf >/dev/null 2>&1; then
    echo "'perf' is required but not installed."
    exit 1
fi

if [ ! -d "${BUILD_DIR}" ]; then
    echo " Build directory not found. Building with debug info..."
    ./scripts/build.sh "${BUILD_TYPE}"
fi

BINARY_PATH=""
if [ -x "${BUILD_DIR}/${EXE_NAME}" ]; then
    BINARY_PATH="${BUILD_DIR}/${EXE_NAME}"
else
    BINARY_PATH=$(find "${BUILD_DIR}" -maxdepth 3 -type f -name "${EXE_NAME}" -perm -u+x | head -n 1)
fi

if [ -z "${BINARY_PATH}" ] || [ ! -x "${BINARY_PATH}" ]; then
    echo "Executable '${EXE_NAME}' not found in ${BUILD_DIR}." >&2
    echo "   Available executables:" >&2
    find "${BUILD_DIR}" -maxdepth 3 -type f -perm -u+x -printf "   %P\n" | sort >&2
    exit 1
fi

if [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    echo "Build artifacts missing. Rebuilding..."
    ./scripts/build.sh "${BUILD_TYPE}"
fi

echo "Running performance analysis on ${BINARY_PATH} using perf..."
perf record -g -- "${BINARY_PATH}" "$@"
perf report --inline
