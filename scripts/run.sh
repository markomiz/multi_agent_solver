#!/bin/bash

MODE=${1:-Release}
BUILD_DIR="build/${MODE,,}/"

echo "🚀 Running MultiAgentSolver examples in ${MODE} mode..."

if [ -d "${BUILD_DIR}" ]; then
    for exe in ${BUILD_DIR}/*; do
        if [ -x "$exe" ] && [ ! -d "$exe" ]; then
            echo "▶  Running ${exe}"
            $exe
        else
            echo " Skipping ${exe} (not executable)"
        fi
    done
else
    echo "❌ Build not found. Run './scripts/build.sh ${MODE}' first."
fi
