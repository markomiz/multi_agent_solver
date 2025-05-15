#!/bin/bash

BUILD_DIR="build/relwithdebinfo/bin"
EXE_NAME=${1:-main_example}

if [ ! -f "${BUILD_DIR}/${EXE_NAME}" ]; then
    echo "⚠️  Build not found. Building with debug info..."
    ./scripts/build.sh RelWithDebInfo
fi

echo "📊 Running performance analysis using perf..."
perf record -g -- "${BUILD_DIR}/${EXE_NAME}"
perf report
