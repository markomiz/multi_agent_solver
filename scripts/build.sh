#!/bin/bash

BUILD_TYPE=${1:-Release}
BUILD_DIR="build/${BUILD_TYPE,,}" # lowercase

echo "🛠️  Building MultiAgentSolver in ${BUILD_TYPE} mode..."

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ../..
cmake --build . --target all -j$(nproc)

# Generate CPack configuration
cmake --build . --target package

echo "✅ Build complete. Binaries are in ${BUILD_DIR}"
