#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"
BUILD_DIR="${PROJECT_ROOT}/build"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Multi-Agent Solver Tests...${NC}"

# Create build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
  mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# Configure properly
if [ ! -f "Makefile" ] && [ ! -f "build.ninja" ]; then
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
fi

# Build tests
cmake --build . --target multi_agent_solver_tests

echo -e "${GREEN}Running Tests...${NC}"

if [ -f "./multi_agent_solver_tests" ]; then
    ./multi_agent_solver_tests
elif [ -f "./tests/multi_agent_solver_tests" ]; then
     ./tests/multi_agent_solver_tests
else
    # Fallback to ctest if we can't find the binary directly (depends on cmake layout)
    ctest -V
fi

echo -e "${GREEN}Tests Completed.${NC}"
