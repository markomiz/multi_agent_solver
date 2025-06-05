#!/bin/bash
set -e

echo "🔄 Setting up dependencies..."
apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libeigen3-dev \
    libomp-dev \
    git

echo "🔽 Cloning OSQP repository..."
git clone https://github.com/osqp/osqp.git /tmp/osqp
cd /tmp/osqp
mkdir build && cd build
cmake -G "Unix Makefiles" ..
cmake --build . --target install
cd /
rm -rf /tmp/osqp

git clone https://github.com/robotology/osqp-eigen.git /tmp/osqp-eigen
cd /tmp/osqp-eigen
mkdir build && cd build
cmake ..
make -j$(nproc)
make install
cd /
rm -rf /tmp/osqp-eigen

echo "✅ OSQP installed successfully."
