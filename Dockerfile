### Stage 1: Build the MultiAgentSolver Library and Examples
FROM ubuntu:22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Create a working directory
WORKDIR /workspace

# Copy the source code into the container
COPY . /workspace
COPY cmake /workspace/cmake

# Make scripts executable
RUN chmod +x /workspace/scripts/*.sh

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libeigen3-dev \
    libomp-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Run the dependency setup script
RUN /workspace/scripts/setup_dependencies.sh

# Build the project using the build script
RUN /workspace/scripts/build.sh Release && /workspace/scripts/run.sh


# ### Stage 2: Create a minimal runtime image
# FROM ubuntu:22.04 AS runner

# # Install required runtime dependencies
# RUN apt-get update && apt-get install -y \
#     libeigen3-dev \
#     libomp-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Set working directory
# WORKDIR /opt/multi_agent_solver

# # Copy built binaries and necessary files from builder
# COPY --from=builder /workspace/build/release /opt/multi_agent_solver/bin

# # Copy the run script for convenience
# COPY --from=builder /workspace/scripts/run.sh /opt/multi_agent_solver/run.sh

# # Make the script executable
# RUN chmod +x /opt/multi_agent_solver/run.sh

# # Set the PATH
# ENV PATH="/opt/multi_agent_solver/bin:${PATH}"

# # Default command to run the script
# CMD ["/opt/multi_agent_solver/run.sh", "Release"]
