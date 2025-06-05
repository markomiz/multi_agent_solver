#!/bin/bash

# Stop and remove any previous Docker containers
if [ $(docker ps -a -q --filter "name=multi_agent_solver") ]; then
    echo "🛑 Stopping existing multi_agent_solver container..."
    docker stop multi_agent_solver
    echo "🗑️  Removing existing container..."
    docker rm multi_agent_solver
fi

# Build the Docker image
echo "🐳 Building Docker image..."
docker build --no-cache -t multi_agent_solver .

# Run the Docker container
echo "🚀 Running Docker container..."
docker run --name multi_agent_solver --rm -it multi_agent_solver

