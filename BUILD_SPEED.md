# Enable BuildKit for faster parallel builds with better caching
# Add to docker-compose.yml:
#   compose.yaml
# Build:
#   context: .
#   cache_from:
#     - core-barter-system-backend:latest

# Or set environment:
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Run builds:
# docker compose build --parallel #Uses BuildKit cache mounts for pip
# docker compose build --no-cache #Fresh build
# docker compose up -d --build       #Build and start