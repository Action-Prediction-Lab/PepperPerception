#!/bin/bash
# Run pytest inside the pepper-perception image, mounting the working tree over the image's copy.
#
# Usage: ./tests/run-in-docker.sh [pytest-args...]
#   ./tests/run-in-docker.sh tests/test_yolo_parsing.py -v
#
# The integration guard needs a running service; run it directly:
#   python3 tests/test_service_contract.py

set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

exec docker run --rm \
    -v "$REPO_ROOT:/app" \
    -w /app \
    ghcr.io/action-prediction-lab/pepper-perception:latest \
    bash -c 'python -m pip install --quiet --user pytest && python -m pytest "$@"' -- "$@"
