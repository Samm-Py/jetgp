#!/bin/bash
# Master script: run all benchmarks.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  Running ALL benchmarks"
echo "=========================================="

bash "$DIR/run_gek.sh"
bash "$DIR/run_gpytorch.sh"


echo ""
echo "=========================================="
echo "  ALL benchmarks complete!"
echo "=========================================="
