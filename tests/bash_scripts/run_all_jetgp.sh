#!/bin/bash
# Run all JetGP benchmark scripts (excludes GPyTorch/GEK comparisons).
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  Running ALL JetGP benchmarks"
echo "=========================================="

bash "$DIR/run_jetgp_degp.sh"
bash "$DIR/run_jetgp_sparse_degp.sh"
bash "$DIR/run_jetgp_gddegp.sh"
bash "$DIR/run_jetgp_wdegp.sh"

echo ""
echo "=========================================="
echo "  ALL JetGP benchmarks complete!"
echo "=========================================="
