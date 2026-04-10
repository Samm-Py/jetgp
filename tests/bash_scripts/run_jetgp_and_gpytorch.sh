#!/bin/bash
# Run JetGP benchmarks (DEGP, GDDEGP, WDEGP) then GPyTorch capped at 1000 iterations.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  JetGP + GPyTorch benchmarks"
echo "=========================================="

# JetGP first
bash "$DIR/run_jetgp_degp.sh"
bash "$DIR/run_jetgp_gddegp.sh"
bash "$DIR/run_jetgp_wdegp.sh"

# GPyTorch second (1000 iterations only)
bash "$DIR/run_gpytorch_1000.sh"

echo ""
echo "=========================================="
echo "  All benchmarks complete!"
echo "=========================================="
