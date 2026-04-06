#!/bin/bash
# Master script: run all benchmarks.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  Running ALL benchmarks"
echo "=========================================="

bash "$DIR/run_gek.sh"
bash "$DIR/run_gpytorch.sh"
bash "$DIR/run_gpytorch_2nd_order.sh"
bash "$DIR/run_jetgp_degp.sh"
bash "$DIR/run_jetgp_gddegp.sh"
bash "$DIR/run_jetgp_wdegp.sh"
bash "$DIR/run_jetgp_2nd_order.sh"

echo ""
echo "=========================================="
echo "  ALL benchmarks complete!"
echo "=========================================="
