#!/bin/bash
# Benchmark: WDEGP on Morris 20D
# DEGP results come from jetgp_morris.py (run_jetgp_benchmarks.sh)
# Usage: bash run_wdegp_morris_benchmark.sh

set -e

PYTHON=/home/sam/anaconda3/envs/pyoti_2/bin/python

echo "============================================================"
echo "  WDEGP Morris Benchmark (20D)"
echo "  $(date)"
echo "============================================================"

cd "$(dirname "$0")"

$PYTHON jetgp_wdegp_morris.py

echo ""
echo "============================================================"
echo "  WDEGP Morris benchmark complete!"
echo "  $(date)"
echo "============================================================"
