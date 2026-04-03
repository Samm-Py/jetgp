#!/bin/bash
# Benchmark: GDDEGP on Morris 20D with 1, 2, 3 directions per point
# GPyTorch and GEKPLS results are reused from the main benchmark suite.
# Usage: bash run_gddegp_morris_benchmark.sh

set -e

PYTHON=/home/sam/anaconda3/envs/pyoti_2/bin/python

echo "============================================================"
echo "  GDDEGP Morris Benchmark (20D, 1/2/3 dirs per point)"
echo "  $(date)"
echo "============================================================"

cd "$(dirname "$0")"

$PYTHON jetgp_gddegp_morris.py

echo ""
echo "============================================================"
echo "  GDDEGP Morris benchmark complete!"
echo "  $(date)"
echo "============================================================"
