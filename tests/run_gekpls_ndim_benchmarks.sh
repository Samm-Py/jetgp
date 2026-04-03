#!/bin/bash
# Benchmark: GEKPLS with extra_points=DIM (N_DIM extra points per training point)
# Runs all three test functions: OTL Circuit, Borehole, Morris
# Usage: bash run_gekpls_ndim_benchmarks.sh

set -e

PYTHON=/home/sam/anaconda3/envs/pyoti_2/bin/python

echo "============================================================"
echo "  GEKPLS (extra_points=DIM) Benchmark Suite"
echo "  $(date)"
echo "============================================================"

cd "$(dirname "$0")"

echo ""
echo "--- GEKPLS (extra_points=DIM): OTL Circuit (6D) ---"
$PYTHON gekpls_otl_circuit_ndim.py

echo ""
echo "--- GEKPLS (extra_points=DIM): Borehole (8D) ---"
$PYTHON gekpls_borehole_ndim.py

echo ""
echo "--- GEKPLS (extra_points=DIM): Morris (20D) ---"
$PYTHON gekpls_morris_ndim.py

echo ""
echo "============================================================"
echo "  GEKPLS (extra_points=DIM) benchmarks complete!"
echo "  $(date)"
echo "============================================================"
