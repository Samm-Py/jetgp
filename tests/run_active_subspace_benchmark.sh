#!/bin/bash
# Benchmark: 10D active subspace function
# Compares DDEGP (2 known directions), DEGP (all 10 partials), GEKPLS, GPyTorch
# Usage: bash run_active_subspace_benchmark.sh

set -e

PYTHON=/home/sam/anaconda3/envs/pyoti_2/bin/python

echo "============================================================"
echo "  Active Subspace Benchmark Suite (10D)"
echo "  $(date)"
echo "============================================================"

cd "$(dirname "$0")"

echo ""
echo "--- JetGP DDEGP (2 active subspace directions) ---"
$PYTHON jetgp_ddegp_active_subspace.py

echo ""
echo "--- JetGP DEGP (all 10 partial derivatives) ---"
$PYTHON jetgp_degp_active_subspace.py

echo ""
echo "--- GEKPLS (all 10 partial derivatives) ---"
$PYTHON gekpls_active_subspace.py

echo ""
echo "--- GEKPLS (extra_points=DIM, all 10 partial derivatives) ---"
$PYTHON gekpls_active_subspace_ndim.py

echo ""
echo "--- GPyTorch DEGP (multiple iteration counts) ---"
for NITER in 100 200 500 1000 2000; do
    echo ""
    echo "  GPyTorch ($NITER iter)"
    $PYTHON gpytorch_active_subspace.py $NITER
done

echo ""
echo "============================================================"
echo "  Active subspace benchmark complete!"
echo "  $(date)"
echo "============================================================"
