#!/bin/bash
# Run all remaining benchmarks serially.
# Usage: bash run_all_benchmarks.sh
# Estimated total time: several hours.

set -e
cd "$(dirname "$0")"

PYTHON="/home/sam/anaconda3/envs/pyoti_2/bin/python"

echo "=========================================="
echo "  Starting all benchmarks"
echo "  $(date)"
echo "=========================================="

# --- GEKPLS ndim (extra_points=DIM) ---
echo ""
echo ">>> GEKPLS ndim benchmarks (OTL Circuit, Borehole, Morris)"
echo "=========================================="
bash run_gekpls_ndim_benchmarks.sh

echo ""
echo ">>> GEKPLS Active Subspace ndim"
echo "=========================================="
$PYTHON gekpls_active_subspace_ndim.py

# --- GPyTorch Active Subspace ---
echo ""
echo ">>> GPyTorch Active Subspace (100 iter)"
echo "=========================================="
$PYTHON gpytorch_active_subspace.py 100

echo ""
echo ">>> GPyTorch Active Subspace (500 iter)"
echo "=========================================="
$PYTHON gpytorch_active_subspace.py 500

echo ""
echo ">>> GPyTorch Active Subspace (1000 iter)"
echo "=========================================="
$PYTHON gpytorch_active_subspace.py 1000

echo ""
echo ">>> GPyTorch Active Subspace (2000 iter)"
echo "=========================================="
$PYTHON gpytorch_active_subspace.py 2000

# --- JetGP (all re-run with OpenMP library) ---
echo ""
echo ">>> JetGP DEGP — OTL Circuit"
echo "=========================================="
$PYTHON jetgp_otl_circuit.py

echo ""
echo ">>> JetGP DEGP — Borehole"
echo "=========================================="
$PYTHON jetgp_borehole.py

echo ""
echo ">>> JetGP DEGP — Morris"
echo "=========================================="
$PYTHON jetgp_morris.py

echo ""
echo ">>> JetGP DDEGP — Active Subspace"
echo "=========================================="
$PYTHON jetgp_ddegp_active_subspace.py

echo ""
echo ">>> JetGP DEGP — Active Subspace"
echo "=========================================="
$PYTHON jetgp_degp_active_subspace.py

echo ""
echo ">>> JetGP GDDEGP — Morris"
echo "=========================================="
$PYTHON jetgp_gddegp_morris.py

echo ""
echo ">>> JetGP WDEGP — Morris"
echo "=========================================="
$PYTHON jetgp_wdegp_morris.py

echo ""
echo "=========================================="
echo "  All benchmarks complete!"
echo "  $(date)"
echo "=========================================="
