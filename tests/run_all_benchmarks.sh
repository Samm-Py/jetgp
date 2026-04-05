#!/bin/bash
# Run all benchmark scripts.
# Order: GEKPLS -> GPyTorch -> JetGP (DEGP/WDEGP Morris last)
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "============================================"
echo "  GPyTorch Benchmarks (2000 iters)"
echo "============================================"

for NITER in 2000; do
  echo ""
  echo "--- GPyTorch Borehole (${NITER} iters) ---"
  python gpytorch_borehole.py $NITER

  echo "--- GPyTorch OTL Circuit (${NITER} iters) ---"
  python gpytorch_otl_circuit.py $NITER

  echo "--- GPyTorch Morris (${NITER} iters) ---"
  python gpytorch_morris.py $NITER

  echo "--- GPyTorch Active Subspace (${NITER} iters) ---"
  python gpytorch_active_subspace.py $NITER
done

echo ""
echo "============================================"
echo "  JetGP Benchmarks"
echo "============================================"

echo "--- JetGP Borehole (DEGP) ---"
python jetgp_borehole.py

echo "--- JetGP OTL Circuit (DEGP) ---"
python jetgp_otl_circuit.py

echo "--- JetGP DDEGP Active Subspace ---"
python jetgp_ddegp_active_subspace.py

echo "--- JetGP DEGP Active Subspace ---"
python jetgp_degp_active_subspace.py

echo "--- JetGP GDDEGP Morris ---"
python jetgp_gddegp_morris.py

echo "--- JetGP WDEGP Morris ---"
python jetgp_wdegp_morris.py

echo "--- JetGP DEGP Morris ---"
python jetgp_morris.py

echo ""
echo "============================================"
echo "  All benchmarks complete!"
echo "============================================"
