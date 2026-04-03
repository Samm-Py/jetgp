#!/bin/bash
# Run JetGP benchmark experiments serially
# Usage: bash run_jetgp_benchmarks.sh

set -e  # Exit on error

echo "============================================================"
echo "  Starting JetGP benchmark suite"
echo "  $(date)"
echo "============================================================"

echo ""
echo "--- JetGP DEGP: OTL Circuit ---"
python jetgp_otl_circuit.py

echo ""
echo "--- JetGP DEGP: Borehole ---"
python jetgp_borehole.py

echo ""
echo "--- JetGP DEGP: Active Subspace ---"
python jetgp_degp_active_subspace.py

echo ""
echo "--- JetGP DDEGP: Active Subspace ---"
python jetgp_ddegp_active_subspace.py

echo ""
echo "--- JetGP GDDEGP: Morris ---"
python jetgp_gddegp_morris.py

echo ""
echo "--- JetGP WDEGP: Morris ---"
python jetgp_wdegp_morris.py

echo ""
echo "--- JetGP DEGP: Morris ---"
python jetgp_morris.py

echo ""
echo "============================================================"
echo "  JetGP benchmarks complete!"
echo "  $(date)"
echo "============================================================"
