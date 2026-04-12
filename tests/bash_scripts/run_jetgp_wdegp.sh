#!/bin/bash
# Run JetGP WDEGP benchmarks (per-replicate processes).
# Morris 20D: standard WDEGP + grouped (optimal) WDEGP + DDEGP active subspace.
set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

N_REPS=5

# --- WDEGP Morris (standard) ---
rm -f "data/results_wdegp_morris.json"

echo ""
echo "============================================"
echo "  WDEGP — Morris (20D)"
echo "============================================"

for N in 20 100 200; do
  for REP in $(seq 1 $N_REPS); do
    SEED=$((999 + REP))
    echo "  wdegp morris n_train=$N, seed=$SEED, rep=$REP"
    python jetgp_wdegp/jetgp_wdegp_morris.py --single $N $SEED $REP
    sleep 15
  done
done

# --- WDEGP Grouped Morris (optimal) ---
rm -f "data/results_wdegp_grouped_morris.json"

echo ""
echo "============================================"
echo "  WDEGP Grouped — Morris (20D)"
echo "============================================"

for N in 20 100 200; do
  for REP in $(seq 1 $N_REPS); do
    SEED=$((999 + REP))
    echo "  wdegp grouped morris n_train=$N, seed=$SEED, rep=$REP"
    python jetgp_wdegp/jetgp_wdegp_grouped_morris.py --single $N $SEED $REP
    sleep 15
  done
done
