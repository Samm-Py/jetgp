#!/bin/bash
# Run JetGP GDDEGP benchmarks (per-replicate processes).
# Morris 20D with 1, 2, and 3 directions per point.
set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

N_REPS=5
DIM=20
SIZES="20,100,200"

for N_DIRS in 1 2 3; do
  rm -f "data/results_jetgp_gddegp_${N_DIRS}dirs_morris.json"

  echo ""
  echo "============================================"
  echo "  JetGP GDDEGP — Morris (${DIM}D, ${N_DIRS} dirs)"
  echo "============================================"

  for N in $(echo "$SIZES" | tr ',' ' '); do
    for REP in $(seq 1 $N_REPS); do
      SEED=$((999 + REP))
      echo "  morris n_dirs=$N_DIRS, n_train=$N, seed=$SEED, rep=$REP"
      python jetgp_gddegp/jetgp_gddegp_morris.py --single $N_DIRS $N $SEED $REP
      sleep 15
    done
  done
done

echo ""
echo "All JetGP GDDEGP benchmarks complete!"
