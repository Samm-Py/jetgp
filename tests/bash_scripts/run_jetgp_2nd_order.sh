#!/bin/bash
# Run JetGP 2nd-order benchmarks (per-replicate processes).
set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

N_REPS=5

# func_name  dim  sizes(D, 3D, 5D)
TASKS=(
  "borehole:8:8,24,40"
  "otl_circuit:6:6,18,30"
  "morris:20:20,60,100"
)

for ENTRY in "${TASKS[@]}"; do
  FUNC=$(echo "$ENTRY" | cut -d: -f1)
  DIM=$(echo "$ENTRY" | cut -d: -f2)
  SIZES=$(echo "$ENTRY" | cut -d: -f3)

  rm -f "data/results_jetgp_2nd_order_${FUNC}.json"

  echo ""
  echo "============================================"
  echo "  JetGP 2nd Order — ${FUNC} (${DIM}D)"
  echo "============================================"

  for N in $(echo "$SIZES" | tr ',' ' '); do
    for REP in $(seq 1 $N_REPS); do
      SEED=$((999 + REP))
      echo "  ${FUNC} n_train=$N, seed=$SEED, rep=$REP"
      python jetgp_degp/jetgp_2nd_order.py --single "$FUNC" $N $SEED $REP
      sleep 15
    done
  done
done

echo ""
echo "All JetGP 2nd-order benchmarks complete!"
