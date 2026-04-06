#!/bin/bash
# Run GPyTorch 2nd-order benchmarks (per-replicate processes).
# Runs at multiple iteration counts: 100, 200, 500, 1000, 2000.
set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

N_REPS=5
ITERS="100 200 500 1000 2000"

# func_name  dim  sizes(D, 3D, 5D)
TASKS=(
  "borehole:8:8,24,40"
  "otl_circuit:6:6,18,30"
  "morris:20:20,60,100"
)

for NITER in $ITERS; do
  for ENTRY in "${TASKS[@]}"; do
    FUNC=$(echo "$ENTRY" | cut -d: -f1)
    DIM=$(echo "$ENTRY" | cut -d: -f2)
    SIZES=$(echo "$ENTRY" | cut -d: -f3)

    rm -f "data/results_gpytorch_2nd_order_${FUNC}_${NITER}iter.json"

    echo ""
    echo "============================================"
    echo "  GPyTorch 2nd Order — ${FUNC} (${DIM}D, ${NITER} iter)"
    echo "============================================"

    for N in $(echo "$SIZES" | tr ',' ' '); do
      for REP in $(seq 1 $N_REPS); do
        SEED=$((999 + REP))
        echo "  ${FUNC} n_train=$N, seed=$SEED, rep=$REP"
        python gpytorch/gpytorch_2nd_order.py --single $NITER "$FUNC" $N $SEED $REP
      done
    done
  done
done

echo ""
echo "All GPyTorch 2nd-order benchmarks complete!"
