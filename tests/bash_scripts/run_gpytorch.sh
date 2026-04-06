#!/bin/bash
# Run all GPyTorch 1st-order benchmarks (per-replicate processes).
# Runs at multiple iteration counts: 100, 200, 500, 1000, 2000.
set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

N_REPS=5
ITERS="100 200 500 1000 2000"

# func_name  script  dim  sizes(D, 5D, 10D)
TASKS=(
  "borehole:gpytorch/gpytorch_borehole.py:8:8,40,80"
  "otl_circuit:gpytorch/gpytorch_otl_circuit.py:6:6,30,60"
  "morris:gpytorch/gpytorch_morris.py:20:20,100,200"
  "active_subspace_10d:gpytorch/gpytorch_active_subspace.py:10:10,50,100"
)

for NITER in $ITERS; do
  for ENTRY in "${TASKS[@]}"; do
    FUNC=$(echo "$ENTRY" | cut -d: -f1)
    SCRIPT=$(echo "$ENTRY" | cut -d: -f2)
    DIM=$(echo "$ENTRY" | cut -d: -f3)
    SIZES=$(echo "$ENTRY" | cut -d: -f4)

    rm -f "data/results_gpytorch_${FUNC}_${NITER}iter.json"

    echo ""
    echo "============================================"
    echo "  GPyTorch — ${FUNC} (${DIM}D, ${NITER} iter)"
    echo "============================================"

    for N in $(echo "$SIZES" | tr ',' ' '); do
      for REP in $(seq 1 $N_REPS); do
        SEED=$((999 + REP))
        echo "  ${FUNC} n_train=$N, seed=$SEED, rep=$REP"
        python "$SCRIPT" --single $NITER $N $SEED $REP
      done
    done
  done
done

echo ""
echo "All GPyTorch 1st-order benchmarks complete!"
