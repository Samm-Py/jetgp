#!/bin/bash
# Run all JetGP Sparse DEGP benchmarks (per-replicate processes).
set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

N_REPS=5
RHO_VALUES=".5 1.0 1.5"

# func_name  script  dim  sizes(D, 5D, 10D)
TASKS=(
  "borehole:jetgp_degp/jetgp_sparse_borehole.py:8:8,40,80"
  "otl_circuit:jetgp_degp/jetgp_sparse_otl_circuit.py:6:6,30,60"
  "morris:jetgp_degp/jetgp_sparse_morris.py:20:20,100,200"
)

for ENTRY in "${TASKS[@]}"; do
  FUNC=$(echo "$ENTRY" | cut -d: -f1)
  SCRIPT=$(echo "$ENTRY" | cut -d: -f2)
  DIM=$(echo "$ENTRY" | cut -d: -f3)
  SIZES=$(echo "$ENTRY" | cut -d: -f4)

  rm -f "data/results_sparse_degp_${FUNC}*.json"

  echo ""
  echo "============================================"
  echo "  Sparse DEGP — ${FUNC} (${DIM}D)"
  echo "============================================"

  for N in $(echo "$SIZES" | tr ',' ' '); do
    for REP in $(seq 1 $N_REPS); do
      SEED=$((999 + REP))

      for RHO in $RHO_VALUES; do
        echo "  ${FUNC} n_train=$N, rho=$RHO, seed=$SEED, rep=$REP"
        python "$SCRIPT" --single $RHO $N $SEED $REP
      done
    done
  done
done

echo ""
echo "All Sparse DEGP benchmarks complete!"
