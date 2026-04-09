#!/bin/bash
# Run all GEKPLS benchmarks (per-replicate processes).
set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

N_REPS=5

# func_name  script  dim  sizes(D, 5D, 10D)

TASKS=(
  "borehole:gek/gekpls_borehole.py:8:8,40,80"
  "otl_circuit:gek/gekpls_otl_circuit.py:6:6,30,60"
  "morris:gek/gekpls_morris.py:20:20,100,200"
  "active_subspace:gek/gekpls_active_subspace.py:10:10,50,100"
)

for ENTRY in "${TASKS[@]}"; do
  FUNC=$(echo "$ENTRY" | cut -d: -f1)
  SCRIPT=$(echo "$ENTRY" | cut -d: -f2)
  DIM=$(echo "$ENTRY" | cut -d: -f3)
  SIZES=$(echo "$ENTRY" | cut -d: -f4)

  rm -f "data/results_gekpls_${FUNC}*.json"

  echo ""
  echo "============================================"
  echo "  GEKPLS — ${FUNC} (${DIM}D)"
  echo "============================================"

  for N in $(echo "$SIZES" | tr ',' ' '); do
    for REP in $(seq 1 $N_REPS); do
      SEED=$((999 + REP))
      echo "  ${FUNC} n_train=$N, seed=$SEED, rep=$REP"
      python "$SCRIPT" --single $N $SEED $REP
    done
  done
done


echo ""
echo "All GEKPLS benchmarks complete!"
