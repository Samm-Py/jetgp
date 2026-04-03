#!/bin/bash
# OpenMP threshold sweep for get_all_derivs — each config in its own process.

PYTHON="/home/sam/anaconda3/envs/pyoti_2/bin/python"
SCRIPT="omp_sweep_derivs.py"
cd "$(dirname "$0")"

declare -A SIZES
SIZES[m6n2]="50 100 200 300 420 600 800 1000"
SIZES[m10n2]="50 100 200 300 420 600 800"
SIZES[m15n2]="50 100 200 300 420 600"
SIZES[m20n2]="50 100 200 300 420 600"

MODULES="m6n2 m10n2 m15n2 m20n2"

printf "%-7s %5s %10s %9s | %10s %10s | %7s\n" "Mod" "N" "Elems" "TotalMB" "OMP(ms)" "1Thr(ms)" "Ratio"
printf '%s\n' "========================================================================"

for mod in $MODULES; do
    for N in ${SIZES[$mod]}; do
        # Run with all threads
        omp_result=$(unset OMP_NUM_THREADS; $PYTHON $SCRIPT $mod $N 2>/dev/null)
        omp_ms=$(echo "$omp_result" | python3 -c "import sys,json; print(json.load(sys.stdin)['time_ms'])")
        total_mb=$(echo "$omp_result" | python3 -c "import sys,json; print(json.load(sys.stdin)['total_mb'])")
        elems=$(echo "$omp_result" | python3 -c "import sys,json; print(json.load(sys.stdin)['elements'])")

        # Run with 1 thread
        serial_result=$(OMP_NUM_THREADS=1 $PYTHON $SCRIPT $mod $N 2>/dev/null)
        serial_ms=$(echo "$serial_result" | python3 -c "import sys,json; print(json.load(sys.stdin)['time_ms'])")

        # Compute ratio
        ratio=$(python3 -c "o=$omp_ms; s=$serial_ms; print(f'{s/o:.2f}' if o > 0 else 'inf')")
        marker=""
        python3 -c "exit(0 if $serial_ms/$omp_ms > 1.15 else 1)" 2>/dev/null && marker=" <<<"
        python3 -c "exit(0 if $serial_ms/$omp_ms < 0.85 else 1)" 2>/dev/null && marker=" !!!"

        printf "%-7s %5d %10d %9s | %10s %10s | %5sx%s\n" \
            "$mod" "$N" "$elems" "$total_mb" "$omp_ms" "$serial_ms" "$ratio" "$marker"
    done
    echo "------------------------------------------------------------------------"
done

echo "Done."
