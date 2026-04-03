#!/bin/bash
# OpenMP threshold sweep — runs each config in a separate process
# with OMP_NUM_THREADS set BEFORE python starts (the only reliable way).

PYTHON="/home/sam/anaconda3/envs/pyoti_2/bin/python"
SCRIPT="omp_sweep.py"
cd "$(dirname "$0")"

# Module -> matrix sizes (keep memory safe)
declare -A SIZES
SIZES[m6n2]="10 25 50 75 100 150 200 250 300"
SIZES[m10n2]="10 25 50 75 100 150 200 250 300"
SIZES[m15n2]="10 25 50 75 100 150 200 250 300"
SIZES[m20n2]="10 25 50 75 100 150 200 250 300"

MODULES="m6n2 m10n2 m15n2 m20n2"

printf "%-7s %5s %10s %9s | %10s %10s | %7s\n" "Mod" "N" "Elems" "TotalMB" "OMP(ms)" "1Thr(ms)" "Ratio"
printf '%s\n' "========================================================================"

for mod in $MODULES; do
    for N in ${SIZES[$mod]}; do
        # Run with all threads (unset OMP_NUM_THREADS so libgomp uses default)
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
