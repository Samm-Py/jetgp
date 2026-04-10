#!/bin/bash
# Rebuild all static modules in parallel
# Usage: bash rebuild_all_static.sh [max_parallel]
#   max_parallel defaults to 4 (each module uses -j4 internally)

MAX_PARALLEL=${1:-4}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/build_logs"
mkdir -p "$LOG_DIR"

# Get all module names from source files
MODULES=$(ls /home/sam/research_head/otilib-master/src/c/static/onumm*.c \
    | sed 's|.*/onumm||;s|\.c||' \
    | sed 's/^/m/')

TOTAL=$(echo "$MODULES" | wc -w)
echo "Rebuilding $TOTAL static modules ($MAX_PARALLEL in parallel)..."
echo ""

RUNNING=0
DONE=0
FAILED=0

for mod in $MODULES; do
    # Wait if we've hit the parallel limit
    while [ "$RUNNING" -ge "$MAX_PARALLEL" ]; do
        wait -n 2>/dev/null
        RUNNING=$((RUNNING - 1))
    done

    echo "Starting: $mod"
    (
        python "$SCRIPT_DIR/build_static.py" "$mod" > "$LOG_DIR/$mod.log" 2>&1
        if [ $? -eq 0 ]; then
            echo "  Done: $mod"
        else
            echo "  FAILED: $mod (see $LOG_DIR/$mod.log)"
        fi
    ) &
    RUNNING=$((RUNNING + 1))
done

# Wait for remaining
wait

echo ""
echo "All builds complete. Check $LOG_DIR for any failures."
