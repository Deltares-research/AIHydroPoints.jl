#!/usr/bin/env bash
# check_training_scripts.sh
#
# Smoke-tests all training scripts to verify they run end-to-end without errors.
# Each script already has nepochs=2 (or similar) for a fast test run.
# Scripts whose required data is not present are skipped with a note.
#
# Usage:
#   bash check_training_scripts.sh
#
# Uses `pixi run julia --project` if pixi is available, otherwise falls back
# to plain `julia --project`.
#
# Exit code:
#   0 — all attempted scripts passed
#   1 — one or more scripts failed

cd "$(dirname "$0")"

# Prefer pixi if available, fall back to plain julia
if command -v pixi &>/dev/null; then
    JULIA="pixi run julia --project"
else
    JULIA="julia --project"
fi
echo "Using: $JULIA"

PASS=0
FAIL=0
SKIP=0
RESULTS=()

# ──────────────────────────────────────────────────────────────────────────────
# Helper: run one script, record PASS / FAIL / SKIP
# Usage: run_script <label> <script.jl> [required_path ...]
# ──────────────────────────────────────────────────────────────────────────────
run_script() {
    local label="$1"
    local script="$2"
    shift 2
    local required=("$@")

    echo ""
    echo "━━━ $label ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Check required data paths
    for path in "${required[@]}"; do
        if [ ! -e "$path" ]; then
            echo "  SKIP — required data not found: $path"
            RESULTS+=("SKIP  $label")
            ((SKIP++))
            return
        fi
    done

    local start
    start=$(date +%s)

    if $JULIA "$script"; then
        local elapsed=$(( $(date +%s) - start ))
        echo "  PASS — ${elapsed}s"
        RESULTS+=("PASS  $label  (${elapsed}s)")
        ((PASS++))
    else
        local elapsed=$(( $(date +%s) - start ))
        echo "  FAIL — ${elapsed}s  (exit code $?)"
        RESULTS+=("FAIL  $label  (${elapsed}s)")
        ((FAIL++))
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# Scripts
# ──────────────────────────────────────────────────────────────────────────────

run_script "new_train_tide" "new_train_tide.jl" \
    "test_data/tides_schureman_2011.nc" \
    "test_data/tides_schureman_2012.nc"

run_script "new_train_surge" "new_train_surge.jl" \
    "test_data/surge_schureman_2011.nc" \
    "test_data/era5_wind_stress_2011_testing.jld2" \
    "test_data/surge_schureman_2012.nc" \
    "test_data/era5_wind_stress_2012_validation.jld2"

run_script "train_waves" "train_waves.jl" \
    "data/waves_2021_2024_10to11"

run_script "train_waves_don" "train_waves_don.jl" \
    "data/waves_2021_2024_10to11"

run_script "analyse_tides_schureman" "analyse_tides_schureman.jl" \
    "test_data/DCSM-FM_0_5nm_2010_5stations_his.jld2"

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "  Passed: $PASS   Failed: $FAIL   Skipped: $SKIP"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[ "$FAIL" -eq 0 ]
