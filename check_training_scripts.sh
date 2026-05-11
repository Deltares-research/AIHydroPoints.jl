#!/usr/bin/env bash
# check_training_scripts.sh
#
# Smoke-tests all training and inference scripts end-to-end.
#
# Usage:
#   bash check_training_scripts.sh
#
# Exit code:
#   0 — all attempted scripts passed
#   1 — one or more scripts failed

cd "$(dirname "$0")"

PASS=0
FAIL=0
RESULTS=()

# ──────────────────────────────────────────────────────────────────────────────
# Helper: run <label> <cmd...>
# ──────────────────────────────────────────────────────────────────────────────
run() {
    local label="$1"; shift

    echo ""
    echo "━━━ $label ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local start
    start=$(date +%s)

    if "$@"; then
        local elapsed=$(( $(date +%s) - start ))
        echo "  PASS — ${elapsed}s"
        RESULTS+=("PASS  $label  (${elapsed}s)")
        ((PASS++))
    else
        local elapsed=$(( $(date +%s) - start ))
        echo "  FAIL — ${elapsed}s"
        RESULTS+=("FAIL  $label  (${elapsed}s)")
        ((FAIL++))
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# Misc scripts (not TOML-driven)
# ──────────────────────────────────────────────────────────────────────────────

if command -v pixi &>/dev/null; then
    JULIA="pixi run julia --project"
elif command -v julia &>/dev/null; then
    JULIA="julia --project"
else
    echo "Error: neither 'pixi' nor 'julia' found on PATH." >&2
    exit 1
fi

run "analyse_tides_schureman" $JULIA analyse_tides_schureman.jl

# ── train ─────────────────────────────────────────────────────────────────────

run "train LinearSurgeModel"    bin/train examples/LinearSurgeModel.toml
run "train ConvSurgeModel"      bin/train examples/ConvSurgeModel.toml
run "train AttentionSurgeModel" bin/train examples/AttentionSurgeModel.toml
run "train DeepONetTideModel"   bin/train examples/DeepONetTideModel.toml
run "train ProductTideModel"    bin/train examples/ProductTideModel.toml
run "train ConvWaveModel"       bin/train examples/ConvWaveModel.toml
run "train DeepONetWaveModel"   bin/train examples/DeepONetWaveModel.toml
run "train ConvInteractionModel" bin/train examples/ConvInteractionModel.toml

# ── predict (depend on trained models above) ──────────────────────────────────

run "predict ConvSurgeModel" bin/predict examples/predict_ConvSurgeModel.toml

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "  Passed: $PASS   Failed: $FAIL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[ "$FAIL" -eq 0 ]
