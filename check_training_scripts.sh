#!/usr/bin/env bash
# check_training_scripts.sh
#
# Smoke-tests all training and inference scripts end-to-end.
# Runs jobs in parallel when GNU parallel is available via pixi.
#
# Usage:
#   bash check_training_scripts.sh
#
# Exit code:
#   0 — all attempted scripts passed
#   1 — one or more scripts failed

cd "$(dirname "$0")"

PASS=0; FAIL=0; RESULTS=()

# ── Julia runner ──────────────────────────────────────────────────────────────
if command -v pixi &>/dev/null; then
    JULIA="pixi run julia --project"
elif command -v julia &>/dev/null; then
    JULIA="julia --project"
else
    echo "Error: neither 'pixi' nor 'julia' found on PATH." >&2; exit 1
fi

# ── Detect GNU parallel via pixi ──────────────────────────────────────────────
if command -v pixi &>/dev/null && pixi run parallel --version &>/dev/null 2>&1; then
    PARALLEL="pixi run parallel"
else
    PARALLEL=""
fi

# ── Sequential runner ─────────────────────────────────────────────────────────
run() {
    local label="$1"; shift
    echo ""; echo "━━━ $label ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    local start; start=$(date +%s)
    if "$@"; then
        local e=$(( $(date +%s) - start ))
        echo "  PASS — ${e}s"; RESULTS+=("PASS  $label  (${e}s)"); ((PASS++))
    else
        local e=$(( $(date +%s) - start ))
        echo "  FAIL — ${e}s"; RESULTS+=("FAIL  $label  (${e}s)"); ((FAIL++))
    fi
}

# ── Group runner — parallel if available, sequential otherwise ────────────────
# Arguments: "Label:::cmd arg1 arg2" ...
run_group() {
    if [[ -n "$PARALLEL" ]]; then
        local joblog results_tmp
        joblog=$(mktemp); results_tmp=$(mktemp)

        printf '%s\n' "$@" | \
            $PARALLEL --colsep ':::' \
                      --jobs 0 \
                      --line-buffer \
                      --tagstring '[{1}] ' \
                      --joblog "$joblog" \
                      {2}

        # Parse joblog (tab-separated), skip header
        while IFS=$'\t' read -r _ _ _ runtime _ _ exitval _ cmd; do
            [[ "$_" == "Seq" ]] && continue  # header row uses first field
            local elapsed="${runtime%.*}"
            for entry in "$@"; do
                if [[ "${entry#*:::}" == "$cmd" ]]; then
                    local label="${entry%%:::*}"
                    if [[ "$exitval" == "0" ]]; then
                        echo "PASS $label (~${elapsed}s)" >> "$results_tmp"
                    else
                        echo "FAIL $label (~${elapsed}s)" >> "$results_tmp"
                    fi
                    break
                fi
            done
        done < "$joblog"

        while read -r status rest; do
            [[ "$status" == "PASS" ]] && { RESULTS+=("PASS  $rest"); ((PASS++)); } \
                                      || { RESULTS+=("FAIL  $rest"); ((FAIL++)); }
        done < "$results_tmp"

        rm -f "$joblog" "$results_tmp"
    else
        for entry in "$@"; do
            local label="${entry%%:::*}" cmd="${entry#*:::}"
            run "$label" $cmd
        done
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# Jobs
# ──────────────────────────────────────────────────────────────────────────────

run "analyse_tides_schureman" $JULIA analyse_tides_schureman.jl

run_group \
    "train LinearSurgeModel:::bin/train examples/LinearSurgeModel.toml" \
    "train ConvSurgeModel:::bin/train examples/ConvSurgeModel.toml" \
    "train AttentionSurgeModel:::bin/train examples/AttentionSurgeModel.toml" \
    "train DeepONetTideModel:::bin/train examples/DeepONetTideModel.toml" \
    "train ProductTideModel:::bin/train examples/ProductTideModel.toml" \
    "train ConvWaveModel:::bin/train examples/ConvWaveModel.toml" \
    "train DeepONetWaveModel:::bin/train examples/DeepONetWaveModel.toml" \
    "train ConvInteractionModel:::bin/train examples/ConvInteractionModel.toml"

run_group \
    "predict ConvSurgeModel:::bin/predict examples/predict_ConvSurgeModel.toml" \
    "predict LinearSurgeModel:::bin/predict examples/predict_LinearSurgeModel.toml"

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo ""
echo "  Passed: $PASS   Failed: $FAIL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[ "$FAIL" -eq 0 ]
