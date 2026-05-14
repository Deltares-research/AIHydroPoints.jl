#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export XDG_RUNTIME_DIR="${TMPDIR:-/tmp}/quarto-runtime-$$"
mkdir -p "$XDG_RUNTIME_DIR"
pixi run quarto render experiments/leaderboard.qmd
