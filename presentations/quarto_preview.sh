#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Each presentation lives in its own subfolder, e.g. presentations/overview/index.qmd
PRESENTATION="${1:-overview}"
PRESENTATION_DIR="$SCRIPT_DIR/$PRESENTATION"

if [[ ! -f "$PRESENTATION_DIR/index.qmd" ]]; then
    echo "Error: cannot find $PRESENTATION_DIR/index.qmd" >&2
    echo "Available presentations:" >&2
    find "$SCRIPT_DIR" -mindepth 2 -maxdepth 2 -name index.qmd -exec dirname {} \; | xargs -n1 basename >&2
    exit 1
fi

QMD_FILE="index.qmd"

find_pixi_manifest() {
    local dir="$1"
    while [[ "$dir" != "/" ]]; do
        [[ -f "$dir/pixi.toml" ]] && echo "$dir/pixi.toml" && return 0
        dir="$(dirname "$dir")"
    done
    return 1
}

PIXI_MANIFEST="$(find_pixi_manifest "$PRESENTATION_DIR")" || PIXI_MANIFEST=""

if [[ -n "$PIXI_MANIFEST" ]] && command -v pixi &>/dev/null && pixi run --manifest-path "$PIXI_MANIFEST" quarto --version &>/dev/null 2>&1; then
    echo "Starting quarto preview via pixi (auto-reloads on save)..."
    pixi run --manifest-path "$PIXI_MANIFEST" quarto preview "$PRESENTATION_DIR/$QMD_FILE" --to revealjs
elif command -v quarto &>/dev/null; then
    echo "Starting quarto preview (auto-reloads on save)..."
    quarto preview "$PRESENTATION_DIR/$QMD_FILE" --to revealjs
else
    echo "Error: quarto is not available. Install it via 'pixi install' or from https://quarto.org/docs/get-started/" >&2
    exit 1
fi
