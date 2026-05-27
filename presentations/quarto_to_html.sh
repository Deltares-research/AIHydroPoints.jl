#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Support running from project root or presentation folder
if [[ -f "$SCRIPT_DIR/index.qmd" ]]; then
    PRESENTATION_DIR="$SCRIPT_DIR"
elif [[ -f "$SCRIPT_DIR/coding_agents_presentation/index.qmd" ]]; then
    PRESENTATION_DIR="$SCRIPT_DIR/coding_agents_presentation"
else
    echo "Error: cannot find index.qmd in $SCRIPT_DIR or $SCRIPT_DIR/coding_agents_presentation/" >&2
    exit 1
fi

QMD_FILE="${1:-index.qmd}"

# Find pixi.toml by walking up from the presentation directory
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
    echo "Using quarto via pixi..."
    pixi run --manifest-path "$PIXI_MANIFEST" quarto render "$PRESENTATION_DIR/$QMD_FILE" --to revealjs
elif command -v quarto &>/dev/null; then
    echo "Using quarto directly..."
    quarto render "$PRESENTATION_DIR/$QMD_FILE" --to revealjs
else
    echo "Error: quarto is not available. Install it via 'pixi install' or from https://quarto.org/docs/get-started/" >&2
    exit 1
fi
