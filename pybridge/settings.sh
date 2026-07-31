# pybridge/settings.sh — source this to set up the pybridge pixi environment
#
# Usage:
#   source pybridge/settings.sh
#
# Ensures the pixi-managed environment is installed and puts pybridge/bin
# on PATH, so `python` resolves to the pixi-wrapped interpreter directly.

PYBRIDGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v pixi &>/dev/null; then
    echo "Warning: 'pixi' not found on PATH — install from https://pixi.sh" >&2
elif [ ! -d "$PYBRIDGE_DIR/.pixi/envs/default" ]; then
    (cd "$PYBRIDGE_DIR" && pixi install)
fi

case ":$PATH:" in
    *":$PYBRIDGE_DIR/bin:"*) ;;
    *) export PATH="$PYBRIDGE_DIR/bin:$PATH" ;;
esac

unset PYBRIDGE_DIR
