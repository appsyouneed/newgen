#!/bin/bash
# Copy qwenimage module from picgen if not already present
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PICGEN_DIR="/root/termux/picgen"

if [ ! -d "$SCRIPT_DIR/qwenimage" ]; then
    if [ -d "$PICGEN_DIR/qwenimage" ]; then
        cp -r "$PICGEN_DIR/qwenimage" "$SCRIPT_DIR/qwenimage"
        echo "Copied qwenimage from picgen."
    else
        echo "WARNING: qwenimage not found in picgen. Qwen pipeline may fail to import."
    fi
else
    echo "qwenimage already present."
fi

if [ ! -d "$SCRIPT_DIR/starters" ]; then
    if [ -d "$PICGEN_DIR/starters" ]; then
        cp -r "$PICGEN_DIR/starters" "$SCRIPT_DIR/starters"
        echo "Copied starters from picgen."
    fi
fi

echo "Done. Run setup.sh next."
