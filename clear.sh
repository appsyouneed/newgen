#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Clearing temporary and output files ==="

# --- 1. Clear tmp/gradio (everything except vibe_edit_history) ---
GRADIO_DIR="$SCRIPT_DIR/tmp/gradio"

if [ -d "$GRADIO_DIR" ]; then
    echo "Clearing $GRADIO_DIR (preserving vibe_edit_history)..."
    find "$GRADIO_DIR" -mindepth 1 -maxdepth 1 ! -name "vibe_edit_history" -exec rm -rf {} +
    echo "  Done."
else
    echo "  $GRADIO_DIR does not exist, skipping."
fi

# --- 2. Clear outputs/images (contents only, keep the folder) ---
IMAGES_DIR="$SCRIPT_DIR/outputs/images"

if [ -d "$IMAGES_DIR" ]; then
    echo "Clearing $IMAGES_DIR..."
    find "$IMAGES_DIR" -mindepth 1 -delete
    echo "  Done."
else
    echo "  $IMAGES_DIR does not exist, skipping."
fi

# --- 3. Clear outputs/videos (contents only, keep the folder) ---
VIDEOS_DIR="$SCRIPT_DIR/outputs/videos"

if [ -d "$VIDEOS_DIR" ]; then
    echo "Clearing $VIDEOS_DIR..."
    find "$VIDEOS_DIR" -mindepth 1 -delete
    echo "  Done."
else
    echo "  $VIDEOS_DIR does not exist, skipping."
fi

echo "=== Done ==="
