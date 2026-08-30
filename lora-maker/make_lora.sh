#!/usr/bin/env bash
# make_lora.sh — LoRA Maker launcher (Gradio web UI)
# Works headless on SimplePod VPS — no display required.
set -e

echo ""
echo " LoRA Maker — WAMU v2 / Wan 2.2 I2V Lightning Subject Trainer"
echo " =============================================================="
echo " Gradio web UI — open http://0.0.0.0:7861 in your browser"
echo ""

# ── Dependency checks ─────────────────────────────────────────────────────────

if ! command -v python3 &>/dev/null; then
    echo " [ERROR] python3 not found."
    echo " Install with:  sudo apt install python3 python3-venv"
    exit 1
fi

if ! command -v git &>/dev/null; then
    echo " [ERROR] git not found."
    echo " Install with:  sudo apt install git"
    exit 1
fi

# Gradio must be installed (it's in requirements.txt so app.py setup covers this)
if ! python3 -c "import gradio" 2>/dev/null; then
    echo " [ERROR] gradio not installed."
    echo " Install with:  pip3 install gradio --break-system-packages"
    exit 1
fi

# ── GPU check (informational, non-fatal) ─────────────────────────────────────

if command -v nvidia-smi &>/dev/null; then
    echo " GPU:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
        | sed 's/^/   /'
    echo ""
else
    echo " [WARNING] nvidia-smi not found — CUDA may not be available."
    echo ""
fi

# ── Launch ────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${LORA_PORT:-7861}"

echo " Launching on http://0.0.0.0:${PORT}"
echo " (app.py uses 7860 by default — LoRA Maker uses 7861 to avoid conflict)"
echo ""
echo " Tip: to expose via SSH tunnel:"
echo "   ssh -L 7861:localhost:7861 root@<your-vps-ip>"
echo ""

exec python3 "${SCRIPT_DIR}/make_lora.py" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --no-browser \
    "$@"
