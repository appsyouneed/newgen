#!/usr/bin/env bash
# setup_loramaker.sh — Install everything make_lora.py needs
# Safe to run alongside app.py — does NOT touch CUDA, PyTorch, or
# any package that app.py already has installed system-wide.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LORAMAKER_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/LoRAMaker"
VENV="$LORAMAKER_DIR/venv"
TUNER_DIR="$LORAMAKER_DIR/musubi-tuner"
FILTERED_REQS="$LORAMAKER_DIR/musubi_reqs_filtered.txt"

echo ""
echo "=== LoRA Maker Setup ==="
echo "    WAMU v2 / Wan 2.2 I2V Lightning"
echo "    Installs into: $LORAMAKER_DIR"
echo "    CUDA / PyTorch: NOT touched"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────

echo "[1/6] Checking system packages..."

need_apt=()
for pkg in python3-venv python3-tk git; do
    if ! dpkg -l "$pkg" 2>/dev/null | grep -q '^ii'; then
        need_apt+=("$pkg")
    fi
done

if [ ${#need_apt[@]} -gt 0 ]; then
    echo "      Installing: ${need_apt[*]}"
    if [ "$EUID" -ne 0 ]; then
        sudo apt-get install -y "${need_apt[@]}"
    else
        apt-get install -y "${need_apt[@]}"
    fi
else
    echo "      ✓ python3-venv, python3-tk, git already installed"
fi

# ── 2. Confirm CUDA is present (read-only check, no changes) ─────────────────

echo ""
echo "[2/6] Verifying CUDA (read-only)..."

CUDA_OK=false
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
    echo "      ✓ PyTorch CUDA $CUDA_VER — $GPU_NAME"
    CUDA_OK=true
else
    echo "      ⚠  System torch not CUDA-capable (or not installed)."
    echo "      The venv will be built with --system-site-packages and we'll"
    echo "      install cu128 wheels only if still needed after that."
fi

# ── 3. Create / update the venv ───────────────────────────────────────────────

echo ""
echo "[3/6] Setting up Python venv..."
mkdir -p "$LORAMAKER_DIR"

if [ "$CUDA_OK" = true ]; then
    # Inherit system torch (already CUDA-capable) — no extra install needed
    if [ ! -f "$VENV/bin/python" ]; then
        python3 -m venv --system-site-packages "$VENV"
        echo "      ✓ Venv created with --system-site-packages (inherits system PyTorch)"
    else
        # Recreate if it wasn't built with system-site-packages
        if ! grep -q "include-system-site-packages = true" "$VENV/pyvenv.cfg" 2>/dev/null; then
            echo "      Rebuilding venv with --system-site-packages..."
            rm -rf "$VENV"
            python3 -m venv --system-site-packages "$VENV"
        fi
        echo "      ✓ Venv ready (system PyTorch inherited)"
    fi
else
    if [ ! -f "$VENV/bin/python" ]; then
        python3 -m venv "$VENV"
        echo "      ✓ Venv created (isolated — will install cu128 PyTorch)"
    else
        echo "      ✓ Venv already exists"
    fi
fi

PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

# Upgrade pip quietly
"$PIP" install --quiet --upgrade pip

# ── 4. Install Python packages (no torch) ────────────────────────────────────

echo ""
echo "[4/6] Installing Python packages..."

# Core packages make_lora.py needs at the GUI level
"$PIP" install --quiet \
    pillow \
    requests \
    tqdm \
    toml \
    huggingface_hub

echo "      ✓ Core packages installed (pillow, requests, tqdm, toml, huggingface_hub)"

# PyTorch — only if the venv still can't see CUDA after inheriting system packages
if "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    CUDA_VER_VENV=$("$PY" -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "?")
    echo "      ✓ PyTorch CUDA $CUDA_VER_VENV visible in venv — skipping install"
else
    echo "      PyTorch not CUDA-capable in venv — installing cu128 wheels..."
    "$PIP" install --quiet \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128
    echo "      ✓ PyTorch + CUDA 12.8 installed"
fi

# ── 5. Clone / update musubi-tuner ───────────────────────────────────────────

echo ""
echo "[5/6] Setting up musubi-tuner..."

if [ ! -d "$TUNER_DIR/.git" ]; then
    echo "      Cloning musubi-tuner..."
    git clone --depth=1 https://github.com/kohya-ss/musubi-tuner.git "$TUNER_DIR"
    echo "      ✓ musubi-tuner cloned"
else
    echo "      Updating musubi-tuner..."
    git -C "$TUNER_DIR" pull --ff-only 2>/dev/null || echo "      (already up-to-date or local changes present)"
    echo "      ✓ musubi-tuner up to date"
fi

# Install musubi-tuner's requirements — but skip torch lines to avoid
# overwriting the CUDA 12.8 install we just confirmed above.
if [ -f "$TUNER_DIR/requirements.txt" ]; then
    grep -iv '^\s*torch\|^\s*torchvision\|^\s*torchaudio' \
        "$TUNER_DIR/requirements.txt" > "$FILTERED_REQS"
    "$PIP" install --quiet -r "$FILTERED_REQS"
    echo "      ✓ musubi-tuner requirements installed (torch lines skipped)"
fi

# ── 6. Verify the training script exists ─────────────────────────────────────

echo ""
echo "[6/6] Checking training scripts..."

FOUND_SCRIPT=""
for name in train_wan_i2v.py train_wan.py train.py; do
    if [ -f "$TUNER_DIR/$name" ]; then
        FOUND_SCRIPT="$name"
        break
    fi
done

if [ -n "$FOUND_SCRIPT" ]; then
    echo "      ✓ Training script: $FOUND_SCRIPT"
    # Check if --wan_transformer_index is supported (needed for dual-transformer targeting)
    if grep -q "wan_transformer_index" "$TUNER_DIR/$FOUND_SCRIPT" 2>/dev/null; then
        echo "      ✓ --wan_transformer_index supported (dual-expert targeting ready)"
    else
        echo "      ⚠  --wan_transformer_index not found in $FOUND_SCRIPT"
        echo "      The trainer will still run but may target only one transformer."
        echo "      Run: git -C $TUNER_DIR pull   to get the latest musubi-tuner."
    fi
else
    echo "      ⚠  No training script found in $TUNER_DIR"
    echo "      This is unexpected — check the musubi-tuner clone above."
fi

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "=== Setup complete ==="
echo ""
echo "  Venv:         $VENV"
echo "  musubi-tuner: $TUNER_DIR"
echo ""

# Final sanity check
echo "Sanity check:"
"$PY" -c "
import torch, PIL, huggingface_hub, tqdm, toml
cuda = torch.cuda.is_available()
ver  = torch.version.cuda if cuda else 'N/A'
gpu  = torch.cuda.get_device_name(0) if cuda else 'none'
print(f'  torch       {torch.__version__}  CUDA {ver}  ({gpu})')
print(f'  PIL         {PIL.__version__}')
print(f'  hf_hub      {huggingface_hub.__version__}')
print(f'  CUDA ready: {cuda}')
if not cuda:
    print()
    print('  WARNING: CUDA not available — training will not work.')
    print('  Check your NVIDIA driver and CUDA toolkit installation.')
"

echo ""
echo "To launch:"
echo "  python3 $SCRIPT_DIR/make_lora.py"
echo "  — or —"
echo "  bash $SCRIPT_DIR/make_lora.sh"
echo ""
