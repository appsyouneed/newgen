#!/bin/bash
set -e

echo "=== Newgen Setup (Video + Photo Generator) ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$EUID" -ne 0 ]; then
    exec sudo bash "$0" "$@"
fi

echo "Installing system dependencies..."
if lsof /var/lib/dpkg/lock-frontend > /dev/null 2>&1; then
    lsof -t /var/lib/dpkg/lock-frontend | xargs kill -9 2>/dev/null || true
fi
rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/cache/apt/archives/lock
rm -f /var/lib/dpkg/updates/*
dpkg --configure -a || true
apt-get update
apt-get install -y --fix-missing ffmpeg wget unzip git python3-pip git-lfs

echo "Creating directories..."
mkdir -p /root/newgen/tmp
mkdir -p /root/newgen/mmaudio
mkdir -p /root/newgen/wan22_distill
mkdir -p /root/.cache/huggingface
chmod 1777 /root/newgen/tmp

echo "Installing PyTorch..."
pip3 uninstall -y torch torchvision torchaudio --break-system-packages 2>/dev/null || true
pip3 install torch torchvision --break-system-packages --no-cache-dir

echo "Installing Python dependencies..."
pip3 install -r "$SCRIPT_DIR/requirements.txt" --break-system-packages --ignore-installed typing-extensions --no-cache-dir

echo "Ensuring critical packages..."
pip3 install Pillow "transformers>=4.50.0,<5.0" "huggingface-hub>=0.34.0,<1.0" "numpy<2.1" "diffusers>=0.33.0,<0.38.0" "safetensors>=0.4.0" torchao accelerate --break-system-packages --no-cache-dir --force-reinstall

echo "Installing SageAttention for accelerated inference..."
pip3 install sageattention --break-system-packages --no-cache-dir 2>/dev/null || {
    echo "Pre-built SageAttention not available for this GPU arch — building from source..."
    pip3 install "git+https://github.com/thu-ml/SageAttention.git" --break-system-packages --no-cache-dir 2>/dev/null || {
        echo "⚠️  SageAttention install failed — will fall back to standard SDPA at runtime."
    }
}

echo "Fixing pyOpenSSL..."
python3 -c "from OpenSSL import SSL" 2>/dev/null || pip3 install pyopenssl --break-system-packages

echo "Setting up RIFE interpolation model..."
if [ ! -d "/root/newgen/train_log/model" ] || [ ! -f "/root/newgen/train_log/RIFE_HDv3.py" ]; then
    rm -rf /root/newgen/train_log /root/newgen/__MACOSX /root/newgen/RIFEv4.26_0921.zip
    git clone --depth 1 https://github.com/hzwer/Practical-RIFE.git /tmp/rife
    mkdir -p /root/newgen/train_log
    cp -r /tmp/rife/model /root/newgen/train_log/
    wget -q -P /root/newgen https://huggingface.co/r3gm/RIFE/resolve/main/RIFEv4.26_0921.zip
    unzip -o /root/newgen/RIFEv4.26_0921.zip -d /root/newgen
    rm -rf /tmp/rife /root/newgen/__MACOSX
    echo "RIFE installed."
else
    echo "RIFE already installed, skipping."
fi

echo "Preparing Wan 2.2 distilled weights directory..."
mkdir -p /root/newgen/wan22_distill
echo "Note: the two merged 4-step BF16 experts (~28.6 GB each, ~57 GB total)"
echo "download automatically on first video generation. The base repo's own"
echo "transformer weights are never downloaded."
echo "Plan for roughly 150-200 GB total across Wan, Qwen and MMAudio."

echo "Copying app files to /root/newgen/..."
if [ "$SCRIPT_DIR" != "/root/newgen" ]; then
    cp "$SCRIPT_DIR/app.py" /root/newgen/app.py
    cp "$SCRIPT_DIR/prompts.py" /root/newgen/prompts.py
    cp "$SCRIPT_DIR/requirements.txt" /root/newgen/requirements.txt
else
    echo "Already in /root/newgen — skipping file copy."
fi

# Copy qwenimage module if present
if [ -d "$SCRIPT_DIR/qwenimage" ] && [ "$SCRIPT_DIR/qwenimage" != "/root/newgen/qwenimage" ]; then
    cp -r "$SCRIPT_DIR/qwenimage" /root/newgen/qwenimage
fi

# Copy starters if present
if [ -d "$SCRIPT_DIR/starters" ] && [ "$SCRIPT_DIR/starters" != "/root/newgen/starters" ]; then
    cp -r "$SCRIPT_DIR/starters" /root/newgen/starters
fi

# Check if running in Docker (no systemd)
if [ ! -d /run/systemd/system ]; then
    echo "Docker environment — skipping systemd setup"
    echo ""
    echo "=== Docker Startup Instructions ==="
    echo "Run manually: cd /root/newgen && python3 app.py"
    echo ""
    echo "🎬 Video Generator:"
    echo "   • Background replacement via Qwen, animation via Wan 2.2"
    echo "   • Merged 4-step BF16 checkpoints, no LoRA"
    echo "   • Zero configuration required"
    echo "   • Unrestricted content generation"
    exit 0
fi

echo "Setting up systemd service..."
cp "$SCRIPT_DIR/newgen.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable newgen
systemctl start newgen

echo ""
echo "=== Setup Complete ==="
echo ""
echo "🎬 New Features Available:"
echo "   • Wan 2.2 I2V A14B merged 4-step distill (BF16, no LoRA)"
echo "   • Qwen relocates subjects, Wan animates them"
echo "   • 4-step MoE (high-noise then low-noise expert)"
echo "   • One model resident on GPU at a time"
echo "   • 720p native generation (1280x720)"
echo "   • Zero configuration interface"
echo ""
echo "Service commands:"
echo "  systemctl status newgen"
echo "  systemctl restart newgen"
echo "  tail -f /root/newgen/newgen.log"
echo ""
echo "App: http://0.0.0.0:7860"
echo "📋 Video Tab: Qwen relocate -> Wan 2.2 merged 4-step animate"
echo "🖼️  Image Tab: Unchanged (Qwen Image Edit)"
