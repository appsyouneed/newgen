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
pip3 install torch torchvision torchaudio --break-system-packages --no-cache-dir

echo "Installing Python dependencies..."
pip3 install -r "$SCRIPT_DIR/requirements.txt" --break-system-packages --ignore-installed typing-extensions --no-cache-dir

echo "Ensuring critical packages..."
pip3 install Pillow "transformers>=4.50.0,<5.0" "huggingface-hub>=0.34.0,<1.0" "numpy<2.1" "diffusers>=0.33.0,<0.38.0" "safetensors>=0.4.0" torchao accelerate --break-system-packages --no-cache-dir --force-reinstall

echo "Pinning gradio to 4.43.0 (supports js= on event handlers; avoids 4.44.1 jinja2 regression)..."
pip3 install "gradio==4.43.0" --break-system-packages --no-cache-dir --force-reinstall

echo "Patching gradio/oauth.py for huggingface_hub >= 0.26 compatibility..."
# huggingface_hub removed HfFolder in 0.26.0. gradio 4.43.0 still imports it at module
# level, crashing the entire process on startup even when OAuth is never used.
# This patch replaces the broken import with a shim class that replicates the only
# method gradio calls (HfFolder.get_token), backed by the replacement API (get_token).
# The shim is only activated when HfFolder is actually absent, so this is a no-op on
# older huggingface_hub installations.
python3 - <<'PYPATCH'
import sys, pathlib

oauth_path = pathlib.Path(
    next(p for p in sys.path if "dist-packages" in p or "site-packages" in p),
    "gradio", "oauth.py"
)
if not oauth_path.exists():
    # try the other path variant
    import site
    for sp in site.getsitepackages():
        candidate = pathlib.Path(sp, "gradio", "oauth.py")
        if candidate.exists():
            oauth_path = candidate
            break

text = oauth_path.read_text()

OLD = "from huggingface_hub import HfFolder, whoami"
NEW = """\
try:
    from huggingface_hub import HfFolder, whoami
except ImportError:
    # huggingface_hub >= 0.26 removed HfFolder. Provide a shim so gradio can
    # still import cleanly. The real HfFolder.get_token() path is only reached
    # when running locally with a mocked OAuth login button, which this app
    # does not use.
    from huggingface_hub import whoami
    try:
        from huggingface_hub import get_token as _get_token
    except ImportError:
        _get_token = lambda: None  # noqa: E731

    class HfFolder:  # noqa: N801
        @staticmethod
        def get_token():
            return _get_token()"""

if OLD in text:
    oauth_path.write_text(text.replace(OLD, NEW, 1))
    print(f"  Patched: {oauth_path}")
else:
    print(f"  Already patched or pattern not found: {oauth_path}")
PYPATCH

echo "Patching gradio_client/utils.py for pydantic v2 bool-schema compatibility..."
# pydantic v2 emits additionalProperties: false (a bool) in JSON schemas.
# gradio_client's _json_schema_to_python_type and get_type both assume schema is
# always a dict, crashing with "argument of type 'bool' is not iterable".
# This adds an isinstance guard so bools are safely returned as "Any".
python3 - <<'PYPATCH'
import sys, pathlib, site

for sp in [*site.getsitepackages(), *sys.path]:
    candidate = pathlib.Path(sp, "gradio_client", "utils.py")
    if candidate.exists():
        text = candidate.read_text()
        if "if not isinstance(schema, dict):" in text:
            print(f"  Already patched: {candidate}")
            break
        OLD = 'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    """Convert the json schema into a python type hint"""\n    if schema == {}:'
        NEW = 'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    """Convert the json schema into a python type hint"""\n    if not isinstance(schema, dict):\n        return "Any"\n    if schema == {}:'
        if OLD in text:
            candidate.write_text(text.replace(OLD, NEW, 1))
            print(f"  Patched: {candidate}")
            break
        # fallback: patch get_type directly
        OLD2 = 'def get_type(schema: dict):\n    if "const" in schema:'
        NEW2 = 'def get_type(schema: dict):\n    if not isinstance(schema, dict):\n        return "unknown"\n    if "const" in schema:'
        if OLD2 in text:
            candidate.write_text(text.replace(OLD2, NEW2, 1))
            print(f"  Patched get_type fallback: {candidate}")
            break
        print(f"  ERROR: no matching pattern in {candidate}")
        break
PYPATCH

echo "Installing SageAttention for accelerated inference..."
pip3 install sageattention --break-system-packages --no-cache-dir 2>/dev/null || {
    echo "Pre-built SageAttention not available for this GPU arch — building from source..."
    pip3 install "git+https://github.com/thu-ml/SageAttention.git" --break-system-packages --no-cache-dir 2>/dev/null || {
        echo "⚠️  SageAttention install failed — will fall back to standard SDPA at runtime."
    }
}

echo "Installing rembg (BiRefNet background removal for Merge Photos)..."
pip3 install "rembg[gpu]" --break-system-packages --no-cache-dir 2>/dev/null || {
    echo "rembg[gpu] failed — trying CPU-only rembg..."
    pip3 install rembg --break-system-packages --no-cache-dir 2>/dev/null || {
        echo "⚠️  rembg install failed — Merge Photos will use original images without background removal."
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
    bash "$SCRIPT_DIR/autorun.sh"
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

bash "$SCRIPT_DIR/autorun.sh"
